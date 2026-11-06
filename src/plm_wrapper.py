from utils import *
import math
from functools import lru_cache
import glob


# Long sequences: a PLM only sees max_len residues at once, so a longer sequence is cut into evenly spaced windows of
# max_len residues (tiles) whose neighbours overlap by at least min_overlap. Each tile is embedded on its own and every
# residue's embedding is the weighted mean over the tiles that cover it. A tile's weight fades in over its first
# min_overlap / 2 residues and out over its last min_overlap / 2 (sigmoid ramps, centred min_overlap / 4 from the
# edge), except at the sequence's own ends, so residues near a cut, which see little context, count less

def tile_spans(L, max_len=1022, min_overlap=511):
    # (start, end) of the fewest windows of max_len residues covering [0, L) with adjacent overlaps >= min_overlap
    if L <= max_len:
        return [(0, L)]
    n = math.ceil((L - max_len) / (max_len - min_overlap)) + 1
    # floor keeps every gap between starts <= max_len - min_overlap; the last start is exactly L - max_len
    starts = np.floor(np.linspace(0, L - max_len, n)).astype(int)
    return [(int(s), int(s) + max_len) for s in starts]

@lru_cache(maxsize=1024)
def tile_weights(L, max_len=1022, min_overlap=511, scale=16):
    # spans from tile_spans and their blending weights, (n_tiles, max_len) summing to 1 over the tiles at each residue
    spans = tile_spans(L, max_len, min_overlap)
    t, c = np.arange(max_len), min_overlap / 4
    rise, fall = 1 / (1 + np.exp(-(t - c) / scale)), 1 / (1 + np.exp(-(max_len - 1 - t - c) / scale))
    W = np.stack([(rise if s > 0 else 1.) * (fall if e < L else 1.) * np.ones(max_len) for s, e in spans])
    total = np.zeros(L)
    for (s, e), w in zip(spans, W):
        total[s:e] += w
    return spans, np.stack([w / total[s:e] for (s, e), w in zip(spans, W)])


class PLMWrapper(nn.Module, ABC):
    """
    Frozen protein language model that maps sequences to per-residue embeddings, a list of (L_i, hidden_size)
    tensors without special tokens, tiling sequences longer than max_len (see tile_spans).
    unfreeze_last_n: train the last n transformer layers and the final layer norm.
    Subclasses load the model and tokenizer into self.plm / self.tokenizer and implement layers, final_norm and run.
    """
    CHECKPOINTS = {}    # name -> HF hub id, per subclass
    n_prefix = 0        # special tokens before the first residue (ESM's CLS)

    def cls_index(self, n):
        # position of the sequence-summary token in a tokenized tile of n residues (return_cls reads it there)
        raise NotImplementedError(f'{self.model_name} has no CLS token')

    def __init__(self, name, **kwargs):
        super().__init__()
        self.model_name  = name
        self.max_len     = kwargs.get('max_len', 1022)
        self.min_overlap = kwargs.get('min_overlap', 511)
        self.load(self.CHECKPOINTS[name])
        freeze_module(self.plm)
        unfreeze_last_n = kwargs.get('unfreeze_last_n', 0)
        if unfreeze_last_n > 0:
            freeze_module(self.layers()[-unfreeze_last_n:], unfreeze=True)
            if self.final_norm() is not None:
                freeze_module(self.final_norm(), unfreeze=True)

    @staticmethod
    def subclasses(cls=None):
        # every subclass, subclasses of subclasses included
        for sub in (cls or PLMWrapper).__subclasses__():
            yield sub
            yield from PLMWrapper.subclasses(sub)

    @staticmethod
    def from_name(name, **kwargs):
        for cls in PLMWrapper.subclasses():
            if name in cls.__dict__.get('CHECKPOINTS', {}):
                return cls(name, **kwargs)
        raise KeyError(f'unknown PLM {name}, choose from {[n for c in PLMWrapper.subclasses() for n in c.__dict__.get("CHECKPOINTS", {})]}')

    @abstractmethod
    def load(self, hf_id):
        # set self.plm, self.tokenizer and self.hidden_size
        ...

    @abstractmethod
    def layers(self):
        # the transformer layers, in order (nn.ModuleList)
        ...

    @abstractmethod
    def final_norm(self):
        # the layer norm after the last layer, or None
        ...

    @abstractmethod
    def run(self, input_ids, attn_mask, layers=None):
        # -> embeddings (B, T, D'), logits (B, T, V) or None, both still with special tokens
        # layers None: the final hidden state; a list of hidden-state indices (0 = token embeddings, -1 = final): those
        # states concatenated, D' = len(layers) * hidden_size
        ...

    @property
    def device(self):
        return next(self.plm.parameters()).device

    def preprocess(self, seq):
        # a sequence string -> the tokenizer's input string
        return ' '.join(seq)

    def tokenize(self, seqs):
        seqs = [self.preprocess(s).replace(PAD_CHAR, self.tokenizer.pad_token) for s in seqs]
        tok = self.tokenizer(seqs, add_special_tokens=True, padding='longest', return_tensors='pt')
        return tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device)

    def embed_tiles(self, tiles, layers=None, batch_size=None, return_cls=False):
        # tiles: strings of <= max_len residues -> lists of (len, D') embeddings, (len, V) logits (or Nones) and (D',)
        # CLS embeddings (at cls_index)
        embs, logits, cls = [], [], []
        batch_size = batch_size or len(tiles)
        for a in range(0, len(tiles), batch_size):
            chunk = tiles[a:a + batch_size]
            h, lg = self.run(*self.tokenize(chunk), layers=layers)
            for k, t in enumerate(chunk):
                residues = slice(self.n_prefix, self.n_prefix + len(t))
                embs.append(h[k, residues])
                logits.append(None if lg is None else lg[k, residues])
                if return_cls:
                    cls.append(h[k, self.cls_index(len(t))])
                else:
                    cls.append(None)
        return embs, logits, cls

    def forward(self, seqs, layers=None, batch_size=None, return_cls=False):
        # seqs: list of strings -> per-residue embeddings [(L_i, D')], logits [(L_i, V)] (None for a model without an LM
        # head); batch_size: tiles per PLM call, None = all at once
        # return_cls: also the CLS embeddings (B, D') from the same pass (the token at cls_index: ESM's <cls>, ProtT5's
        #   </s>), the mean over its tiles for a tiled sequence
        tiles, owner = [], []
        for i, s in enumerate(seqs):
            for a, b in tile_spans(len(s), self.max_len, self.min_overlap):
                tiles.append(s[a:b])
                owner.append(i)
        tile_embs, tile_logits, tile_cls = self.embed_tiles(tiles, layers, batch_size, return_cls)

        embs, logits, cls, k = [], [], [], 0
        for i, s in enumerate(seqs):
            n = owner.count(i) if len(s) > self.max_len else 1
            if n == 1:
                embs.append(tile_embs[k])
                logits.append(tile_logits[k])
            else:
                spans, W = tile_weights(len(s), self.max_len, self.min_overlap)
                embs.append(self.blend(tile_embs[k:k + n], spans, W, len(s)))
                logits.append(None if tile_logits[k] is None else self.blend(tile_logits[k:k + n], spans, W, len(s)))
            if return_cls:
                cls.append(torch.stack(tile_cls[k:k + n]).mean(0))
            k += n
        return (embs, logits, torch.stack(cls)) if return_cls else (embs, logits)

    @staticmethod
    def blend(parts, spans, W, L):
        # weighted mean of the tiles' (max_len, D) outputs -> (L, D)
        W = torch.as_tensor(W, dtype=parts[0].dtype, device=parts[0].device)
        out = parts[0].new_zeros(L, parts[0].shape[-1])
        for (s, e), x, w in zip(spans, parts, W):
            out[s:e] += x * w[:, None]
        return out


class ESMWrapper(PLMWrapper):
    CHECKPOINTS = {
        'esm2-8M':   'facebook/esm2_t6_8M_UR50D',
        'esm2-35M':  'facebook/esm2_t12_35M_UR50D',
        'esm2-150M': 'facebook/esm2_t30_150M_UR50D',
        'esm2-650M': 'facebook/esm2_t33_650M_UR50D',
    }
    n_prefix = 1        # CLS

    def cls_index(self, n):
        return 0            # <cls>

    def load(self, hf_id):
        from transformers import AutoTokenizer, EsmForMaskedLM
        self.plm = EsmForMaskedLM.from_pretrained(hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.hidden_size = self.plm.config.hidden_size

    def layers(self):
        return self.plm.esm.encoder.layer

    def final_norm(self):
        return getattr(self.plm.esm.encoder, 'emb_layer_norm_after', None)

    def run(self, input_ids, attn_mask, layers=None):
        # the trunk, then the LM head on its final output (after emb_layer_norm_after, same as hidden_states[-1])
        out = self.plm.esm(input_ids, attention_mask=attn_mask, output_hidden_states=layers is not None, return_dict=True)
        logits = self.plm.lm_head(out.last_hidden_state)
        h = out.last_hidden_state if layers is None else torch.cat([out.hidden_states[l] for l in layers], -1)
        return h, logits


class ProtT5Wrapper(PLMWrapper):
    """ProtT5 encoder only (T5EncoderModel); it has no LM head over residues, so logits are None and no mask token"""
    CHECKPOINTS = {
        'prott5':      'Rostlab/prot_t5_xl_uniref50',
        'prott5-half': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    }

    def load(self, hf_id):
        from transformers import T5Tokenizer, T5EncoderModel
        self.plm = T5EncoderModel.from_pretrained(hf_id, torch_dtype=torch.float32)
        self.tokenizer = T5Tokenizer.from_pretrained(hf_id, do_lower_case=False, legacy=True)
        self.hidden_size = self.plm.config.d_model

    def layers(self):
        return self.plm.encoder.block

    def final_norm(self):
        return self.plm.encoder.final_layer_norm

    def cls_index(self, n):
        # no <cls>: the </s> right after the residues stands in for it. ProtT5 was not pretrained to summarize the
        # sequence there, so it needs training (e.g. ClsAugmentedInfoNCE) to become one
        return n

    def preprocess(self, seq):
        # ProtT5's vocabulary maps the rare residues U, Z, O, B to X
        return ' '.join(seq.translate(str.maketrans('UZOB', 'XXXX')))

    def run(self, input_ids, attn_mask, layers=None):
        out = self.plm(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=layers is not None,
                       return_dict=True)
        h = out.last_hidden_state if layers is None else torch.cat([out.hidden_states[l] for l in layers], -1)
        return h, None


class ProstT5Wrapper(ProtT5Wrapper):
    """
    ProstT5 encoder (ProtT5-XL-U50 fine-tuned on amino acids <-> Foldseek 3Di). Residues go in upper case after the
    <AA2fold> prefix token (lower-case 3Di after <fold2AA> is the other direction, not used here), so a tokenized tile
    is <AA2fold>, residues, </s>. Same architecture as ProtT5, so the rest (layers, cached runs) is ProtT5Wrapper's
    """
    CHECKPOINTS = {'prostt5': 'Rostlab/ProstT5'}
    n_prefix = 1        # <AA2fold>

    def cls_index(self, n):
        # the </s> after the residues, as ProtT5Wrapper's
        return self.n_prefix + n

    def preprocess(self, seq):
        return '<AA2fold> ' + super().preprocess(seq.upper())


# cached inputs to a PLM's first trained layer: PLM_CACHE_DIR/<model>_L<layer>/<name>.f16, a flat (tokens, D) fp16
# array of every cached tile's rows (its tokens as the PLM sees them, special ones included: ProtT5's residues then
# </s>, ESM's <cls>, residues, <eos>) back to back, and <name>_index.pt, {'meta': {...}, 'tiles': [tile strings],
# 'offsets', 'lengths'}; written by experiments/cache_plm.py
PLM_CACHE_DIR = os.environ.get('ARCHER_PLM_CACHE_DIR', os.path.join(REPO_ROOT, 'plm_cache'))


class CachedPLM:
    """
    Mixin for a PLM run from the input of its first trained layer, read from a cache (experiments/cache_plm.py) instead
    of computed. Sequences are looked up by their string in every cache under PLM_CACHE_DIR/<cached_model>_L<first>/ (a
    sequence not in one is an error). The frozen layers ran in eval mode when cached,
    so they apply no dropout. Subclasses set self.first and call load_cache, and implement run from those inputs
    """
    def load_cache(self, cache_dir):
        # tile string -> (array, offset, length) over every cache in cache_dir
        self.cache, self.arrays = {}, []
        for path in sorted(glob.glob(os.path.join(cache_dir, '*_index.pt'))):
            index = torch.load(path)
            assert index['meta']['complete'], f'{path} is not complete'
            rows = sum(index['lengths'])
            arr = np.memmap(path[:-len('_index.pt')] + '.f16', dtype=np.float16, mode='r', shape=(rows, self.hidden_size))
            self.arrays.append(arr)
            for t, o, n in zip(index['tiles'], index['offsets'], index['lengths']):
                self.cache[t] = (arr, o, n)
        assert self.cache, f'no cache in {cache_dir}'
        print(f'{type(self).__name__}: {len(self.cache)} cached sequences from {len(self.arrays)} caches in {cache_dir}')

    def tokenize(self, seqs):
        # the cached layer inputs in place of token ids: (B, T, D) float and the (B, T) attention mask
        missing = [s for s in seqs if s not in self.cache]
        if missing:
            raise KeyError(f'{len(missing)} of {len(seqs)} sequences are not cached, e.g. {missing[0][:50]}')
        rows = [self.cache[s] for s in seqs]
        T = max(n for _, _, n in rows)
        h = np.zeros((len(seqs), T, self.hidden_size), dtype=np.float16)
        mask = torch.zeros(len(seqs), T, dtype=torch.long)
        for b, (arr, o, n) in enumerate(rows):
            h[b, :n] = arr[o:o + n]
            mask[b, :n] = 1
        return torch.from_numpy(h).to(self.device, non_blocking=True).float(), mask.to(self.device)

    def drop_frozen_layers(self, name, n):
        # replace the layers below the last n by placeholders; parameter names stay those of the uncached wrapper, so
        # checkpoints load into either
        blocks = self.layers()
        self.first = len(blocks) - n
        assert 0 < self.first < len(blocks), f'{name} needs 0 < unfreeze_last_n < {len(blocks)}, got {n}'
        for i in range(self.first):
            blocks[i] = nn.Identity()


class CachedProtT5Wrapper(CachedPLM, ProtT5Wrapper):
    """
    ProtT5 from cached inputs of its first trained layer (CachedPLM): only the last unfreeze_last_n layers, the final
    layer norm and layer 0's relative position bias table are kept
    """
    CHECKPOINTS = {'prott5-half-cached': 'Rostlab/prot_t5_xl_half_uniref50-enc'}
    cached_model = 'prott5-half'                # the model whose caches are read

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # layer 0's attention, without its projections, computes the position bias the stack shares between layers
        self.rel_attn = self.layers()[0].layer[0].SelfAttention
        for p in ('q', 'k', 'v', 'o'):
            delattr(self.rel_attn, p)
        self.drop_frozen_layers(name, kwargs.get('unfreeze_last_n', 0))
        self.plm.shared = self.plm.encoder.embed_tokens = None
        self.load_cache(os.path.join(PLM_CACHE_DIR, f'{self.cached_model}_L{self.first}'))

    def run(self, h, attn_mask, layers=None):
        # h: cached inputs to layer self.first -> the final hidden state, as T5Stack computes it from there
        assert layers is None, 'a cached PLM has only the final hidden state'
        T = h.shape[1]
        bias = self.rel_attn.compute_bias(T, T, device=h.device)
        bias = bias + (1.0 - attn_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        enc = self.plm.encoder
        for block in enc.block[self.first:]:
            h = block(h, position_bias=bias)[0]
        return enc.dropout(enc.final_layer_norm(h)), None


class CachedProstT5Wrapper(CachedProtT5Wrapper, ProstT5Wrapper):
    # ProstT5 from cached inputs of its first trained layer: CachedProtT5Wrapper's run, ProstT5Wrapper's <AA2fold> prefix
    CHECKPOINTS = {'prostt5-cached': 'Rostlab/ProstT5'}
    cached_model = 'prostt5'


class CachedESMWrapper(CachedPLM, ESMWrapper):
    """
    ESM-2 from cached inputs of its first trained layer (CachedPLM): only the last unfreeze_last_n layers and
    emb_layer_norm_after are kept (rotary positions live in each layer's attention). No logits: the LM head is dropped
    """
    CHECKPOINTS = {f'{n}-cached': h for n, h in ESMWrapper.CHECKPOINTS.items()}

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.cached_model = name[:-len('-cached')]
        self.drop_frozen_layers(name, kwargs.get('unfreeze_last_n', 0))
        self.plm.esm.embeddings = self.plm.lm_head = None
        self.load_cache(os.path.join(PLM_CACHE_DIR, f'{self.cached_model}_L{self.first}'))

    def run(self, h, attn_mask, layers=None):
        # h: cached inputs to layer self.first -> the final hidden state, as EsmEncoder computes it from there
        assert layers is None, 'a cached PLM has only the final hidden state'
        mask = (1.0 - attn_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        enc = self.plm.esm.encoder
        for layer in enc.layer[self.first:]:
            h = layer(h, mask)[0]
        return enc.emb_layer_norm_after(h), None


if __name__ == '__main__':
    # tiling check: sequences longer than max_len come back with one embedding per residue
    plm = PLMWrapper.from_name('esm2-35M').to('cuda')
    seqs = [random_protein_sequence(random.randint(500, 2000)) for _ in range(10)]
    embs, logits = plm(seqs)
    for s, e, lg in zip(seqs, embs, logits):
        print(len(s), tuple(e.shape), tuple(lg.shape), len(tile_spans(len(s))))
