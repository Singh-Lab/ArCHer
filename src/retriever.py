from utils import *
import math
import json
from torch.optim import AdamW
from data import *
from encoder import *
from infonce import *


def artifact_path():
    # where checkpoints go; read at call time, so setting ARCHER_ARTIFACT_PATH any time before fit() works
    return os.environ.get('ARCHER_ARTIFACT_PATH', os.path.join(REPO_ROOT, 'artifacts'))


class RetrieverTrainer:
    """
    Contrastive training of an encoder: each batch is (anchor, positive) pairs from a RetrievalDataset, scored by
    loss_fn(q_emb, q_mask, d_emb, d_mask, exclude), where exclude masks the in-batch negatives that share the anchor's
    sf (the diagonal, the positive, is kept)
    """
    def __init__(self, dataset, encoder, loss_fn, **kwargs):
        self.dataset     = dataset
        self.encoder     = encoder
        self.loss_fn     = loss_fn
        self.batch_size  = kwargs.get('batch_size', 32)
        self.ckpt_prefix = kwargs.get('ckpt_prefix', 'test')

    def param_groups(self, lr_backbone=1e-5, lr_head=1e-4, wd=1e-2):
        # PLM parameters (under plm_wrapper) at lr_backbone, the rest at lr_head; no weight decay on biases and 1-D
        groups = {k: [] for k in ('bb_decay', 'bb_plain', 'head_decay', 'head_plain')}
        for n, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            prefix = 'bb' if n.startswith('plm_wrapper') else 'head'
            kind = 'plain' if (p.ndim == 1 or n.endswith('.bias')) else 'decay'
            groups[f'{prefix}_{kind}'].append(p)
        out = [
            {'params': groups['bb_decay'],   'lr': lr_backbone, 'weight_decay': wd},
            {'params': groups['bb_plain'],   'lr': lr_backbone, 'weight_decay': 0.0},
            {'params': groups['head_decay'], 'lr': lr_head,     'weight_decay': wd},
            {'params': groups['head_plain'], 'lr': lr_head,     'weight_decay': 0.0},
        ]
        # a frozen backbone, or an encoder without a head, leaves some groups empty
        return [g for g in out if g['params']]

    def training_step(self, batch):
        anchor, positive, labels = batch['anchor'], batch['positive'], batch['sf']
        exclude = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.encoder(q_seqs=anchor, d_seqs=positive)
            loss = self.loss_fn(*out, exclude)
        return loss

    def epoch_batches(self, ep, seed, length_buckets=0):
        # the epoch's batches, a function of (seed, ep) only, so a resumed run skips exactly the batches already done:
        # length_buckets 0: lists of anchor indices in random order, each positive drawn when its item is loaded
        # length_buckets n: lists of (anchor, positive) pairs, the positives drawn here; the shuffled pairs are cut into
        #   chunks of n batches, each chunk sorted by the pair's longer sequence before it is cut into batches, and the
        #   batches shuffled. Batches then hold sequences of similar length (less padding in the PLM and the scorer's
        #   residue grid) and still change from epoch to epoch; their in-batch negatives are length-matched
        n_batches, B = len(self.dataset) // self.batch_size, self.batch_size   # drop_last
        if not length_buckets:
            order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(seed + ep))
            return order[:n_batches * B].view(n_batches, B).tolist()
        rng = random.Random(seed + ep)
        anchors = list(range(len(self.dataset)))
        rng.shuffle(anchors)
        pairs = [(a, self.dataset.draw_positive(a, rng)) for a in anchors[:n_batches * B]]
        size = lambda pair: max(len(self.dataset.data[k]['seq']) for k in pair)
        batches, chunk = [], length_buckets * B
        for c in range(0, len(pairs), chunk):
            part = sorted(pairs[c:c + chunk], key=size)
            batches += [part[a:a + B] for a in range(0, len(part), B)]
        rng.shuffle(batches)
        return batches

    def fit(self, num_epochs=101, save_every=5, save_dir=None, resume=False, time_budget=None, state_every=500,
            seed=2603, length_buckets=0, log_every=100, **opt_kwargs):
        # save_every:  write {ckpt_prefix}_ckpt_ep{ep:03d}.pt (encoder.trainable_state_dict()) every this many epochs
        #              and after the last one
        # resume:      continue from {ckpt_prefix}_state.pt (encoder, optimizer, epoch, batch) when it exists
        # time_budget: seconds; after the first batch past it, save the state and return False (True = all epochs done)
        # state_every: also save the state every this many batches, so a killed job loses little
        # log_every:   a scorer with learned parameters that has log_params() (qNWSim), and a loss that has one
        #              (ClsAugmentedInfoNCE's align slot wins): append them every this many batches as one json line
        #              ({epoch, batch, ...}) to {ckpt_prefix}_params.jsonl; 0 = off
        # length_buckets: batches of length-sorted pairs drawn per epoch, sorted within chunks of this many batches
        #              (epoch_batches); 0 = random batches
        # opt_kwargs:  lr_backbone, lr_head, wd (param_groups)
        save_dir = save_dir or artifact_path()
        opt = AdamW(self.param_groups(**opt_kwargs))
        os.makedirs(save_dir, exist_ok=True)
        state_path = os.path.join(save_dir, f'{self.ckpt_prefix}_state.pt')
        log_path = os.path.join(save_dir, f'{self.ckpt_prefix}_params.jsonl')
        scorer = getattr(self.encoder, 'scorer', None)
        loggers = [x for x in (scorer, self.loss_fn) if log_every and hasattr(x, 'log_params')]
        start_ep, start_batch, cum_loss, skipped = 0, 0, 0., 0
        if resume and os.path.exists(state_path):
            state = torch.load(state_path, map_location='cuda')
            self.encoder.load_checkpoint(state['encoder'])
            opt.load_state_dict(state['opt'])
            start_ep, start_batch, cum_loss, skipped = state['epoch'], state['batch'], state['cum_loss'], state['skipped']
            print(f'Resuming {self.ckpt_prefix} from epoch {start_ep}, batch {start_batch}')

        def save_state(ep, batch):
            state = {'encoder': self.encoder.trainable_state_dict(), 'opt': opt.state_dict(),
                     'epoch': ep, 'batch': batch, 'cum_loss': cum_loss, 'skipped': skipped}
            torch.save(state, state_path + '.tmp')
            os.replace(state_path + '.tmp', state_path)          # never leaves a half-written state

        t0 = time.time()
        n_batches = len(self.dataset) // self.batch_size         # drop_last
        self.encoder.train()
        for ep in range(start_ep, num_epochs):
            order = self.epoch_batches(ep, seed, length_buckets)
            first = start_batch if ep == start_ep else 0
            if ep != start_ep:
                cum_loss, skipped = 0., 0
            loader = DataLoader(self.dataset, batch_sampler=order[first:])
            bar = tqdm(loader, desc=f'Epoch {ep}', initial=first, total=n_batches)
            for i, batch in enumerate(bar, start=first):
                opt.zero_grad(set_to_none=True)
                oom = False
                try:
                    loss = self.training_step(batch)
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    oom = True                                   # cleanup outside the except, so the traceback frees its tensors
                if oom:
                    # long sequences on both sides of a batch; skip it rather than end the run
                    loss = None
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    skipped += 1
                    print(f'Epoch {ep} batch {i}: out of memory, skipped ({skipped} this epoch)')
                    continue
                cum_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                opt.step()
                bar.set_postfix_str(f'Cum Loss = {cum_loss:.3f}')
                if loggers and (i + 1) % log_every == 0:
                    logged = {k: v for x in loggers for k, v in x.log_params().items()}
                    if logged:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({'epoch': ep, 'batch': i + 1, **logged}) + '\n')
                if time_budget is not None and time.time() - t0 > time_budget:
                    save_state(ep, i + 1)
                    print(f'Time budget reached at epoch {ep}, batch {i + 1}/{n_batches}; state saved to {state_path}')
                    return False
                if state_every and (i + 1) % state_every == 0:
                    save_state(ep, i + 1)
            if save_every and (ep % save_every == 0 or ep == num_epochs - 1):
                path = os.path.join(save_dir, f'{self.ckpt_prefix}_ckpt_ep{ep:03d}.pt')
                torch.save(self.encoder.trainable_state_dict(), path)   # frozen pretrained weights left out
            cum_loss, skipped = 0., 0
            save_state(ep + 1, 0)
        return True


class DistillTrainer(RetrieverTrainer):
    """
    RetrieverTrainer with a frozen teacher (an Encoder whose .scorer, e.g. qNWSim, scores its embeddings) for
    infonce.DistillInfoNCE: each batch's pairs are also scored by the teacher, (B, B) with the anchors as queries, without
    gradients, and passed to the loss. The teacher stays in eval mode and out of the optimizer
    """
    def __init__(self, dataset, encoder, loss_fn, teacher, **kwargs):
        super().__init__(dataset, encoder, loss_fn, **kwargs)
        self.teacher = teacher.eval().requires_grad_(False)

    def training_step(self, batch):
        anchor, positive, labels = batch['anchor'], batch['positive'], batch['sf']
        exclude = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                teacher_scores = self.teacher.scorer(*self.teacher(q_seqs=anchor, d_seqs=positive))
            out = self.encoder(q_seqs=anchor, d_seqs=positive)
            loss = self.loss_fn(*out, exclude, teacher_scores)
        return loss


class RetrieverEval:
    """
    All-vs-all retrieval on a RetrievalDataset: every sequence queries all the others, ranked by sim_fn (a SequenceSim).
    Reports capped recall@k = |relevant in top-k| / min(k, |relevant|), micro (mean over queries) and macro (mean
    within each sf, then over sf), for two definitions of relevant:
      all_*:    same sf
      remote_*: same sf, different fa (same-fa documents are removed from the ranking; queries without one are skipped)
    pair_budget: max Bq * Bd * Lq * Ld per sim_fn call. symmetric: sim_fn(q, d) == sim_fn(d, q).T, score the upper
    triangle of chunk pairs only
    queries: dataset indices of the queries (default all); each is ranked against the whole dataset, itself excluded
    groups: a dataset level (e.g. 'arch' of CATH) that every sf lies within: each query is ranked against the documents
      of its own group only (the rest stay at -inf, unscored), so only within-group pairs cost scorer calls
    """
    def __init__(self, dataset, sim_fn, **kwargs):
        self.dataset      = dataset
        self.sim_fn       = sim_fn
        self.device       = kwargs.get('device', 'cuda')
        self.pair_budget  = kwargs.get('pair_budget', 2**29)
        self.symmetric    = kwargs.get('symmetric', False)
        self.seqs         = [x['seq'] for x in dataset.data]
        self.sf           = torch.tensor([x['sf'] for x in dataset.data], device=self.device)
        self.fa           = torch.tensor([x['fa'] for x in dataset.data], device=self.device)
        queries           = kwargs.get('queries', None)
        self.queries      = None if queries is None else torch.as_tensor(queries, device=self.device)
        groups            = kwargs.get('groups', None)
        self.groups       = None
        if groups is not None:
            assert queries is None, 'groups ranks every sequence against its group: no queries subset'
            members = getattr(dataset, f'{groups}_members')
            assert all(len({dataset.data[i][groups] for i in m}) == 1 for m in dataset.sf_members.values()), \
                f'an sf spans several {groups}: its positives would go unranked'
            self.groups = list(members.values())

    def embed_group_by_length(self, encoder, idx=None):
        # length-sorted batches of <= sqrt(pair_budget) padded tokens, used for embedding and scoring:
        # any batch pair then has Bq * Bd * Lq * Ld <= pair_budget -> one sim_fn call each. idx: these dataset indices only
        max_tokens_per_batch = np.sqrt(self.pair_budget)
        order = sorted(range(len(self.seqs)) if idx is None else idx, key=lambda k: len(self.seqs[k]))
        batches = [[]]
        for k in order:
            if batches[-1] and (len(batches[-1]) + 1) * len(self.seqs[k]) > max_tokens_per_batch:
                batches.append([])
            batches[-1].append(k)

        emb_chunks = []
        for batch in tqdm(batches, desc='Eval -- Embedding'):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask = encoder([self.seqs[idx] for idx in batch])
            emb_chunks.append((torch.tensor(batch, device=emb.device), emb.bfloat16(), mask))   # (dataset indices, emb, mask)
        return emb_chunks

    @torch.no_grad()
    def similarity(self, encoder):
        # N x N scores in dataset order, self-matches at -inf; with queries, Q x N (row r: query self.queries[r])
        was_training = encoder.training
        encoder.eval()
        N = len(self.seqs)
        if self.queries is not None:
            chunks = self.embed_group_by_length(encoder)
            q_chunks = self.embed_group_by_length(encoder, self.queries.tolist())
            row = torch.full((N,), -1, dtype=torch.long, device=self.device)
            row[self.queries] = torch.arange(len(self.queries), device=self.device)
            sim = torch.empty(len(self.queries), N, device=self.device)
            for (qi, q_emb, q_mask), (dj, d_emb, d_mask) in tqdm([(a, b) for a in q_chunks for b in chunks],
                                                                 desc='Eval -- Sim scores'):
                sim[row[qi][:, None], dj[None, :]] = self.sim_fn(q_emb, q_mask, d_emb, d_mask).float()
            del chunks, q_chunks
            encoder.train(was_training)
            sim[torch.arange(len(self.queries), device=self.device), self.queries] = float('-inf')
            return sim
        sim = torch.full((N, N), float('-inf'), device=self.device)
        # one group at a time (the whole dataset without groups), embedded on its own: its chunks pair only with each other
        for group in ([None] if self.groups is None else self.groups):
            chunks = self.embed_group_by_length(encoder, group)
            pairs = [(ci, cj) for ci in range(len(chunks)) for cj in range(len(chunks)) if not self.symmetric or cj >= ci]
            for ci, cj in tqdm(pairs, desc='Eval -- Sim scores'):
                (qi, q_emb, q_mask), (dj, d_emb, d_mask) = chunks[ci], chunks[cj]
                s = self.sim_fn(q_emb, q_mask, d_emb, d_mask).float()   # index_put won't cast bf16 -> fp32
                sim[qi[:, None], dj[None, :]] = s
                if self.symmetric and ci != cj:
                    sim[dj[:, None], qi[None, :]] = s.T
            del chunks
        encoder.train(was_training)
        return sim.fill_diagonal_(float('-inf'))

    def recall(self, sim, ks=(1, 5, 10, 25, 50), remote=False, per_group=False):
        # capped recall@k = |no. positives in top-k| / min(k, |relevant|)
        # per_group: also each sf's mean ({prefix}sf_recall@k, a list in {prefix}sf_ids order: the sf labels), the values
        # the macro score averages
        # positive: same sf; when remote=True: same sf but different fa (same fa dropped from the ranking)
        # sim rows: the queries (all of the dataset by default), columns: the dataset
        sf, fa = self.sf, self.fa
        qsf, qfa = (sf, fa) if self.queries is None else (sf[self.queries], fa[self.queries])
        if remote:
            sim = sim.masked_fill(qfa[:, None] == fa[None, :], float('-inf'))
            n_rel = torch.bincount(sf)[qsf] - torch.bincount(fa)[qfa]
        else:
            n_rel = torch.bincount(sf)[qsf] - 1
        keep = n_rel > 0                                         # nothing to retrieve
        top = sim.topk(min(max(ks), sim.shape[1] - 1), dim=-1)
        topk = top.indices[keep]
        # a -inf document was not ranked (self-match, a dropped same-fa one, or one a pruned search such as PLAID did
        # not return) and never counts, even where topk runs into them
        rel = (sf[topk] == qsf[keep, None]) & torch.isfinite(top.values[keep])
        if remote:
            rel &= fa[topk] != qfa[keep, None]
        hits = rel.cumsum(-1)
        n_rel = n_rel[keep]
        sf_ids, q_sf = torch.unique(qsf[keep], return_inverse=True)
        n_sf = int(q_sf.max()) + 1

        out, prefix = {}, 'remote_' if remote else 'all_'
        if per_group:
            out[f'{prefix}sf_ids'] = sf_ids.tolist()
        for k in sorted(ks):
            rec = hits[:, min(k, topk.shape[1]) - 1] / n_rel.clamp(max=k)
            per_sf = torch.zeros(n_sf, device=sim.device).index_add_(0, q_sf, rec) / torch.bincount(q_sf)
            out[f'{prefix}micro_recall@{k}'] = rec.mean().item()
            out[f'{prefix}macro_recall@{k}'] = per_sf.mean().item()   # mean over sf first
            if per_group:
                out[f'{prefix}sf_recall@{k}'] = per_sf.tolist()
        out[f'{prefix}n_queries'], out[f'{prefix}n_sf'] = int(keep.sum()), n_sf
        return out

    @torch.no_grad()
    def average_precision(self, sim, remote=False, chunk=1024, per_group=False):
        # AP = mean over a query's positives of the precision at each one's rank, an unranked (-inf) positive adding 0;
        # micro (mean over queries) and macro (mean within each sf, then over sf) MAP, positives as in recall.
        # sim may be a CPU (e.g. fp16, saved) matrix: scored chunk rows at a time on self.device. per_group: as recall's
        # ({prefix}sf_ids, {prefix}sf_map)
        sf, fa = self.sf, self.fa
        qsf, qfa = (sf, fa) if self.queries is None else (sf[self.queries], fa[self.queries])
        n_rel = torch.bincount(sf)[qsf] - (torch.bincount(fa)[qfa] if remote else 1)
        ranks = torch.arange(1, sim.shape[1] + 1, device=self.device)
        ap = torch.zeros(len(qsf), device=self.device)
        for a in range(0, len(qsf), chunk):
            b = min(a + chunk, len(qsf))
            s = sim[a:b].to(self.device).float()
            if remote:
                s.masked_fill_(qfa[a:b, None] == fa[None, :], float('-inf'))
            vals, order = s.sort(dim=-1, descending=True)
            rel = (sf[order] == qsf[a:b, None]) & torch.isfinite(vals)
            if remote:
                rel &= fa[order] != qfa[a:b, None]
            ap[a:b] = (rel * rel.cumsum(-1) / ranks).sum(-1) / n_rel[a:b].clamp(min=1)
        keep = n_rel > 0                                         # nothing to retrieve
        ap = ap[keep]
        sf_ids, q_sf = torch.unique(qsf[keep], return_inverse=True)
        per_sf = torch.zeros(int(q_sf.max()) + 1, device=self.device).index_add_(0, q_sf, ap) / torch.bincount(q_sf)
        prefix = 'remote_' if remote else 'all_'
        out = {f'{prefix}micro_map': ap.mean().item(), f'{prefix}macro_map': per_sf.mean().item()}
        if per_group:
            out.update({f'{prefix}sf_ids': sf_ids.tolist(), f'{prefix}sf_map': per_sf.tolist()})
        return out



if __name__ == '__main__':
    # untrained ESM2-35M with symmetric hard DTW on astral40 test
    encoder = Encoder(backbone='esm2-35M', unfreeze_last_n=3).to('cuda')
    evaluator = RetrieverEval(SCOPe('astral40', 'test'), DTWSim(), symmetric=True)
    sim = evaluator.similarity(encoder)
    pprint({**evaluator.recall(sim), **evaluator.recall(sim, remote=True)})
