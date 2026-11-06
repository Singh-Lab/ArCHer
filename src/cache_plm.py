from retriever import *
import argparse

# Caches a PLM's hidden state at the input of layer --layer (the first trained one, len(layers) - unfreeze) for every
# tile of a dataset, for plm_wrapper's CachedPLM wrappers (ProtT5, ESM-2), as PLM_CACHE_DIR/<backbone>_L<layer>/<name>.f16 and
# <name>_index.pt (layout in plm_wrapper.py). Frozen layers in eval mode (no dropout), under the bf16 autocast training
# uses; rows stored fp16 (an error if one leaves its range). Tiles go length-sorted, in batches of <= --max_tokens
# padded tokens, written in that order, so the file fills front to back. Resumable: --time_budget seconds, then it
# saves its progress and exits 3 (rerun the same command to continue); exit 0 once the cache is complete.
#   PYTHONPATH=. python experiments/cache_plm.py --data pfam --split train --max_len 768
#   PYTHONPATH=. python experiments/cache_plm.py --data astral40 --split full
#   PYTHONPATH=. python experiments/cache_plm.py --backbone esm2-650M --layer 30 --data pfam --split train --max_len 768


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='prott5-half')
    ap.add_argument('--layer', type=int, default=21)                # cache the input of this layer
    ap.add_argument('--data', choices=('pfam',) + SCOPE_SUBSETS, default='pfam')
    ap.add_argument('--split', default='train')
    ap.add_argument('--max_len', type=int, default=768)             # as train.py's: drop longer sequences; 0 keeps all
    ap.add_argument('--max_tokens', type=int, default=2**15)        # padded tokens per PLM call
    ap.add_argument('--time_budget', type=float, default=None)      # seconds
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    print(vars(args))
    t0 = time.time()
    data = (Pfam(args.split, label='clan', max_len=args.max_len or None) if args.data == 'pfam'
            else SCOPe(args.data, args.split, max_len=args.max_len or None))
    plm = PLMWrapper.from_name(args.backbone)
    assert isinstance(plm, (ProtT5Wrapper, ESMWrapper)) and not isinstance(plm, CachedPLM), 'ProtT5 or ESM-2 only'
    # layers 0 .. layer-1 and no final norm: the stack's last hidden state is then the input of layer `layer`
    assert 0 < args.layer < len(plm.layers()), f'--layer must be in 1..{len(plm.layers()) - 1}'
    if isinstance(plm, ProtT5Wrapper):
        plm.plm.encoder.block = plm.plm.encoder.block[:args.layer]
        plm.plm.encoder.final_layer_norm = nn.Identity()            # T5Stack's closing dropout is off in eval mode
        special = 1 + plm.n_prefix                                  # ProtT5 appends </s>; ProstT5 also leads with <AA2fold>
    else:
        plm.plm.esm.encoder.layer = plm.plm.esm.encoder.layer[:args.layer]
        plm.plm.esm.encoder.emb_layer_norm_after = None
        plm.plm.lm_head = nn.Identity()                             # ESMWrapper.run's logits, unused here
        special = 2                                                 # ESM adds <cls> and <eos>
    plm = plm.to('cuda').eval()
    out_dir = os.path.join(PLM_CACHE_DIR, f'{args.backbone}_L{args.layer}')
    name = f'{args.data}_{args.split}' + (f'_max{args.max_len}' if args.max_len else '')
    os.makedirs(out_dir, exist_ok=True)
    index_path, array_path = os.path.join(out_dir, f'{name}_index.pt'), os.path.join(out_dir, f'{name}.f16')

    # every distinct tile, as PLMWrapper.forward cuts them, shortest first
    tiles = list(dict.fromkeys(s[a:b] for s in (x['seq'] for x in data.data)
                               for a, b in tile_spans(len(s), plm.max_len, plm.min_overlap)))
    tiles.sort(key=len)
    lengths = [len(t) + special for t in tiles]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    batches = [[]]
    for k, t in enumerate(tiles):
        if batches[-1] and (len(batches[-1]) + 1) * lengths[k] > args.max_tokens:
            batches.append([])
        batches[-1].append(k)

    if os.path.exists(index_path):
        index = torch.load(index_path)
        assert index['tiles'] == tiles, f'{index_path} was built from other data; delete it to rebuild'
        meta = index['meta']
    else:
        meta = {'backbone': args.backbone, 'layer': args.layer, 'data': args.data, 'split': args.split,
                'max_len': args.max_len, 'tile_max_len': plm.max_len, 'tile_min_overlap': plm.min_overlap,
                'dtype': 'float16', 'hidden_size': plm.hidden_size, 'batches_done': 0, 'max_abs': 0., 'complete': False}
        index = {'meta': meta, 'tiles': tiles, 'offsets': offsets, 'lengths': lengths}
    rows = sum(lengths)
    mode = 'r+' if os.path.exists(array_path) and meta['batches_done'] > 0 else 'w+'
    arr = np.memmap(array_path, dtype=np.float16, mode=mode, shape=(rows, plm.hidden_size))
    print(f'{name}: {len(tiles)} tiles from {len(data)} sequences, {rows} rows, {rows * plm.hidden_size * 2 / 1e9:.1f} GB '
          f'-> {array_path}; {len(batches)} batches, {meta["batches_done"]} done')

    def save_index():
        arr.flush()
        torch.save(index, index_path + '.tmp')
        os.replace(index_path + '.tmp', index_path)

    fp16_max = torch.finfo(torch.float16).max
    stopped = False
    with torch.no_grad():
        for bi in trange(meta['batches_done'], len(batches), initial=meta['batches_done'], total=len(batches),
                         desc='Caching'):
            batch = batches[bi]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h, _ = plm.run(*plm.tokenize([tiles[k] for k in batch]))
            h = h.float()
            m = h.abs().max().item()
            assert m < fp16_max, f'batch {bi}: |h| up to {m}, beyond fp16; cache in bf16 instead'
            meta['max_abs'] = max(meta['max_abs'], m)
            h = h.half().cpu().numpy()
            # the batch's tiles are consecutive in the file
            a = offsets[batch[0]]
            arr[a:offsets[batch[-1]] + lengths[batch[-1]]] = np.concatenate([h[b, :lengths[k]] for b, k in enumerate(batch)])
            meta['batches_done'] = bi + 1
            if (bi + 1) % 500 == 0:
                save_index()
            if args.time_budget is not None and time.time() - t0 > args.time_budget:
                stopped = True
                break
    meta['complete'] = meta['batches_done'] == len(batches)
    save_index()
    print(f'{meta["batches_done"]}/{len(batches)} batches done, max |h| {meta["max_abs"]:.1f}'
          + ('' if meta['complete'] else '; time budget reached, rerun to continue'))
    raise SystemExit(0 if meta['complete'] else 3)
