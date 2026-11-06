"""
ArCHer command line: two-phase remote homology search of query sequences against a sequence database.

  phase 1  the dense model embeds every sequence as one vector (mean of its residue embeddings) and shortlists each
           query's top --shortlist database sequences by cosine similarity
  phase 2  the qNW-affine model re-scores each query's shortlist by a learned alignment of residue embeddings
           (query-global, database-local Needleman-Wunsch with affine gaps); hits are ranked by this score

  archer makedb database.fasta mydb/ [--store-residues]    embed a database once and store it
  archer search -q queries.fasta -d mydb/ -o hits.tsv       search it (-d also takes a FASTA, embedded on the fly)

Both phases are the models and scorers of train.py, loaded from <checkpoint-dir>/<name>.json (the run's training flags, as
train.py saved them; "prefix" is the original run name) and <name>.pt (its weights); embedding and scoring follow
retriever.RetrieverEval (length-sorted batches, bf16 autocast). Needs a CUDA GPU (the alignment kernels are Triton).
  ArCHer_shortlist  phase 1, run pfam_prott5c_dense_cos_ft_pfdev_kdonly_lr1e-6_s2707_lb50, epoch 9
  ArCHer_reranker   phase 2, run pfam_prott5c_qnw_affine_geo_b0.3_lr1e-6_s2707_lb50, epoch 9

A database folder (archer makedb) holds
  sequences.fasta        the database sequences, as read
  shortlist.pt           (N, D) bf16 phase-1 vectors, in sequences.fasta order
  residues.bin           with --store-residues: phase-2 residue embeddings of every sequence back to back, a flat
  residues_offsets.npy   (total residues, D) array of bf16 bits (uint16), sequence i at rows offsets[i]:offsets[i+1]
  archer_db.json         written last (a folder without it is an unfinished build): sizes, and the sha256 of each
                         checkpoint the stored embeddings came from; search refuses a database whose checkpoints differ
"""
import argparse
import hashlib
import json
import os
import sys
import time

PHASE1 = 'ArCHer_shortlist'
PHASE2 = 'ArCHer_reranker'
DB_FORMAT = 1
DB_META = 'archer_db.json'


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--checkpoint-dir',
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints'),
                        help='folder holding the checkpoints, <name>.pt and <name>.json for each phase')
    common.add_argument('--phase1', default=PHASE1, help='phase-1 (shortlist) checkpoint name')
    common.add_argument('--phase2', default=PHASE2, help='phase-2 (reranker) checkpoint name')
    common.add_argument('--batch-tokens', type=int, default=23170,
                        help='max padded residues per embedding batch; lower it on a small GPU')
    fmt = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(prog='archer', formatter_class=fmt,
                                description='ArCHer: two-phase remote homology search.')
    sub = p.add_subparsers(dest='command', required=True, metavar='{makedb,search}')

    m = sub.add_parser('makedb', parents=[common], formatter_class=fmt,
                       help='embed a database FASTA once and store it for repeated searches',
                       description='Embed a database FASTA with the shortlist model (and, with --store-residues, the '
                                   'reranker) and store it in a folder that "archer search -d" reads.')
    m.add_argument('fasta', help='FASTA of database sequences')
    m.add_argument('db', help='output database folder (must not exist yet, or be empty)')
    m.add_argument('--store-residues', action='store_true',
                   help='also store the reranker residue embeddings: searches then skip embedding the shortlisted '
                        'sequences, at about 2 KB per residue on disk')

    s = sub.add_parser('search', parents=[common], formatter_class=fmt,
                       help='search query sequences against a database',
                       description='Rank the database sequences shortlisted for each query and write the hits as a '
                                   'TSV.')
    s.add_argument('-q', '--query', required=True, help='FASTA of query sequences')
    s.add_argument('-d', '--database', required=True,
                   help='database folder from "archer makedb", or a FASTA of database sequences (embedded on the fly)')
    s.add_argument('-o', '--output', required=True, help='output TSV of hits')
    s.add_argument('-k', '--shortlist', type=int, default=4096,
                   help='phase-1 shortlist size per query, the sequences phase 2 re-scores; 0 = re-score the whole '
                        'database (exhaustive)')
    s.add_argument('-n', '--max-hits', type=int, default=0,
                   help='hits written per query; 0 = the whole shortlist (--shortlist hits)')
    s.add_argument('--exclude-self', action='store_true',
                   help='drop hits whose database id equals the query id (e.g. when searching a set against itself)')
    return p.parse_args()


def read_fasta(path):
    ids, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                ids.append(line[1:].split()[0] if line[1:].strip() else f'seq{len(ids)}')
                seqs.append([])
            else:
                if not ids:
                    sys.exit(f'{path}: not a FASTA file (sequence before the first ">" header)')
                seqs[-1].append(line.replace(' ', '').rstrip('*'))
    seqs = [''.join(s).upper() for s in seqs]
    empty = [i for i, s in zip(ids, seqs) if not s]
    if not ids:
        sys.exit(f'{path}: no sequences')
    if empty:
        sys.exit(f'{path}: empty sequence(s), e.g. {empty[0]}')
    if len(set(ids)) != len(ids):
        sys.exit(f'{path}: duplicate sequence ids')
    return ids, seqs


def checkpoint_paths(ckpt_dir, name):
    paths = os.path.join(ckpt_dir, f'{name}.json'), os.path.join(ckpt_dir, f'{name}.pt')
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f'{path} not found: download the checkpoints (README, "Download the checkpoints")')
    return paths


HASH_CACHE = '.archer_sha256_cache.json'


def checkpoint_sha256(ckpt_dir, name):
    # one hash over the checkpoint's settings and weights: any change to either changes it. Hashing reads ~600 MB, so
    # the result is cached in <ckpt_dir>/.archer_sha256_cache.json, keyed by each file's size, mtime, inode and ctime
    # (ctime changes on any write or replacement and cannot be set by copying); an unwritable folder just skips the cache
    paths = checkpoint_paths(ckpt_dir, name)
    stat = [[s.st_size, s.st_mtime_ns, s.st_ino, s.st_ctime_ns] for s in map(os.stat, paths)]
    cache_path = os.path.join(ckpt_dir, HASH_CACHE)
    try:
        cache = json.load(open(cache_path))
    except (OSError, ValueError):
        cache = {}
    if cache.get(name, {}).get('stat') == stat:
        return cache[name]['sha256']
    h = hashlib.sha256()
    for path in paths:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
    cache[name] = {'stat': stat, 'sha256': h.hexdigest()}
    try:
        tmp = f'{cache_path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, cache_path)                                 # never leaves a half-written cache
    except OSError:
        pass
    return h.hexdigest()


def load_model(ckpt_dir, name):
    # the run's encoder and scorer as train.py built them (its saved flags over train.py's defaults), the scorer's
    # learned parameters (qNW gap costs) loaded with the checkpoint; a run trained on cached PLM states runs on the full
    # PLM, whose checkpoints it shares (as experiments/eval_ckpts.run_config)
    import torch
    from train import parser, scorer, attach_scorer
    from encoder import Encoder
    args_path, ckpt_path = checkpoint_paths(ckpt_dir, name)
    cfg = vars(parser().parse_args(['--prefix', name]))
    cfg.update(json.load(open(args_path)))
    if cfg['backbone'].endswith('-cached'):
        cfg['backbone'] = cfg['backbone'][:-len('-cached')]
    cfg = argparse.Namespace(**cfg)
    encoder = Encoder(backbone=cfg.backbone, unfreeze_last_n=cfg.unfreeze).to('cuda')
    sim_fn = scorer(cfg)
    attach_scorer(encoder, sim_fn)
    encoder.load_checkpoint(torch.load(ckpt_path, map_location='cuda'))
    return encoder.eval(), sim_fn


def length_batches(seqs, max_tokens):
    # length-sorted batches of <= max_tokens padded residues (RetrieverEval.embed_group_by_length)
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
    batches = [[]]
    for k in order:
        if batches[-1] and (len(batches[-1]) + 1) * len(seqs[k]) > max_tokens:
            batches.append([])
        batches[-1].append(k)
    return batches


def embed_batches(encoder, seqs, max_tokens, desc):
    # yields (batch indices, (B, L, D) bf16 residue embeddings, (B, L) mask), length-sorted batches
    import torch
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(length_batches(seqs, max_tokens), desc=desc):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask = encoder([seqs[k] for k in batch])
            yield batch, emb.bfloat16(), mask


def embed(encoder, seqs, max_tokens, desc, pooled):
    # pooled: (N, D) mean residue embeddings on the GPU; else a list of (L_i, D) residue embeddings on the CPU
    import torch
    from utils import masked_mean
    out = [None] * len(seqs)
    for batch, emb, mask in embed_batches(encoder, seqs, max_tokens, desc):
        if pooled:
            p = masked_mean(emb, mask)
            for b, k in enumerate(batch):
                out[k] = p[b]
        else:
            for b, k in enumerate(batch):
                out[k] = emb[b, :int(mask[b].sum())].cpu()
    return torch.stack(out) if pooled else out


def shortlist(sim_fn, q, d, k, exclude):
    # each query's top-k database sequences by sim_fn (DenseSim) on the pooled vectors, each a one-residue sequence
    # (its mean is itself); exclude: (Nq,) database index to drop per query (-1: none), or None
    import torch
    nq, nd = len(q), len(d)
    ones = lambda n: torch.ones(n, 1, dtype=torch.bool, device='cuda')
    step = max(1, 2**28 // nd)
    cand, score = [], []
    with torch.no_grad():
        for a in range(0, nq, step):
            s = sim_fn(q[a:a + step, None], ones(len(q[a:a + step])), d[:, None], ones(nd)).float()
            if exclude is not None:
                rows = (exclude[a:a + step] >= 0).nonzero().flatten()
                s[rows, exclude[a:a + step][rows]] = float('-inf')
            top = s.topk(k, dim=-1)
            cand.append(top.indices)
            score.append(top.values)
    return torch.cat(cand), torch.cat(score)


def rerank(sim_fn, q_embs, d_embs, cand, pair_budget=2**25, max_tokens=2**16):
    # sim_fn(query i, each of its candidates) -> (Nq, K); a query's candidates in length-sorted groups of
    # n * Lq * Ld_max <= pair_budget and n * Ld_max <= max_tokens (the batching and budget of the authors' two-stage
    # evaluation); candidates at -1 (a query with fewer than K allowed) score -inf
    import torch
    from tqdm import trange
    from utils import pad_embeddings
    nq, K = cand.shape
    lens = torch.tensor([len(e) for e in d_embs])
    out = torch.full((nq, K), float('-inf'))
    with torch.no_grad():
        for i in trange(nq, desc='phase 2: rerank'):
            q = q_embs[i].cuda()[None]
            qm = torch.ones(1, q.shape[1], dtype=torch.bool, device='cuda')
            valid = (cand[i] >= 0).nonzero().flatten()
            pos = valid[lens[cand[i, valid]].argsort()].tolist()
            a = 0
            while a < len(pos):
                b = a + 1
                while b < len(pos):
                    n, Ld = b + 1 - a, len(d_embs[cand[i, pos[b]]])
                    if n * q.shape[1] * Ld > pair_budget or n * Ld > max_tokens:
                        break
                    b += 1
                idx = pos[a:b]
                d, dm = pad_embeddings([d_embs[j].cuda() for j in cand[i, idx].tolist()])
                out[i, idx] = sim_fn(q, qm, d.bfloat16(), dm)[0].float().cpu()
                a = b
    return out


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit('ArCHer needs a CUDA GPU (its alignment kernels are written in Triton)')


def makedb(args):
    import numpy as np
    import torch
    require_gpu()
    t0 = time.time()
    if os.path.exists(args.db) and os.listdir(args.db):
        sys.exit(f'{args.db} exists and is not empty: choose a new folder, or delete it to rebuild')
    ids, seqs = read_fasta(args.fasta)
    os.makedirs(args.db, exist_ok=True)
    n_res = sum(len(s) for s in seqs)
    print(f'ArCHer makedb: {len(ids)} sequences, {n_res} residues -> {args.db}', flush=True)
    with open(os.path.join(args.db, 'sequences.fasta'), 'w') as f:
        for i, s in zip(ids, seqs):
            f.write(f'>{i}\n{s}\n')

    meta = {'format': DB_FORMAT, 'n_sequences': len(ids), 'n_residues': n_res, 'store_residues': args.store_residues,
            'checkpoints': {'shortlist': {'name': args.phase1, 'sha256': checkpoint_sha256(args.checkpoint_dir, args.phase1)}}}
    encoder, _ = load_model(args.checkpoint_dir, args.phase1)
    vec = embed(encoder, seqs, args.batch_tokens, 'makedb: shortlist vectors', pooled=True).cpu()
    torch.save(vec, os.path.join(args.db, 'shortlist.pt'))
    meta['shortlist_dim'] = vec.shape[1]
    del encoder
    torch.cuda.empty_cache()

    if args.store_residues:
        meta['checkpoints']['reranker'] = {'name': args.phase2, 'sha256': checkpoint_sha256(args.checkpoint_dir, args.phase2)}
        encoder, _ = load_model(args.checkpoint_dir, args.phase2)
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(s) for s in seqs])
        res, dim = None, None
        for batch, emb, mask in embed_batches(encoder, seqs, args.batch_tokens, 'makedb: residue embeddings'):
            if res is None:
                dim = emb.shape[-1]
                res = np.memmap(os.path.join(args.db, 'residues.bin'), dtype=np.uint16, mode='w+', shape=(n_res, dim))
            for b, k in enumerate(batch):
                L = int(mask[b].sum())
                assert L == offsets[k + 1] - offsets[k], f'{ids[k]}: {L} embeddings for {len(seqs[k])} residues'
                res[offsets[k]:offsets[k + 1]] = emb[b, :L].cpu().view(torch.int16).numpy().view(np.uint16)
        res.flush()
        del res
        np.save(os.path.join(args.db, 'residues_offsets.npy'), offsets)
        meta['residue_dim'] = dim
    meta['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta['source'] = os.path.abspath(args.fasta)
    with open(os.path.join(args.db, DB_META), 'w') as f:            # last: marks the build complete
        json.dump(meta, f, indent=1)
    print(f'database written to {args.db} ({time.time() - t0:.0f} s)')


def open_db(args):
    # a makedb folder -> (ids, seqs, (N, D) shortlist vectors on the GPU, residue reader or None)
    import numpy as np
    import torch
    meta_path = os.path.join(args.database, DB_META)
    if not os.path.exists(meta_path):
        sys.exit(f'{args.database}: no {DB_META}, so not a finished "archer makedb" database')
    meta = json.load(open(meta_path))
    if meta.get('format') != DB_FORMAT:
        sys.exit(f'{args.database}: database format {meta.get("format")}, this ArCHer reads {DB_FORMAT}: rebuild it')
    for role, name in (('shortlist', args.phase1), ('reranker', args.phase2)):
        built = meta['checkpoints'].get(role)
        if built is not None and built['sha256'] != checkpoint_sha256(args.checkpoint_dir, name):
            sys.exit(f'{args.database} was built with different {role} checkpoints than {args.checkpoint_dir}/{name}: '
                     f'rebuild it with "archer makedb"')
    ids, seqs = read_fasta(os.path.join(args.database, 'sequences.fasta'))
    vec = torch.load(os.path.join(args.database, 'shortlist.pt')).cuda()
    assert len(ids) == len(vec) == meta['n_sequences'], f'{args.database}: inconsistent database files'
    residues = None
    if meta['store_residues']:
        offsets = np.load(os.path.join(args.database, 'residues_offsets.npy'))
        mm = np.memmap(os.path.join(args.database, 'residues.bin'), dtype=np.uint16, mode='r',
                       shape=(int(offsets[-1]), meta['residue_dim']))
        residues = lambda j: torch.from_numpy(np.array(mm[offsets[j]:offsets[j + 1]]).view(np.int16)).view(torch.bfloat16)
    return ids, seqs, vec, residues


def search(args):
    import torch
    require_gpu()
    t0 = time.time()
    q_ids, q_seqs = read_fasta(args.query)
    if os.path.isdir(args.database):
        d_ids, d_seqs, d_vec, residues = open_db(args)
    else:
        d_ids, d_seqs = read_fasta(args.database)
        d_vec, residues = None, None
    nq, nd = len(q_ids), len(d_ids)
    k = nd if args.shortlist <= 0 else min(args.shortlist, nd)
    exclude = None
    if args.exclude_self:
        d_index = {s: j for j, s in enumerate(d_ids)}
        exclude = torch.tensor([d_index.get(s, -1) for s in q_ids], device='cuda')
    print(f'ArCHer: {nq} queries x {nd} database sequences, phase-1 shortlist {k}'
          + (f', database {args.database}' if d_vec is not None else ''), flush=True)

    # phase 1
    encoder, sim_fn = load_model(args.checkpoint_dir, args.phase1)
    q_vec = embed(encoder, q_seqs, args.batch_tokens, 'phase 1: embed queries', pooled=True)
    if d_vec is None:
        d_vec = embed(encoder, d_seqs, args.batch_tokens, 'phase 1: embed database', pooled=True)
    cand, p1 = shortlist(sim_fn, q_vec, d_vec, k, exclude)
    if exclude is not None:
        cand[p1 == float('-inf')] = -1                          # the excluded self-hit, when it fell in the top k
    cand, p1 = cand.cpu(), p1.cpu()
    del encoder, sim_fn, q_vec, d_vec
    torch.cuda.empty_cache()

    # phase 2: the queries and every shortlisted database sequence (stored, or embedded now)
    encoder, sim_fn = load_model(args.checkpoint_dir, args.phase2)
    need = torch.unique(cand[cand >= 0]).tolist()
    local = torch.full((nd,), -1, dtype=torch.long)
    local[need] = torch.arange(len(need))
    q_embs = embed(encoder, q_seqs, args.batch_tokens, 'phase 2: embed queries', pooled=False)
    if residues is not None:
        d_embs = [residues(j) for j in need]
    else:
        d_embs = embed(encoder, [d_seqs[j] for j in need], args.batch_tokens, 'phase 2: embed shortlisted',
                       pooled=False)
    cand_local = torch.where(cand >= 0, local[cand.clamp(min=0)], torch.full_like(cand, -1))
    p2 = rerank(sim_fn, q_embs, d_embs, cand_local)

    # hits: each query's shortlist by phase-2 score
    n = k if args.max_hits <= 0 else min(args.max_hits, k)
    order = p2.argsort(dim=-1, descending=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = 0
    with open(args.output, 'w') as f:
        f.write('query\ttarget\trank\tscore\tphase1_score\n')
        for i in range(nq):
            r = 0
            for c in order[i].tolist():
                if r == n or cand[i, c] < 0:
                    break
                r += 1
                f.write(f'{q_ids[i]}\t{d_ids[cand[i, c]]}\t{r}\t{p2[i, c].item():.6f}\t{p1[i, c].item():.6f}\n')
            rows += r
    print(f'wrote {rows} hits to {args.output} ({time.time() - t0:.0f} s)')


def main():
    args = parse_args()
    {'makedb': makedb, 'search': search}[args.command](args)


if __name__ == '__main__':
    main()
