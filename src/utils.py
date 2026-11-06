import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn as nn
import torch.nn.functional as F  # used by subclass towers
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from tqdm import trange
from pprint import pprint
import pickle
import time
from functools import partial
from abc import ABC, abstractmethod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this file is <repo>/src/utils.py
NEG_INF = -1e8  # safer than -inf: fully-padded rows won't produce NaN
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_CHAR = '!'   # one-character stand-in for the PLM's pad token in a sequence string

def set_seed(seed=2603):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_module(module, unfreeze=False):
    for param in module.parameters():
        param.requires_grad = unfreeze

def num_params(net):
    return sum(p.numel() for p in net.parameters())

def assess_cuda_memory(header=''):
    print(f'{header}-------------------------------')
    print(f'Allocated : {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
    print(f'Reserved  : {torch.cuda.memory_reserved()  / 1024**2:.2f} MB')


def random_protein_sequence(length):
    return "".join(random.choices(AMINO_ACIDS, k=length))

def read_fasta(path):
    sid, sccs, buf = None, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if sid is not None:
                    yield sid, sccs, ''.join(buf)
                sid, sccs = (line[1:].split() + [None])[:2]          # a bare '>sid' header (CATH) has no sccs
                buf = []
            else:
                buf.append(line.strip())
    if sid is not None:
        yield sid, sccs, ''.join(buf)



# ---- data preparation (data_prep/*.py)

MMSEQS = os.environ.get('MMSEQS', 'mmseqs')   # MMseqs2 18.8cc5c, on PATH unless MMSEQS names it


def check_outputs(paths, force):
    # the --force guard of every data_prep step: stop if an output exists (unless force), create the output folder
    existing = [p for p in paths if os.path.exists(p)]
    assert force or not existing, f'{existing} already exist; pass --force to overwrite'
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)


def filter_fasta(query_fa, db_fa, out_dir, k=25, evalue=1e-3, sensitivity=7.5, max_seqs=300, threads=32, force=False):
    # Filters one FASTA against another: drops every sequence of query_fa that overlaps any sequence of db_fa, e.g. a
    # test set against a training set, so no test sequence is homologous to (or a near copy of) something trained on.
    # A query overlaps when either test flags it:
    #   - homology: an MMseqs2 alignment to a db sequence with E-value <= evalue (easy-search -s <sensitivity>; 7.5 is
    #     MMseqs2's high-sensitivity setting; --max-seqs only caps the hits kept per query, not which queries hit)
    #   - near-identity: an exact run of k residues (k-mer) shared with a db sequence, a backstop for near copies
    # Sequence ids are the first word of each FASTA header (what MMseqs2 reports too). Writes to out_dir:
    #   removed.txt   the dropped query ids, one per line (build --exclude of data_prep/cath.py, balibase.py)
    #   kept.fa       the remaining queries, headers and sequences as in query_fa
    #   hits.m8       the MMseqs2 hits, one row per (query, db) pair: query, target, pident, alnlen, qlen, tlen,
    #                 qstart, qend, tstart, tend, evalue, bits
    #   kmer<k>.tsv   per query: id, k-mers shared with db, k-mers in total
    # Returns the sorted dropped ids. Needs the MMseqs2 binary (MMSEQS); ~10 min on 32 CPUs for CATH S40 (35k) against
    # Pfam train (400k), and a few GB of RAM for db's k-mer set
    import subprocess
    removed = os.path.join(out_dir, 'removed.txt')
    check_outputs([removed], force)                                 # refuses to overwrite without force

    # 1. homology: every query against all of db. MMseqs2 builds its own databases in out_dir/tmp, removed afterwards
    m8, tmp = os.path.join(out_dir, 'hits.m8'), os.path.join(out_dir, 'tmp')
    subprocess.run([MMSEQS, 'easy-search', query_fa, db_fa, m8, tmp, '--threads', str(threads), '-s', f'{sensitivity:g}',
                    '-e', f'{evalue:g}', '--max-seqs', str(max_seqs), '--format-output',
                    'query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits'], check=True)
    subprocess.run(['rm', '-rf', tmp], check=True)
    hits = {l.split('\t', 1)[0] for l in open(m8)}                  # first column: the query id

    # 2. near-identity: db's k-mers in one set (upper case, as the PLMs see residues), then each query's looked up
    kmers = set()
    for _, _, seq in read_fasta(db_fa):
        seq = seq.upper()
        kmers.update(seq[i:i + k] for i in range(len(seq) - k + 1))
    shared = set()
    with open(os.path.join(out_dir, f'kmer{k}.tsv'), 'w') as f:
        for sid, _, seq in read_fasta(query_fa):
            seq = seq.upper()
            n = sum(seq[i:i + k] in kmers for i in range(len(seq) - k + 1))
            f.write(f'{sid}\t{n}\t{max(len(seq) - k + 1, 0)}\n')
            if n:
                shared.add(sid)
    del kmers

    # 3. drop the union; copy the other queries' FASTA records unchanged (header lines included) into kept.fa
    drop = sorted(hits | shared)
    with open(removed, 'w') as f:
        f.writelines(f'{sid}\n' for sid in drop)
    n_query, keep = 0, False
    with open(os.path.join(out_dir, 'kept.fa'), 'w') as f:
        for line in open(query_fa):
            if line.startswith('>'):
                n_query += 1
                keep = line[1:].split()[0] not in hits | shared
            if keep:
                f.write(line)
    print(f'{os.path.basename(query_fa)} vs {os.path.basename(db_fa)}: {len(drop)} of {n_query} removed '
          f'({len(hits)} MMseqs2 hits, {len(shared)} sharing a {k}-mer, {len(shared - hits)} by the {k}-mer test only) -> {out_dir}')
    return drop


def pad_embeddings(embs):
    """Stack list of variable-length (T_i, D) tensors into (B, T_max, D) with a boolean mask."""
    B, D = len(embs), embs[0].size(-1)
    T_max = max(e.size(0) for e in embs)
    device = embs[0].device
    padded = torch.zeros(B, T_max, D, device=device)
    mask   = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for i, e in enumerate(embs):
        T = e.size(0)
        padded[i, :T] = e
        mask[i, :T]   = True
    return padded, mask

def masked_mean(emb, mask):
    """emb: B x L x D, mask: B x L (1 = real token) -> B x D"""
    mask = mask.unsqueeze(-1).to(emb.dtype)
    return (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)