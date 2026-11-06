# ArCHer

Traditional search tools identify homologs by comparing amino acid sequences or sequence-derived profiles. While
effective for closely related proteins, these methods have limited sensitivity in the low-identity "twilight zone"
where remote homologs can be difficult to detect.

ArCHer, an **A**lignment-awa**r**e **C**ontrastive **H**omolog R**e**t**r**ieval uses
the alignment score between residue embeddings as the contrastive similarity, directly training PLM representations
for alignment-based homolog discrimination.

For scalable search, ArCHer has two phases:

1. **Shortlist** (`ArCHer_shortlist`): first retrieves candidates by vector similarity,
2. **Rerank** (`ArCHer_reranker`): and then re-ranks them by alignment using fine-tuned embeddings.

---

## Quickstart

### 1) Clone
```
git clone https://github.com/Singh-Lab/ArCHer.git
cd ArCHer
```

### 2) Create the environment and install ArCHer
```
conda env create -f environment.yml
conda activate ArCHer
pip install -e .
```
This installs the `archer` command. Install with `-e` (editable): `archer` then finds the `checkpoints/` folder of
this clone from any directory.

ArCHer currently requires an NVIDIA GPU (CUDA). Its alignment kernels are written in
[Triton](https://github.com/triton-lang/triton).

### 3) Download the checkpoints
```
hf download iarmourgarb/ArCHer --include "ArCHer_*" --local-dir checkpoints
```
This downloads four files into `checkpoints/`: `ArCHer_shortlist.pt` and `ArCHer_reranker.pt` (the weights, about 600 MB
each) and a `.json` file of training settings for each. The checkpoints are a pre-release and may still be updated.

The base ProtT5 model ([Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc),
about 2.3 GB) is downloaded automatically from Hugging Face the first time `archer` runs. If your GPU machines have no
internet access, download it once from a machine that does, into the same
Hugging Face cache:
```
hf download Rostlab/prot_t5_xl_half_uniref50-enc
```

### 4) Run the example
```
archer search -q examples/queries.fasta -d examples/database.fasta -o outputs/example_hits.tsv
```
`examples/` holds 5 query domains from SCOPe ASTRAL40 (one each from SCOPe classes a to e) and a 500-sequence
database. For each query, the database holds 4 to 14 remote homologs (same superfamily, different family) among
unrelated ASTRAL40 domains; the queries' own families are left out. Each FASTA header carries the domain's SCOPe
family (e.g. `>d2cria1 b.1.11.0`), so you can check that the top hits share the query's superfamily (the first three
fields). Once ProtT5 is downloaded, the example should take under a minute on an A100 GPU, most of it loading the models.

---

## Databases

If you will search the same database more than once, build it
once with `archer makedb` and point every search at the result:
```
archer makedb my_database.fasta my_db/
archer search -q my_queries.fasta -d my_db/ -o my_hits.tsv
```
`makedb` takes a FASTA of protein sequences (same format rules as queries, see below) and writes a database folder.
Each later search then only embeds its queries, plus the shortlisted database sequences for the reranker. Searching a
database folder gives the same hits as searching the FASTA it was built from.

### Storing reranker embeddings

By default, a database stores one shortlist vector per sequence, about 2 KB per sequence (2 GB per million
sequences). The reranker still embeds each search's shortlisted sequences on the fly. With `--store-residues`, the
database also stores the reranker's per-residue embeddings, so searches skip that step:
```
archer makedb my_database.fasta my_db/ --store-residues
```
This takes about 2 KB per residue on disk. Measured sizes:

| database | sequences | residues | shortlist vectors | residue embeddings |
|---|---|---|---|---|
| `examples/database.fasta` | 500 | 90,000 | 1 MB | 180 MB |
| SCOPe ASTRAL40 | 15,000 | 2.8 million | 30 MB | 5.7 GB |

A database of 100,000 sequences averaging 300 residues needs about 60 GB of residue embeddings. Storing residues suits
small and medium databases that are searched often. With stored residues, reranker scores can differ from on-the-fly
embedding by up to about 0.001, which can swap hits with near-identical scores.

### Database folder contents

| file | contents |
|---|---|
| `sequences.fasta` | the database sequences |
| `shortlist.pt` | shortlist vectors |
| `residues.bin`, `residues_offsets.npy` | reranker residue embeddings (`--store-residues` only) |
| `archer_db.json` | database description, written last: a folder without it is an unfinished build |

A database is tied to the checkpoints that built it. `archer_db.json` records a fingerprint of each
checkpoint used, and `archer search` refuses a database built with different checkpoints. If you download updated
checkpoints, rebuild your databases. `makedb` will not write into a folder that already has files in it; delete the
old folder or choose a new name.

For a one-off search you can skip `makedb` and pass a FASTA as `-d`. The database is then embedded on the fly and not
kept.

---

## Searching your own sequences

```
archer search -q my_queries.fasta -d my_db/ -o my_hits.tsv
```

- **Input:** a FASTA of query protein sequences, and a database: a folder from `archer makedb` or a FASTA. Sequence
  ids (the first word of each header) must be unique within a file. The rare residues U, Z, O and B are read as X,
  as in ProtT5's training data. Sequences of any length are accepted; ones longer than
  1022 residues are embedded in overlapping windows.
- **Searching a set against itself:** use the same sequences as queries and database and add `--exclude-self`, so no
  query reports itself as a hit.
- **Exhaustive search:** `-k 0` reranks the whole database for every query instead of a shortlist. This is more
  accurate but its cost grows with queries × database size, so it suits small databases. With a database smaller than
  `--shortlist`, the search is exhaustive anyway.

### Output

A tab-separated file with one row per hit, ranked best first within each query:

| column | meaning |
|---|---|
| `query` | query id |
| `target` | database id |
| `rank` | rank of the hit for this query (1 = best) |
| `score` | reranker score, which sets the ranking: the alignment score per query residue. Higher is more similar. Scores are comparable between hits of the same query, not across queries. |
| `phase1_score` | shortlist score: cosine similarity of the two sequences' vectors |

By default, every shortlisted sequence is written (`--shortlist` hits per query, 4,096 by default). Use `-n` to
write only each query's top hits, e.g. `-n 100`.

---

## CLI Reference

```
usage: archer makedb FASTA DB [--store-residues] [common options]

FASTA                Database sequences.
DB                   Output database folder (must not exist yet, or be empty).
--store-residues     Also store the reranker's residue embeddings (about 2 KB per residue).

usage: archer search -q QUERY -d DATABASE -o OUTPUT [-k SHORTLIST] [-n MAX_HITS]
                     [--exclude-self] [common options]

--query, -q          FASTA of query sequences.
--database, -d       Database folder from "archer makedb", or a FASTA of database sequences.
--output, -o         Output TSV of hits. Its folder is created if missing.
--shortlist, -k      Shortlist size per query: the database sequences the reranker scores.
                     0 = rerank the whole database (exhaustive). Default: 4096
--max-hits, -n       Hits written per query; 0 = the whole shortlist. Default: 0
--exclude-self       Drop hits whose database id equals the query id.

common options (both commands):
--checkpoint-dir     Folder holding the checkpoints. Default: checkpoints/ in this repository
--phase1             Shortlist checkpoint name (<name>.pt and <name>.json). Default: ArCHer_shortlist
--phase2             Reranker checkpoint name. Default: ArCHer_reranker
--batch-tokens       Max padded residues per embedding batch. Default: 23170
```
Run `archer makedb -h` or `archer search -h` for the same lists.

---

## Training

ArCHer is meant to be used with the released checkpoints. The training code is in this repository (`src/train.py`),
and `data/pfam/` holds the Pfam training and validation splits:
`train.pt` and `dev.pt`, the Pfam seed random split with every family that hits an ASTRAL40 domain removed, along with
all other families in its clan (`manifest.json` records how they were built).

## Repository layout

| file | contents |
|---|---|
| `src/run_archer.py` | the `archer` command |
| `src/encoder.py`, `src/plm_wrapper.py` | ProtT5 encoder with the fine-tuned last layers |
| `src/sim.py`, `src/dtw_triton_kernels.py` | similarity scorers, including the qNW alignment and its Triton GPU kernels |
| `src/utils.py` | shared helpers |
| `src/train.py`, `src/retriever.py`, `src/infonce.py`, `src/data.py`, `src/cache_plm.py` | training |
| `examples/` | example queries and database |
| `data/pfam/` | Pfam training and validation splits |
| `checkpoints/` | model checkpoints (downloaded, not in git) |
