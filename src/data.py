from utils import *
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

# split files: DATA_DIR/<subset>/<split>.pt (SCOPe, written by data_prep/scope.py) and PFAM_DIR/<split>.pt, each a dict from
# sequence id to its record
DATA_DIR = os.environ.get('ARCHER_DATA_DIR', os.path.join(REPO_ROOT, 'data'))
PFAM_DIR = os.environ.get('ARCHER_PFAM_DIR', os.path.join(REPO_ROOT, 'data', 'pfam'))
SCOPE_SUBSETS = ('astral40', 'astral95', 'scope')
SCOPE_SPLITS = ('train', 'val', 'test', 'full')
PFAM_SPLITS = ('train', 'dev', 'test')
BALIBASE_REFS = ('rv11', 'rv12', 'rv20', 'rv30', 'rv40', 'rv50')          # BAliBASE 3 reference sets


class RetrievalDataset(Dataset):
    """
    Sequences with integer labels at several levels; 'sf' is the retrieval group (a positive shares it) and 'fa' the
    finer level inside it (a positive with another 'fa' is a remote homolog). In SCOPe they are the superfamily and
    family; in Pfam (label='clan') sf is the clan and fa the family. Each item is an anchor and a positive:
    another member of the anchor's sf, from another fa with probability remote_prob when there is one, and the anchor
    itself only when it is alone in its sf.
    data[i] = {'seq', <level>: int id}; for each level, <level>_to_id maps the label string to its id and
    <level>_members the id to its dataset indices.
    max_len: drop sequences longer than this when loading (None keeps all). For training sets: a single long sequence
      pads the whole batch in the PLM and multiplies the DTW size of every pair, so batches holding one tend to run
      out of memory; leave evaluation sets complete
    """
    levels = ('sf', 'fa')

    def __init__(self, records, **kwargs):
        # records: dict sequence id -> {'seq', <level>: label string for each of self.levels}
        max_len = kwargs.get('max_len', None)
        if max_len is not None:
            n = len(records)
            records = {sid: r for sid, r in records.items() if len(r['seq']) <= max_len}
            print(f'{type(self).__name__}: dropped {n - len(records)} of {n} sequences longer than {max_len}')
        self.remote_prob = kwargs.get('remote_prob', 1.0)

        self.sid_list = list(records)
        self.sid_to_id = {sid: i for i, sid in enumerate(self.sid_list)}
        self.data = []
        for lvl in self.levels:
            setattr(self, f'{lvl}_to_id', {})
            setattr(self, f'{lvl}_members', defaultdict(list))
        for i, sid in enumerate(self.sid_list):
            item = {'seq': records[sid]['seq']}
            for lvl in self.levels:
                to_id = getattr(self, f'{lvl}_to_id')
                item[lvl] = to_id.setdefault(records[sid][lvl], len(to_id))
                getattr(self, f'{lvl}_members')[item[lvl]].append(i)
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def draw_positive(self, idx, rng):
        # the positive __getitem__ draws, from the same distribution, with rng (a random.Random): by rejection, so a
        # whole epoch's positives cost O(N) rather than O(sum of sf sizes^2)
        sf, fa = self.data[idx]['sf'], self.data[idx]['fa']
        members = self.sf_members[sf]
        if len(members) == 1:
            return idx
        remote = len(members) > len(self.fa_members[fa])          # a fa lies within one sf
        want_remote = remote and rng.random() < self.remote_prob
        for _ in range(64):
            k = members[rng.randrange(len(members))]
            if k != idx and (not want_remote or self.data[k]['fa'] != fa):
                return k
        # a sf (nearly) all of the anchor's fa: draw from the explicit list
        return rng.choice([k for k in members if k != idx and (not want_remote or self.data[k]['fa'] != fa)])

    def __getitem__(self, idx):
        # idx: an anchor index, its positive drawn here, or an (anchor, positive) pair drawn beforehand (draw_positive)
        if isinstance(idx, (tuple, list)):
            idx, positive_idx = idx
            sf = self.data[idx]['sf']
        else:
            sf, fa = self.data[idx]['sf'], self.data[idx]['fa']
            same_sf = [k for k in self.sf_members[sf] if k != idx] or [idx]
            remote = [k for k in same_sf if self.data[k]['fa'] != fa]
            positive_idx = random.choice(remote if remote and random.random() < self.remote_prob else same_sf)
        return {
            'anchor_idx': idx,
            'positive_idx': positive_idx,
            'anchor': self.data[idx]['seq'],
            'positive': self.data[positive_idx]['seq'],
            'sf': sf,
        }


class SCOPe(RetrievalDataset):
    """
    subset: 'astral40' / 'astral95' (the ASTRAL genetic-domain subsets at < 40% / < 95% identity) or 'scope' (every
    SCOPe 2.08 domain). split: 'train' / 'val' / 'test', disjoint by fold, so val and test folds and superfamilies are
    unseen in train; or 'full', the whole subset. The retrieval group 'sf' and its finer level 'fa' are
      label='sf':   the superfamily, fa the family (remote: a distant homolog)
      label='fold': the fold, fa the superfamily (remote: a structural analog)
    """
    levels = ('fold', 'sf', 'fa')

    def __init__(self, subset='astral40', split='test', label='sf', **kwargs):
        assert subset in SCOPE_SUBSETS and split in SCOPE_SPLITS and label in ('sf', 'fold')
        records = torch.load(os.path.join(DATA_DIR, subset, f'{split}.pt'))
        if label == 'fold':
            records = {sid: {'seq': r['seq'], 'fold': r['fold'], 'sf': r['fold'], 'fa': r['sf']} for sid, r in records.items()}
        super().__init__(records, **kwargs)


class Pfam(RetrievalDataset):
    """
    Pfam seed random split ('train' / 'dev' / 'test', the same families in all three; Pfam 32.0 sequences, Pfam 35.0
    clans), filtered against SCOPe (data_prep/pfam.py), loaded from pfam_dir (default PFAM_DIR). 'fa' is the Pfam
    family; the retrieval group
    'sf' is
      label='clan':   the family's clan, or the family alone when it has none
      label='family': the family itself (remote_* metrics then have no queries)
    Records without a 'clan' field ({'seq', 'fam'}) get it from the clan table the split was filtered with, the
    'clans' source in pfam_dir/manifest.json, so the labels match the clan-level exclusion
    """
    def __init__(self, split='train', label='clan', pfam_dir=None, **kwargs):
        assert split in PFAM_SPLITS and label in ('clan', 'family')
        pfam_dir = pfam_dir or PFAM_DIR
        records = torch.load(os.path.join(pfam_dir, f'{split}.pt'))
        if label == 'clan' and 'clan' not in next(iter(records.values())):
            # Pfam-A.clans.tsv: family accession, clan accession, ...; families without a clan have no entry
            clans_tsv = json.load(open(os.path.join(pfam_dir, 'manifest.json')))['sources']['clans']
            clans = pd.read_csv(clans_tsv, sep='\t', header=None, usecols=[0, 1], names=['pf', 'clan']).dropna()
            pf2clan = dict(zip(clans.pf, clans.clan))
            for r in records.values():
                r['clan'] = pf2clan.get(r['fam'].split('.')[0])
        records = {sid: {'seq': r['seq'], 'fa': r['fam'], 'sf': (r['clan'] or r['fam']) if label == 'clan' else r['fam']}
                   for sid, r in records.items()}
        super().__init__(records, **kwargs)


class CATH(RetrievalDataset):
    """
    CATH v4.4.0 S40 domains (data_prep/cath.py), an evaluation set; name: the set's file, 's40' or one with
    domains excluded, e.g. 's40-nopfamtrain'. 'arch' is the CATH architecture (C.A); the
    retrieval group 'sf' and its finer level 'fa' are
      label='hsf':  the homologous superfamily (C.A.T.H), fa the S35 sequence cluster within it
      label='topo': the topology / fold (C.A.T), fa the homologous superfamily (remote: a structural analog)
    Both levels lie within one architecture, so ranking only same-'arch' documents (RetrieverEval groups='arch')
    keeps every positive and leaves negatives with the same secondary-structure arrangement but another connectivity
    """
    levels = ('arch', 'sf', 'fa')

    def __init__(self, label='hsf', name='s40', **kwargs):
        assert label in ('hsf', 'topo')
        fa = {'hsf': 's35', 'topo': 'hsf'}[label]
        records = torch.load(os.path.join(DATA_DIR, 'cath', f'{name}.pt'))
        super().__init__({sid: {'seq': r['seq'], 'arch': r['arch'], 'sf': r[label], 'fa': r[fa]}
                          for sid, r in records.items()}, **kwargs)


class BAliBASE(RetrievalDataset):
    """
    The BAliBASE 3 reference alignments pooled into one evaluation set (data_prep/balibase.py); name: 'all', one
    reference set ('rv11'), or either with sequences excluded ('all-nopfamtrain'). The retrieval group 'sf' is the family (alignments sharing a sequence merged);
    'fa' is the sequence itself, so remote_* metrics equal all_*. alns[i]: the reference alignments holding sequence i
    (e.g. {'BB11001', 'BB20001'}), to pick the queries of one reference set
    """
    def __init__(self, name='all', **kwargs):
        records = torch.load(os.path.join(DATA_DIR, 'balibase', f'{name}.pt'))
        super().__init__({sid: {'seq': r['seq'], 'sf': r['fam'], 'fa': sid} for sid, r in records.items()}, **kwargs)
        self.alns = [set(records[sid]['alns'].split(',')) for sid in self.sid_list]


if __name__ == '__main__':
    for subset in SCOPE_SUBSETS:
        for split in SCOPE_SPLITS:
            d = SCOPe(subset, split)
            print(f'{subset} {split}: {len(d)} seqs, {len(d.fold_to_id)} folds, {len(d.sf_to_id)} sf, {len(d.fa_to_id)} fa')
    for split in PFAM_SPLITS:
        d = Pfam(split)
        print(f'pfam {split}: {len(d)} seqs, {len(d.sf_to_id)} clans/families, {len(d.fa_to_id)} families')
    loader = DataLoader(Pfam('train'), batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(batch['anchor'][0], batch['positive'][0], batch['sf'][:8])
