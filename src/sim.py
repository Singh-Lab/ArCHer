from utils import *
from dtw_triton_kernels import _align, _soft_hvp, _qnw_align


# ---- Residue similarities: how similar two residue embeddings are

class ResidueSim(ABC):
    """
    Residue similarity used by every SequenceSim, split in two so each scorer keeps its own precision handling:
      features(x): per-residue transform, applied outside the scorer's bf16 autocast block
      matrix(q, d, q_mask, d_mask): (Bq, Lq, D) x (Bd, Ld, D) -> (Bq, Bd, Lq, Ld) similarity of every residue pair,
        called inside it; the masks (True = real residue) are for similarities that normalize over residues
    DenseSim applies it to the pooled vectors (L = 1)
    """
    def features(self, x):
        return x

    @abstractmethod
    def matrix(self, q, d, q_mask=None, d_mask=None):
        ...

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(f"{k}={v}" for k, v in vars(self).items())})'


class CosineSim(ResidueSim):
    def features(self, x):
        return F.normalize(x, dim=-1)

    def matrix(self, q, d, q_mask=None, d_mask=None):
        return torch.einsum('bid,cjd->bcij', q, d)

COSINE = CosineSim()


class AffineMutualSoftmax(ResidueSim):
    """
    How strongly q_i and d_j pick each other among the other sequence's residues, from the L2 distance of the raw
    embeddings: R = softmax_j, C = softmax_i of -|q_i - d_j| / temperature, averaged and mapped affinely,
      sim(i, j) = (1 + beta) * avg(R_ij, C_ij) - beta,  in [-beta, 1]
    average='arithmetic': (R + C) / 2, already 1/2 when only one side picks the other
    average='geometric':  sqrt(R C), high only when both do; computed as exp((log R + log C) / 2), so its gradient stays
      finite where R C underflows
    beta > 0 pushes pairs that neither side picks below 0 (local DTW can then leave them out of an alignment).
    Padded residues are left out of both softmaxes and score 0. Computed in float32 even inside autocast:
    |q|^2 + |d|^2 - 2<q, d> cancels badly in bf16. Needs residue axes, so it is meaningless for DenseSim (a softmax
    over one vector is 1)
    """
    def __init__(self, temperature=1., average='arithmetic', beta=0.):
        assert average in ('arithmetic', 'geometric')
        self.temperature, self.average, self.beta = temperature, average, beta

    def logits(self, q, d, q_mask=None, d_mask=None):
        # float32 -|q_i - d_j| / T, (Bq, Bd, Lq, Ld), padded cells at NEG_INF; and the valid-cell mask (None without masks).
        # Call inside torch.autocast('cuda', enabled=False)
        q, d = q.float(), d.float()
        sq = q.pow(2).sum(-1)[:, None, :, None] + d.pow(2).sum(-1)[None, :, None, :] - 2 * torch.einsum('bid,cjd->bcij', q, d)
        logits = -sq.clamp_min(1e-8).sqrt() / self.temperature               # clamp: finite gradient at distance 0
        if q_mask is None and d_mask is None:
            return logits, None
        qv = q_mask.bool() if q_mask is not None else torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        dv = d_mask.bool() if d_mask is not None else torch.ones(d.shape[:2], dtype=torch.bool, device=d.device)
        valid = qv[:, None, :, None] & dv[None, :, None, :]
        return logits.masked_fill(~valid, NEG_INF), valid

    def matrix(self, q, d, q_mask=None, d_mask=None):
        with torch.autocast('cuda', enabled=False):
            logits, valid = self.logits(q, d, q_mask, d_mask)
            if self.average == 'arithmetic':
                S = 0.5 * (logits.softmax(-1) + logits.softmax(-2))
            else:
                S = (0.5 * (logits.log_softmax(-1) + logits.log_softmax(-2))).exp()
            S = (1 + self.beta) * S - self.beta
            return S if valid is None else S.masked_fill(~valid, 0.)


# ---- Sequence similarities: (q_emb (Bq, Lq, D), q_mask (Bq, Lq), d_emb (Bd, Ld, D), d_mask (Bd, Ld)) -> (Bq, Bd)

class SequenceSim(ABC):
    """
    Scores every query against every document from their per-residue embeddings, with the residue similarity sim.
    Instances are called like functions, so they plug into InfoNCE and RetrieverEval as the scorer. All handle Bq != Bd
    """
    def __init__(self, sim=COSINE):
        self.sim = sim

    @abstractmethod
    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        ...

    def pair_matrix(self, q_emb, q_mask, d_emb, d_mask):
        # residue similarity of every (query, document) pair flattened to (Bq * Bd, Lq, Ld), computed in bf16 autocast,
        # and the pairs' masks (Bq * Bd, Lq), (Bq * Bd, Ld)
        Bq, Lq, _ = q_emb.shape
        Bd, Ld, _ = d_emb.shape
        q, d = self.sim.features(q_emb), self.sim.features(d_emb)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            M = self.sim.matrix(q, d, q_mask, d_mask).reshape(Bq * Bd, Lq, Ld)
        qm = q_mask[:, None, :].expand(Bq, Bd, Lq).reshape(Bq * Bd, Lq)
        dm = d_mask[None, :, :].expand(Bq, Bd, Ld).reshape(Bq * Bd, Ld)
        return M, qm, dm

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(f"{k}={v}" for k, v in vars(self).items())})'


class DenseSim(SequenceSim):
    # masked mean of the residue features, then sim on the pooled vectors (cosine of the mean embeddings by default)
    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        q = self.sim.features(masked_mean(q_emb, q_mask))           # Bq x D
        d = self.sim.features(masked_mean(d_emb, d_mask))           # Bd x D
        return self.sim.matrix(q[:, None], d[:, None])[:, :, 0, 0]


class MaxSim(SequenceSim):
    # ColBERT late interaction: each query residue takes its most similar document residue; reduce 'mean' or 'sum' over
    # the query residues
    def __init__(self, sim=COSINE, reduce='mean'):
        super().__init__(sim)
        assert reduce in ('mean', 'sum')
        self.reduce = reduce

    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        S = self.sim.matrix(self.sim.features(q_emb), self.sim.features(d_emb), q_mask, d_mask)   # Bq x Bd x Lq x Ld
        S = S.masked_fill(~d_mask.bool()[None, :, None, :], NEG_INF)
        S = S.max(dim=-1).values                                  # Bq x Bd x Lq
        qm = q_mask.bool().to(S.dtype)[:, None, :]
        out = (S * qm).sum(-1)
        return out / qm.sum(-1).clamp(min=1.0) if self.reduce == 'mean' else out


class _HardDTWScore(torch.autograd.Function):
    # s = <E, M> / l per pair, E the optimal path of hard DTW on cost -M with its step weights (0, 1 or 2), l = sum E
    # ('mean') or 1 ('sum'). The path does not move with M, so ds/dM = E / l. The kernels run in chunks of pairs so only
    # one chunk's buffers are alive at a time, and E is kept for the backward as uint8 (1 byte per cell, exact).
    # The sum runs in float32 with autocast off: under autocast einsum would round M to bf16 first

    @staticmethod
    @torch.autocast('cuda', enabled=False)
    def forward(ctx, M, qm, dm, chunk, symmetric, mean):
        P = M.shape[0]
        n, m = qm.sum(-1).long().contiguous(), dm.sum(-1).long().contiguous()
        s, l = M.new_zeros(P, dtype=torch.float32), M.new_ones(P, dtype=torch.float32)
        W = torch.empty(M.shape, dtype=torch.uint8, device=M.device)
        for a in range(0, P, chunk):
            b = slice(a, a + chunk)
            Mc = M[b].float()
            _, E = _align((-Mc).contiguous(), n[b], m[b], symmetric=symmetric, omega='hard')
            s[b] = torch.einsum('bij,bij->b', Mc, E)
            if mean:
                l[b] = E.sum((1, 2)).clamp_min(1)
            W[b] = E.to(torch.uint8)
            del Mc, E
        ctx.save_for_backward(W, l)
        ctx.m_dtype = M.dtype
        return s / l

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        W, l = ctx.saved_tensors
        return (W.float() * (g / l)[:, None, None]).to(ctx.m_dtype), None, None, None, None, None


class DTWSim(SequenceSim):
    """
    Hard DTW: the optimal alignment path of the residue similarity M (dtw_triton_kernels on cost -M), score = the sum of
    M along the path, divided by the path's weight when reduce='mean'. The path does not move with M, so gradients flow
    only through the path cells; ties break diagonal, then up, then left. The kernels run in chunks of at most
    chunk_cells DP cells (pairs * Lq * Ld), so their buffers do not grow with Bq * Bd (_HardDTWScore).
    symmetric: diagonal steps weigh 2, the others 1, so every path weighs n + m and the max-sum path is also the
      max-mean path (plain steps favor the longest staircase when M > 0)
    """
    def __init__(self, sim=COSINE, reduce='mean', symmetric=True, chunk_cells=2**28):
        super().__init__(sim)
        assert reduce in ('mean', 'sum')
        self.reduce, self.symmetric, self.chunk_cells = reduce, symmetric, chunk_cells

    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        M, qm, dm = self.pair_matrix(q_emb, q_mask, d_emb, d_mask)
        chunk = max(1, self.chunk_cells // (M.shape[1] * M.shape[2]))
        return _HardDTWScore.apply(M, qm, dm, chunk, self.symmetric, self.reduce == 'mean').reshape(Bq, Bd)


class _QueryDTWScore(torch.autograd.Function):
    # s = (1/n) sum_i max_{j: (i, j) on P} M_ij per pair, P the optimal hard DTW path (as _HardDTWScore's). The path and
    # each row's pick do not move with M, so ds/dM = 1/n at each query row's picked cell; the picks are kept for the
    # backward as int32 (B, Lq). In float32 with autocast off, as _HardDTWScore

    @staticmethod
    @torch.autocast('cuda', enabled=False)
    def forward(ctx, M, qm, dm, chunk, symmetric):
        P = M.shape[0]
        n, m = qm.sum(-1).long().contiguous(), dm.sum(-1).long().contiguous()
        rows = qm.bool()
        s = M.new_zeros(P, dtype=torch.float32)
        J = torch.zeros(M.shape[:2], dtype=torch.int32, device=M.device)
        for a in range(0, P, chunk):
            b = slice(a, a + chunk)
            Mc = M[b].float()
            _, E = _align((-Mc).contiguous(), n[b], m[b], symmetric=symmetric, omega='hard')
            val, j = Mc.masked_fill(E == 0, float('-inf')).max(-1)   # every valid query row has a path cell
            s[b] = val.masked_fill(~rows[b], 0.).sum(-1)
            J[b] = j.to(torch.int32)
            del Mc, E, val, j
        ctx.save_for_backward(J, rows, n)
        ctx.m_shape, ctx.m_dtype = M.shape, M.dtype
        return s / n.clamp_min(1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        J, rows, n = ctx.saved_tensors
        gM = torch.zeros(ctx.m_shape, dtype=torch.float32, device=g.device)
        w = (g / n.clamp_min(1))[:, None].expand_as(J) * rows
        gM.scatter_(2, J.long()[..., None], w[..., None].float())
        return gM.to(ctx.m_dtype), None, None, None, None


class QueryDTWSim(SequenceSim):
    """
    Query-side DTW: the hard DTW path as DTWSim finds it, scored on the query only: each query residue takes its best
    cell among the path cells in its row, and the score is their mean over the query residues. Document residues the
    path crosses besides those picks do not count, so a document longer than the query is not penalized for its extra
    residues. Asymmetric: score(q, d) != score(d, q). Gradients flow only through the picked cells (_QueryDTWScore).
    symmetric: the step weights of the path search, as DTWSim's
    """
    def __init__(self, sim=COSINE, symmetric=True, chunk_cells=2**28):
        super().__init__(sim)
        self.symmetric, self.chunk_cells = symmetric, chunk_cells

    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        M, qm, dm = self.pair_matrix(q_emb, q_mask, d_emb, d_mask)
        chunk = max(1, self.chunk_cells // (M.shape[1] * M.shape[2]))
        return _QueryDTWScore.apply(M, qm, dm, chunk, self.symmetric).reshape(Bq, Bd)


# ---- Gap relations (qNWSim): how much a residue inserted into an alignment gap resembles an anchor residue of the other sequence

class GapRelation(nn.Module):
    """
    phi(x, a) for qNWSim's gap scores: x a residue inserted into a gap, a a residue of the other sequence at the
    insertion point (each gap takes phi to the better of its two flanking anchors). Higher = the insertion resembles
    the anchor (a stretch), so the gap is cheaper.
    Define a new form by subclassing and implementing relate(x, a): (B1, L1, D) x (B2, L2, D) -> (B1, B2, L1, L2), phi
    of every (x_k, a_l) pair, from the raw residue embeddings. symmetric = True when phi(x, a) == phi(a, x): one matrix
    then serves both gap directions. Learned parameters are created in build(dim), called once the embedding size is
    known (qNWSim.bind, from train.attach_scorer), so they train with the encoder and load from its checkpoints.
    reads_match = True: phi is the match's own residue similarity S, which qNWSim passes in (flank_mu) instead of
    calling the relation
    """
    symmetric = False
    reads_match = False

    def build(self, dim):
        pass

    @abstractmethod
    def relate(self, x, a):
        ...

    def forward(self, q, d, q_mask=None, d_mask=None):
        # (CD, CQ), each (Bq * Bd, Lq, Ld): CD[i, j] = phi(d_j, q_i) (doc gaps), CQ[i, j] = phi(q_i, d_j) (query gaps);
        # the masks are for relations that normalize over residues (PickGap), the others ignore them
        Bq, Lq, _ = q.shape
        Bd, Ld, _ = d.shape
        CQ = self.relate(q, d).reshape(Bq * Bd, Lq, Ld)
        CD = CQ if self.symmetric else self.relate(d, q).permute(1, 0, 3, 2).reshape(Bq * Bd, Lq, Ld)
        return CD, CQ


class CosineGap(GapRelation):
    # phi = cos(x, a) of the raw embeddings; absolute, unlike the competitive mutual softmax: a stretch, where two
    # residues compete for one partner, keeps a high cosine
    symmetric = True

    def relate(self, x, a):
        return torch.einsum('bid,cjd->bcij', F.normalize(x, dim=-1), F.normalize(a, dim=-1))


class BilinearGap(GapRelation):
    # phi = <U x^, V a^>, x^ a^ unit-normalized, U V (rank x dim): a learned, asymmetric compatibility of the inserted
    # residue with its anchor; U = V = a random orthonormal projection at init (scaled so phi starts on the cosine's scale)
    def __init__(self, rank=32):
        super().__init__()
        self.rank = rank

    def build(self, dim):
        P = torch.linalg.qr(torch.randn(dim, self.rank))[0].T * (dim / self.rank) ** 0.5   # E <P x, P a> = cos(x, a)
        self.U, self.V = nn.Parameter(P.clone()), nn.Parameter(P.clone())

    def relate(self, x, a):
        x, a = F.normalize(x, dim=-1), F.normalize(a, dim=-1)
        return torch.einsum('bir,cjr->bcij', x @ self.U.T.to(x.dtype), a @ self.V.T.to(a.dtype))

    def extra_repr(self):
        return f'rank={self.rank}'


class DistanceGap(GapRelation):
    # phi = -|L (x^ - a^)|^2, x^ a^ unit-normalized, L (rank x dim): closeness in a learned metric, symmetric; L a random
    # orthonormal projection at init (scaled so phi starts on the scale of -|x^ - a^|^2 = 2 cos - 2)
    symmetric = True

    def __init__(self, rank=32):
        super().__init__()
        self.rank = rank

    def build(self, dim):
        self.L = nn.Parameter(torch.linalg.qr(torch.randn(dim, self.rank))[0].T * (dim / self.rank) ** 0.5)   # E |L v|^2 = |v|^2

    def relate(self, x, a):
        x, a = F.normalize(x, dim=-1), F.normalize(a, dim=-1)
        lx, la = x @ self.L.T.to(x.dtype), a @ self.L.T.to(a.dtype)
        dot = torch.einsum('bir,cjr->bcij', lx, la)
        sq = lx.pow(2).sum(-1)[:, None, :, None] + la.pow(2).sum(-1)[None, :, None, :] - 2 * dot
        return -sq

    def extra_repr(self):
        return f'rank={self.rank}'


class PickGap(GapRelation):
    # phi(x, a) = how strongly x picks a among the other sequence's residues: the one-sided half of the geometric mutual
    # softmax, softmax over a of -|x - a| / T, so CQ = R (q_i's pick of d_j) and CD = C (d_j's pick of q_i). One-sided
    # because a gap's inserted residue is what is being explained: in a stretch (q_{i-1} matched to d_j, q_i a copy of
    # q_{i-1}) q_i still picks d_j, while d_j's own pick splits between the two copies and would dilute sqrt(R C)
    def __init__(self, temperature=0.5):
        super().__init__()
        self.msm = AffineMutualSoftmax(temperature, 'geometric')

    def forward(self, q, d, q_mask=None, d_mask=None):
        Bq, Lq, _ = q.shape
        Bd, Ld, _ = d.shape
        with torch.autocast('cuda', enabled=False):
            logits, _ = self.msm.logits(q, d, q_mask, d_mask)        # (Bq, Bd, Lq, Ld), padded cells at NEG_INF
            CQ, CD = logits.softmax(-1), logits.softmax(-2)
        return CD.reshape(Bq * Bd, Lq, Ld), CQ.reshape(Bq * Bd, Lq, Ld)

    def relate(self, x, a):
        raise NotImplementedError('PickGap normalizes over the other sequence: forward only')

    def extra_repr(self):
        return f'temperature={self.msm.temperature:g}'


class NullGap(GapRelation):
    # the match's own logits l(x, a) = -|x - a| / T, unnormalized (qNWNullSim normalizes them against a null, not over
    # the other sequence); symmetric, so one matrix serves both gap directions. Padded cells at NEG_INF
    def __init__(self, temperature=0.5):
        super().__init__()
        self.msm = AffineMutualSoftmax(temperature, 'geometric')

    def forward(self, q, d, q_mask=None, d_mask=None):
        Bq, Lq, _ = q.shape
        Bd, Ld, _ = d.shape
        with torch.autocast('cuda', enabled=False):
            logits, _ = self.msm.logits(q, d, q_mask, d_mask)
        logits = logits.reshape(Bq * Bd, Lq, Ld)
        return logits, logits

    def relate(self, x, a):
        raise NotImplementedError('NullGap needs the masks: forward only')

    def extra_repr(self):
        return f'temperature={self.msm.temperature:g}'


class GeoGap(GapRelation):
    # phi = S, the match's residue similarity itself (qNW-geogap: the geometric mutual softmax with its b, in [-b, 1]),
    # so a gap is cheap when the inserted residue would have been a good mutual match for a flanking anchor. Read from
    # qNWSim's pair matrix, nothing recomputed. Competitive, unlike the cosine: in a stretch, the anchor's softmax
    # splits between the copies and dilutes sqrt(R C) at the flank
    symmetric = True
    reads_match = True

    def forward(self, q, d, q_mask=None, d_mask=None):
        raise NotImplementedError('GeoGap reads qNWSim\'s match matrix S: flank_mu passes it in')

    def relate(self, x, a):
        raise NotImplementedError('GeoGap reads qNWSim\'s match matrix S: flank_mu passes it in')


GAP_RELATIONS = {'cosine': CosineGap, 'bilinear': BilinearGap, 'distance': DistanceGap, 'geo': GeoGap}


class _qNWScore(torch.autograd.Function):
    # s = v / n per pair, v the qNW value (dtw_triton_kernels._qnw_align) of the substitution matrix S and the gap
    # score matrices GDO, GQO (and, affine, GDE, GQE; None for linear gaps), each step reading its own entry. On the
    # optimal path P, s = (1/n) [sum_{match} S + sum_{gap steps} G], and P does not move with the inputs, so
    # ds/dS = 1/n on the match cells and ds/dG = 1/n on the gap steps that read it; the gap functions' parameters get
    # theirs through autograd. The forward keeps P as uint8 step codes (1 match, 2 / 4 query gap open / extend,
    # 3 / 5 doc gap open / extend; linear gaps use 1-3); the backward runs in the same chunks of pairs
    @staticmethod
    @torch.autocast('cuda', enabled=False)
    def forward(ctx, S, GDO, GQO, GDE, GQE, qm, dm, chunk, store):
        # store: keep the paths for a backward (False under no_grad: only the values are computed)
        P = S.shape[0]
        n, m = qm.sum(-1).long().contiguous(), dm.sum(-1).long().contiguous()
        affine = GDE is not None
        val = S.new_zeros(P, dtype=torch.float32)
        paths = []
        c = lambda G, b: None if G is None else G[b].float().contiguous()
        for a in range(0, P, chunk):
            b = slice(a, a + chunk)
            v, path = _qnw_align(c(S, b), c(GDO, b), c(GQO, b), n[b], m[b], GDE=c(GDE, b), GQE=c(GQE, b), store=store)
            val[b] = v
            if store:
                paths.append(path)
        if store:
            ctx.save_for_backward(torch.cat(paths), n)
            ctx.chunk, ctx.affine = chunk, affine
            ctx.dtypes = [None if G is None else G.dtype for G in (S, GDO, GQO, GDE, GQE)]
        return val / n.clamp_min(1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        path, n = ctx.saved_tensors
        w = (g.float() / n.clamp_min(1))[:, None, None]
        inner, rows = path[:, 1:, 1:], path[:, 1:, :]                   # S, GD cells (i, j >= 1); GQ cells (i >= 1, j >= 0)
        out = lambda codes, code, dt: None if dt is None else ((codes == code) * w).to(dt)
        dS, dGD, dGQ, dGDE, dGQE = ctx.dtypes
        return (out(inner, 1, dS), out(inner, 3, dGD), out(rows, 2, dGQ), out(inner, 5, dGDE), out(rows, 4, dGQE),
                None, None, None, None)


class qNWSim(nn.Module, SequenceSim):
    """
    qNW: Needleman-Wunsch with parameterized gaps (scorer name 'qnw' in train.py's --scorer), query-global,
    document-local alignment (dtw_triton_kernels._qnw_align). Every query
    residue is either matched to one document residue (scored S = sim) or inserted (a query gap); document residues
    between the first and last match are matched or inserted (a doc gap), those outside are free. Score = the optimal
    alignment's value / Lq, so each query residue adds exactly one term and a longer document can only add gap costs
    (G <= 0), never score.
    A gap is cheap when the inserted residue resembles a residue of the other sequence flanking the insertion point (a
    stretch, e.g. a repeat or an extended helix turn), costly when it does not (a true insertion):
      doc gap   d_j between q_i, q_{i+1}:  G = -softplus(alpha - beta * max(phi(d_j, q_i), phi(d_j, q_{i+1})))
      query gap q_i between d_j, d_{j+1}:  G = -softplus(alpha - beta * max(phi(q_i, d_j), phi(q_i, d_{j+1})))
    phi = relation, a GapRelation (an instance, or a GAP_RELATIONS name): 'cosine' (default; the raw embeddings'
    cosine), 'bilinear', 'distance', or a subclass of your own. The gap scores are computed in torch for every cell (the
    kernel reads them). form picks how G depends on c = max(phi(., flank), phi(., flank)), per direction:
      'softplus': G = -softplus(alpha - beta c), alpha and beta (> 0, kept as log) learned; nothing bounds beta, which
        sets the switch's sharpness and the slope below it at once (the old runs)
      'hinge': G = -kappa softplus(beta (tau - c)) / beta, one job per parameter: tau the switch point (in cosine units),
        beta its sharpness (the softplus' width ~ 1 / beta; capped at beta_max, beta = beta_max sigmoid(raw), or
        unbounded, kept as log, when beta_max is None), kappa > 0 (kept as log) the slope below it: as beta grows,
        G -> -kappa max(0, tau - c). Starts as the softplus form's init (tau = alpha / beta, kappa = beta)
    gaps='linear': every gap step scores G (one (alpha, beta) per direction; with the cosine, the defaults put a
      dissimilar gap, phi ~ 0.2, at ~ -0.5 and a near-copy, phi ~ 0.9, at ~ -0.05)
    gaps='affine': Gotoh, a gap's first step scores G_open, the next ones G_ext (an (alpha, beta) pair each, per
      direction; G_ext defaults to ~ -0.15 at phi ~ 0.2), so one long insertion costs less than as many scattered ones
    A module so they train: train.attach_scorer binds it to the embedding size (gap relations with parameters create
    them then) and attaches it to the encoder (encoder.scorer), so checkpoints carry them.
    Hard path; gradients flow through the path cells (_qNWScore). Asymmetric: score(q, d) != score(d, q)
    """
    def __init__(self, sim=COSINE, gaps='linear', relation='cosine', alpha=0.3, beta=3.6, ext_alpha=-1.1, ext_beta=3.6,
                 form='softplus', beta_max=None, chunk_cells=2**28):
        assert gaps in ('linear', 'affine') and form in ('softplus', 'hinge')
        assert beta_max is None or (form == 'hinge' and beta_max > max(beta, ext_beta)), 'beta_max: hinge, above the init'
        nn.Module.__init__(self)
        self.sim, self.gaps, self.form, self.beta_max, self.chunk_cells = sim, gaps, form, beta_max, chunk_cells
        self.bound = False
        self.relation = GAP_RELATIONS[relation]() if isinstance(relation, str) else relation
        two = lambda v: torch.tensor([v, v], dtype=torch.float32)            # (doc, query) gaps; affine: their opening
        if form == 'softplus':
            self.gap_alpha = nn.Parameter(two(alpha))
            self.gap_log_beta = nn.Parameter(two(beta).log())
            if gaps == 'affine':
                self.gap_ext_alpha = nn.Parameter(two(ext_alpha))
                self.gap_ext_log_beta = nn.Parameter(two(ext_beta).log())
            return
        raw = lambda b: two(b).log() if beta_max is None else torch.logit(two(b / beta_max))
        self.gap_tau = nn.Parameter(two(alpha / beta))
        self.gap_beta_raw = nn.Parameter(raw(beta))
        self.gap_log_kappa = nn.Parameter(two(beta).log())
        if gaps == 'affine':
            self.gap_ext_tau = nn.Parameter(two(ext_alpha / ext_beta))
            self.gap_ext_beta_raw = nn.Parameter(raw(ext_beta))
            self.gap_ext_log_kappa = nn.Parameter(two(ext_beta).log())

    def bind(self, dim):
        # the residue embeddings' size: the gap function creates its parameters (once)
        if not self.bound:
            self.relation.build(dim)
            self.bound = True
        return self

    def gap(self):
        # softplus: (alpha, beta) of (doc open, doc extend, query open, query extend); linear: extend = open
        beta = self.gap_log_beta.exp()
        a_ext, b_ext = (self.gap_ext_alpha, self.gap_ext_log_beta.exp()) if self.gaps == 'affine' else (self.gap_alpha, beta)
        return torch.stack([self.gap_alpha[0], beta[0], a_ext[0], b_ext[0], self.gap_alpha[1], beta[1], a_ext[1], b_ext[1]])

    def hinge(self):
        # (4, 3): (tau, beta, kappa) of (doc open, doc extend, query open, query extend), either form (softplus:
        # tau = alpha / beta, kappa = beta); linear: extend = open
        if self.form == 'softplus':
            g = self.gap().reshape(4, 2)
            return torch.stack([g[:, 0] / g[:, 1], g[:, 1], g[:, 1]], 1)
        beta = lambda raw: raw.exp() if self.beta_max is None else self.beta_max * torch.sigmoid(raw)
        o = torch.stack([self.gap_tau, beta(self.gap_beta_raw), self.gap_log_kappa.exp()], 1)          # (2, 3): doc, query
        e = torch.stack([self.gap_ext_tau, beta(self.gap_ext_beta_raw), self.gap_ext_log_kappa.exp()], 1) \
            if self.gaps == 'affine' else o
        return torch.stack([o[0], e[0], o[1], e[1]])

    def log_params(self):
        # {name: value} of the gap parameters, logged during training (RetrieverTrainer, log_every): the switch point tau,
        # the sharpness beta and the slope below the switch kappa of each gap kind (softplus: kappa = beta)
        h, out = self.hinge().tolist(), {}
        kinds = (('doc', 0), ('query', 2)) if self.gaps == 'linear' else \
            (('doc_open', 0), ('doc_ext', 1), ('query_open', 2), ('query_ext', 3))
        for name, k in kinds:
            out.update({f'{name}_tau': h[k][0], f'{name}_beta': h[k][1], f'{name}_kappa': h[k][2],
                        f'{name}_alpha': h[k][0] * h[k][1]})
        return out

    def flank_mu(self, q_emb, qm, d_emb, dm, S=None):
        # (mu_d, mu_q, CQ): mu_d (P, Lq, Ld), mu_d[i-1, j-1] phi of d_j to its better flank when gapped into cell (i, j);
        # mu_q (P, Lq, Ld + 1), mu_q[i-1, j] that of q_i gapped into (i, j); CQ[i-1, j-1] = phi(q_i, d_j).
        # S: the residue similarity (P, Lq, Ld) of pair_matrix, phi itself for a relation that reads_match (GeoGap)
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        if self.relation.reads_match:
            CD = CQ = S
        else:
            CD, CQ = self.relation(q_emb, d_emb, qm.view(Bq, Bd, -1)[:, 0], dm.view(Bq, Bd, -1)[0])   # (P, Lq, Ld) each
        valid = qm.bool()[:, :, None] & dm.bool()[:, None, :]
        Dc = CD.float().masked_fill(~valid, float('-inf'))
        Qc = CQ.float().masked_fill(~valid, float('-inf'))
        # doc gap into (i, j): max(CD[i-1, j-1], CD[i, j-1]), q_{n+1} missing; query gap into (i, j), j = 0..m:
        # max(CQ[i-1, j-1], CQ[i-1, j]), d_0 and d_{m+1} missing
        mu_d = torch.maximum(Dc, torch.cat([Dc[:, 1:], Dc.new_full((Dc.shape[0], 1, Dc.shape[2]), float('-inf'))], 1))
        neg = Qc.new_full((Qc.shape[0], Qc.shape[1], 1), float('-inf'))
        Cp = torch.cat([neg, Qc, neg], 2)
        mu_q = torch.maximum(Cp[:, :, :-1], Cp[:, :, 1:])
        mu_d, mu_q = (torch.where(torch.isfinite(mu), mu, 0.) for mu in (mu_d, mu_q))   # off the valid cells: unread
        return mu_d, mu_q, CQ

    def flank_gaps(self, q_emb, qm, d_emb, dm, S=None):
        # (GDO, GQO, GDE, GQE): GD (P, Lq, Ld), GD[i-1, j-1] the doc gap step into cell (i, j); GQ (P, Lq, Ld + 1), GQ[i-1, j]
        # the query gap step into (i, j); linear: GDE = GQE = None. mu = phi to the better flanking anchor (flank_mu)
        mu_d, mu_q, _ = self.flank_mu(q_emb, qm, d_emb, dm, S)
        if self.form == 'softplus':
            g = self.gap()
            G = lambda mu, k: -F.softplus(g[2 * k] - g[2 * k + 1] * mu)
        else:
            h = self.hinge()
            G = lambda mu, k: -h[k, 2] * F.softplus(h[k, 1] * (h[k, 0] - mu)) / h[k, 1]
        if self.gaps == 'linear':
            return G(mu_d, 0), G(mu_q, 2), None, None
        return G(mu_d, 0), G(mu_q, 2), G(mu_d, 1), G(mu_q, 3)

    def forward(self, q_emb, q_mask, d_emb, d_mask):
        if not self.bound:
            self.bind(q_emb.shape[-1])                              # parameter-free gap functions; learned ones are bound earlier
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        S, qm, dm = self.pair_matrix(q_emb, q_mask, d_emb, d_mask)
        M = self.match_scores(S)
        G = self.flank_gaps(q_emb, qm, d_emb, dm, S)
        chunk = max(1, self.chunk_cells // (M.shape[1] * M.shape[2]))
        store = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (M,) + G)
        return _qNWScore.apply(M, *G, qm, dm, chunk, store).reshape(Bq, Bd)

    def match_scores(self, M):
        # the score of a match step from the residue similarity S = sim (P, Lq, Ld); qNW: S itself
        return M

    def extra_repr(self):
        if self.form == 'hinge':
            h = self.hinge().tolist()
            rows = [('doc', 0), ('query', 2)] if self.gaps == 'linear' else \
                [('doc open', 0), ('doc ext', 1), ('query open', 2), ('query ext', 3)]
            cap = 'none' if self.beta_max is None else f'{self.beta_max:g}'
            return (f'sim={self.sim!r}, gaps={self.gaps}, form=hinge (beta cap {cap}), '
                    + ', '.join(f'{n} (tau {h[k][0]:.3f}, beta {h[k][1]:.3f}, kappa {h[k][2]:.3f})' for n, k in rows))
        g = self.gap().tolist()
        if self.gaps == 'linear':
            return f'sim={self.sim!r}, gaps=linear, doc (alpha {g[0]:.3f}, beta {g[1]:.3f}), query (alpha {g[4]:.3f}, beta {g[5]:.3f})'
        return (f'sim={self.sim!r}, gaps=affine, doc open ({g[0]:.3f}, {g[1]:.3f}) ext ({g[2]:.3f}, {g[3]:.3f}), '
                f'query open ({g[4]:.3f}, {g[5]:.3f}) ext ({g[6]:.3f}, {g[7]:.3f})  [(alpha, beta)]')


class qNWTestSim(qNWSim):
    """
    qNW-test: qNW with every score read from the geometric mutual softmax's two picks, R = softmax_j and C = softmax_i
    of -|q_i - d_j| / T, with no fixed bias:
      match:      M = tanh(alpha_S * sqrt(R C) + beta_S)
      query gap:  q_i inserted between d_j and d_{j+1}: G_Q = -softplus(alpha_Q - beta_Q * max(R_ij, R_i,j+1))
      doc gap:    d_j inserted between q_i and q_{i+1}: G_D = -softplus(alpha_D - beta_D * max(C_ij, C_i+1,j))
    alpha_S > 0 (kept as log) and beta_S learned: beta_S replaces qNW-geo's fixed b (tanh(beta_S) is the score of a
    pair neither side picks) and tanh bounds the match in (-1, 1), so a growing alpha_S cannot rescale the alignment
    score the way a linear map could (which would act as a lower InfoNCE temperature). The gaps keep qNWSim's form
    (softplus, or hinge; linear or affine) on the inserted residue's own pick (PickGap): sim must therefore be
    AffineMutualSoftmax(average='geometric', beta=0), whose matrix is sqrt(R C) itself, and the gaps use its temperature.
    Init: the match follows qNW-geo's 1.3 P - 0.3 where it can (tanh(-0.31) = -0.30 for unpicked pairs, slope
    1.43 (1 - 0.3^2) = 1.3 at P = 0, flattening to tanh(1.12) = 0.81 at P = 1); the gaps switch at a pick of
    alpha / beta = 0.7 / 3.6 ~ 0.19 (qNW's 0.3 / 3.6 was set for cosines; an unpicked residue's pick is ~ 1 / length).
    gaps='affine' (qNW-test-affine): the extension is shifted the same + 0.4 from qNW's, ext_alpha = -0.7, so an
    opening costs -1.10 (pick 0) to -0.05 (pick 1) and an extension -0.40 to -0.01
    """
    def __init__(self, sim=COSINE, match_alpha=1.43, match_beta=-0.31, alpha=0.7, ext_alpha=-0.7, **kwargs):
        assert isinstance(sim, AffineMutualSoftmax) and sim.average == 'geometric' and sim.beta == 0, \
            'qNW-test reads P = sqrt(R C): sim must be the geometric mutual softmax with beta 0 (--sim geometric --beta 0)'
        super().__init__(sim, relation=PickGap(sim.temperature), alpha=alpha, ext_alpha=ext_alpha, **kwargs)
        self.match_log_alpha = nn.Parameter(torch.tensor(float(match_alpha)).log())
        self.match_beta = nn.Parameter(torch.tensor(float(match_beta)))

    def match_scores(self, P):
        # padded cells (P = 0) get tanh(beta_S); the kernel never reads them
        return torch.tanh(self.match_log_alpha.exp() * P.float() + self.match_beta)

    def log_params(self):
        return {**super().log_params(), 'match_alpha': self.match_log_alpha.exp().item(), 'match_beta': self.match_beta.item()}

    def extra_repr(self):
        return (f'match tanh(alpha {self.match_log_alpha.exp().item():.3f} P + beta {self.match_beta.item():.3f}), '
                f'gaps on {self.relation!r}, ' + super().extra_repr())


class qNWGammaSim(qNWSim):
    """
    qNW-gamma: qNW with its match bias learned, M = (1 + gamma) sqrt(R C) - gamma, gamma = sigmoid(raw) in (0, 1).
    qNW-geo fixes gamma = b = 0.3; here -gamma, the score of a pair neither side picks, is learned, while M stays in
    [-gamma, 1] with a slope in (1, 2), so gamma cannot rescale the alignment score (no InfoNCE temperature drift).
    Gaps as qNWSim's (cosine flanks by default; linear or affine, softplus or hinge). sim must be
    AffineMutualSoftmax(average='geometric', beta=0), whose matrix is sqrt(R C) itself. Init gamma = 0.3: at init it
    scores as qNW-geo b0.3 with the same gap init
    """
    def __init__(self, sim=COSINE, gamma=0.3, **kwargs):
        assert isinstance(sim, AffineMutualSoftmax) and sim.average == 'geometric' and sim.beta == 0, \
            'qNW-gamma reads P = sqrt(R C): sim must be the geometric mutual softmax with beta 0 (--sim geometric --beta 0)'
        assert 0 < gamma < 1, 'gamma in (0, 1)'
        super().__init__(sim, **kwargs)
        assert not self.relation.reads_match, 'qNW-gamma: gaps on S would move with the learned gamma'
        self.match_gamma_raw = nn.Parameter(torch.logit(torch.tensor(float(gamma))))

    def gamma(self):
        return torch.sigmoid(self.match_gamma_raw)

    def match_scores(self, P):
        # padded cells (P = 0) get -gamma; the kernel never reads them
        g = self.gamma()
        return (1 + g) * P.float() - g

    def log_params(self):
        return {**super().log_params(), 'match_gamma': self.gamma().item()}

    def extra_repr(self):
        return f'match (1 + gamma) P - gamma, gamma {self.gamma().item():.3f}, ' + super().extra_repr()


class qNWMirrorSim(qNWSim):
    """
    qNW-mirror: qNW with match and gaps read from the same picks of the geometric mutual softmax, R = softmax_j and
    C = softmax_i of -|q_i - d_j| / T, each an expected payoff in units of a mismatch:
      match:      both residues pick each other, P = sqrt(R C):  M = P - gamma (1 - P)    in [-gamma, 1]
                  (a true pair earns 1, a false one costs gamma; = qNW-geo's 1.3 P - 0.3 at gamma = 0.3)
      doc gap:    d_j inserted between q_i and q_{i+1} picks a flank, pi = C_ij + C_i+1,j:  G = -kappa gamma (1 - pi)
      query gap:  q_i inserted between d_j and d_{j+1}, pi = R_ij + R_i,j+1:            G = -kappa gamma (1 - pi)
    so G in [-kappa gamma, 0]: an insertion that picks neither flank costs kappa mismatches, one that picks a flank is
    free. The gap's pick is one-sided because the flank is matched elsewhere and spends its own pick there (a mutual
    pick, qNW-geogap, fails exactly on a duplication). pi is a probability (each softmax sums to 1), padded flanks add 0.
    gamma fixed (qNW's match); kappa > 0 learned per direction (kept as log); gaps='affine': kappa_ext =
    kappa_open sigmoid(raw), so an extension never costs more than an opening. Init at qNW's gaps on an unrelated
    insertion (cosine ~ 0.2): open ~ -0.5 = 1.67 mismatches, extend ~ -0.15 = 0.3 of an opening.
    sim must be AffineMutualSoftmax(average='geometric', beta=0), whose matrix is sqrt(R C) itself; the gaps use its
    temperature
    """
    def __init__(self, sim=COSINE, gaps='linear', gamma=0.3, kappa=5 / 3, ext_frac=0.3, form='softplus', beta_max=None,
                 chunk_cells=2**28):
        assert isinstance(sim, AffineMutualSoftmax) and sim.average == 'geometric' and sim.beta == 0, \
            'qNW-mirror reads P = sqrt(R C): sim must be the geometric mutual softmax with beta 0 (--sim geometric --beta 0)'
        assert gaps in ('linear', 'affine') and form == 'softplus' and beta_max is None, \
            'qNW-mirror: its own gap form (--gap_form, --gap_beta_max do not apply)'
        nn.Module.__init__(self)
        self.sim, self.gaps, self.gamma, self.chunk_cells = sim, gaps, gamma, chunk_cells
        self.bound = False
        self.relation = PickGap(sim.temperature)
        self.gap_log_kappa = nn.Parameter(torch.tensor([kappa, kappa], dtype=torch.float32).log())   # (doc, query)
        if gaps == 'affine':
            self.gap_ext_raw = nn.Parameter(torch.logit(torch.tensor([ext_frac, ext_frac], dtype=torch.float32)))

    def kappa(self):
        # (doc open, doc extend, query open, query extend), in mismatches; linear: extend = open
        o = self.gap_log_kappa.exp()
        e = o * torch.sigmoid(self.gap_ext_raw) if self.gaps == 'affine' else o
        return torch.stack([o[0], e[0], o[1], e[1]])

    def flank_gaps(self, q_emb, qm, d_emb, dm, S=None):
        # as qNWSim.flank_gaps, with pi the inserted residue's pick of either flank (a sum, not the better one)
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        CD, CQ = self.relation(q_emb, d_emb, qm.view(Bq, Bd, -1)[:, 0], dm.view(Bq, Bd, -1)[0])   # (P, Lq, Ld) each
        valid = qm.bool()[:, :, None] & dm.bool()[:, None, :]
        Dc, Qc = CD.float().masked_fill(~valid, 0.), CQ.float().masked_fill(~valid, 0.)
        # doc gap into (i, j): CD[i-1, j-1] + CD[i, j-1], q_{n+1} missing; query gap into (i, j), j = 0..m:
        # CQ[i-1, j-1] + CQ[i-1, j], d_0 and d_{m+1} missing
        pi_d = Dc + torch.cat([Dc[:, 1:], Dc.new_zeros(Dc.shape[0], 1, Dc.shape[2])], 1)
        Cp = F.pad(Qc, (1, 1))
        pi_q = Cp[:, :, :-1] + Cp[:, :, 1:]
        k = self.kappa() * self.g()
        G = lambda pi, i: -k[i] * (1 - pi.clamp(max=1))              # clamp: float rounding of a sum of two softmax entries
        if self.gaps == 'linear':
            return G(pi_d, 0), G(pi_q, 2), None, None
        return G(pi_d, 0), G(pi_q, 2), G(pi_d, 1), G(pi_q, 3)

    def g(self):
        # gamma, the cost of a false match (fixed here; qNWMirrorGammaSim learns it)
        return self.gamma

    def match_scores(self, P):
        # padded cells (P = 0) get -gamma; the kernel never reads them
        g = self.g()
        return (1 + g) * P.float() - g

    def log_params(self):
        k = self.kappa().tolist()
        kinds = (('doc', 0), ('query', 2)) if self.gaps == 'linear' else \
            (('doc_open', 0), ('doc_ext', 1), ('query_open', 2), ('query_ext', 3))
        return {f'{name}_kappa': k[i] for name, i in kinds}

    def extra_repr(self):
        k = self.kappa().tolist()
        gaps = f'doc {k[0]:.3f}, query {k[2]:.3f}' if self.gaps == 'linear' else \
            f'doc open {k[0]:.3f} ext {k[1]:.3f}, query open {k[2]:.3f} ext {k[3]:.3f}'
        return (f'sim={self.sim!r}, match P - gamma (1 - P), gamma {float(self.g()):.3f}; gaps={self.gaps}, '
                f'-kappa gamma (1 - pick of a flank), kappa (mismatches): {gaps}')


class qNWMirrorGammaSim(qNWMirrorSim):
    """
    qNW-mirror-gamma: qNW-mirror with gamma, the cost of a false match, learned: gamma = sigmoid(raw) in (0, 1), init
    0.3 (at init it scores as qNW-mirror). gamma is the unit of both terms, M = P - gamma (1 - P) and
    G = -kappa gamma (1 - pi), so a moving gamma rescales the gaps with the mismatch and kappa stays the gap cost in
    mismatches
    """
    def __init__(self, sim=COSINE, gamma=0.3, **kwargs):
        assert 0 < gamma < 1, 'gamma in (0, 1)'
        super().__init__(sim, gamma=gamma, **kwargs)
        self.match_gamma_raw = nn.Parameter(torch.logit(torch.tensor(float(gamma))))

    def g(self):
        return torch.sigmoid(self.match_gamma_raw)

    def log_params(self):
        return {**super().log_params(), 'match_gamma': self.g().item()}

    def extra_repr(self):
        return 'gamma learned, ' + super().extra_repr()


# qNW-null's delta init: where untrained ProtT5's residue distances of mutual nearest neighbours across a Pfam positive
# pair (median 4.28, 90th pct 5.88) meet those of random cross pairs (median 7.93, 10th pct 6.47), so a flank as far as
# delta is as likely a partner as unrelated: at T 0.5, pi(4.28) = 0.97, pi(7.93) = 0.03 (one flank)
DELTA_INIT = 6.1


class qNWNullSim(qNWMirrorSim):
    """
    qNW-null: qNW-mirror with the gap's probability normalized against a null instead of over the other sequence. Both
    terms are expected payoffs, in mismatches, of one distance kernel w(x, a) = exp(l(x, a)), l = -|x - a| / T:
      match:  which residue is the partner? Normalized over the other sequence (R, C), P = sqrt(R C):
              M = P - gamma (1 - P)
      gap:    is the inserted residue x a continuation of a flanking residue a_L or a_R of the other sequence (a repeat,
              a stretch), or new (a true indel)? Normalized over {a_L, a_R, null}, the null's weight exp(-delta / T):
                pi = (w_L + w_R) / (w_L + w_R + exp(-delta / T)) = sigmoid(logaddexp(l_L, l_R) + delta / T)
              doc gap   d_j between q_i, q_{i+1}: a_L = q_i, a_R = q_{i+1}
              query gap q_i between d_j, d_{j+1}: a_L = d_j, a_R = d_{j+1}       (a missing flank has weight 0)
              G = -kappa gamma (1 - pi)
    so a flank closer than ~delta makes the insertion nearly free, one far beyond it costs kappa mismatches, graded in
    between; unlike qNW-mirror's pick, pi does not dilute when x resembles many residues elsewhere (a repeat).
    gamma fixed, 1 by default (every learned gamma went to 0.9-1): M = 2 P - 1 in [-1, 1], G in [-kappa, 0].
    Learned per direction: delta (a distance), kappa > 0 (kept as log); gaps='affine': kappa_ext = kappa_open
    sigmoid(raw), the same pi for both. Init: delta = DELTA_INIT (see there); linear kappa 1 (an unexplained residue
    costs one false match); affine kappa_open 2, kappa_ext 1 (qNW-mirror-affine's 0.15 start, below a mismatch, made long
    query gaps cheaper than matching; training took it to ~1)
    """
    def __init__(self, sim=COSINE, gaps='linear', gamma=1., kappa=None, ext_frac=0.5, delta=None, **kwargs):
        kappa = kappa if kappa is not None else (1. if gaps == 'linear' else 2.)
        super().__init__(sim, gaps=gaps, gamma=gamma, kappa=kappa, ext_frac=ext_frac, **kwargs)
        self.relation = NullGap(sim.temperature)
        delta = DELTA_INIT if delta is None else delta
        self.gap_delta = nn.Parameter(torch.tensor([delta, delta], dtype=torch.float32))   # (doc, query)

    def flank_gaps(self, q_emb, qm, d_emb, dm, S=None):
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        L, _ = self.relation(q_emb, d_emb, qm.view(Bq, Bd, -1)[:, 0], dm.view(Bq, Bd, -1)[0])   # (P, Lq, Ld), padded at NEG_INF
        null = self.gap_delta / self.sim.temperature
        # doc gap into (i, j): flanks q_i, q_{i+1} of d_j, L[i-1, j-1] and L[i, j-1], q_{n+1} missing (a padded row,
        # or the appended one); query gap into (i, j), j = 0..m: flanks d_j, d_{j+1} of q_i, L[i-1, j-1] and L[i-1, j],
        # d_0 and d_{m+1} missing
        Ld = torch.cat([L[:, 1:], L.new_full((L.shape[0], 1, L.shape[2]), NEG_INF)], 1)
        pi_d = torch.sigmoid(torch.logaddexp(L, Ld) + null[0])
        Lp = F.pad(L, (1, 1), value=NEG_INF)
        pi_q = torch.sigmoid(torch.logaddexp(Lp[:, :, :-1], Lp[:, :, 1:]) + null[1])
        k = self.kappa() * self.g()
        G = lambda pi, i: -k[i] * (1 - pi)
        if self.gaps == 'linear':
            return G(pi_d, 0), G(pi_q, 2), None, None
        return G(pi_d, 0), G(pi_q, 2), G(pi_d, 1), G(pi_q, 3)

    def log_params(self):
        d = self.gap_delta.tolist()
        return {**super().log_params(), 'doc_delta': d[0], 'query_delta': d[1]}

    def extra_repr(self):
        d = self.gap_delta.tolist()
        return (super().extra_repr().replace('pick of a flank', 'pi, flanks vs null')
                + f'; delta (null distance): doc {d[0]:.3f}, query {d[1]:.3f}')


class _SoftDTWScore(torch.autograd.Function):
    # s = <E, M> / l per pair, E = d softDTW_gamma(-M) / dtheta, l = sum E, with the full gradient computed in chunks of
    # pairs so only one chunk's kernel buffers are alive at a time:
    #   ds/dM = E / l - H Z,  H = grad^2 R (Hessian of soft DTW in theta = -M),  Z = (M - s) / l  (-s: through l)
    # the backward re-runs the forward + alignment kernels per chunk instead of keeping Q and E for the whole batch

    @staticmethod
    def forward(ctx, M, n, m, gamma, chunk, symmetric=False):
        P = M.shape[0]
        dt = M.dtype if M.dtype in (torch.float32, torch.float64) else torch.float32   # half precision runs in float32
        s, l = M.new_zeros(P, dtype=dt), M.new_zeros(P, dtype=dt)
        for a in range(0, P, chunk):
            Mc = M[a:a + chunk].to(dt)
            _, E = _align(-Mc, n[a:a + chunk], m[a:a + chunk], gamma, symmetric)
            l[a:a + chunk] = E.sum((1, 2))
            s[a:a + chunk] = torch.einsum('bij,bij->b', E, Mc) / l[a:a + chunk].clamp_min(1e-6)
            del E
        ctx.save_for_backward(M, n, m, s, l)
        ctx.gamma, ctx.chunk, ctx.symmetric = gamma, chunk, symmetric
        return s

    @staticmethod
    def backward(ctx, g):
        M, n, m, s, l = ctx.saved_tensors
        gM = torch.empty_like(M)
        for a in range(0, M.shape[0], ctx.chunk):
            b = slice(a, a + ctx.chunk)
            Mc = M[b].to(s.dtype)
            lc = l[b].clamp_min(1e-6)[:, None, None]
            Q, E = _align(-Mc, n[b], m[b], ctx.gamma, ctx.symmetric)
            HZ = _soft_hvp(Q, E, (Mc - s[b][:, None, None]) / lc, n[b], m[b], ctx.gamma, ctx.symmetric)
            gM[b] = ((E / lc - HZ) * g[b][:, None, None]).to(M.dtype)
            del Q, E, HZ
        return gM, None, None, None, None, None


class SoftDTWSim(SequenceSim):
    """
    Soft DTW: R = softDTW_gamma(-M), E = dR/dtheta the expected alignment (P(path) ~ exp(<A_path, M> / gamma)),
    score = <E, M> / sum(E), the expected path mean (gamma -> 0: DTWSim). Full gradient, also through how M moves the
    alignment and through the normaliser (_SoftDTWScore), computed in chunks of at most chunk_cells DP cells
    (pairs * Lq * Ld) so memory does not grow with Bq * Bd.
    symmetric: soft version of DTWSim(symmetric=True), every path weighs n + m, so sum E = n + m
    """
    def __init__(self, sim=COSINE, gamma=0.1, symmetric=True, chunk_cells=2**28):
        super().__init__(sim)
        self.gamma, self.symmetric, self.chunk_cells = gamma, symmetric, chunk_cells

    def __call__(self, q_emb, q_mask, d_emb, d_mask):
        Bq, Bd = q_emb.shape[0], d_emb.shape[0]
        M, qm, dm = self.pair_matrix(q_emb, q_mask, d_emb, d_mask)
        n, m = qm.sum(-1).long().contiguous(), dm.sum(-1).long().contiguous()
        chunk = max(1, self.chunk_cells // (M.shape[1] * M.shape[2]))
        return _SoftDTWScore.apply(M, n, m, self.gamma, chunk, self.symmetric).reshape(Bq, Bd)


if __name__ == '__main__':
    # every scorer on random embeddings: (Bq, Bd) scores and a finite gradient
    set_seed(2603)
    Bq, Bd, D = 6, 5, 32
    q_emb = torch.randn(Bq, 90, D, device='cuda', requires_grad=True)
    d_emb = torch.randn(Bd, 120, D, device='cuda', requires_grad=True)
    q_mask = torch.arange(90, device='cuda')[None] < torch.randint(20, 91, (Bq, 1), device='cuda')
    d_mask = torch.arange(120, device='cuda')[None] < torch.randint(20, 121, (Bd, 1), device='cuda')
    msm = AffineMutualSoftmax(0.5, 'geometric', beta=0.5)
    for scorer in (DenseSim(), MaxSim(), MaxSim(msm), DTWSim(), DTWSim(msm), SoftDTWSim(gamma=0.25),
                   qNWSim(msm).cuda(),
                   qNWSim(msm, gaps='affine').cuda(), qNWSim(msm, relation='bilinear').bind(32).cuda(),
                   qNWSim(msm, gaps='affine', relation='distance').bind(32).cuda(), qNWTestSim(AffineMutualSoftmax(0.5, 'geometric')).cuda()):
        S = scorer(q_emb, q_mask, d_emb, d_mask)
        gq, = torch.autograd.grad(S.sum(), q_emb)
        print(f'{scorer!r}: {tuple(S.shape)}, finite grad {bool(gq.isfinite().all())}')
