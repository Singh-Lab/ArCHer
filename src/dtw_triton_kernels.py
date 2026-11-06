# Differentiable DTW on CUDA (Triton): global DTW (soft and hard, with Hessian-vector products) and qNW (query-global
# Needleman-Wunsch with parameterized gaps, hard); every kernel runs one program per pair over the anti-diagonals
#
# ---- Global DTW
#
# DTW_Omega of Mensch & Blondel 2018 (arXiv 1802.03676), Algorithms 5 and 6 of appendix B.3:
#   v[i, j] = theta[i, j] + min_Omega(v[i, j - 1], v[i - 1, j - 1], v[i - 1, j]),  DTW_Omega(theta) = v[n, m]
# theta is a COST matrix (pass -M for a similarity M); omega picks the smoothed min:
#   'soft'  soft-DTW (Cuturi & Blondel 2017), min_Omega = -gamma logsumexp(-x / gamma) (Omega = negative entropy)
#   'hard'  plain min (Omega = 0): the classic DTW, its gradient is the optimal path
# the gradient E = grad DTW_Omega(theta) is the (expected) alignment matrix
#
# the backward never autograds through the DP: the forward stores the local weights q[i, j] = grad min_Omega
# and E is the reverse recursion over them (Algo 5); the double backward is Algo 6 (appendix A.7: grad^2 DTW Z is
# the gradient of <grad DTW, Z>), so a Hessian-vector product costs two more sweeps
#
# one Triton program per pair runs a whole sweep over the anti-diagonals s = i + j (1-indexed), with a block-wide
# barrier between diagonals; the running values (v, e and their derivatives) only need the last 3 diagonals, kept
# in a ring buffer indexed by i, and each diagonal also overwrites the two cells just outside its valid range
# (lo - 1, hi + 1) with the border value: they are the only stale ring cells the next two diagonals read
# no CUDA: TRITON_INTERPRET=1 runs the kernels on CPU (slowly)

import torch
import torch.nn as nn
import triton
import triton.language as tl

DTW_BIG = 1e10  # the inf border of the DP; finite so no inf - inf
OMEGAS = {'soft': 0, 'hard': 1}  # the OMEGA constexpr of the global DTW kernels


@triton.jit
def _min_omega(a, d, u, gamma, OMEGA: tl.constexpr):
    # min_Omega over (left, diag, up) and its gradient q
    if OMEGA == 0:
        za, zd, zu = -a / gamma, -d / gamma, -u / gamma
        mx = tl.maximum(tl.maximum(za, zd), zu)
        ea, ed, eu = tl.exp(za - mx), tl.exp(zd - mx), tl.exp(zu - mx)
        sm = ea + ed + eu
        return -gamma * (mx + tl.log(sm)), ea / sm, ed / sm, eu / sm
    else:
        # one-hot argmin, ties -> diag, then up, then left (dtw.py's backtrack order, so tied paths match it)
        bd = (d <= a) & (d <= u)
        bu = ~bd & (u <= a)
        ba = ~bd & ~bu
        return tl.minimum(tl.minimum(a, d), u), ba.to(a.dtype), bd.to(a.dtype), bu.to(a.dtype)


@triton.jit
def _dtw_fwd_kernel(TH, Q, VAL, R, NN, MM, GAMMA, N, M,
                    BIG: tl.constexpr, OMEGA: tl.constexpr, BLOCK: tl.constexpr, SYM: tl.constexpr = False):
    # Algo 5 forward pass: VAL = v[n, m], Q[i, j] = q at cell (i, j), (N + 2) x (M + 2) x 3 (zero padding around)
    # SYM: v[i, j] = min_Omega(v[i, j - 1] + theta, v[i - 1, j - 1] + 2 theta, v[i - 1, j] + theta)
    b = tl.program_id(0).to(tl.int64)
    n, m, gamma = tl.load(NN + b), tl.load(MM + b), tl.load(GAMMA)
    N2, M2 = N + 2, M + 2
    TH += b * N * M
    Q += b * N2 * M2 * 3
    R += b * 3 * N2
    for s in range(2, n + m + 1):
        lo, hi = tl.maximum(1, s - m), tl.minimum(n, s - 1)
        cur, p1, p2 = R + (s % 3) * N2, R + ((s + 2) % 3) * N2, R + ((s + 1) % 3) * N2
        for i0 in range(lo - 1, hi + 2, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = (i >= lo) & (i <= hi)
            a = tl.load(p1 + i, mask=ok, other=BIG)                      # left (i, j - 1)
            d = tl.load(p2 + i - 1, mask=ok, other=BIG)                  # diag (i - 1, j - 1)
            u = tl.load(p1 + i - 1, mask=ok, other=BIG)                  # up (i - 1, j)
            if SYM:
                th = tl.load(TH + (i - 1) * M + j - 1, mask=ok, other=0.)
                mn, qa, qd, qu = _min_omega(a + th, d + 2 * th, u + th, gamma, OMEGA)
                v = tl.where(ok, mn, BIG)
            else:
                mn, qa, qd, qu = _min_omega(a, d, u, gamma, OMEGA)
                v = tl.where(ok, tl.load(TH + (i - 1) * M + j - 1, mask=ok, other=0.) + mn, BIG)
            tl.store(cur + i, v, mask=i <= hi + 1)
            qp = Q + (i * M2 + j) * 3
            tl.store(qp, qa, mask=ok)
            tl.store(qp + 1, qd, mask=ok)
            tl.store(qp + 2, qu, mask=ok)
            tl.store(VAL + b + 0 * i, v, mask=ok & (s == n + m) & (i == n))
        tl.debug_barrier()


@triton.jit
def _dtw_bwd_kernel(Q, G, EG, R, NN, MM, N, M, BLOCK: tl.constexpr, SYM: tl.constexpr = False):
    # Algo 5 backward pass: EG = g * grad DTW; e is seeded with g at (n, m) (the paper's virtual end node)
    # SYM: e (= dv[n, m] / dv[i, j]) runs the same recursion, and theta[i, j] enters v[i, j] with weight
    # q_left + 2 q_diag + q_up = 1 + q_diag, so EG = e * (1 + q_diag)
    b = tl.program_id(0).to(tl.int64)
    n, m, g = tl.load(NN + b), tl.load(MM + b), tl.load(G + b)
    N3, M2 = N + 3, M + 2
    Q += b * (N + 2) * M2 * 3
    EG += b * N * M
    R += b * 3 * N3
    for t in range(0, n + m - 1):
        s = n + m - t
        lo, hi = tl.maximum(1, s - m), tl.minimum(n, s - 1)
        cur, c1, c2 = R + (s % 3) * N3, R + ((s + 1) % 3) * N3, R + ((s + 2) % 3) * N3
        for i0 in range(lo - 1, hi + 2, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = (i >= lo) & (i <= hi)
            e = tl.where(ok & (s == n + m) & (i == n), g, 0.)
            e += tl.load(Q + (i * M2 + j + 1) * 3, mask=ok, other=0.) * tl.load(c1 + i, mask=ok, other=0.)
            e += tl.load(Q + ((i + 1) * M2 + j + 1) * 3 + 1, mask=ok, other=0.) * tl.load(c2 + i + 1, mask=ok, other=0.)
            e += tl.load(Q + ((i + 1) * M2 + j) * 3 + 2, mask=ok, other=0.) * tl.load(c1 + i + 1, mask=ok, other=0.)
            e = tl.where(ok, e, 0.)
            tl.store(cur + i, e, mask=i <= hi + 1)
            if SYM:
                tl.store(EG + (i - 1) * M + j - 1, e * (1 + tl.load(Q + (i * M2 + j) * 3 + 1, mask=ok, other=0.)), mask=ok)
            else:
                tl.store(EG + (i - 1) * M + j - 1, e, mask=ok)
        tl.debug_barrier()


@triton.jit
def _dtw_hvp_kernel(Q, E, Z, QD, HD, GD, R1, R2, NN, MM, GAMMA, N, M,
                    OMEGA: tl.constexpr, BLOCK: tl.constexpr, SYM: tl.constexpr = False):
    # Algo 6: HD = g * grad^2 DTW Z, GD = <grad DTW, Z>
    # SYM: the min_Omega arguments move by (z, 2 z, z) with theta, the recursion carries the raw e = E / (1 + q_diag),
    # and E = e (1 + q_diag) gives HD = ed (1 + q_diag) + e qd_diag (sum qd = 0, so d(q_left + 2 q_diag + q_up) = qd_diag)
    b = tl.program_id(0).to(tl.int64)
    n, m, gamma = tl.load(NN + b), tl.load(MM + b), tl.load(GAMMA)
    N2, N3, M2 = N + 2, N + 3, M + 2
    Q += b * N2 * M2 * 3
    QD += b * N2 * M2 * 3
    E += b * N * M
    Z += b * N * M
    HD += b * N * M
    R1 += b * 3 * N2
    R2 += b * 3 * N3
    # forward mode: vd = z + <q, vd of the parents>, qd = -J_Omega(q) (vd of the parents)
    for s in range(2, n + m + 1):
        lo, hi = tl.maximum(1, s - m), tl.minimum(n, s - 1)
        cur, p1, p2 = R1 + (s % 3) * N2, R1 + ((s + 2) % 3) * N2, R1 + ((s + 1) % 3) * N2
        for i0 in range(lo - 1, hi + 2, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = (i >= lo) & (i <= hi)
            xa = tl.load(p1 + i, mask=ok, other=0.)
            xd = tl.load(p2 + i - 1, mask=ok, other=0.)
            xu = tl.load(p1 + i - 1, mask=ok, other=0.)
            qp = Q + (i * M2 + j) * 3
            qa = tl.load(qp, mask=ok, other=0.)
            qd = tl.load(qp + 1, mask=ok, other=0.)
            qu = tl.load(qp + 2, mask=ok, other=0.)
            z = tl.load(Z + (i - 1) * M + j - 1, mask=ok, other=0.)
            if SYM:
                xa, xd, xu = xa + z, xd + 2 * z, xu + z
                vd = qa * xa + qd * xd + qu * xu
            else:
                vd = z + qa * xa + qd * xd + qu * xu
            vd = tl.where(ok, vd, 0.)
            tl.store(cur + i, vd, mask=i <= hi + 1)
            tl.store(GD + b + 0 * i, vd, mask=ok & (s == n + m) & (i == n))
            if OMEGA == 0:                                                # hard: J = 0, qd stays 0
                dot = qa * xa + qd * xd + qu * xu
                ja, jd, ju = qa * (xa - dot), qd * (xd - dot), qu * (xu - dot)
                qdp = QD + (i * M2 + j) * 3
                tl.store(qdp, -ja / gamma, mask=ok)
                tl.store(qdp + 1, -jd / gamma, mask=ok)
                tl.store(qdp + 2, -ju / gamma, mask=ok)
        tl.debug_barrier()
    # reverse mode: ed = sum over the children of qd e + q ed
    for t in range(0, n + m - 1):
        s = n + m - t
        lo, hi = tl.maximum(1, s - m), tl.minimum(n, s - 1)
        cur, c1, c2 = R2 + (s % 3) * N3, R2 + ((s + 1) % 3) * N3, R2 + ((s + 2) % 3) * N3
        for i0 in range(lo - 1, hi + 2, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = (i >= lo) & (i <= hi)
            o1 = (i * M2 + j + 1) * 3                                      # child (i, j + 1)
            o2 = ((i + 1) * M2 + j + 1) * 3 + 1                            # child (i + 1, j + 1)
            o3 = ((i + 1) * M2 + j) * 3 + 2                                # child (i + 1, j)
            e1 = tl.load(E + (i - 1) * M + j, mask=ok & (j < m), other=0.)
            e2 = tl.load(E + i * M + j, mask=ok & (i < n) & (j < m), other=0.)
            e3 = tl.load(E + i * M + j - 1, mask=ok & (i < n), other=0.)
            if SYM:                                                        # raw e of the children
                e1 = e1 / (1 + tl.load(Q + o1 + 1, mask=ok, other=0.))
                e2 = e2 / (1 + tl.load(Q + o2, mask=ok, other=0.))
                e3 = e3 / (1 + tl.load(Q + o3 - 1, mask=ok, other=0.))
            ed = (tl.load(QD + o1, mask=ok, other=0.) * e1
                  + tl.load(Q + o1, mask=ok, other=0.) * tl.load(c1 + i, mask=ok, other=0.)
                  + tl.load(QD + o2, mask=ok, other=0.) * e2
                  + tl.load(Q + o2, mask=ok, other=0.) * tl.load(c2 + i + 1, mask=ok, other=0.)
                  + tl.load(QD + o3, mask=ok, other=0.) * e3
                  + tl.load(Q + o3, mask=ok, other=0.) * tl.load(c1 + i + 1, mask=ok, other=0.))
            ed = tl.where(ok, ed, 0.)
            tl.store(cur + i, ed, mask=i <= hi + 1)
            if SYM:
                w = 1 + tl.load(Q + (i * M2 + j) * 3 + 1, mask=ok, other=0.)
                e0 = tl.load(E + (i - 1) * M + j - 1, mask=ok, other=0.) / w
                tl.store(HD + (i - 1) * M + j - 1, ed * w + e0 * tl.load(QD + (i * M2 + j) * 3 + 1, mask=ok, other=0.), mask=ok)
            else:
                tl.store(HD + (i - 1) * M + j - 1, ed, mask=ok)
        tl.debug_barrier()


def _launch(N, M):
    # one block spans a whole diagonal (at most min(N, M) + 2 cells written) when it fits
    block = min(triton.next_power_of_2(min(N, M) + 3), 2048)
    return block, 4 if block <= 256 else 8 if block <= 1024 else 16


class _DTWGrad(torch.autograd.Function):
    # E = g * grad DTW_Omega(theta) (Algo 5 backward pass); its own backward is Algo 6

    @staticmethod
    def forward(ctx, theta, g, Q, n, m, gamma, omega, symmetric):
        B, N, M = theta.shape
        E = theta.new_zeros(B, N, M)
        block, warps = _launch(N, M)
        _dtw_bwd_kernel[(B,)](Q, g.contiguous(), E, theta.new_zeros(B, 3, N + 3), n, m, N, M,
                              BLOCK=block, SYM=symmetric, num_warps=warps)
        ctx.save_for_backward(Q, E, n, m)
        ctx.gamma, ctx.omega, ctx.symmetric = gamma, omega, symmetric
        return E

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, Z):
        Q, E, n, m = ctx.saved_tensors
        B, N, M = Z.shape
        H, gd = torch.zeros_like(Z), Z.new_zeros(B)
        block, warps = _launch(N, M)
        _dtw_hvp_kernel[(B,)](Q, E, Z.contiguous(), torch.zeros_like(Q), H, gd, Z.new_zeros(B, 3, N + 2),
                              Z.new_zeros(B, 3, N + 3), n, m, Z.new_tensor([ctx.gamma]), N, M,
                              OMEGA=OMEGAS[ctx.omega], BLOCK=block, SYM=ctx.symmetric, num_warps=warps)
        # d<E, Z>/dg = <grad DTW, Z>
        return H, gd, None, None, None, None, None, None


class _DTW(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, n, m, gamma, omega, symmetric):
        B, N, M = theta.shape
        val = theta.new_zeros(B)                                          # empty q or d: 0
        Q = theta.new_zeros(B, N + 2, M + 2, 3)                           # cells outside the real n x m: q = 0
        R = theta.new_full((B, 3, N + 2), DTW_BIG)
        R[:, 0, 0] = 0.
        block, warps = _launch(N, M)
        _dtw_fwd_kernel[(B,)](theta, Q, val, R, n, m, theta.new_tensor([gamma]), N, M,
                              BIG=DTW_BIG, OMEGA=OMEGAS[omega], BLOCK=block, SYM=symmetric, num_warps=warps)
        ctx.save_for_backward(theta, Q, n, m)
        ctx.gamma, ctx.omega, ctx.symmetric = gamma, omega, symmetric
        return val

    @staticmethod
    def backward(ctx, g):
        theta, Q, n, m = ctx.saved_tensors
        # an autograd Function itself, so create_graph=True gets the Hessian-vector product of Algo 6
        return _DTWGrad.apply(theta, g.to(theta.dtype), Q, n, m, ctx.gamma, ctx.omega, ctx.symmetric), None, None, None, None, None


def dtw(theta, q_mask=None, d_mask=None, gamma=1., omega='soft', symmetric=False):
    # theta: B x Lq_max x Ld_max cost matrix -> B values of DTW_Omega; masks: B x L prefix masks of the real lengths
    # symmetric: a diagonal step costs 2 theta[i, j], a horizontal / vertical one theta[i, j], so every path weighs
    #   n + m (dtw.py's symmetric=True, on the cost side)
    # float32 / float64 run as is, half precision in float32
    assert omega in OMEGAS and gamma > 0
    B, N, M = theta.shape
    n = q_mask.sum(-1).long() if q_mask is not None else torch.full((B,), N, device=theta.device)
    m = d_mask.sum(-1).long() if d_mask is not None else torch.full((B,), M, device=theta.device)
    x = theta if theta.dtype in (torch.float32, torch.float64) else theta.float()
    return _DTW.apply(x.contiguous(), n.contiguous(), m.contiguous(), float(gamma), omega, bool(symmetric)).to(theta.dtype)

def soft_dtw(theta, q_mask=None, d_mask=None, gamma=1., symmetric=False):
    return dtw(theta, q_mask, d_mask, gamma, 'soft', symmetric)

def hard_dtw(theta, q_mask=None, d_mask=None, symmetric=False):
    # min-cost alignment; the gradient is the optimal path (ties -> diagonal, then up, then left), the Hessian is 0
    # (symmetric: the path weighted 2 on cells entered diagonally, the Hessian is still 0)
    return dtw(theta, q_mask, d_mask, 1., 'hard', symmetric)

def dtw_alignment(theta, q_mask=None, d_mask=None, gamma=1., omega='soft', symmetric=False):
    # values and alignments E = grad DTW_Omega(theta), B x Lq_max x Ld_max: expected (soft) or optimal (hard) path
    # symmetric: cells weighted by their step (2 diagonal, 1 otherwise), so sum E = n + m
    with torch.enable_grad():
        th = theta.detach().requires_grad_()
        val = dtw(th, q_mask, d_mask, gamma, omega, symmetric)
        E, = torch.autograd.grad(val.sum(), th)
    return val.detach(), E


def _align(theta, n, m, gamma=1., symmetric=False, omega='soft'):
    # the DTW forward + alignment kernels without autograd: step weights Q and E = dR / dtheta (soft: the expected
    # alignment; hard: the optimal path, step-weighted when symmetric). theta float32 / float64 and contiguous, n and m
    # the lengths as long tensors
    B, N, M = theta.shape
    Q = theta.new_zeros(B, N + 2, M + 2, 3)
    R = theta.new_full((B, 3, N + 2), DTW_BIG)
    R[:, 0, 0] = 0.
    block, warps = _launch(N, M)
    _dtw_fwd_kernel[(B,)](theta, Q, theta.new_zeros(B), R, n, m, theta.new_tensor([gamma]), N, M,
                          BIG=DTW_BIG, OMEGA=OMEGAS[omega], BLOCK=block, SYM=symmetric, num_warps=warps)
    del R
    E = theta.new_zeros(B, N, M)
    _dtw_bwd_kernel[(B,)](Q, theta.new_ones(B), E, theta.new_zeros(B, 3, N + 3), n, m, N, M, BLOCK=block,
                          SYM=symmetric, num_warps=warps)
    return Q, E

def _soft_hvp(Q, E, Z, n, m, gamma, symmetric=False):
    # the Hessian-vector product kernel: grad^2 R Z (E taken with an upstream gradient of 1)
    B, N, M = Z.shape
    H = torch.zeros_like(Z)
    block, warps = _launch(N, M)
    _dtw_hvp_kernel[(B,)](Q, E, Z.contiguous(), torch.zeros_like(Q), H, Z.new_zeros(B), Z.new_zeros(B, 3, N + 2),
                          Z.new_zeros(B, 3, N + 3), n, m, Z.new_tensor([gamma]), N, M,
                          OMEGA=OMEGAS['soft'], BLOCK=block, SYM=symmetric, num_warps=warps)
    return H


class SoftDTW(nn.Module):
    # cost matrix -> soft DTW values

    def __init__(self, gamma=1., symmetric=False):
        super().__init__()
        self.gamma, self.symmetric = gamma, symmetric

    def forward(self, theta, q_mask=None, d_mask=None):
        return soft_dtw(theta, q_mask, d_mask, self.gamma, self.symmetric)

    def extra_repr(self):
        return f'gamma={self.gamma}, symmetric={self.symmetric}'


class HardDTW(nn.Module):
    # cost matrix -> DTW values; backprop routes the gradient along the optimal path

    def __init__(self, symmetric=False):
        super().__init__()
        self.symmetric = symmetric

    def forward(self, theta, q_mask=None, d_mask=None):
        return hard_dtw(theta, q_mask, d_mask, self.symmetric)


# --------------------------------------------------------------------------------------------------------------------
# qNW (Needleman-Wunsch with parameterized gaps): query-global, document-local alignment of a substitution matrix S
# with gap scores precomputed per cell (sim.qNWSim; hard max, one program per pair)
#
# cells (i, j), i in 0..n, j in 0..m (1-indexed residues; row / column 0 = before the first residue); per pair
#   S  (N, M):      S[i-1, j-1] scores matching q_i with d_j
#   GD (N, M):      GD[i-1, j-1] scores the doc gap step into (i, j): d_j inserted after q_i (j >= 1)
#   GQ (N, M + 1):  GQ[i-1, j] scores the query gap step into (i, j): q_i inserted after d_j (j = 0..m)
# affine gaps take an open and an extend version of each (GDO, GDE, GQO, GQE); linear gaps use GDO and GQO only
#
# linear gaps (one state):
#   v[0, j] = 0                                                  free leading document overhang
#   v[i, j] = max(v[i-1, j-1] + S, v[i-1, j] + GQ, v[i, j-1] + GD)
#   value = max_j v[n, j]                                        free trailing document overhang
# affine gaps (Gotoh, three states: A match, Q query gap, D doc gap):
#   A[0, j] = 0, Q[0, j] = D[0, j] = -inf                        free leading document overhang
#   A[i, j] = S + max(A, Q, D)[i-1, j-1]
#   Q[i, j] = max(A[i-1, j] + GQO, Q[i-1, j] + GQE, D[i-1, j] + GQO)
#   D[i, j] = max(A[i, j-1] + GDO, Q[i, j-1] + GDO, D[i, j-1] + GDE)
#   value = max_j max(A[n, j], Q[n, j])                          free trailing document overhang
# ties -> the earlier option as written (linear: diagonal, vertical, horizontal; affine: A, Q, D). The forward stores
# the last row (the value, the end column and, affine, the end state) and, for the backtrace, each cell's step(s) as
# uint8 (linear: 1 diagonal, 2 vertical, 3 horizontal; affine: the predecessor state of A, Q, D in bits 0-1, 2-3, 4-5);
# the backtrace marks the optimal path's cells with their step: 1 match, 2 query gap open, 3 doc gap open, 4 query gap
# extend, 5 doc gap extend (linear gaps only use 1-3). The gradient is then taken over those cells in torch

@triton.jit
def _qnw_linear_fwd_kernel(S, GD, GQ, LAST, STEP, R, NN, MM, N, M,
                            BIG: tl.constexpr, BLOCK: tl.constexpr, STORE: tl.constexpr):
    # LAST: (M + 1) per pair, v[n, j] (-BIG beyond m); STEP: (N + 1) x (M + 1) uint8 per pair when STORE; R: ring buffer
    # of the last 3 diagonals, 3 x (N + 1)
    b = tl.program_id(0).to(tl.int64)
    n, m = tl.load(NN + b), tl.load(MM + b)
    N1, M1 = N + 1, M + 1
    S += b * N * M
    GD += b * N * M
    GQ += b * N * M1
    LAST += b * M1
    R += b * 3 * N1
    if STORE:
        STEP += b * N1 * M1
    for s in range(0, n + m + 1):
        lo, hi = tl.maximum(0, s - m), tl.minimum(n, s)
        cur, p1, p2 = R + (s % 3) * N1, R + ((s + 2) % 3) * N1, R + ((s + 1) % 3) * N1
        for i0 in range(lo, hi + 1, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = i <= hi
            inner = ok & (i >= 1)                                        # row 0 is the free start
            hasj = inner & (j >= 1)
            sij = tl.load(S + (i - 1) * M + j - 1, mask=hasj, other=0.)
            gd = tl.load(GD + (i - 1) * M + j - 1, mask=hasj, other=0.)
            gq = tl.load(GQ + (i - 1) * M1 + j, mask=inner, other=0.)
            dg = tl.load(p2 + i - 1, mask=hasj, other=-BIG) + sij        # diag (i-1, j-1)
            up = tl.load(p1 + i - 1, mask=inner, other=-BIG) + gq        # up (i-1, j)
            lf = tl.load(p1 + i, mask=hasj, other=-BIG) + gd             # left (i, j-1)
            take_d = (dg >= up) & (dg >= lf)
            take_u = ~take_d & (up >= lf)
            v = tl.where(take_d, dg, tl.where(take_u, up, lf))
            v = tl.where(inner, v, 0.)
            tl.store(cur + i, v, mask=ok)
            if STORE:
                step = tl.where(take_d, 1, tl.where(take_u, 2, 3)).to(tl.uint8)
                tl.store(STEP + i * M1 + j, step, mask=inner)
            tl.store(LAST + j, v, mask=ok & (i == n))
        tl.debug_barrier()


@triton.jit
def _qnw_linear_backtrace_kernel(STEP, PATH, NN, JEND, N, M):
    # from (n, JEND) back to row 0, copying each visited cell's step into PATH
    b = tl.program_id(0).to(tl.int64)
    n, j = tl.load(NN + b), tl.load(JEND + b)
    M1 = M + 1
    STEP += b * (N + 1) * M1
    PATH += b * (N + 1) * M1
    i = n
    while i > 0:
        c = tl.load(STEP + i * M1 + j)
        tl.store(PATH + i * M1 + j, c)
        c = c.to(tl.int64)
        i -= (c != 3).to(tl.int64)
        j -= (c != 2).to(tl.int64)


@triton.jit
def _best3(x0, x1, x2):
    # max of three and its index, ties -> the lower index
    t0 = (x0 >= x1) & (x0 >= x2)
    t1 = ~t0 & (x1 >= x2)
    return tl.where(t0, x0, tl.where(t1, x1, x2)), tl.where(t0, 0, tl.where(t1, 1, 2))


@triton.jit
def _qnw_affine_fwd_kernel(S, GDO, GDE, GQO, GQE, LAST, STEP, RA, RQ, RD, NN, MM, N, M,
                            BIG: tl.constexpr, BLOCK: tl.constexpr, STORE: tl.constexpr):
    # LAST: 2 x (M + 1) per pair, A[n, j] and Q[n, j]; STEP: the predecessor states of A, Q, D packed in 2 bits each;
    # RA, RQ, RD: ring buffers as the linear one
    b = tl.program_id(0).to(tl.int64)
    n, m = tl.load(NN + b), tl.load(MM + b)
    N1, M1 = N + 1, M + 1
    S += b * N * M
    GDO += b * N * M
    GDE += b * N * M
    GQO += b * N * M1
    GQE += b * N * M1
    LAST += b * 2 * M1
    RA += b * 3 * N1
    RQ += b * 3 * N1
    RD += b * 3 * N1
    if STORE:
        STEP += b * N1 * M1
    for s in range(0, n + m + 1):
        lo, hi = tl.maximum(0, s - m), tl.minimum(n, s)
        o0, o1, o2 = (s % 3) * N1, ((s + 2) % 3) * N1, ((s + 1) % 3) * N1      # this diagonal, s - 1, s - 2
        for i0 in range(lo, hi + 1, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            j = s - i
            ok = i <= hi
            inner = ok & (i >= 1)
            hasj = inner & (j >= 1)
            sij = tl.load(S + (i - 1) * M + j - 1, mask=hasj, other=0.)
            gdo = tl.load(GDO + (i - 1) * M + j - 1, mask=hasj, other=0.)
            gde = tl.load(GDE + (i - 1) * M + j - 1, mask=hasj, other=0.)
            gqo = tl.load(GQO + (i - 1) * M1 + j, mask=inner, other=0.)
            gqe = tl.load(GQE + (i - 1) * M1 + j, mask=inner, other=0.)
            # A: diagonal from any state at (i-1, j-1)
            va, pa = _best3(tl.load(RA + o2 + i - 1, mask=hasj, other=-BIG), tl.load(RQ + o2 + i - 1, mask=hasj, other=-BIG),
                            tl.load(RD + o2 + i - 1, mask=hasj, other=-BIG))
            va = tl.where(hasj, va + sij, -BIG)
            # Q: vertical from (i-1, j), opening from A or D, extending Q
            vq, pq = _best3(tl.load(RA + o1 + i - 1, mask=inner, other=-BIG) + gqo,
                            tl.load(RQ + o1 + i - 1, mask=inner, other=-BIG) + gqe,
                            tl.load(RD + o1 + i - 1, mask=inner, other=-BIG) + gqo)
            vq = tl.where(inner, vq, -BIG)
            # D: horizontal from (i, j-1), opening from A or Q, extending D
            vd, pd = _best3(tl.load(RA + o1 + i, mask=hasj, other=-BIG) + gdo,
                            tl.load(RQ + o1 + i, mask=hasj, other=-BIG) + gdo,
                            tl.load(RD + o1 + i, mask=hasj, other=-BIG) + gde)
            vd = tl.where(hasj, vd, -BIG)
            va = tl.where(ok & (i == 0), 0., va)                         # row 0: the free start
            tl.store(RA + o0 + i, va, mask=ok)
            tl.store(RQ + o0 + i, vq, mask=ok)
            tl.store(RD + o0 + i, vd, mask=ok)
            if STORE:
                tl.store(STEP + i * M1 + j, (pa + 4 * pq + 16 * pd).to(tl.uint8), mask=inner)
            tl.store(LAST + j, va, mask=ok & (i == n))
            tl.store(LAST + M1 + j, vq, mask=ok & (i == n))
        tl.debug_barrier()


@triton.jit
def _qnw_affine_backtrace_kernel(STEP, PATH, NN, JEND, SEND, N, M):
    # from (n, JEND) in state SEND (0 A, 1 Q) back to row 0; PATH gets 1 match, 2 / 4 query gap open / extend,
    # 3 / 5 doc gap open / extend
    b = tl.program_id(0).to(tl.int64)
    n, j, st = tl.load(NN + b), tl.load(JEND + b), tl.load(SEND + b)
    M1 = M + 1
    STEP += b * (N + 1) * M1
    PATH += b * (N + 1) * M1
    i = n
    while i > 0:
        c = tl.load(STEP + i * M1 + j).to(tl.int64)
        pa, pq, pd = c % 4, (c // 4) % 4, (c // 16) % 4
        code = tl.where(st == 0, 1, tl.where(st == 1, tl.where(pq == 1, 4, 2), tl.where(pd == 2, 5, 3)))
        tl.store(PATH + i * M1 + j, code.to(tl.uint8))
        nst = tl.where(st == 0, pa, tl.where(st == 1, pq, pd))
        i -= (st != 2).to(tl.int64)
        j -= (st != 1).to(tl.int64)
        st = nst


def _qnw_align(S, GDO, GQO, n, m, GDE=None, GQE=None, store=True):
    # qNW value per pair and, with store, the optimal path as a (B, N + 1, M + 1) uint8 map of step codes (0 off the
    # path). S, GDO, GDE (B, N, M) and GQO, GQE (B, N, M + 1), float32 and contiguous; n, m long. Affine gaps when the
    # extend scores GDE, GQE are given, linear (GDO, GQO on every gap step) otherwise
    B, N, M = S.shape
    affine = GDE is not None
    last = S.new_full((B, 2 if affine else 1, M + 1), -DTW_BIG)
    step = torch.zeros(B, N + 1, M + 1, dtype=torch.uint8, device=S.device) if store else S.new_zeros(1)
    block, warps = _launch(N + 1, M + 1)
    ring = lambda: S.new_zeros(B, 3, N + 1)
    if affine:
        _qnw_affine_fwd_kernel[(B,)](S, GDO, GDE, GQO, GQE, last, step, ring(), ring(), ring(), n, m, N, M,
                                      BIG=DTW_BIG, BLOCK=block, STORE=store, num_warps=warps)
    else:
        _qnw_linear_fwd_kernel[(B,)](S, GDO, GQO, last, step, ring(), n, m, N, M,
                                      BIG=DTW_BIG, BLOCK=block, STORE=store, num_warps=warps)
    val, k = last.reshape(B, -1).max(-1)                            # ties -> state A, then the lower column
    if not store:
        return val, None
    jend, send = (k % (M + 1)).contiguous(), (k // (M + 1)).contiguous()
    path = torch.zeros_like(step)
    if affine:
        _qnw_affine_backtrace_kernel[(B,)](step, path, n, jend, send, N, M)
    else:
        _qnw_linear_backtrace_kernel[(B,)](step, path, n, jend, N, M)
    return val, path
