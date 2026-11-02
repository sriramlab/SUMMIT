"""Direct-effect SNP heritability from sibships, computed exactly with GRMs of the siblings (the version run in All of
Us; `summit relatives direct` implements the same estimator). fit and jkfit take --as-cols, the covariates removed in the
whole sample before centering within sibships.

Model. y = sum_j g_j b_j + sum_f 1_f a_f + e, with b_j iid (0, s_c/M_c) in class c,
a_f arbitrary for each sibship f, and Cov(e) zero between members of different extended families. P centers within
sibships (members with the trait; others get zero rows), K_c = P A_c P, and S* holds every pair of the same extended
family, the diagonal included. Then E(y~' K_c^u y~) = sum_d s_d tr(K_c^u K_d^u) exactly, which gives the direct
estimator (one equation per class, no residual parameter). The population estimator on the same people uses
P_W A_c P_W with covariates W, leaves out the off-diagonal pairs of each extended family and keeps the diagonal and the
residual equation, as in the main paper. Standard errors: delete-one-group jackknife over 200 groups of whole
extended families (rows are sorted by extended family and sibship, so each group is a contiguous block).

Subcommands
  selftest                      dense-matrix check of every block formula on a small random problem
  grm   --bfile --sibs --annot --out [--chunk --threads]       class GRMs A_c (float32 .npy, sorted row order)
  fit   --grm --pheno --cov --covpc --traits --out [--workers] direct and population estimates of real traits
  sim   --grm --bfile --sibs --pairs --covpc --pcs20 --out ...   simulations with family-level confounding
"""
import argparse, json, os, sys, time

import numpy as np
import pandas as pd

T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.0f} s] {m}', flush=True)


# ------------------------------------------------------------------------------------------------ layout
class Layout:
    """Row order sorted by (extended family, sibship); contiguous ranges of sibships, families and jackknife groups."""

    def __init__(self, sib_df, ngroups=200):
        d = sib_df.sort_values(['ext', 'sib', 'IID'], kind='mergesort').reset_index(drop=True)
        self.iid = d.IID.to_numpy()
        self.N = len(d)
        self.sib = pd.factorize(d.sib)[0]                     # increasing along the rows
        self.ext = pd.factorize(d.ext)[0]
        assert (np.diff(self.sib) >= 0).all() and (np.diff(self.ext) >= 0).all()
        self.sib_start = np.flatnonzero(np.r_[True, np.diff(self.sib) > 0])
        self.ext_start = np.flatnonzero(np.r_[True, np.diff(self.ext) > 0])
        self.ext_end = np.r_[self.ext_start[1:], self.N]
        # groups of whole extended families with about N/ngroups rows each
        cut = np.minimum(np.searchsorted(self.ext_start, np.linspace(0, self.N, ngroups + 1)[1:-1]), len(self.ext_start) - 1)
        starts = np.unique(np.r_[0, self.ext_start[cut]])
        self.g_start = starts
        self.g_end = np.r_[starts[1:], self.N]


def right_P(B, sib, sib_start, m, n_obs):
    """B P for the within-sibship centering P over members m: columns centered within sibships, non-members zero."""
    S = np.add.reduceat(B * m[None, :], sib_start, axis=1)               # rows x sibships
    mean = np.divide(S, n_obs[None, :], out=np.zeros_like(S), where=n_obs[None, :] > 0)
    return (B - mean[:, sib]) * m[None, :]


def left_P(B, sib_rows, sib_start_rows, m_rows, n_obs_rows):
    """P B for a row block that holds whole sibships (sib_rows relabelled 0.. within the block)."""
    S = np.add.reduceat(B * m_rows[:, None], sib_start_rows, axis=0)
    mean = np.divide(S, n_obs_rows[:, None], out=np.zeros_like(S), where=n_obs_rows[:, None] > 0)
    return (B - mean[sib_rows]) * m_rows[:, None]


def members(lay, obs):
    """Members: observed individuals in sibships with at least two observed members."""
    n_obs = np.add.reduceat(obs.astype(float), lay.sib_start)
    m = obs & (n_obs[lay.sib] >= 2)
    n_m = np.add.reduceat(m.astype(float), lay.sib_start)
    return m.astype(float), n_m


# ------------------------------------------------------------------------------------------------ core accumulation
def accumulate(lay, getA, C, yd, obs_pop=None, Wq=None, yp=None):
    """One pass over the row groups. getA(c, r0, r1) returns rows r0:r1 of A_c (float64, all N columns); getA.m and
    getA.n_m are the member mask and the members per sibship. yd: N x R within-sibship-centered phenotypes (zero for
    non-members). Returns per-group sums for the direct estimator and, if obs_pop is given, for the population estimator
    with covariate basis Wq (orthonormal on observed rows, zero elsewhere) and yp = P_W y."""
    m, n_m = getA.m, getA.n_m
    G = len(lay.g_start)
    R = yd.shape[1]
    out = dict(T=np.zeros((G, C, R)), W=np.zeros((G, C, R)), Gr=np.zeros((G, C, C)), Wg=np.zeros((G, C, C)))
    pop = obs_pop is not None
    if pop:
        o = obs_pop.astype(float)
        # Z_c = Wq' A_c P_W  (q x N), needed for the left projection
        Z = []
        for c in range(C):
            QA = np.zeros((Wq.shape[1], lay.N))
            for g in range(G):
                r0, r1 = lay.g_start[g], lay.g_end[g]
                QA += Wq[r0:r1].T @ getA(c, r0, r1)
            Z.append((QA - (QA @ Wq) @ Wq.T) * o[None, :])
        Rp = yp.shape[1]
        out.update(Tp=np.zeros((G, C, Rp)), Wp=np.zeros((G, C, Rp)), Gp=np.zeros((G, C, C)), Wgp=np.zeros((G, C, C)),
                   tr=np.zeros((G, C)), yy=np.zeros((G, Rp)))
    for g in range(G):
        r0, r1 = lay.g_start[g], lay.g_end[g]
        rows = slice(r0, r1)
        sib_r = lay.sib[rows] - lay.sib[r0]
        ss_r = np.flatnonzero(np.r_[True, np.diff(sib_r) > 0])
        e_s = lay.ext_start[(lay.ext_start >= r0) & (lay.ext_start < r1)]
        e_e = lay.ext_end[(lay.ext_start >= r0) & (lay.ext_start < r1)]
        Kd, Kp = [], []
        for c in range(C):
            B = getA(c, r0, r1)
            Kc = left_P(right_P(B, lay.sib, lay.sib_start, m, n_m), sib_r, ss_r, m[rows], n_m[lay.sib[rows]][ss_r])
            for s, e in zip(e_s, e_e):                     # S*: every pair of the same extended family, diagonal included
                Kc[s - r0:e - r0, s:e] = 0.0
            Kd.append(Kc)
            if pop:
                BP = (B - (B @ Wq) @ Wq.T) * o[None, :]
                Kq = (BP - Wq[rows] @ Z[c]) * o[rows][:, None]
                out['tr'][g, c] = np.trace(Kq[:, r0:r1])
                dg = np.diagonal(Kq[:, r0:r1]).copy()
                for s, e in zip(e_s, e_e):                 # population: off-diagonal pairs of each extended family out
                    Kq[s - r0:e - r0, s:e] = 0.0
                Kq[np.arange(r1 - r0), np.arange(r0, r1)] = dg
                Kp.append(Kq)
        yr = yd[rows]
        for c in range(C):
            u = Kd[c] @ yd
            out['T'][g, c] = (yr * u).sum(0)
            out['W'][g, c] = (yr * (Kd[c][:, rows] @ yr)).sum(0)
            for d in range(c + 1):
                v = (Kd[c] * Kd[d]).sum()
                w = (Kd[c][:, rows] * Kd[d][:, rows]).sum()
                out['Gr'][g, c, d] = out['Gr'][g, d, c] = v
                out['Wg'][g, c, d] = out['Wg'][g, d, c] = w
            if pop:
                ypr = yp[rows]
                u = Kp[c] @ yp
                out['Tp'][g, c] = (ypr * u).sum(0)
                out['Wp'][g, c] = (ypr * (Kp[c][:, rows] @ ypr)).sum(0)
                for d in range(c + 1):
                    v = (Kp[c] * Kp[d]).sum()
                    w = (Kp[c][:, rows] * Kp[d][:, rows]).sum()
                    out['Gp'][g, c, d] = out['Gp'][g, d, c] = v
                    out['Wgp'][g, c, d] = out['Wgp'][g, d, c] = w
        if pop:
            out['yy'][g] = (yp[rows] ** 2).sum(0)
    return out


def solve_direct(acc, wts=None):
    """Direct estimates (classes x R) for the full sample and each deleted group. wts: class weights (M_c/M) to
    collapse to one class (returns 1 x R)."""
    T, W, Gr, Wg = acc['T'], acc['W'], acc['Gr'], acc['Wg']
    rhs, gram = T.sum(0), Gr.sum(0)
    if wts is not None:
        w = np.asarray(wts)
        rhs1, gram1 = (w[:, None] * rhs).sum(0, keepdims=True), np.array([[w @ gram @ w]])
        full = rhs1 / gram1[0, 0]
        dele = np.array([((w[:, None] * (rhs - 2 * T[g] + W[g])).sum(0, keepdims=True)) / (w @ (gram - 2 * Gr[g] + Wg[g]) @ w)
                         for g in range(T.shape[0])])
        return full, dele
    full = np.linalg.solve(gram, rhs)
    dele = np.array([np.linalg.solve(gram - 2 * Gr[g] + Wg[g], rhs - 2 * T[g] + W[g]) for g in range(T.shape[0])])
    return full, dele


def solve_pop(acc, nu, wts=None):
    """Population estimates: rows = classes then residual, for the full sample and each deleted group (P_W fixed)."""
    Tp, Wp, Gp, Wgp, tr, yy = acc['Tp'], acc['Wp'], acc['Gp'], acc['Wgp'], acc['tr'], acc['yy']
    G = Tp.shape[0]

    def one(rhs, gram, t, yty, n):
        if wts is not None:
            w = np.asarray(wts)
            rhs, gram, t = (w[:, None] * rhs).sum(0, keepdims=True), np.array([[w @ gram @ w]]), np.array([w @ t])
        C = gram.shape[0]
        Mx = np.zeros((C + 1, C + 1))
        Mx[:C, :C], Mx[:C, C], Mx[C, :C], Mx[C, C] = gram, t, t, n
        return np.linalg.solve(Mx, np.vstack([rhs, yty[None, :]]))

    full = one(Tp.sum(0), Gp.sum(0), tr.sum(0), yy.sum(0), nu)
    sizes = acc['n_obs_g']
    dele = np.array([one(Tp.sum(0) - 2 * Tp[g] + Wp[g], Gp.sum(0) - 2 * Gp[g] + Wgp[g], tr.sum(0) - tr[g],
                         yy.sum(0) - yy[g], nu - sizes[g]) for g in range(G)])
    return full, dele


def jk_se(full, dele):
    G = dele.shape[0]
    return np.sqrt((G - 1) / G * ((dele - dele.mean(0)) ** 2).sum(0))


# ------------------------------------------------------------------------------------------------ simulation pass
VARIANTS = (('dir', 'ext'), ('dir', 'sib'), ('pop', 'ext'), ('pop', 'none'))


def sim_accumulate(lay, getA1, m, n_m, Yd, Wq, Yp):
    """Single-class accumulation for the simulation variants (VARIANTS): direct kernel with S* = extended family or
    sibship (diagonal included), population kernel (covariate basis Wq on everyone) leaving out the off-diagonal pairs
    of each extended family or nothing. Yd = P_sib Y, Yp = P_W Y (N x R). Returns per-group sums."""
    G, R = len(lay.g_start), Yd.shape[1]
    out = {v: dict(T=np.zeros((G, R)), W=np.zeros((G, R)), Gr=np.zeros(G), Wg=np.zeros(G), tr=np.zeros(G)) for v in VARIANTS}
    QA = np.zeros((Wq.shape[1], lay.N))
    for g in range(G):
        r0, r1 = lay.g_start[g], lay.g_end[g]
        QA += Wq[r0:r1].T @ getA1(r0, r1)
    Z = QA - (QA @ Wq) @ Wq.T
    for g in range(G):
        r0, r1 = lay.g_start[g], lay.g_end[g]
        rows = slice(r0, r1)
        sib_r = lay.sib[rows] - lay.sib[r0]
        ss_r = np.flatnonzero(np.r_[True, np.diff(sib_r) > 0])
        sel = (lay.ext_start >= r0) & (lay.ext_start < r1)
        e_s, e_e = lay.ext_start[sel], lay.ext_end[sel]
        s_s = lay.sib_start[(lay.sib_start >= r0) & (lay.sib_start < r1)]
        s_e = np.r_[s_s[1:], r1]
        B = getA1(r0, r1)
        Kd0 = left_P(right_P(B, lay.sib, lay.sib_start, m, n_m), sib_r, ss_r, m[rows], n_m[lay.sib[r0 + ss_r]])
        BP = B - (B @ Wq) @ Wq.T
        Kp0 = BP - Wq[rows] @ Z
        for kind, ex in VARIANTS:
            K = (Kd0 if kind == 'dir' else Kp0).copy()
            o = out[(kind, ex)]
            o['tr'][g] = np.trace(K[:, rows])
            if ex != 'none':
                dg = np.diagonal(K[:, rows]).copy()
                for s, e in (zip(e_s, e_e) if ex == 'ext' else zip(s_s, s_e)):
                    K[s - r0:e - r0, s:e] = 0.0
                if kind == 'pop':
                    K[np.arange(r1 - r0), np.arange(r0, r1)] = dg
            Y = Yd if kind == 'dir' else Yp
            o['T'][g] = (Y[rows] * (K @ Y)).sum(0)
            o['W'][g] = (Y[rows] * (K[:, rows] @ Y[rows])).sum(0)
            o['Gr'][g] = (K * K).sum()
            o['Wg'][g] = (K[:, rows] * K[:, rows]).sum()
    return out


def sim_solve(o, kind, yy=None, nu=None, nsize=None):
    """Estimates (R,) and delete-group estimates (G x R): direct sigma^2, or population h2 from two equations."""
    T, W, Gr, Wg, tr = o['T'], o['W'], o['Gr'], o['Wg'], o['tr']
    G = T.shape[0]
    rhs, gram = T.sum(0), Gr.sum()
    if kind == 'dir':
        return rhs / gram, np.array([(rhs - 2 * T[g] + W[g]) / (gram - 2 * Gr[g] + Wg[g]) for g in range(G)])

    def h2(rh, gm, t, yty, n):
        sg = (rh - t / n * yty) / (gm - t * t / n)
        se = (yty - t * sg) / n
        return sg / (sg + se)
    full = h2(rhs, gram, tr.sum(), yy.sum(0), nu)
    dele = np.array([h2(rhs - 2 * T[g] + W[g], gram - 2 * Gr[g] + Wg[g], tr.sum() - tr[g], yy.sum(0) - yy[g], nu - nsize[g])
                     for g in range(G)])
    return full, dele


# ------------------------------------------------------------------------------------------------ selftest
def cmd_selftest(a):
    rng = np.random.default_rng(1)
    # random extended families with sibships, plus singletons
    rows = []
    sid = eid = 0
    for _ in range(60):
        nsib = rng.integers(1, 4)
        for _ in range(nsib):
            for _ in range(rng.integers(1, 4)):
                rows.append((f'i{len(rows)}', sid, eid))
            sid += 1
        eid += 1
    d = pd.DataFrame(rows, columns=['IID', 'sib', 'ext']).sample(frac=1, random_state=3)
    lay = Layout(d, ngroups=7)
    N, C, M = lay.N, 3, 400
    X = rng.normal(size=(N, M))
    cls = rng.integers(0, C, M)
    A = [X[:, cls == c] @ X[:, cls == c].T / (cls == c).sum() for c in range(C)]
    obs = rng.random(N) > 0.1
    m, n_m = members(lay, obs)
    y = rng.normal(size=(N, 2)) + X[:, :1]
    # dense direct
    Pm = np.zeros((N, N))
    for f in np.unique(lay.sib):
        ix = np.flatnonzero((lay.sib == f) & (m > 0))
        if len(ix) >= 2:
            Pm[np.ix_(ix, ix)] = np.eye(len(ix)) - 1 / len(ix)
    same = lay.ext[:, None] == lay.ext[None, :]
    Kd = [Pm @ Ac @ Pm * ~same for Ac in A]
    yd = Pm @ y
    gram = np.array([[(Kd[c] * Kd[d]).sum() for d in range(C)] for c in range(C)])
    rhs = np.array([[yd[:, r] @ Kd[c] @ yd[:, r] for r in range(2)] for c in range(C)])
    dense = np.linalg.solve(gram, rhs)
    # dense population with covariates
    Wc = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
    o = obs.astype(float)
    Q = np.linalg.qr(Wc[obs])[0]
    Wq = np.zeros((N, Q.shape[1])); Wq[obs] = Q
    PW = np.diag(o) - Wq @ Wq.T
    Kp = []
    off = same & ~np.eye(N, dtype=bool)
    for Ac in A:
        K = PW @ Ac @ PW
        Kp.append(K * ~off)
    yp = PW @ np.nan_to_num(y)
    nu = obs.sum() - Q.shape[1]
    Mx = np.zeros((C + 1, C + 1))
    for c in range(C):
        for dd in range(C):
            Mx[c, dd] = (Kp[c] * Kp[dd]).sum()
        Mx[c, C] = Mx[C, c] = np.trace(PW @ A[c] @ PW)
    Mx[C, C] = nu
    rp = np.vstack([np.array([[yp[:, r] @ Kp[c] @ yp[:, r] for r in range(2)] for c in range(C)]), (yp ** 2).sum(0)])
    dense_p = np.linalg.solve(Mx, rp)

    def getA(c, r0, r1):
        return A[c][r0:r1].copy()
    getA.m, getA.n_m = m, n_m
    acc = accumulate(lay, getA, C, yd, obs_pop=obs, Wq=Wq, yp=yp)
    acc['n_obs_g'] = np.array([obs[s:e].sum() for s, e in zip(lay.g_start, lay.g_end)])
    full, dele = solve_direct(acc)
    fp, dp = solve_pop(acc, nu)
    print('direct max abs diff', np.abs(full - dense).max())
    print('population max abs diff', np.abs(fp - dense_p).max())
    # deleted group 2, recomputed from scratch
    g = 2
    keep = np.ones(N, bool); keep[lay.g_start[g]:lay.g_end[g]] = False
    kk = np.ix_(keep, keep)
    gram2 = np.array([[(Kd[c][kk] * Kd[dd][kk]).sum() for dd in range(C)] for c in range(C)])
    rhs2 = np.array([[yd[keep, r] @ Kd[c][kk] @ yd[keep, r] for r in range(2)] for c in range(C)])
    print('deleted-group direct max abs diff', np.abs(np.linalg.solve(gram2, rhs2) - dele[g]).max())
    w = np.array([(cls == c).mean() for c in range(C)])
    f1, _ = solve_direct(acc, wts=w)
    Asg = sum(w[c] * A[c] for c in range(C))
    K1 = Pm @ Asg @ Pm * ~same
    print('single-class direct max abs diff', np.abs(f1[0] - np.array([yd[:, r] @ K1 @ yd[:, r] for r in range(2)]) / (K1 ** 2).sum()).max())
    # unbiasedness: E(y~'K^u y~) = s tr(K^u K^u) under y = sqrt(s) X b + family effects + within-family noise
    # simulation variants against dense matrices (single class, everyone observed)
    A1 = sum(w[c] * A[c] for c in range(C))
    allobs = np.ones(N, bool)
    m1, n1 = members(lay, allobs)
    P1 = np.zeros((N, N))
    for f in np.unique(lay.sib):
        ix = np.flatnonzero((lay.sib == f) & (m1 > 0))
        if len(ix) >= 2:
            P1[np.ix_(ix, ix)] = np.eye(len(ix)) - 1 / len(ix)
    Q1 = np.linalg.qr(Wc)[0]
    PW1 = np.eye(N) - Q1 @ Q1.T
    Y = rng.normal(size=(N, 3))
    o = sim_accumulate(lay, lambda r0, r1: A1[r0:r1].copy(), m1, n1, P1 @ Y, Q1, PW1 @ Y)
    sames = lay.sib[:, None] == lay.sib[None, :]
    yyg = np.array([((PW1 @ Y)[s:e] ** 2).sum(0) for s, e in zip(lay.g_start, lay.g_end)])
    nsz = np.array([e - s for s, e in zip(lay.g_start, lay.g_end)])
    for kind, ex in VARIANTS:
        K = (P1 @ A1 @ P1) if kind == 'dir' else (PW1 @ A1 @ PW1)
        if ex == 'ext':
            K = K * ~same if kind == 'dir' else K * ~off
        elif ex == 'sib':
            K = K * ~sames
        Yk = P1 @ Y if kind == 'dir' else PW1 @ Y
        q = np.array([Yk[:, r] @ K @ Yk[:, r] for r in range(3)])
        if kind == 'dir':
            ref = q / (K * K).sum()
            got = sim_solve(o[(kind, ex)], 'dir')[0]
        else:
            t, n = np.trace(PW1 @ A1 @ PW1), N - Q1.shape[1]
            yy = (Yk ** 2).sum(0)
            sg = (q - t / n * yy) / ((K * K).sum() - t * t / n)
            ref = sg / (sg + (yy - t * sg) / n)
            got = sim_solve(o[(kind, ex)], 'pop', yy=yyg, nu=n, nsize=nsz)[0]
        print(f'simulation variant {kind}/{ex} max abs diff', np.abs(got - ref).max())
    # summary-statistic form (PLAN section 3): within-sibship GWAS moments m_j, LD scores of the centered genotypes and
    # their S* parts give the same estimate as the kernel form
    Gs = X[:, cls == 0]
    Ms = Gs.shape[1]
    PG = Pm @ Gs
    nu = int(m.sum() - (n_m >= 2).sum())
    Vj = (PG ** 2).sum(0) / nu
    xj = PG / np.sqrt(Vj)[None, :]
    Kx = (xj * Vj[None, :]) @ xj.T / Ms
    assert np.allclose(Kx, Pm @ A[0] @ Pm)
    yt = yd[:, 0] * np.sqrt(nu / (yd[:, 0] @ yd[:, 0]))
    mj = (xj.T @ yt) ** 2 / nu
    rr = (xj.T @ xj) / nu
    Vp = Vj.sum()
    lj = (rr ** 2 * Vj[None, :]).sum(1) - Vp / nu
    Sm = same.astype(float)                                  # S*: same extended family, diagonal included
    mstar = np.einsum('aj,bj,ab,a,b->j', xj, xj, Sm, yt, yt) / nu
    lstar = Ms / nu ** 2 * np.einsum('aj,bj,ab->j', xj, xj, Sm * Kx)
    cstar = (Vj * mstar).sum() / Vp
    summ = Ms / nu * (Vj * (mj - cstar)).sum() / (Vj * (lj + Vp / nu - lstar)).sum()
    Ku = Kx * ~same
    exact = yt @ Ku @ yt / (Ku * Ku).sum()
    print('summary-statistic form vs exact', summ, exact, abs(summ - exact))
    est = []; nrep = 20000
    for _ in range(nrep):
        b = rng.normal(size=M) / np.sqrt(np.array([(cls == c).sum() for c in range(C)])[cls])
        sv = np.array([0.3, 0.1, 0.2])[cls]
        g_ = X @ (b * np.sqrt(sv))
        fam_eff = rng.normal(size=lay.sib.max() + 1)[lay.sib] * 2 + rng.normal(size=lay.ext.max() + 1)[lay.ext]
        yy = g_ + fam_eff + rng.normal(size=N)
        ydd = Pm @ yy
        est.append(np.linalg.solve(gram, np.array([ydd @ Kd[c] @ ydd for c in range(C)])))
    est = np.array(est)
    print('Monte Carlo mean of direct estimates', np.round(est.mean(0), 4), 'MC SE', np.round(est.std(0) / np.sqrt(nrep), 4),
          'truth [0.3 0.1 0.2]')


# ------------------------------------------------------------------------------------------------ grm
def read_sibs(path, fam_iid):
    d = pd.read_csv(path, sep='\t', dtype={'IID': str})
    assert set(d.IID) == set(fam_iid), 'sibship table and fam differ'
    return d


def cmd_grm(a):
    from bed_reader import open_bed
    from scipy.linalg import blas
    os.makedirs(a.out, exist_ok=True)
    fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
    lay = Layout(read_sibs(a.sibs, fam))
    pos = pd.Series(np.arange(len(fam)), index=fam)
    perm = pos[lay.iid].to_numpy()
    bim = pd.read_csv(a.bfile + '.bim', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
    an = pd.read_csv(a.annot, sep='\t')
    assert (an.SNP.astype(str).to_numpy() == bim).all(), 'annotation and bim SNP order differ'
    ccols = list(an.columns[4:])
    Z = an[ccols].to_numpy()
    assert (Z.sum(1) == 1).all()
    cls = Z.argmax(1)
    N = lay.N
    pd.DataFrame({'IID': lay.iid, 'sib': lay.sib, 'ext': lay.ext}).to_csv(f'{a.out}/order.tsv', sep='\t', index=False)
    bed = open_bed(a.bfile + '.bed', count_A1=True, num_threads=a.threads)
    meta = json.load(open(f'{a.out}/meta.json')) if os.path.exists(f'{a.out}/meta.json') else {'classes': ccols, 'M': {}}
    for c, name in enumerate(ccols):
        path = f'{a.out}/A_{name}.npy'
        if os.path.exists(path) and name in meta['M']:
            continue
        idx = np.flatnonzero(cls == c)
        acc = np.zeros((N, N), order='F')
        used = 0
        for s in range(0, len(idx), a.chunk):
            X = bed.read(index=np.s_[perm, idx[s:s + a.chunk]], dtype='float64', order='F')
            mu = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
            sd = np.sqrt((X ** 2).sum(0) / (N - 1))
            ok = sd > 0
            X = np.asfortranarray(X[:, ok] / sd[ok])
            r = blas.dsyrk(1.0, X, beta=1.0, c=acc, lower=1, overwrite_c=1)
            assert np.shares_memory(r, acc)
            used += int(ok.sum())
        Am = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.float32, shape=(N, N))
        step = 2048
        for i in range(0, N, step):                       # symmetrize from the lower triangle while writing
            j = min(N, i + step)
            blk = acc[i:j, :].copy()
            low = np.tril(np.ones((j - i, N), bool), k=i)
            blk = np.where(low, blk, acc[:, i:j].T)
            Am[i:j] = (blk / used).astype(np.float32)
        Am.flush(); del Am, acc
        os.replace(path + '.tmp', path)
        meta['M'][name] = used
        json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
        log(f'class {name}: {used} SNPs')
    log('GRMs done')


# ------------------------------------------------------------------------------------------------ fit
def load_grm(path):
    meta = json.load(open(f'{path}/meta.json'))
    order = pd.read_csv(f'{path}/order.tsv', sep='\t', dtype={'IID': str})
    As = [np.load(f'{path}/A_{n}.npy', mmap_mode='r') for n in meta['classes']]
    Mc = np.array([meta['M'][n] for n in meta['classes']], float)
    return meta, order, As, Mc


def make_getA(As, m, n_m):
    def getA(c, r0, r1):
        return np.asarray(As[c][r0:r1], dtype=np.float64)
    getA.m, getA.n_m = m, n_m
    return getA


def cmd_fit(a):
    meta, order, As, Mc = load_grm(a.grm)
    lay = Layout(order[['IID', 'sib', 'ext']].assign(sib=order.sib, ext=order.ext))
    assert (lay.iid == order.IID.to_numpy()).all()
    C = len(As)
    wts = Mc / Mc.sum()
    ph = pd.read_csv(a.pheno, sep='\t', dtype={'FID': str, 'IID': str}).set_index('IID')
    cv = pd.read_csv(a.covpc, sep='\t', dtype={'FID': str, 'IID': str}).set_index('IID').reindex(ph.index)
    as_cols = a.as_cols.split(',')
    pc_cols = [c for c in cv.columns if c.startswith('PC')]
    assert cv[as_cols + pc_cols].notna().all().all()
    traits = a.traits.split(',') if a.traits else [c for c in ph.columns if c != 'FID']
    os.makedirs(a.out, exist_ok=True)
    for t in traits:
        if os.path.exists(f'{a.out}/{t}.json'):
            continue
        y = ph[t].to_numpy(float)
        ok = np.isfinite(y)
        W1 = np.column_stack([np.ones(ok.sum()), cv.loc[ok, as_cols].to_numpy(float)])
        W2 = np.column_stack([W1, cv.loc[ok, pc_cols].to_numpy(float)])
        b1 = np.linalg.lstsq(W1, y[ok], rcond=None)[0]
        res2 = y[ok] - W2 @ np.linalg.lstsq(W2, y[ok], rcond=None)[0]
        var_full = float(res2 @ res2 / (ok.sum() - W2.shape[1]))     # population scale of h2
        y_as = pd.Series(y[ok] - W1 @ b1, index=ph.index[ok]).reindex(lay.iid).to_numpy()
        y_raw = ph[t].reindex(lay.iid).to_numpy(float)
        obs = np.isfinite(y_raw)
        m, n_m = members(lay, obs)
        yd = np.zeros((lay.N, 1))
        yc = np.where(m > 0, y_as, 0.0)
        mean = np.divide(np.add.reduceat(yc, lay.sib_start), n_m, out=np.zeros(len(n_m)), where=n_m > 0)
        yd[:, 0] = (yc - mean[lay.sib]) * m
        Wsib = np.column_stack([np.ones(lay.N), cv[as_cols + pc_cols].reindex(lay.iid).to_numpy(float)])
        Q = np.linalg.qr(Wsib[obs])[0]
        Wq = np.zeros((lay.N, Q.shape[1])); Wq[obs] = Q
        yp = np.zeros((lay.N, 1))
        yo = np.where(obs, y_raw, 0.0)
        yp[:, 0] = (yo - Wq @ (Wq.T @ yo)) * obs
        nu_p = int(obs.sum()) - Q.shape[1]
        getA = make_getA(As, m, n_m)
        acc = accumulate(lay, getA, C, yd, obs_pop=obs, Wq=Wq, yp=yp)
        acc['n_obs_g'] = np.array([obs[s:e].sum() for s, e in zip(lay.g_start, lay.g_end)])
        res = dict(trait=t, n_members=int(m.sum()), n_sibships=int((n_m >= 2).sum()), nu_dir=int(m.sum() - (n_m >= 2).sum()),
                   n_pop=int(obs.sum()), var_full=var_full, var_within=float((yd[:, 0] ** 2).sum() / (m.sum() - (n_m >= 2).sum())))
        res['gram_dir'] = acc['Gr'].sum(0).tolist(); res['rhs_dir'] = acc['T'].sum(0)[:, 0].tolist()
        for model, w in (('single', wts), ('8bin', None)):
            f, dl = solve_direct(acc, wts=w)
            s, sd = f.sum(0)[0], dl.sum(1)[:, 0]
            res[f'dir_{model}'] = s / var_full
            res[f'dir_{model}_se'] = float(jk_se(np.array(s), sd)) / var_full
            fp, dp = solve_pop(acc, nu_p, wts=w)
            sg, se_ = fp[:-1].sum(0)[0], fp[-1, 0]
            dsg, dse = dp[:, :-1].sum(1)[:, 0], dp[:, -1, 0]
            h, dh = sg / (sg + se_), dsg / (dsg + dse)
            res[f'pop_{model}'] = float(h)
            res[f'pop_{model}_se'] = float(jk_se(np.array(h), dh))
            ratio, dr = (s / var_full) / h, (sd / var_full) / dh       # direct over population, same people
            res[f'ratio_{model}'] = float(ratio)
            res[f'ratio_{model}_se'] = float(jk_se(np.array(ratio), dr))
            diff, dd = h - s / var_full, dh - sd / var_full
            res[f'diff_{model}'] = float(diff)
            res[f'diff_{model}_se'] = float(jk_se(np.array(diff), dd))
            if model == '8bin':
                res['dir_8bin_classes'] = (f[:, 0] / var_full).tolist()
        json.dump(res, open(f'{a.out}/{t}.json', 'w'), indent=1)
        log(f"{t}: direct {res['dir_8bin']:.3f} ({res['dir_8bin_se']:.3f}), population {res['pop_8bin']:.3f} "
            f"({res['pop_8bin_se']:.3f}), ratio {res['ratio_8bin']:.2f} ({res['ratio_8bin_se']:.2f})")


# ------------------------------------------------------------------------------------------------ sim
SIM_SETTINGS = {           # variances; h2 = direct genetic variance; the noise variance makes the total one
    'base':      dict(h2=0.4),
    'sibenv':    dict(h2=0.4, c=0.2),
    'extfam':    dict(h2=0.4, c=0.1, x=0.05, u=0.1),
    'nurture':   dict(h2=0.4, n=0.1),
    'ancestry':  dict(h2=0.4, a=0.05),
    'sibeffect': dict(h2=0.4, s=0.02),
}


def cmd_sim(a):
    from bed_reader import open_bed
    meta, order, As, Mc = load_grm(a.grm)
    lay = Layout(order[['IID', 'sib', 'ext']])
    assert (lay.iid == order.IID.to_numpy()).all()
    N, wts = lay.N, Mc / Mc.sum()
    os.makedirs(a.out, exist_ok=True)
    p1 = f'{a.grm}/A_single.npy'
    if not os.path.exists(p1):
        A1 = np.lib.format.open_memmap(p1 + '.tmp', mode='w+', dtype=np.float32, shape=(N, N))
        for i in range(0, N, 2048):
            j = min(N, i + 2048)
            A1[i:j] = sum(w * np.asarray(Ac[i:j], dtype=np.float64) for w, Ac in zip(wts, As)).astype(np.float32)
        A1.flush(); del A1
        os.replace(p1 + '.tmp', p1)
    A1 = np.load(p1, mmap_mode='r')
    # genetic values with the standardization of the GRMs: E(g g') = A exactly for effects iid N(0, 1/M)
    gpath = f'{a.out}/genetic.npz'
    R = a.nrep
    if not os.path.exists(gpath):
        fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
        perm = pd.Series(np.arange(len(fam)), index=fam)[lay.iid].to_numpy()
        bed = open_bed(a.bfile + '.bed', count_A1=True, num_threads=8)
        Mt = bed.sid_count
        rng = np.random.default_rng([a.seed, 1])
        Gd, Gn, used = np.zeros((N, R)), np.zeros((N, R)), 0
        for s in range(0, Mt, 4096):
            X = bed.read(index=np.s_[perm, s:min(Mt, s + 4096)], dtype='float64', order='F')
            mu = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
            sd = np.sqrt((X ** 2).sum(0) / (N - 1))
            ok = sd > 0
            X = X[:, ok] / sd[ok]
            b1 = rng.standard_normal((X.shape[1], R))
            b2 = a.rho * b1 + np.sqrt(1 - a.rho ** 2) * rng.standard_normal((X.shape[1], R))
            Gd += X @ b1; Gn += X @ b2
            used += int(ok.sum())
        assert used == int(Mc.sum()), (used, Mc.sum())
        Gd /= np.sqrt(used); Gn /= np.sqrt(used)
        np.savez(gpath, Gd=Gd, Gn=Gn)
        log(f'genetic values: mean variance {Gd.var(0).mean():.4f} (direct), {Gn.var(0).mean():.4f} (nurture effects)')
    gz = np.load(gpath); Gd, Gn = gz['Gd'], gz['Gn']
    # family structure: relatedness 2 x kinship of released pairs among the siblings' extended families
    pr = pd.read_csv(a.pairs, sep='\t', dtype={'IID1': str, 'IID2': str})
    pos = pd.Series(np.arange(N), index=lay.iid)
    pr = pr[pr.IID1.isin(pos.index) & pr.IID2.isin(pos.index)]
    pi, pj, pk = pos[pr.IID1].to_numpy(), pos[pr.IID2].to_numpy(), 2 * pr.Kinship.to_numpy()
    assert (lay.ext[pi] == lay.ext[pj]).all()
    roots = []
    for s, e in zip(lay.ext_start, lay.ext_end):
        if e - s > 1:
            Phi = np.eye(e - s)
            sel = (pi >= s) & (pi < e)
            Phi[pi[sel] - s, pj[sel] - s] = pk[sel]; Phi[pj[sel] - s, pi[sel] - s] = pk[sel]
            wv, v = np.linalg.eigh(Phi)
            roots.append((s, e, v * np.sqrt(np.clip(wv, 0, None))[None, :]))
    sc = pd.read_csv(a.pcs20, sep='\t', dtype={'IID': str}).set_index('IID').reindex(lay.iid)
    hidden = sc[a.hidden_pc + '_AVG'].to_numpy(float)
    hidden = (hidden - hidden.mean()) / hidden.std()
    cv = pd.read_csv(a.covpc, sep='\t', dtype={'IID': str}).set_index('IID').reindex(lay.iid)
    Wc = np.column_stack([np.ones(N), cv[[f'PC{k}' for k in range(1, 11)]].to_numpy(float)])
    assert np.isfinite(Wc).all() and np.isfinite(hidden).all()
    Q = np.linalg.qr(Wc)[0]
    nsib = np.bincount(lay.sib)
    mean_sib = lambda v: (np.add.reduceat(v, lay.sib_start, axis=0) / nsib[:, None])[lay.sib]
    Ys, names = [], list(SIM_SETTINGS)
    for k, name in enumerate(names):
        S = SIM_SETTINGS[name]
        rng = np.random.default_rng([a.seed, 2, k])
        Y = np.sqrt(S['h2']) * Gd
        if S.get('c'):
            Y += np.sqrt(S['c']) * rng.standard_normal((lay.sib.max() + 1, R))[lay.sib]
        if S.get('x'):
            Y += np.sqrt(S['x']) * rng.standard_normal((lay.ext.max() + 1, R))[lay.ext]
        if S.get('u'):
            U = rng.standard_normal((N, R))
            for s, e, L in roots:
                U[s:e] = L @ U[s:e]
            Y += np.sqrt(S['u']) * U
        if S.get('n'):
            nb = mean_sib(Gn)
            Y += np.sqrt(S['n']) * nb / nb.std(0)
        if S.get('a'):                                   # environment following the sibship's mean score on a PC
            hb = mean_sib(hidden[:, None])[:, 0]
            Y += np.sqrt(S['a']) * (hb / hb.std())[:, None]
        if S.get('s'):                                   # effect of each sibling's genetic value on the others
            Y += np.sqrt(S['s']) * (mean_sib(Gd) * nsib[lay.sib][:, None] - Gd)
        rest = 1 - sum(v for kk, v in S.items() if kk != 'h2') - S['h2']
        Y += np.sqrt(rest) * rng.standard_normal((N, R))
        Ys.append(Y)
    Y = np.concatenate(Ys, axis=1)
    allobs = np.ones(N, bool)
    m1, n1 = members(lay, allobs)
    Yd = (Y - mean_sib(Y)) * m1[:, None]
    Yp = Y - Q @ (Q.T @ Y)
    o = sim_accumulate(lay, lambda r0, r1: np.asarray(A1[r0:r1], dtype=np.float64), m1, n1, Yd, Q, Yp)
    np.save(f'{a.out}/Yd.npy', Yd)
    np.savez(f'{a.out}/exact_dir.npz', gram=o[('dir', 'ext')]['Gr'].sum(), rhs=o[('dir', 'ext')]['T'].sum(0))
    yyg = np.array([(Yp[s:e] ** 2).sum(0) for s, e in zip(lay.g_start, lay.g_end)])
    nsz = np.array([e - s for s, e in zip(lay.g_start, lay.g_end)])
    rows = []
    for kind, ex in VARIANTS:
        f, d = sim_solve(o[(kind, ex)], kind, yy=yyg, nu=N - Q.shape[1], nsize=nsz)
        se = jk_se(f, d)
        for k, name in enumerate(names):
            sl = slice(k * R, (k + 1) * R)
            for r in range(R):
                rows.append(dict(setting=name, estimator=f'{kind}_{ex}', rep=r, est=f[sl][r], se=se[sl][r],
                                 truth=SIM_SETTINGS[name]['h2']))
    res = pd.DataFrame(rows)
    res.to_csv(f'{a.out}/sim_results.tsv', sep='\t', index=False, float_format='%.6g')
    T = 1.9720                                            # t quantile, 199 df (200 groups)
    smry = res.assign(err=res.est - res.truth, cover=(res.est - res.truth).abs() <= T * res.se).groupby(
        ['setting', 'estimator']).agg(mean=('est', 'mean'), bias=('err', 'mean'), mcse=('err', lambda e: e.std() / np.sqrt(len(e))),
                                      sd=('est', 'std'), mean_se=('se', 'mean'), coverage=('cover', 'mean')).reset_index()
    smry.to_csv(f'{a.out}/sim_summary.tsv', sep='\t', index=False, float_format='%.5g')
    print(smry.round(4).to_string())


# ------------------------------------------------------------------------------------------------ SNP-block jackknife
def spairs(lay):
    """Ordered pairs (a, b) of the same extended family, the diagonal included (the set S*)."""
    pa, pb = [], []
    for s, e in zip(lay.ext_start, lay.ext_end):
        ix = np.arange(s, e)
        pa.append(np.repeat(ix, e - s)); pb.append(np.tile(ix, e - s))
    return np.concatenate(pa), np.concatenate(pb)


def cmd_blocks(a):
    """Phenotype-free parts of the SNP-block jackknife, with every sibling a member (all are genotyped):
    D[b, c, d] = M_c^-1 sum_{j in b, class c} (x~_j' A_d x~_j - S* part), the contribution of block b to
    tr(K_c^u K_d^u) with x~_j = P g_j (LD-score form, one-sided as in SUMMIT's jackknife), and
    AS[b, c, :] = M_c^-1 sum_{j in b, class c} g_aj g_bj at the pairs of S* (for the S* parts of the numerators)."""
    from bed_reader import open_bed
    meta, order, As, Mc = load_grm(a.grm)
    lay = Layout(order[['IID', 'sib', 'ext']])
    N, C = lay.N, len(As)
    fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
    perm = pd.Series(np.arange(len(fam)), index=fam)[lay.iid].to_numpy()
    an = pd.read_csv(a.annot, sep='\t')
    cls = an[meta['classes']].to_numpy().argmax(1)
    Mt = len(cls)
    blk = np.minimum((np.arange(Mt) * a.nblocks) // Mt, a.nblocks - 1)    # contiguous blocks of equal SNP counts
    pa, pb = spairs(lay)
    m1, n1 = members(lay, np.ones(N, bool))
    log(f'loading {C} GRMs into memory')
    Af = [np.asarray(A, dtype=np.float32) for A in As]
    # S* blocks of K_d = P A_d P (within extended families)
    Ks = []
    for A in Af:
        v = np.empty(len(pa))
        k = 0
        for s, e in zip(lay.ext_start, lay.ext_end):
            B = A[s:e, s:e].astype(np.float64)
            sib = lay.sib[s:e] - lay.sib[s]
            P = (sib[:, None] == sib[None, :]) / np.bincount(sib)[sib][None, :]
            P = np.eye(e - s) - P
            v[k:k + (e - s) ** 2] = (P @ B @ P).ravel(); k += (e - s) ** 2
        Ks.append(v)
    D = np.zeros((a.nblocks, C, C)); AS = np.zeros((a.nblocks, C, len(pa)), dtype=np.float32)
    bed = open_bed(a.bfile + '.bed', count_A1=True, num_threads=8)
    used = np.zeros(C, int)
    for s in range(0, Mt, a.chunk):
        X = bed.read(index=np.s_[perm, s:min(Mt, s + a.chunk)], dtype='float64', order='F')
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
        sd = np.sqrt((X ** 2).sum(0) / (N - 1))
        ok = sd > 0
        X[:, ok] /= sd[ok]; X[:, ~ok] = 0.0
        Xt = right_P(X.T, lay.sib, lay.sib_start, m1, n1).T             # P g_j (within-sibship centering)
        Xf = np.asfortranarray(Xt, dtype=np.float32)
        cj, bj = cls[s:s + X.shape[1]], blk[s:s + X.shape[1]]
        q = np.empty((X.shape[1], C))
        sst = np.empty((X.shape[1], C))
        prod = Xt[pa] * Xt[pb]                                          # |S*| x k
        for d in range(C):
            Y = Af[d] @ Xf
            q[:, d] = (Xf * Y).sum(0, dtype=np.float64)
            sst[:, d] = Ks[d] @ prod
        gg = (X[pa] * X[pb]).astype(np.float64)                         # population-standardized g at S*
        for c in range(C):
            for b in np.unique(bj[(cj == c) & ok]):
                sel = (cj == c) & (bj == b) & ok
                D[b, c] += (q[sel] - sst[sel]).sum(0)
                AS[b, c] += gg[:, sel].sum(1).astype(np.float32)
            used[c] += int(((cj == c) & ok).sum())
        if (s // a.chunk) % 10 == 0:
            log(f'SNPs {s + X.shape[1]} of {Mt}')
    assert (used == Mc).all(), (used, Mc)
    D /= Mc[None, :, None]
    AS /= Mc[None, :, None].astype(np.float32)
    np.savez(f'{a.grm}/blocks.npz', D=D, AS=AS, pa=pa, pb=pb, blk=blk, cls=cls)
    log('blocks done')


def block_numerators(lay, bed, perm, cls, blk, Mc, AS, pa, pb, Y, nblocks, chunk=4096):
    """rhs[b, c, r] = y~_r' K_{b,c}^u y~_r for centered phenotypes Y (N x R, zero for non-members):
    M_c^-1 sum_{j in b, c} (g_j' y~)^2 minus the S* part sum_{S*} AS[b, c] y_a y_b."""
    N, R = Y.shape
    C = len(Mc)
    out = np.zeros((nblocks, C, R))
    Mt = len(cls)
    for s in range(0, Mt, chunk):
        X = bed.read(index=np.s_[perm, s:min(Mt, s + chunk)], dtype='float64', order='F')
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
        sd = np.sqrt((X ** 2).sum(0) / (N - 1))
        ok = sd > 0
        X[:, ok] /= sd[ok]; X[:, ~ok] = 0.0
        Z = (X.T @ Y) ** 2                                               # k x R
        cj, bj = cls[s:s + X.shape[1]], blk[s:s + X.shape[1]]
        for c in range(C):
            for b in np.unique(bj[cj == c]):
                out[b, c] += Z[(cj == c) & (bj == b)].sum(0) / Mc[c]
    P = len(pa)
    out -= (AS.reshape(-1, P).astype(np.float64) @ (Y[pa] * Y[pb])).reshape(nblocks, C, R)
    return out


def jk_direct(Gram, D, rhs_b, wts=None):
    """Direct estimates and their delete-one-SNP-block jackknife SEs. Gram: exact tr(K_c^u K_d^u) of the analysed
    sample; D[b]: block contributions with every sibling a member, used as fractions; rhs_b: block numerators."""
    frac = D / D.sum(0, keepdims=True)
    rhs = rhs_b.sum(0)
    nb = D.shape[0]
    if wts is not None:
        w = np.asarray(wts)
        g1 = w @ Gram @ w
        g1b = np.array([w @ (Gram * (1 - frac[b])) @ w for b in range(nb)])
        r1 = (w[:, None] * rhs).sum(0)
        r1b = r1[None, :] - (w[None, :, None] * rhs_b).sum(1)
        full = r1 / g1
        dele = r1b / g1b[:, None]
        return full[None, :], dele[:, None, :]
    full = np.linalg.solve(Gram, rhs)
    dele = np.array([np.linalg.solve(Gram * (1 - frac[b]), rhs - rhs_b[b]) for b in range(nb)])
    return full, dele


def cmd_jkfit(a):
    """SNP-block jackknife SEs of the direct estimates of every trait fitted by `fit` (one genotype pass)."""
    from bed_reader import open_bed
    meta, order, As, Mc = load_grm(a.grm)
    lay = Layout(order[['IID', 'sib', 'ext']])
    bz = np.load(f'{a.grm}/blocks.npz')
    D, AS, pa, pb, blk, cls = bz['D'], bz['AS'], bz['pa'], bz['pb'], bz['blk'], bz['cls']
    wts = Mc / Mc.sum()
    ph = pd.read_csv(a.pheno, sep='\t', dtype={'FID': str, 'IID': str}).set_index('IID')
    cv = pd.read_csv(a.covpc, sep='\t', dtype={'FID': str, 'IID': str}).set_index('IID').reindex(ph.index)
    as_cols = a.as_cols.split(',')
    res = {t: json.load(open(f'{a.fit}/{t}.json')) for t in sorted(os.path.basename(f)[:-5] for f in
           __import__('glob').glob(f'{a.fit}/*.json'))}
    traits = list(res)
    Y = np.zeros((lay.N, len(traits)))
    for k, t in enumerate(traits):                       # the centered phenotypes of `fit`
        y = ph[t].to_numpy(float); ok = np.isfinite(y)
        W1 = np.column_stack([np.ones(ok.sum()), cv.loc[ok, as_cols].to_numpy(float)])
        y_as = pd.Series(y[ok] - W1 @ np.linalg.lstsq(W1, y[ok], rcond=None)[0], index=ph.index[ok]).reindex(lay.iid).to_numpy()
        obs = np.isfinite(ph[t].reindex(lay.iid).to_numpy(float))
        m, n_m = members(lay, obs)
        yc = np.where(m > 0, y_as, 0.0)
        mean = np.divide(np.add.reduceat(yc, lay.sib_start), n_m, out=np.zeros(len(n_m)), where=n_m > 0)
        Y[:, k] = (yc - mean[lay.sib]) * m
    fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
    perm = pd.Series(np.arange(len(fam)), index=fam)[lay.iid].to_numpy()
    bed = open_bed(a.bfile + '.bed', count_A1=True, num_threads=8)
    rb = block_numerators(lay, bed, perm, cls, blk, Mc, AS, pa, pb, Y, D.shape[0])
    rows = []
    for k, t in enumerate(traits):
        r = res[t]
        G, rhs = np.array(r['gram_dir']), np.array(r['rhs_dir'])
        err = np.abs(rb[:, :, k].sum(0) - rhs).max() / np.abs(rhs).max()
        assert err < 1e-5, (t, err)                       # block numerators add up to the exact numerators
        for model, w in (('single', wts), ('8bin', None)):
            f, d = jk_direct(G, D, rb[:, :, [k]], wts=w)
            s, sd = f.sum(0)[0], d.sum(1)[:, 0]
            r[f'dir_{model}_jk'] = float(s) / r['var_full']
            r[f'dir_{model}_se_snpjk'] = float(jk_se(np.array(s), sd)) / r['var_full']
            r[f'dir_{model}_se_famjk'] = r.pop(f'dir_{model}_se', r.get(f'dir_{model}_se_famjk', np.nan))
        r['block_check'] = float(err)
        json.dump(r, open(f'{a.fit}/{t}.json', 'w'), indent=1)
        log(f"{t}: direct {r['dir_8bin']:.3f}, SE {r['dir_8bin_se_snpjk']:.3f} (SNP blocks) vs {r['dir_8bin_se_famjk']:.3f} (families)")


def cmd_simjk(a):
    """SNP-block jackknife SEs of the simulated direct estimates (dir_ext), single class."""
    from bed_reader import open_bed
    meta, order, As, Mc = load_grm(a.grm)
    lay = Layout(order[['IID', 'sib', 'ext']])
    bz = np.load(f'{a.grm}/blocks.npz')
    D, AS, pa, pb, blk, cls = bz['D'], bz['AS'], bz['pa'], bz['pb'], bz['blk'], bz['cls']
    wts = Mc / Mc.sum()
    Yd = np.load(f'{a.sim}/Yd.npy')
    ex = np.load(f'{a.sim}/exact_dir.npz')
    fam = pd.read_csv(a.bfile + '.fam', sep=r'\s+', header=None, usecols=[1], dtype=str)[1].to_numpy()
    perm = pd.Series(np.arange(len(fam)), index=fam)[lay.iid].to_numpy()
    bed = open_bed(a.bfile + '.bed', count_A1=True, num_threads=8)
    rb = block_numerators(lay, bed, perm, cls, blk, Mc, AS, pa, pb, Yd, D.shape[0])
    r1 = (wts[None, :, None] * rb).sum(1)                 # single class: blocks x R
    err = np.abs(r1.sum(0) - ex['rhs']).max() / np.abs(ex['rhs']).max()
    g1 = float(ex['gram'])
    assert abs(wts @ D.sum(0) @ wts - g1) / g1 < 1e-4, (wts @ D.sum(0) @ wts, g1)
    print('block numerators vs exact: max relative difference', err, flush=True)
    frac = np.array([wts @ D[b] @ wts for b in range(D.shape[0])]) / (wts @ D.sum(0) @ wts)
    full = r1.sum(0) / g1
    dele = (r1.sum(0)[None, :] - r1) / (g1 * (1 - frac))[:, None]
    se = jk_se(full, dele)
    res = pd.read_csv(f'{a.sim}/sim_results.tsv', sep='\t')
    names = list(SIM_SETTINGS)
    R = Yd.shape[1] // len(names)
    x = res[res.estimator == 'dir_ext'].copy()
    key = {(n, r): k * R + r for k, n in enumerate(names) for r in range(R)}
    idx = [key[(s_, r_)] for s_, r_ in zip(x.setting, x.rep)]
    assert np.allclose(x.est.to_numpy(), full[idx], rtol=1e-4)   # sim_results.tsv keeps six significant digits
    x['se_snpjk'] = se[idx]
    T = 1.9720
    x['cover_snpjk'] = (x.est - x.truth).abs() <= T * x.se_snpjk
    x['cover_famjk'] = (x.est - x.truth).abs() <= T * x.se
    s = x.groupby('setting').agg(bias=('est', lambda e: e.mean()), sd=('est', 'std'), se_snpjk=('se_snpjk', 'mean'),
                                 se_famjk=('se', 'mean'), cover_snpjk=('cover_snpjk', 'mean'), cover_famjk=('cover_famjk', 'mean'))
    s['bias'] -= 0.4
    s.to_csv(f'{a.sim}/sim_snpjk_summary.tsv', sep='\t', float_format='%.5g')
    x.to_csv(f'{a.sim}/sim_results_dir_snpjk.tsv', sep='\t', index=False, float_format='%.6g')
    print(s.round(4).to_string())


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('selftest')
    g = sp.add_parser('grm')
    g.add_argument('--bfile', required=True); g.add_argument('--sibs', required=True); g.add_argument('--annot', required=True)
    g.add_argument('--out', required=True); g.add_argument('--chunk', type=int, default=4096); g.add_argument('--threads', type=int, default=8)
    f = sp.add_parser('fit')
    f.add_argument('--grm', required=True); f.add_argument('--pheno', required=True); f.add_argument('--covpc', required=True)
    f.add_argument('--traits', default=''); f.add_argument('--out', required=True)
    f.add_argument('--as-cols', default='age,sex,age_sex,age2,age2_sex')
    s = sp.add_parser('sim')
    s.add_argument('--grm', required=True); s.add_argument('--bfile', required=True); s.add_argument('--pairs', required=True)
    s.add_argument('--pcs20', required=True); s.add_argument('--covpc', required=True); s.add_argument('--out', required=True)
    s.add_argument('--nrep', type=int, default=200); s.add_argument('--rho', type=float, default=0.5)
    s.add_argument('--hidden-pc', default='PC15'); s.add_argument('--seed', type=int, default=20261027)
    b = sp.add_parser('blocks')
    b.add_argument('--grm', required=True); b.add_argument('--bfile', required=True); b.add_argument('--annot', required=True)
    b.add_argument('--nblocks', type=int, default=200); b.add_argument('--chunk', type=int, default=4096)
    j = sp.add_parser('jkfit')
    j.add_argument('--grm', required=True); j.add_argument('--bfile', required=True); j.add_argument('--pheno', required=True)
    j.add_argument('--covpc', required=True); j.add_argument('--fit', required=True)
    j.add_argument('--as-cols', default='age,sex,age_sex,age2,age2_sex')
    k = sp.add_parser('simjk')
    k.add_argument('--grm', required=True); k.add_argument('--bfile', required=True); k.add_argument('--sim', required=True)
    a = ap.parse_args()
    {'selftest': cmd_selftest, 'grm': cmd_grm, 'fit': cmd_fit, 'sim': cmd_sim, 'blocks': cmd_blocks, 'jkfit': cmd_jkfit,
     'simjk': cmd_simjk}[a.cmd](a)


if __name__ == '__main__':
    main()
