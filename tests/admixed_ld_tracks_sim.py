"""Simulated admixed cohort with known local ancestry for the ancestry-track tests (`summit admixed-ld compose --tracks`,
`tractor-sumstats`, `track-rg`).

The model is the founder-copying model of the admixed-ld example with two changes that make the tracks' LD informative
(as in the design prototype, notes/ANCESTRY_TRACK_RG_DESIGN_20261116.md section 6.1): the founders of both sources are
mosaics of one small set of ancestral haplotypes (so the sources share LD structure but differ in it), and haplotypes
copy their source's founders at a source-specific rate (shorter LD in A). An admixed haplotype breaks at T/100 per cM
since admixture and each segment draws its ancestry. The cohort keeps its haplotypes' local ancestry, so the
ancestry-specific allele counts that Tractor regresses on are exact.

`build_fixture` runs in a child process (tests/admixed_ld_sim.in_child): it writes the source panels (PLINK), the
Tractor tables of a set of traits (OLS and sandwich SEs), the GWAS of a disjoint European sample, the annotation and
the exact in-sample track LD scores, and returns only the small truths, so the test process's peak memory stays low.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _mosaic(rng, src, cm, n_out, rate):
    F, M = src.shape
    out = np.empty((n_out, M), dtype=np.int8)
    f = rng.integers(0, F, n_out)
    for j in range(M):
        if j > 0:
            dcm = cm[j] - cm[j - 1]
            sw = np.ones(n_out, bool) if dcm < 0 else rng.random(n_out) < 1 - np.exp(-rate * dcm)
            f = np.where(sw, rng.integers(0, F, n_out), f)
        out[:, j] = src[f, j]
    return out


def _admixed(rng, founders, cm, theta_h, rates, break_rate):
    n, M = theta_h.size, cm.size
    out = np.empty((n, M), dtype=np.int8)
    la = np.empty((n, M), dtype=np.int8)
    a = rng.random(n) < theta_h
    FA, FE = founders["A"].shape[0], founders["E"].shape[0]
    f = np.where(a, rng.integers(0, FA, n), rng.integers(0, FE, n))
    for j in range(M):
        if j > 0:
            dcm = cm[j] - cm[j - 1]
            if dcm < 0:
                brk = sw = np.ones(n, bool)
            else:
                brk = rng.random(n) < 1 - np.exp(-break_rate * dcm)
                sw = rng.random(n) < 1 - np.exp(-np.where(a, rates["A"], rates["E"]) * dcm)
            a = np.where(brk, rng.random(n) < theta_h, a)
            f = np.where(sw | brk, np.where(a, rng.integers(0, FA, n), rng.integers(0, FE, n)), f)
        out[:, j] = np.where(a, founders["A"][np.minimum(f, FA - 1), j], founders["E"][np.minimum(f, FE - 1), j])
        la[:, j] = a
    return out, la


def _orth(C):
    return np.linalg.qr(np.column_stack([np.ones(C.shape[0]), C]))[0]


def tractor(Y, gA, gE, lA, C):
    """Tractor's per-SNP OLS of each trait on [1, C, l^A_j, g^A_j, g^E_j], with OLS and HC1 (sandwich) SEs."""
    N = Y.shape[0]
    Q = _orth(C)
    res = lambda X: X - Q @ (Q.T @ X)
    Yr = res(np.asarray(Y, np.float64))
    xs = [res(np.asarray(x, np.float64)) for x in (lA, gA, gE)]
    J, R = xs[0].shape[1], Yr.shape[1]
    S = np.empty((J, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            S[:, a, b] = S[:, b, a] = np.einsum("ij,ij->j", xs[a], xs[b])
    Si = np.linalg.inv(S)
    XtY = np.stack([x.T @ Yr for x in xs], axis=1)
    B = np.einsum("jab,jbr->jar", Si, XtY)
    rss = np.einsum("ir,ir->r", Yr, Yr)[None, :] - np.einsum("jar,jar->jr", B, XtY)
    df = N - Q.shape[1] - 3
    s2 = rss / df
    Y2 = Yr * Yr
    meat = np.zeros((J, 3, 3, R))
    m3 = {}
    for a in range(3):
        for b in range(a, 3):
            xab = xs[a] * xs[b]
            t = xab.T @ Y2
            for c in range(3):
                key = tuple(sorted((a, b, c)))
                if key not in m3:
                    m3[key] = (xs[key[0]] * xs[key[1]] * xs[key[2]]).T @ Yr
                t = t - 2.0 * B[:, c, :] * m3[key]
            for c in range(3):
                for d in range(3):
                    t = t + B[:, c, :] * B[:, d, :] * np.einsum("ij,ij->j", xab, xs[c] * xs[d])[:, None]
            meat[:, a, b, :] = meat[:, b, a, :] = t
    Vhc = np.einsum("jab,jbcr,jcd->jadr", Si, meat, Si) * (N / df)
    out = dict(df=df, N=N, cov_rank=Q.shape[1] - 1 + 2)
    for s, a in (("A", 1), ("E", 2)):
        out[f"beta_{s}"] = B[:, a, :]
        out[f"se_{s}"] = np.sqrt(s2 * Si[:, a, a][:, None])
        out[f"sehc_{s}"] = np.sqrt(np.maximum(Vhc[:, a, a, :], 1e-300))
    return out


def _gwas(G, y):
    """Marginal OLS of each column of y on each SNP of G with an intercept: BETA, SE."""
    n = G.shape[0]
    X = G - G.mean(0)
    Yc = y - y.mean(0)
    sxx = np.einsum("ij,ij->j", X, X)
    beta = (X.T @ Yc) / sxx[:, None]
    rss = np.einsum("ir,ir->r", Yc, Yc)[None, :] - beta ** 2 * sxx[:, None]
    se = np.sqrt(rss / (n - 2) / sxx[:, None])
    return beta, se


def exact_track_scores(gA, gE, lA, bp, rows, partners, w, window_bp=3_000_000):
    """The cohort's own track LD scores at the rows against the partners, from its haplotypes and true local ancestry:
    u^s_k = g^s_k - p^s_k l^s_k (p^s the track's frequency), q^s_k = mean of (u^s_k)^2, V^s_k = q^s_k w_k, and over
    pairs within the window U-statistics (unbiased in the cohort's own people) of the squared and cross covariances:
        own  V^s_j + sum_{k != j} U[C^s_jk^2] / (q^s_j q^s_k) V^s_k,
        cross sqrt(V^A_j V^E_j) + sum_{k != j} U[C^A_jk C^E_jk] / sqrt(q^A_j q^E_j q^A_k q^E_k) sqrt(V^A_k V^E_k).
    Rows and partners are indices of one chromosome's variants (rows a subset of partners)."""
    N = gA.shape[0]
    lE = 2.0 - lA
    U = {}
    for s, g, l in (("A", gA, lA), ("E", gE, lE)):
        g = g[:, partners].astype(np.float64)
        l = l[:, partners].astype(np.float64)
        p = g.sum(0) / np.maximum(l.sum(0), 1.0)
        U[s] = g - p[None, :] * l
    q = {s: (U[s] ** 2).mean(0) for s in U}
    V = {s: q[s] * w[partners] for s in U}
    pos = {k: i for i, k in enumerate(partners)}
    ri = np.array([pos[r] for r in rows])
    win = np.abs(bp[rows][:, None] - bp[partners][None, :]) <= window_bp
    win[np.arange(ri.size), ri] = False
    nn = N * (N - 1.0)
    out = {}
    for s in U:
        S = U[s][:, ri].T @ U[s]
        P = (U[s][:, ri] ** 2).T @ (U[s] ** 2)
        sq = (S * S - P) / nn
        out[s] = V[s][ri] + np.where(win, sq / (q[s][ri][:, None] * q[s][None, :]), 0.0) @ V[s]
    SA, SE = U["A"][:, ri].T @ U["A"], U["E"][:, ri].T @ U["E"]
    P = (U["A"][:, ri] * U["E"][:, ri]).T @ (U["A"] * U["E"])
    cx = (SA * SE - P) / nn
    den = np.sqrt(q["A"][ri] * q["E"][ri])[:, None] * np.sqrt(q["A"] * q["E"])[None, :]
    out["AE"] = np.sqrt(V["A"][ri] * V["E"][ri]) + np.where(win, cx / den, 0.0) @ np.sqrt(V["A"] * V["E"])
    out["V"] = {s: V[s][ri] for s in V}
    return out


def build_fixture(d, *, seed=3, n_chr=2, chr_cm=20.0, snps_per_cm=40, n_cohort=2500, n_panel=500, n_eur=3000,
                  mean_theta=0.6, sd_theta=0.25, generations=8.0, rows_share=1 / 3, maf=0.05, traits=None):
    """Write the inputs of the track tests to directory d and return the truths.

    traits: {name: dict(causal='rows'|'all', rho, sA, sE, h2, eta, reps, classes=(v_low, v_high), r_ex)}: R traits of
    each scenario with per-allele effects (Var = tau_s,c w_k, w = 1 / Var(g), class variance factors by MAF class),
    correlation rho across tracks, residual variance multiplied by exp(eta z_theta) (eta > 0: variance that follows
    ancestry), and, when r_ex is given, the European GWAS of the same traits with effects correlated r_ex with the E
    track's. Files: panels pA, pE (BED, all eligible variants), annot.tsv (two MAF classes, all eligible variants),
    tractor/<scenario>.<r>.tsv (rows only), eur/<scenario>.<r>.sumstats, exact.npz, truth.json."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from admixed_ld_sim import write_bed
    rng = np.random.default_rng(seed)
    per = int(round(chr_cm * snps_per_cm))
    chrom = np.repeat(np.arange(1, n_chr + 1), per)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, per)) for _ in range(n_chr)])
    M = cm.size
    anc = (rng.random((8, M)) < rng.uniform(0.15, 0.85, M)[None, :]).astype(np.int8)
    founder_rate, copy_rate = dict(A=1.2, E=0.8), dict(A=1.2, E=0.6)
    founders = {s: _mosaic(rng, anc ^ (rng.random(anc.shape) < 0.06).astype(np.int8), cm, 50, founder_rate[s])
                for s in ("A", "E")}
    panel = lambda s, n: (lambda h: (h[0::2] + h[1::2]).astype(np.int8))(_mosaic(rng, founders[s], cm, 2 * n,
                                                                                  copy_rate[s]))
    PA, PE, GX = panel("A", n_panel), panel("E", n_panel), panel("E", n_eur)
    k = mean_theta * (1 - mean_theta) / sd_theta ** 2 - 1
    theta = rng.beta(mean_theta * k, (1 - mean_theta) * k, n_cohort)
    th_h = np.clip(np.repeat(theta, 2) + rng.normal(0, 0.05, 2 * n_cohort), 0.0, 1.0)
    hap, la = _admixed(rng, founders, cm, th_h, copy_rate, generations / 100.0)
    a = la.astype(np.int8)
    gA = (hap[0::2] * a[0::2] + hap[1::2] * a[1::2]).astype(np.int8)
    gE = (hap[0::2] * (1 - a[0::2]) + hap[1::2] * (1 - a[1::2])).astype(np.int8)
    lA = (a[0::2] + a[1::2]).astype(np.int8)
    del hap, la, a
    mafv = lambda p: np.minimum(p, 1 - p)
    afA = gA.sum(0) / np.maximum(lA.sum(0), 1)
    afE = gE.sum(0) / np.maximum((2 - lA).sum(0), 1)
    elig = np.flatnonzero((mafv(PA.mean(0) / 2) >= maf) & (mafv(PE.mean(0) / 2) >= maf) & (mafv(afA) >= maf)
                          & (mafv(afE) >= maf) & (mafv(GX.mean(0) / 2) >= maf))
    rows = np.sort(rng.choice(elig, int(round(elig.size * rows_share)), replace=False))
    bp = np.rint(cm * 1e6).astype(np.int64) + 1
    v = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"t{j:05d}" for j in range(M)], "CM": cm, "BP": bp,
                      "A1": np.where(rng.random(M) < 0.5, "A", "C"), "A2": np.where(rng.random(M) < 0.5, "G", "T")})
    ve = v.iloc[elig].reset_index(drop=True)
    write_bed(os.path.join(d, "pA"), PA[:, elig], ve, "a")
    write_bed(os.path.join(d, "pE"), PE[:, elig], ve, "e")
    write_bed(os.path.join(d, "pX"), GX[:, elig][: n_panel], ve, "x")   # a European panel of other people
    gtot = (gA + gE).astype(np.float64)
    s2 = gtot.var(0, ddof=1)
    w = 1.0 / np.maximum(s2, 1e-12)
    pbar = gtot.mean(0) / 2
    low = mafv(pbar) < 0.2
    meta = {"CHR": chrom[elig], "BP": bp[elig], "SNP": v.SNP.to_numpy()[elig]}
    pd.DataFrame({**meta, "low": low[elig].astype(int), "high": (~low[elig]).astype(int)}).to_csv(
        os.path.join(d, "annot.tsv"), sep="\t", index=False)
    pd.DataFrame({**meta, "all": 1}).to_csv(os.path.join(d, "annot1.tsv"), sep="\t", index=False)
    # exact in-sample track scores, per chromosome, with partners = rows and = all eligible variants
    ex = {}
    for pname, P in (("rows", rows), ("all", elig)):
        parts = {key: np.zeros(rows.size) for key in ("A", "E", "AE", "VA", "VE")}
        for c in np.unique(chrom):
            rc = rows[chrom[rows] == c]
            pc = P[chrom[P] == c]
            e = exact_track_scores(gA, gE, lA, bp, rc, pc, w)
            ii = np.searchsorted(rows, rc)
            for key in ("A", "E", "AE"):
                parts[key][ii] = e[key]
            parts["VA"][ii], parts["VE"][ii] = e["V"]["A"], e["V"]["E"]
        for key, val in parts.items():
            ex[f"{pname}_{key}"] = val
    np.savez(os.path.join(d, "exact.npz"), snps=v.SNP.to_numpy()[rows].astype(str), **ex)
    truth = dict(n=int(n_cohort), n_rows=int(rows.size), n_eligible=int(elig.size), m_A=float(lA.mean() / 2),
                 theta_sd=float(theta.std()), generations=float(generations), scenarios={})
    os.makedirs(os.path.join(d, "tractor"), exist_ok=True)
    os.makedirs(os.path.join(d, "eur"), exist_ok=True)
    C = (lA.mean(1) / 2.0)[:, None]                                  # global ancestry from the calls
    zt = (C[:, 0] - C[:, 0].mean()) / C[:, 0].std()
    for name, sc in (traits or {}).items():
        causal = rows if sc.get("causal", "rows") == "rows" else elig
        R, h2, rho, sA, sE = sc["reps"], sc.get("h2", 0.6), sc["rho"], sc.get("sA", 1.0), sc.get("sE", 1.0)
        cf = np.where(low[causal], sc.get("classes", (1.0, 1.0))[0], sc.get("classes", (1.0, 1.0))[1])
        z0, z1, z2 = (rng.standard_normal((causal.size, R)) for _ in range(3))
        zA = z0
        zE = rho * z0 + np.sqrt(max(1 - rho * rho, 0.0)) * z1
        sd = np.sqrt(w[causal] * cf)[:, None]
        bA, bE = sA * sd * zA, sE * sd * zE
        g = gA[:, causal].astype(np.float64) @ bA + gE[:, causal].astype(np.float64) @ bE
        g -= g.mean(0)
        scale = np.sqrt(h2 / g.var(0))
        bA, bE, g = bA * scale, bE * scale, g * scale
        eps = rng.standard_normal(g.shape)
        if sc.get("eta"):
            sdv = np.exp(0.5 * sc["eta"] * zt)
            eps *= (sdv / np.sqrt(np.mean(sdv ** 2)))[:, None]
        Y = g + np.sqrt(1 - h2) * eps
        TR = tractor(Y, gA[:, rows], gE[:, rows], lA[:, rows], C)
        afr, afe = afA[rows], afE[rows]
        lap = lA[:, rows].mean(0) / 2
        for r in range(R):
            tab = pd.DataFrame({"CHR": chrom[rows], "POS": bp[rows], "ID": v.SNP.to_numpy()[rows],
                                "REF": v.A2.to_numpy()[rows], "ALT": v.A1.to_numpy()[rows], "N": n_cohort})
            for i, (s, af, lp) in enumerate((("A", afr, lap), ("E", afe, 1 - lap))):
                b, se, sh = TR[f"beta_{s}"][:, r], TR[f"se_{s}"][:, r], TR[f"sehc_{s}"][:, r]
                tab[f"AF_anc{i}"], tab[f"LAprop_anc{i}"] = af, lp
                tab[f"beta_anc{i}"], tab[f"se_anc{i}"] = b, np.round(se, 4)
                tab[f"pval_anc{i}"], tab[f"tval_anc{i}"] = 0.5, b / se
                tab[f"se_hc_anc{i}"] = sh
            tab.to_csv(os.path.join(d, "tractor", f"{name}.{r}.tsv"), sep="\t", index=False, float_format="%.10g")
        ent = dict(rho=rho, reps=R, causal=sc.get("causal", "rows"), n_causal=int(causal.size),
                   cov_rank=int(TR["cov_rank"]))
        if sc.get("r_ex") is not None:
            # the European GWAS: per-allele effects correlated r_ex with the E track's (so r with A is r_ex rho)
            bX = sd * (sc["r_ex"] * zE + np.sqrt(1 - sc["r_ex"] ** 2) * z2)
            GXc = GX[n_panel:, causal].astype(np.float64)               # the GWAS people: not the panel's
            gx = GXc @ bX
            gx = (gx - gx.mean(0)) * np.sqrt(h2 / gx.var(0))
            y = gx + np.sqrt(1 - h2) * rng.standard_normal(gx.shape)
            beta, se = _gwas(GX[n_panel:, rows].astype(np.float64), y)
            for r in range(R):
                pd.DataFrame({"SNP": v.SNP.to_numpy()[rows], "CHR": chrom[rows], "BP": bp[rows],
                              "A1": v.A1.to_numpy()[rows], "A2": v.A2.to_numpy()[rows], "N": n_eur - n_panel,
                              "BETA": beta[:, r], "SE": se[:, r], "COV_RANK": 0}).to_csv(
                    os.path.join(d, "eur", f"{name}.{r}.sumstats"), sep="\t", index=False, float_format="%.10g")
            ent.update(r_ex=sc["r_ex"], r_A_ex=sc["r_ex"] * rho)
        truth["scenarios"][name] = ent
    with open(os.path.join(d, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1)
    return truth


def build_all(d, traits):
    """build_fixture and the reference of its three panels (sources A, E and a European panel X of other people)."""
    from summit.admixed_ld.reference import build_reference
    truth = build_fixture(d, traits=traits)
    build_reference([f"{d}/pA", f"{d}/pE", f"{d}/pX"], ["A", "E", "X"], f"{d}/ref", window_kb=3000, step=256,
                    log=lambda *a: None)
    return truth
