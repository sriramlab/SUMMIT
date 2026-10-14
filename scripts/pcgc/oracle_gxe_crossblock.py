"""Dense Monte Carlo oracle for the person-level cross-block covariance.

Small fixed-genotype design (population sampling, rare disease, Q=2). For
each replicate it forms participant-by-participant block kernels and checks
three claims of the derivation independently of the streamed code:

1. sum over blocks F != G of Cov(U_F, U_G), estimated empirically across
   replicates from the block estimating equations U_F(theta_true), is
   matched by the mean of the person-level (Hajek) cross-block estimate.
2. The separable context-by-genotype approximation to the coefficient-side
   person rows agrees with exact dense rows for that between-block term.
3. E[V_JK] = sum_F Var(U_F) - C_between/(B-1) in equation units.

Results are written to a new JSON file; nothing is overwritten.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr


def build(n, m, blocks, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m))
    e = np.linspace(-1, 1, 5)
    level = rng.integers(0, 5, n)
    phi = np.column_stack((np.ones(n), e[level]))
    ids = np.arange(m)*blocks//m
    G = np.stack([x[:, ids == F] @ x[:, ids == F].T/m for F in range(blocks)])
    for g in G:
        np.fill_diagonal(g, 0.)
    return x, phi, level, ids, G, G.sum(0)


STATE = {}


def _init(n, m, blocks, seed):
    from threadpoolctl import threadpool_limits
    STATE["limits"] = threadpool_limits(1)


def replicate(args):
    from summit.sumstats.binary import prepare_binary_risk
    from summit.pcgc.crossblock import _center, context_pair_sums, genotype_factors
    index, spec = args
    x, phi, level, ids, G, Gtotal = STATE["design"]
    n, m = x.shape
    B = len(G)
    rng = np.random.default_rng([spec["seed"], 31, index])
    om = np.array([[spec["h"], 0.], [0., spec["h"]/3]])
    beta = rng.normal(size=(m, 2))*np.sqrt(np.diag(om)/m)
    v = np.einsum('iq,qr,ir->i', phi, beta.T @ beta, phi)
    mu = .3*phi[:, 1]
    from scipy.optimize import brentq
    cut = brentq(lambda t: np.mean(ndtr(mu-t))-spec["K"], -15, 15)
    K = float(np.mean(ndtr(mu-cut)))
    k = ndtr(mu-cut)
    g = np.sum((x @ beta)*phi, axis=1)
    y = (g+rng.normal(size=n)*np.sqrt(1-v) > cut-mu).astype(float)
    if y.sum() < 8:
        return None
    risk = prepare_binary_risk(y, K, population_risk=k)
    w = risk.sensitivity
    psi = phi*w[:, None]
    r = risk.z
    h = [np.outer(psi[:, 0], psi[:, 0]), np.outer(psi[:, 1], psi[:, 1]),
         np.outer(psi[:, 0], psi[:, 1])+np.outer(psi[:, 1], psi[:, 0])]
    Kfull = [hp*Gtotal for hp in h]                  # diagonal zero
    H = np.array([[np.sum(a*b) for b in Kfull] for a in Kfull])
    rr = np.outer(r, r)
    b = np.array([np.sum(a*rr) for a in Kfull])
    theta_hat = np.linalg.solve(H, b)
    theta = np.array([om[0, 0], om[1, 1], om[0, 1]])
    HF = np.empty((B, 3, 3)); bF = np.empty((B, 3))
    alpha = np.empty((B, n, 3)); lam = np.empty((B, n, 3, 3))
    for F in range(B):
        KF = [hp*G[F] for hp in h]
        HF[F] = [[np.sum(a*bb) for bb in Kfull] for a in KF]
        bF[F] = [np.sum(a*rr) for a in KF]
        alpha[F] = np.stack([r*(KF[c] @ r) for c in range(3)], 1)
        lam[F] = np.stack([np.stack([np.sum(KF[c]*Kfull[d], 1) for d in range(3)], 1) for c in range(3)], 1)
        del KF
    U_true = bF-HF @ theta
    # Jackknife (target-row deletion, full source), as in fit_gxe.
    reps = np.array([np.linalg.solve(H-HF[F], b-bF[F]) for F in range(B)])
    centered = reps-reps.mean(0)
    vjk = (B-1)/B*centered.T @ centered
    cases = r > 0
    S = context_pair_sums(psi)
    pc = np.arange(3)
    out = dict(theta_hat=theta_hat, U_true=U_true, vjk=vjk, H=H)
    for label, th in (("hat", theta_hat), ("true", theta)):
        rows_exact, rows_sep = [], []
        for F in range(B):
            lam_sep = genotype_factors(HF[F], S.sum(0))[0, 0]*S[:, pc][:, :, pc]
            rows_exact.append(_center(alpha[F]-lam[F] @ th, cases))
            rows_sep.append(_center(alpha[F]-lam_sep @ th, cases))
        for name, rows in (("exact", rows_exact), ("separable", rows_sep)):
            tot = sum(rows)
            out[f"between_{name}_{label}"] = 4*(tot.T @ tot-sum(v.T @ v for v in rows))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=2400)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--K", type=float, default=.05)
    parser.add_argument("--h", type=float, default=.36)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91013)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    import importlib
    import multiprocessing as mp
    module = importlib.import_module("scripts.pcgc.oracle_gxe_crossblock")
    module.STATE["design"] = build(args.n, args.m, args.blocks, args.seed)
    spec = dict(K=args.K, h=args.h, seed=args.seed)
    started = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers, initializer=module._init,
                                     initargs=(args.n, args.m, args.blocks, args.seed)) as pool:
        rows = [r for r in pool.imap_unordered(module.replicate, [(i, spec) for i in range(args.replicates)], chunksize=2)
                if r is not None]
    B = args.blocks
    U = np.array([r["U_true"] for r in rows])                  # R x B x C
    total = U.sum(1)
    var_total = np.cov(total.T)
    var_within = sum(np.cov(U[:, F].T) for F in range(B))
    empirical_between = var_total-var_within
    H = np.mean([r["H"] for r in rows], 0)
    Hinv = np.linalg.inv(H)
    th = np.array([r["theta_hat"] for r in rows])
    result = dict(args=vars(args), replicates=len(rows), seconds=time.perf_counter()-started,
        empirical_between_diag=np.diag(empirical_between).tolist(),
        empirical_within_diag=np.diag(var_within).tolist(),
        empirical_theta_var=th.var(0, ddof=1).tolist(),
        mean_jk_var=np.mean([np.diag(r["vjk"]) for r in rows], 0).tolist(),
        jk_expectation_prediction=np.diag(Hinv @ (var_within-empirical_between/(B-1)) @ Hinv.T).tolist())
    # Monte Carlo SE of the empirical between-block diagonal by batch means.
    batches = np.array_split(np.arange(len(rows)), 20)
    eb = []
    for idx in batches:
        vt = np.cov(total[idx].T)
        vw = sum(np.cov(U[idx, F].T) for F in range(B))
        eb.append(np.diag(vt-vw))
    result["empirical_between_diag_mcse"] = (np.std(eb, 0, ddof=1)/np.sqrt(len(eb))).tolist()
    for key in ("between_exact_true", "between_separable_true", "between_exact_hat", "between_separable_hat"):
        values = np.array([np.diag(r[key]) for r in rows])
        result[key+"_mean_diag"] = values.mean(0).tolist()
        result[key+"_mcse"] = (values.std(0, ddof=1)/np.sqrt(len(values))).tolist()
    for key in ("between_exact_hat", "between_separable_hat"):
        add = np.array([np.diag(Hinv @ r[key] @ Hinv.T)*B/(B-1) for r in rows])
        result["variance_ratio_jk_plus_"+key] = ((np.mean([np.diag(r["vjk"]) for r in rows], 0)+add.mean(0))/th.var(0, ddof=1)).tolist()
    result["variance_ratio_jk"] = (np.mean([np.diag(r["vjk"]) for r in rows], 0)/th.var(0, ddof=1)).tolist()
    with open(out, "x") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
