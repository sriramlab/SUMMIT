"""Rescaled jackknife for h2 (`summit --h2 ... --jk-traces PREFIX`).

The delete-one-block jackknife cannot see covariance between blocks. With the genotype-only traces of
`summit relatives traces` (relatives.traces), the variance of the method-of-moments estimate under the fitted model,
AV = w'Cw, and the expectation of the jackknife variance under the same model, E_model[V_jk], are both quadratic forms
in the unit covariance C = 2 [e^2 G1 + g e (G2 + G2') + g^2 G3]. The reported SE is

    SE = SE_jk * sqrt(AV / E_model[V_jk]),

so the jackknife keeps what it sees (local excess variance the model misses) and is scaled up by the model's share of
what it cannot see. The estimate's weights w and the replicates' weights are linear in the unit numerators q_u:
theta = 1'T^-1 sum_u q_u (T the moment matrix of the classes), and replicate r uses T_r = T - sum_{u in r} T_u with
the deleted units' numerators left out (one-sided deletion, as SUMMIT's replicates with frozen LD scores). E_model[V_jk]
follows SUMMIT's unequal-unit pseudovalue jackknife (utils._calc_jackknife_se_from_delete_sets, exact delete-1 case).
g is the fitted genetic variance in kernel units (the sum over classes for partitioned fits, whose signal is
approximated by the one-class kernel) and e the fitted residual variance; g < 0 is set to 0.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def load_jk_traces(path: str) -> dict:
    """Traces of `summit relatives traces` from PREFIX (or PREFIX.jktraces.npz)."""
    base = path[: -len(".jktraces.npz")] if str(path).endswith(".jktraces.npz") else str(path)
    with np.load(base + ".jktraces.npz", allow_pickle=False) as z:
        tr = {k: z[k] for k in z.files}
    tr["meta"] = json.loads(str(tr["meta"]))
    tr["classes"] = [str(c) for c in np.asarray(tr["classes"]).ravel()]
    tr["snps"] = pd.read_csv(base + ".jktraces.snps.tsv.gz", sep="\t", dtype={"SNP": str, "CHR": str})
    tr["path"] = base
    return tr


def _cell_map(tr: dict, axis_snps, unit_id: np.ndarray, nunit: int):
    """Fit unit of every trace cell (majority of its SNPs on the fit axis; -1 without any) and the mapping shares."""
    snps = tr["snps"]
    cell_of = pd.Series(snps["CELL"].to_numpy(), index=snps["SNP"].to_numpy())
    cells = cell_of.reindex(np.asarray(axis_snps).astype(str)).to_numpy()
    on = np.isfinite(cells)
    ncell = int(np.asarray(tr["cell_blk"]).size)
    counts = np.zeros((ncell, int(nunit)))
    np.add.at(counts, (cells[on].astype(np.int64), np.asarray(unit_id)[on]), 1.0)
    cmap = np.where(counts.sum(axis=1) > 0, counts.argmax(axis=1), -1)
    tot = counts.sum()
    info = {
        "fit_snps_in_other_unit": float((tot - counts.max(axis=1).sum()) / max(tot, 1.0)),
        "fit_snps_without_trace": float(1.0 - on.mean()) if on.size else 0.0,
        "trace_snps_not_in_fit": float(1.0 - tot / max(len(snps), 1)),
    }
    return cmap, info


def _merge(tr: dict, cmap: np.ndarray, nunit: int, bin_names: list[str], noself: bool):
    """Unit matrices of the fit's (unit, class) layout: G1, G2, G3 (NU x NU) and Tb (U x K x K)."""
    sfx = "n" if noself else ""
    if noself and f"G1{sfx}" not in tr:
        raise ValueError(f"{tr['path']}: these traces have no self-pair-free values; rerun `summit relatives traces` "
                         "with --inbreeding for fits with the --inbreeding values.")
    tcls = tr["classes"]
    K = len(bin_names)
    unit_cell, unit_cls = np.asarray(tr["unit_cell"]), np.asarray(tr["unit_cls"])
    vsum = np.asarray(tr["vsum"], dtype=np.float64)
    if K == 1:
        w = vsum / vsum.sum()                                     # one-class kernel = sum_k (vsum_k / vsum) K~_k
        tgt = cmap[unit_cell]
        wt = w[unit_cls]
        Tb = np.asarray(tr[f"Tb{sfx}"]) @ w
        Tb = Tb[:, None, None] if Tb.ndim == 1 else Tb
    else:
        if sorted(tcls) != sorted(bin_names):
            raise ValueError(f"{tr['path']}: trace classes {tcls[:4]} differ from the fit's bins {bin_names[:4]}; compute "
                             "the traces with the fit's annotation.")
        kmap = np.array([bin_names.index(c) for c in tcls])
        tgt = np.where(cmap[unit_cell] >= 0, cmap[unit_cell] * K + kmap[unit_cls], -1)
        wt = np.ones(unit_cell.size)
        Tb = np.asarray(tr[f"Tb{sfx}"])[:, [tcls.index(b) for b in bin_names]]       # columns in the fit's class order
    NUf = int(nunit) * K
    S = np.zeros((NUf, unit_cell.size))
    use = tgt >= 0
    S[tgt[use], np.flatnonzero(use)] = wt[use]
    G = [S @ np.asarray(tr[f"{g}{sfx}"]) @ S.T for g in ("G1", "G2", "G3")]
    Tm = (S @ Tb.reshape(unit_cell.size, -1)).reshape(int(nunit), K, K)
    return G, Tm


def jackknife_rescaling(tr: dict, *, axis_snps, unit_id, D, unit_sizes, bin_names, sigma, sigma_e, noself: bool,
                        log=None, label: str = "") -> dict | None:
    """AV, E_model[V_jk] and their ratio for one fit (None for jackknife designs other than exact delete-1)."""
    D = np.asarray(D, dtype=np.float64)
    R, U = D.shape
    mask = D > 0.5
    if not (R == U and np.all(mask.sum(axis=1) == 1) and np.all(mask.sum(axis=0) == 1)):
        if log is not None:
            log._log(f"[jk-traces] {label}: only delete-1 jackknife designs are rescaled; SE left unchanged.")
        return None
    cmap, info = _cell_map(tr, axis_snps, np.asarray(unit_id, dtype=np.int64), U)
    (G1, G2, G3), Tm = _merge(tr, cmap, U, list(bin_names), noself)
    K = len(bin_names)
    g = max(float(np.nansum(sigma)), 0.0)
    e = float(sigma_e)
    C = 2.0 * (e * e * G1 + g * e * (G2 + G2.T) + g * g * G3)
    T = Tm.sum(axis=0)
    one = np.ones(K)
    omega = np.tile(np.linalg.solve(T.T, one), U)
    unit_of_rep = mask.argmax(axis=1)
    Wr = np.zeros((U, U * K))                                     # replicate deleting unit u, by unit
    for r in range(R):
        u = unit_of_rep[r]
        w = np.tile(np.linalg.solve((T - Tm[u]).T, one), U)
        w[u * K:(u + 1) * K] = 0.0
        Wr[u] = w
    av = float(omega @ C @ omega)
    sizes = np.asarray(unit_sizes, dtype=np.float64)
    Mtot = float(sizes.sum())
    valid = (sizes > 0) & (sizes < Mtot)
    wu = sizes / Mtot
    L = (omega[None, :] - (1.0 - wu)[:, None] * Wr)[valid] / wu[valid][:, None]   # pseudovalues = L q
    ww = wu[valid] / wu[valid].sum()
    Z = L - ww @ L
    c = ww * ww / (1.0 - ww)
    ejk = float(np.einsum("u,ui,ij,uj->", c, Z, C, Z))
    out = dict(info, av=av, ejk=ejk, factor=av / ejk if ejk > 0 else np.nan, g=g, e=e, noself=bool(noself))
    if log is not None:
        log._log(f"[jk-traces] {label}: model variance {av:.4g}, expected jackknife variance {ejk:.4g}, "
                 f"factor {out['factor']:.4g} ({'self-pairs out' if noself else 'self-pairs in'}; "
                 f"{100 * info['fit_snps_in_other_unit']:.2f}% of the fit's SNPs in a trace cell mapped to another unit, "
                 f"{100 * info['fit_snps_without_trace']:.2f}% without traces, "
                 f"{100 * info['trace_snps_not_in_fit']:.2f}% of trace SNPs not in the fit).")
    return out
