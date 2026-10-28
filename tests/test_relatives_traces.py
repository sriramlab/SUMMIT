import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.inference.jktraces import _merge, jackknife_rescaling, load_jk_traces
from summit.inference.jackknife import _equal_count_block_boundaries
from summit.relatives.moments import read_relative_pairs
from summit.relatives.traces import compute_jackknife_traces
from test_relatives_moments import _annot, _simulate


def _exact_traces(tr, G, covs, pi, pj, Acls):
    """Exact G1, G2, G3, Tb (and the self-pair-free versions) with the definitions of relatives.traces, from dense
    matrices, on the units of the probe run `tr`."""
    G = np.asarray(G, dtype=np.float64)
    n, M = G.shape
    Xc = G - G.mean(0)
    Xs = Xc / np.sqrt((Xc ** 2).sum(0) / (n - 1))
    Q = np.linalg.qr(np.column_stack([np.ones(n), covs]))[0]
    P = np.eye(n) - Q @ Q.T
    H = Q @ Q.T
    X = P @ Xs
    nu = n - Q.shape[1]
    V = (X ** 2).sum(0) / nu
    cls = np.zeros(M, int) if Acls is None else Acls.argmax(1)
    K = int(cls.max()) + 1
    vsum = np.bincount(cls, weights=V, minlength=K)
    cell = tr["snps"]["CELL"].to_numpy()
    ukey = {(c, k): u for u, (c, k) in enumerate(zip(tr["unit_cell"], tr["unit_cls"]))}
    unit = np.array([ukey[(c, k)] for c, k in zip(cell, cls)])
    NU = len(ukey)
    Kall = X @ X.T / V.sum()
    Kd = [X[:, cls == d] @ X[:, cls == d].T / vsum[d] for d in range(K)]
    pair = np.zeros((n, n), bool)
    pair[pi, pj] = True; pair[pj, pi] = True
    out = {}
    for sfx in ("", "n"):
        A = []
        for u in range(NU):
            Ku = X[:, unit == u] @ X[:, unit == u].T / vsum[cls[unit == u][0]]
            B = np.where(pair, 0.0, Ku)
            if sfx:
                lam = np.diag(Ku) @ np.diag(H) + (Ku * H * pair).sum()
                B = B - np.diag(np.diag(Ku)) - lam / nu * np.eye(n)
            else:
                B = B - np.trace(Ku) / nu * np.eye(n)
            A.append(P @ B @ P)
        KA = [Kall @ a for a in A]
        out["G1" + sfx] = np.array([[np.sum(a * b) for b in A] for a in A])
        out["G2" + sfx] = np.array([[np.sum(ka.T * b) for b in A] for ka in KA])
        out["G3" + sfx] = np.array([[np.sum(ka.T * kb) for kb in KA] for ka in KA])      # tr(A_u K A_v K)
        out["Tb" + sfx] = np.array([[np.sum(a * kd) for kd in Kd] for a in A])
        out["A" + sfx] = A
    return out, Kall, P


def _setup(tmp_path, partitioned, pairs, nblocks=10):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=200, M=240, npairs=40 if pairs else 0, ncov=2, missing=0.0)
    if not pairs:
        pd.DataFrame({"IID1": [], "IID2": []}).to_csv(tmp_path / "pairs.txt", sep=" ", index=False)
    annot, Acls = _annot(tmp_path, G.shape[1]) if partitioned else (None, None)
    return prefix, G, y, covs, iid, cov_path, annot, Acls


def _fit_design(M, nblocks):
    starts, ends = _equal_count_block_boundaries(M, nblocks)
    unit_id = np.concatenate([np.full(e - s, b) for b, (s, e) in enumerate(zip(starts, ends))])
    return unit_id, np.eye(nblocks), (ends - starts).astype(float)


@pytest.mark.parametrize("partitioned,noself", [(False, False), (True, False), (False, True), (True, True)])
def test_exact_traces_give_exact_model_and_jackknife_variances(tmp_path, partitioned, noself):
    """With exact traces, AV equals the Gaussian variance 2 tr(B S B S) of the estimate and E_model[V_jk] the expectation
    of SUMMIT's pseudovalue jackknife variance computed directly from quadratic-form moments (1e-8)."""
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, partitioned, pairs=False)
    compute_jackknife_traces(prefix, str(tmp_path / "tr"), covar=cov_path, annot=annot, inbreeding=True, nblocks=10,
                             probes=2, seed=1)
    tr = load_jk_traces(str(tmp_path / "tr"))
    ex, Kall, P = _exact_traces(tr, G, covs, np.zeros(0, int), np.zeros(0, int), Acls)
    tr.update({k: v for k, v in ex.items() if not k.startswith("A")})
    M = G.shape[1]
    unit_id, D, sizes = _fit_design(M, 10)
    names = ["lowmaf", "highmaf"] if partitioned else ["L2_0"]
    sfx = "n" if noself else ""
    g, e = 0.3, 0.7
    sig = g * tr["vsum"] / tr["vsum"].sum() if partitioned else np.array([g])
    out = jackknife_rescaling(tr, axis_snps=tr["snps"]["SNP"], unit_id=unit_id, D=D, unit_sizes=sizes,
                              bin_names=names, sigma=sig, sigma_e=e, noself=noself)
    # brute force on the fit's (unit, class) matrices
    Kf = len(names)
    w = tr["vsum"] / tr["vsum"].sum()
    cellunit = np.array([unit_id[tr["snps"]["CELL"].to_numpy() == c][0] for c in range(tr["cell_blk"].size)])
    Af = [np.zeros_like(P) for _ in range(10 * Kf)]
    for t, (c, k) in enumerate(zip(tr["unit_cell"], tr["unit_cls"])):
        Af[cellunit[c] * Kf + (k if Kf > 1 else 0)] += ex["A" + sfx][t] * (1.0 if Kf > 1 else w[k])
    _, Tm = _merge(tr, cellunit, 10, names, noself)
    T = Tm.sum(0)
    omega = np.tile(np.linalg.solve(T.T, np.ones(Kf)), 10)
    S = g * Kall + e * P

    def var(Bw):
        B = sum(c_ * a for c_, a in zip(Bw, Af))
        BS = B @ S
        return 2.0 * np.sum(BS * BS.T), np.trace(BS)

    assert out["av"] == pytest.approx(var(omega)[0], rel=1e-8)
    Wr = np.zeros((10, 10 * Kf))
    for u in range(10):
        wr = np.tile(np.linalg.solve((T - Tm[u]).T, np.ones(Kf)), 10); wr[u * Kf:(u + 1) * Kf] = 0.0
        Wr[u] = wr
    wu = sizes / sizes.sum()
    L = (omega[None, :] - (1 - wu)[:, None] * Wr) / wu[:, None]
    Z = L - wu @ L
    c = wu * wu / (1 - wu)
    ev = 0.0
    for u in range(10):
        v, m = var(Z[u])
        ev += c[u] * (v + m * m)
    assert out["ejk"] == pytest.approx(ev, rel=1e-8)
    assert out["factor"] == pytest.approx(var(omega)[0] / ev, rel=1e-8)


@pytest.mark.parametrize("partitioned", [False, True])
def test_probe_traces_match_exact(tmp_path, partitioned):
    """Probe estimates of the traces give AV and E_model[V_jk] within 2% of the exact traces (with related pairs)."""
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, partitioned, pairs=True)
    compute_jackknife_traces(prefix, str(tmp_path / "tr"), covar=cov_path, pairs=str(tmp_path / "pairs.txt"),
                             annot=annot, inbreeding=True, nblocks=10, probes=3000, seed=3)
    tr = load_jk_traces(str(tmp_path / "tr"))
    rp = read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid}))
    ex, _, _ = _exact_traces(tr, G, covs, rp.i, rp.j, Acls)
    trx = dict(tr); trx.update({k: v for k, v in ex.items() if not k.startswith("A")})
    unit_id, D, sizes = _fit_design(G.shape[1], 10)
    names = ["lowmaf", "highmaf"] if partitioned else ["L2_0"]
    sig = 0.3 * tr["vsum"] / tr["vsum"].sum() if partitioned else np.array([0.3])
    for noself in (False, True):
        kw = dict(axis_snps=tr["snps"]["SNP"], unit_id=unit_id, D=D, unit_sizes=sizes, bin_names=names, sigma=sig,
                  sigma_e=0.7, noself=noself)
        a, b = jackknife_rescaling(tr, **kw), jackknife_rescaling(trx, **kw)
        print(f"[traces] partitioned={partitioned} noself={noself}: probes/exact AV {a['av'] / b['av']:.4f}, "
              f"E[V_jk] {a['ejk'] / b['ejk']:.4f}, factor {a['factor']:.4f} vs {b['factor']:.4f}")
        for key in ("av", "ejk", "factor"):
            assert a[key] == pytest.approx(b[key], rel=0.02), (noself, key, a[key], b[key])
        np.testing.assert_allclose(tr["Tb" + ("n" if noself else "")], ex["Tb" + ("n" if noself else "")], rtol=0.05,
                                   atol=0.05 * np.abs(ex["Tb"]).max())


@pytest.mark.parametrize("njack", ["10", "chr"])
def test_h2_jk_traces_rescales_the_jackknife_se(tmp_path, njack):
    """summit --h2 --jk-traces keeps the estimate and the jackknife SE (h2_se_jk) and writes h2_se = h2_se_jk x
    sqrt(jk_factor); without self-pairs (the --inbreeding values) it uses the self-pair-free traces."""
    from summit.inference.sumrhe import Sumrhe
    from summit.relatives.moments import append_rel_m_to_sumstats, compute_relative_pair_totals
    from test_relatives_moments import _write_fit_inputs
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, False, pairs=True)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "inb"), inbreeding=True, **kw)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "std"), **kw)
    compute_jackknife_traces(prefix, str(tmp_path / "tr"), inbreeding=True, nblocks=10, probes=200, **kw)
    ss, ld, _ = _write_fit_inputs(tmp_path, G, y, covs, iid)
    append_rel_m_to_sumstats(ss, str(tmp_path / "inb.rel_m.tsv"), "y", str(tmp_path / "a_inb.tsv"))
    append_rel_m_to_sumstats(ss, str(tmp_path / "std.rel_m.tsv"), "y", str(tmp_path / "a_std.tsv"))
    for tag, path in (("self", str(tmp_path / "a_std.tsv")), ("noself", str(tmp_path / "a_inb.tsv"))):
        base = Sumrhe(h2_path=path, ldscores=ld, njack=njack, chisq_threshold=None, out=str(tmp_path / f"b_{tag}"))
        base._run()
        s = Sumrhe(h2_path=path, ldscores=ld, njack=njack, chisq_threshold=None, out=str(tmp_path / f"r_{tag}"),
                   jk_traces=str(tmp_path / "tr"))
        s._run()
        r = pd.read_csv(tmp_path / f"r_{tag}.results.tsv", sep="\t").iloc[0]
        b = pd.read_csv(tmp_path / f"b_{tag}.results.tsv", sep="\t").iloc[0]
        print(f"[traces] njack={njack} {tag}: h2 {r.h2:.4f}, SE_jk {r.h2_se_jk:.4f}, factor {r.jk_factor:.4f}, SE {r.h2_se:.4f}")
        assert r.h2 == pytest.approx(b.h2, rel=1e-12)
        assert r.h2_se_jk == pytest.approx(b.h2_se, rel=1e-12)
        assert 0.3 < r.jk_factor < 3.0
        assert r.h2_se == pytest.approx(r.h2_se_jk * np.sqrt(r.jk_factor), rel=1e-10)
        assert s.jk_rescale[0]["noself"] == (tag == "noself")
        assert s.jk_rescale[0]["fit_snps_in_other_unit"] == 0.0 and s.jk_rescale[0]["fit_snps_without_trace"] == 0.0
