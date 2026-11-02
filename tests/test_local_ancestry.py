import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.local_ancestry.component import JointFit, _project, local_ancestry_test
from summit.local_ancestry.markers import build_markers, merge_markers


def _segments(rng, n_sites, n_segments=4):
    stops = np.r_[np.sort(rng.choice(np.arange(1, n_sites), n_segments - 1, replace=False)), n_sites]
    return stops, rng.integers(0, 2, (n_segments, 2))


def _per_site(stops, anc, n_sites):
    out = np.zeros((n_sites, 2), int)
    prev = 0
    for s, a in zip(stops, anc):
        out[prev:s] = a
        prev = s
    return out


def _fam(path, ids):
    pd.DataFrame({0: ids, 1: ids, 2: 0, 3: 0, 4: 1, 5: -9}).to_csv(path, sep=" ", header=False, index=False)


def test_markers_from_lanc(tmp_path):
    rng = np.random.default_rng(1)
    ids = ["a", "b", "c"]
    _fam(tmp_path / "x.fam", ids + ["u"])
    expected, called, total = [], np.zeros(4), 0
    for c in range(1, 23):
        n = 30
        pos = np.sort(rng.choice(np.arange(1, 3000), n, replace=False)) * 1000
        pd.DataFrame({"#CHROM": c, "POS": pos, "ID": [f"s{c}_{k}" for k in range(n)], "REF": "A", "ALT": "G"}).to_csv(
            tmp_path / f"chr{c}.pvar", sep="\t", index=False)
        pd.DataFrame({"#IID": ids[::-1]}).to_csv(tmp_path / f"chr{c}.psam", sep="\t", index=False)
        segs = {s: _segments(rng, n) for s in ids}
        lines = [" ".join(f"{e}:{a}{b}" for e, (a, b) in zip(*segs[s])) for s in ids[::-1]]
        (tmp_path / f"chr{c}.lanc").write_text(f"{n} 3\n" + "\n".join(lines) + "\n")
        site = {s: _per_site(*segs[s], n) for s in ids}
        idx = np.unique(np.clip(np.searchsorted(pos, np.arange(pos[0], pos[-1] + 1, 250_000)), 0, n - 1))
        expected.append(np.array([(site[s][idx] == 1).sum(1) for s in ids] + [np.full(idx.size, 2)]))
        called += np.r_[[(site[s] == 1).sum() for s in ids], 2 * n]
        total += 2 * n
    build_markers(str(tmp_path / "chr{chr}.lanc"), str(tmp_path / "x.fam"), str(tmp_path / "m"), fmt="lanc", code=1,
                  sites=str(tmp_path / "chr{chr}.pvar"), fill=1, log=lambda m: None)
    got = bed_reader.open_bed(str(tmp_path / "m.bed"), count_A1=True).read(dtype="float64")
    np.testing.assert_array_equal(got, np.concatenate(expected, axis=1))
    theta = pd.read_csv(tmp_path / "m.theta.tsv", sep="\t").THETA.to_numpy()
    np.testing.assert_allclose(theta, called / total, rtol=1e-5)
    with pytest.raises(ValueError, match="no calls"):
        build_markers(str(tmp_path / "chr{chr}.lanc"), str(tmp_path / "x.fam"), str(tmp_path / "m2"), fmt="lanc",
                      code=1, sites=str(tmp_path / "chr{chr}.pvar"), log=lambda m: None)


def test_markers_from_msp(tmp_path):
    rng = np.random.default_rng(2)
    ids = ["a", "b"]
    _fam(tmp_path / "x.fam", ids)
    expected = []
    for c in range(1, 23):
        ends = np.sort(rng.choice(np.arange(2, 40), 5, replace=False)) * 100_000
        starts = np.r_[1, ends[:-1] + 1]
        hap = rng.integers(0, 2, (5, 4))
        cols = ["chm", "spos", "epos", "sgpos", "egpos", "n snps", "a.0", "a.1", "b.0", "b.1"]
        rows = [[c, s, e, 0, 0, 10] + list(h) for s, e, h in zip(starts, ends, hap)]
        with open(tmp_path / f"chr{c}.msp.tsv", "w") as fh:
            fh.write("#Subpopulation order/codes: AFR=0\tEUR=1\n#" + "\t".join(cols) + "\n")
            fh.write("\n".join("\t".join(map(str, r)) for r in rows) + "\n")
        grid = np.arange(starts[0], ends[-1] + 1, 250_000)
        win = np.clip(np.searchsorted(ends, grid), 0, 4)
        expected.append(np.array([(hap[win][:, 0:2] == 0).sum(1), (hap[win][:, 2:4] == 0).sum(1)]))
    build_markers(str(tmp_path / "chr{chr}.msp.tsv"), str(tmp_path / "x.fam"), str(tmp_path / "m"), fmt="msp", code=0,
                  log=lambda m: None)
    got = bed_reader.open_bed(str(tmp_path / "m.bed"), count_A1=True).read(dtype="float64")
    np.testing.assert_array_equal(got, np.concatenate(expected, axis=1))


def test_merge_interleaves_by_position(tmp_path):
    rng = np.random.default_rng(3)
    ids = np.array(["a", "b", "c", "d", "e"])
    def write(prefix, G, chrom, bp, sid):
        bed_reader.to_bed(str(tmp_path / prefix) + ".bed", G.astype(np.float32), properties={
            "fid": ids, "iid": ids, "sid": sid, "chromosome": chrom.astype(str), "bp_position": bp,
            "allele_1": np.full(len(sid), "A"), "allele_2": np.full(len(sid), "B")})
    chs, bps = np.repeat([1, 2], 20), np.tile(np.arange(20) * 1000 + 500, 2)
    Gs = rng.integers(0, 3, (5, 40)).astype(float)
    write("snp", Gs, chs, bps, np.array([f"rs{j}" for j in range(40)]))
    chl, bpl = np.repeat([1, 2], 4), np.tile([1000, 5500, 9000, 30000], 2)
    Gl = rng.integers(0, 3, (5, 8)).astype(float)
    write("la", Gl, chl, bpl, np.array([f"LA:{c}:{b}" for c, b in zip(chl, bpl)]))
    pd.DataFrame({"CHR": chs, "BP": bps, "SNP": [f"rs{j}" for j in range(40)], "CM": 0,
                  "x": np.arange(40) % 2, "y": 1 - np.arange(40) % 2}).to_csv(tmp_path / "an.tsv", sep="\t", index=False)
    merge_markers(str(tmp_path / "snp"), str(tmp_path / "la"), str(tmp_path / "an.tsv"), str(tmp_path / "m"),
                  log=lambda m: None)
    bim = pd.read_csv(tmp_path / "m.bim", sep="\t", header=None)
    assert (np.diff(bim[0] * 10**9 + bim[3]) >= 0).all()
    G = bed_reader.open_bed(str(tmp_path / "m.bed"), count_A1=True).read(dtype="float64")
    for k, sid in enumerate(bim[1]):
        ref = Gl[:, int(np.flatnonzero(np.array([f"LA:{c}:{b}" for c, b in zip(chl, bpl)]) == sid)[0])] \
            if sid.startswith("LA") else Gs[:, int(sid[2:])]
        np.testing.assert_array_equal(G[:, k], ref)
    an = pd.read_csv(tmp_path / "m.annot.tsv.gz", sep="\t")
    assert list(an.columns) == ["CHR", "BP", "SNP", "CM", "x", "y", "LA"]
    np.testing.assert_array_equal(an.LA, bim[1].str.startswith("LA").astype(int))
    np.testing.assert_array_equal(an.loc[an.LA == 0, ["x", "y"]].sum(axis=1), 1)


def test_joint_fit_solves_the_moment_equations():
    rng = np.random.default_rng(4)
    n, K = 60, 3
    W = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Q, _ = np.linalg.qr(W)
    Ks = []
    for _ in range(K):
        X = rng.standard_normal((n, 40))
        Ks.append(_project(X @ X.T / 40, Q))
    pi, pj = np.array([0, 2, 4]), np.array([1, 3, 5])
    unit = np.arange(n)
    unit[[1, 3, 5]] = [0, 2, 4]
    fit = JointFit(Ks, Q, pi, pj, unit)
    y = rng.standard_normal(n)
    y -= Q @ (Q.T @ y)
    theta = fit.fit(y)
    # explicit equations: kernels with the related pairs set to zero in the class rows
    P = np.eye(n) - Q @ Q.T
    Ku = []
    for Kc in Ks:
        Kz = Kc.copy()
        Kz[pi, pj] = Kz[pj, pi] = 0.0
        Ku.append(Kz)
    S = Ks + [P]
    A = np.array([[np.sum(Ku[c] * S[d]) for d in range(K + 1)] for c in range(K)] + [[np.trace(Sd) for Sd in S]])
    b = np.array([y @ Kz @ y for Kz in Ku] + [y @ y])
    np.testing.assert_allclose(theta, np.linalg.solve(A, b), rtol=1e-10)
    w = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(y @ fit.form(w) @ y, w @ theta[:K], rtol=1e-10)


def test_local_ancestry_test_runs(tmp_path):
    rng = np.random.default_rng(5)
    n, ms, nl = 300, 400, 30
    theta = rng.beta(4, 2, n)
    G = rng.binomial(2, 0.3, (n, ms)).astype(float)
    LA = rng.binomial(2, np.clip(theta[:, None] + rng.normal(0, 0.1, (n, nl)), 0, 1)).astype(float)
    X = np.column_stack([G, LA])
    ids = np.array([f"i{k}" for k in range(n)])
    names = np.array([f"rs{j}" for j in range(ms)] + [f"LA{j}" for j in range(nl)])
    bed_reader.to_bed(str(tmp_path / "g.bed"), X.astype(np.float32), properties={
        "fid": ids, "iid": ids, "sid": names, "chromosome": np.full(ms + nl, "1"),
        "bp_position": np.arange(ms + nl) + 1, "allele_1": np.full(ms + nl, "A"), "allele_2": np.full(ms + nl, "B")})
    pd.DataFrame({"CHR": 1, "BP": np.arange(ms + nl) + 1, "SNP": names, "CM": 0,
                  "SNP_all": np.r_[np.ones(ms), np.zeros(nl)].astype(int),
                  "LA": np.r_[np.zeros(ms), np.ones(nl)].astype(int)}).to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    pd.DataFrame({"FID": ids, "IID": ids, "THETA": theta}).to_csv(tmp_path / "c.tsv", sep="\t", index=False)
    y = (X - X.mean(0)) / X.std(0) @ rng.normal(0, np.sqrt(0.4 / (ms + nl)), ms + nl) + rng.normal(0, 0.8, n)
    pd.DataFrame({"FID": ids, "IID": ids, "y": y}).to_csv(tmp_path / "p.tsv", sep="\t", index=False)
    res = local_ancestry_test(str(tmp_path / "g"), str(tmp_path / "a.tsv"), str(tmp_path / "p.tsv"), str(tmp_path / "o"),
                              covar=str(tmp_path / "c.tsv"), min_n=100, log=lambda m: None)
    row = res.iloc[0]
    assert row.n == n and row.la_null_sd > 0 and row.la_se > 0 and 0 <= row.la_p <= 1
    assert (tmp_path / "o.la_test.tsv").exists()
