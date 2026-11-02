import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.relatives.direct import direct_effects


def _data(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    M, rows, G = 300, [], []
    p = rng.uniform(0.1, 0.5, M)
    for f in range(30):
        for s in range(rng.integers(1, 3)):
            par = (rng.random((4, M)) < p).astype(int)
            for _ in range(rng.integers(2, 4)):
                pick = rng.integers(0, 2, (2, M))
                G.append(par[pick[0], np.arange(M)] + par[2 + pick[1], np.arange(M)])
                rows.append((f"i{len(rows)}", 10 * f + s, f))
    G = np.array(G, float)
    n = len(G)
    ids = np.array([r[0] for r in rows])
    bed_reader.to_bed(str(tmp_path / "g.bed"), G.astype(np.float32), properties={
        "fid": ids, "iid": ids, "sid": np.array([f"rs{j}" for j in range(M)]), "chromosome": np.full(M, "1"),
        "bp_position": np.arange(M) + 1, "allele_1": np.full(M, "A"), "allele_2": np.full(M, "B")})
    sibs = pd.DataFrame(rows, columns=["IID", "SIBSHIP", "FAMILY"])
    sibs.to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    cls = np.arange(M) % 2
    pd.DataFrame({"CHR": 1, "BP": np.arange(M) + 1, "SNP": [f"rs{j}" for j in range(M)], "CM": 0, "a": 1 - cls,
                  "b": cls}).to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    cov = pd.DataFrame({"FID": ids, "IID": ids, "age": rng.normal(50, 5, n), "pc": rng.standard_normal(n)})
    cov.to_csv(tmp_path / "c.tsv", sep="\t", index=False)
    sib = sibs.groupby(["FAMILY", "SIBSHIP"]).ngroup().to_numpy()
    Z = (G - G.mean(0)) / G.std(0, ddof=1)
    y = Z @ rng.normal(0, np.sqrt(0.5 / M), M) + rng.normal(0, 0.6, sib.max() + 1)[sib] + rng.normal(0, 0.6, n)
    pd.DataFrame({"FID": ids, "IID": ids, "y": y}).to_csv(tmp_path / "p.tsv", sep="\t", index=False)
    return G, sibs, cls, cov, y


def test_direct_estimate_equals_dense_calculation(tmp_path):
    G, sibs, cls, cov, y = _data(tmp_path)
    res = direct_effects(str(tmp_path / "g"), str(tmp_path / "s.tsv"), str(tmp_path / "a.tsv"), str(tmp_path / "p.tsv"),
                         str(tmp_path / "o"), covar=str(tmp_path / "c.tsv"), within_covar=["age"], nblocks=10,
                         log=lambda m: None).iloc[0]
    n = len(y)
    X = G - G.mean(0)
    X /= np.sqrt((X ** 2).sum(0) / (n - 1))
    sib = sibs.groupby(["FAMILY", "SIBSHIP"]).ngroup().to_numpy()
    fam = sibs.FAMILY.to_numpy()
    P = np.eye(n) - (sib[:, None] == sib[None, :]) / np.bincount(sib)[sib][None, :]
    same_family = fam[:, None] == fam[None, :]
    W = np.column_stack([np.ones(n), cov.age])
    yd = P @ (y - W @ np.linalg.lstsq(W, y, rcond=None)[0])
    K = []
    for c in (0, 1):
        Kc = P @ (X[:, cls == c] @ X[:, cls == c].T / (cls == c).sum()) @ P
        Kc[same_family] = 0.0
        K.append(Kc)
    gram = np.array([[np.sum(K[c] * K[d]) for d in (0, 1)] for c in (0, 1)])
    sigma = np.linalg.solve(gram, [yd @ Kc @ yd for Kc in K])
    Wf = np.column_stack([np.ones(n), cov.age, cov.pc])
    r = y - Wf @ np.linalg.lstsq(Wf, y, rcond=None)[0]
    np.testing.assert_allclose(res.h2_direct, sigma.sum() / (r @ r / (n - 3)), rtol=1e-5)
    assert res.h2_direct_se > 0 and res.h2_population_se > 0
    assert np.isfinite([res.h2_population, res.difference, res.ratio]).all()
