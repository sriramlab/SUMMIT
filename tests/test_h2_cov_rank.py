"""h2 and rg use the GWAS covariate rank (COV_RANK / P_EFF, --cov-rank, --pheno-rg-cov) and then equal exact HE."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.inference.h2_batch_fast import dispatch_h2_batch_fast
from summit.inference.sumcore import Sumcore
from summit.inference.sumrhe import Sumrhe
from summit.logger import Logger


def _setting(tmp_path, *, N=400, M=200, p=6, seed=3):
    """Structured cohort with p covariates, exact in-sample LD scores and two traits' OLS sumstats."""
    rng = np.random.default_rng(seed)
    pop = (rng.random(N) < 0.5).astype(float)
    f0 = rng.uniform(0.08, 0.5, M)
    f1 = np.clip(f0 + rng.normal(0, 0.12, M), 0.03, 0.97)
    G = rng.binomial(2, np.where(pop[:, None] == 1, f1[None, :], f0[None, :])).astype(np.float64)
    covs = np.column_stack([pop + rng.normal(0, 0.3, N), rng.normal(size=(N, p - 1))])
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N), covs]))
    ne = N - p - 1
    X = (G - G.mean(0)) / G.std(0, ddof=1)
    PX = X - Q @ (Q.T @ X)
    ss = (PX ** 2).sum(0)
    Xt = PX * np.sqrt(ne / ss)[None, :]
    meta = pd.DataFrame({"CHR": np.repeat([1, 2], [M // 2, M - M // 2]), "SNP": [f"rs{j}" for j in range(M)],
                         "BP": np.arange(1, M + 1) * 1000})
    ld = meta.copy()
    ld["L2_0"] = ((Xt.T @ Xt / ne) ** 2).sum(1) - M / ne
    ld_path = tmp_path / "in.gw.ldscore.gz"
    ld.to_csv(ld_path, sep="\t", index=False, float_format="%.12g")
    iid = [f"s{i}" for i in range(N)]
    cov = pd.DataFrame(covs, columns=[f"c{k}" for k in range(p)])
    cov.insert(0, "IID", iid)
    cov.insert(0, "FID", iid)
    cov.to_csv(tmp_path / "covar.txt", sep=" ", index=False)
    g1 = X @ rng.normal(0, np.sqrt(0.5 / M), M)
    g2 = 0.7 * g1 + X @ rng.normal(0, np.sqrt(0.2 / M), M)
    y1 = g1 + 0.8 * pop + rng.normal(0, np.sqrt(0.5), N)
    y2 = g2 + 0.3 * pop + 0.3 * (y1 - g1) + rng.normal(0, 0.7, N)
    ys = []
    for t, y in enumerate((y1, y2)):
        yp = y - Q @ (Q.T @ y)
        r = (PX.T @ yp) / np.sqrt(ss * (yp @ yp))
        base = pd.DataFrame({"SNP": meta.SNP, "A1": "A", "A2": "G", "N": N, "BETA": r,
                             "SE": np.sqrt((1 - r ** 2) / (N - p - 2))})
        (tmp_path / "plain").mkdir(exist_ok=True)
        (tmp_path / "ranked").mkdir(exist_ok=True)
        base.to_csv(tmp_path / "plain" / f"t{t}.tsv", sep="\t", index=False, float_format="%.15g")
        base.assign(COV_RANK=p).to_csv(tmp_path / "ranked" / f"t{t}.tsv", sep="\t", index=False, float_format="%.15g")
        pd.DataFrame({"FID": iid, "IID": iid, "y": y}).to_csv(tmp_path / f"ph{t}.txt", sep=" ", index=False)
        ys.append(yp)
    K = Xt @ Xt.T / M
    P = np.eye(N) - Q @ Q.T
    return SimpleNamespace(N=N, M=M, p=p, ne=ne, ld=str(ld_path), K=K, P=P, y=ys, dir=tmp_path)


def _exact(s, a, b):
    L = np.array([[(s.K * s.K).sum(), np.trace(s.K @ s.P)], [np.trace(s.K), np.trace(s.P)]])
    return np.linalg.solve(L, [a @ s.K @ b, a @ s.P @ b])


def _h2(path, s, **kw):
    fit = Sumrhe(h2_path=str(path), ldscores=s.ld, chisq_threshold=None, **kw)
    fit._run()
    return float(fit.results[0].h2[-1, 0])


@pytest.mark.parametrize("njack", ["chr", "10"])
def test_h2_with_the_covariate_rank_equals_exact_he(tmp_path, njack):
    s = _setting(tmp_path)
    e = _exact(s, s.y[0], s.y[0])
    exact = e[0] / e.sum()
    assert _h2(s.dir / "ranked" / "t0.tsv", s, njack=njack) == pytest.approx(exact, rel=1e-9)
    assert _h2(s.dir / "plain" / "t0.tsv", s, njack=njack, cov_rank=str(s.p)) == pytest.approx(exact, rel=1e-9)
    # without the rank, h2 reads (N - p - 1) / (N - 1) of the exact value
    assert _h2(s.dir / "plain" / "t0.tsv", s, njack=njack) == pytest.approx(exact * s.ne / (s.N - 1), rel=2e-3)


def test_batch_h2_uses_the_covariate_rank(tmp_path):
    s = _setting(tmp_path)
    regular = Sumrhe(h2_path=str(s.dir / "ranked"), ldscores=s.ld, chisq_threshold=None, njack="chr")
    regular._run()
    args = SimpleNamespace(
        trace=None, ldscores=s.ld, bim=None, annot=None, h2=str(s.dir / "ranked"), out=str(tmp_path / "fast"),
        max_chisq=None, chisq_action="drop", cov_rank=None, njack="chr", h2_workers=1, h2_batch_size=2,
        h2_fast_reader="stream", h2_checkpoint_every=2, h2_cache_dir=None, h2_cache_mode="readwrite",
        h2_cache_only=False, h2_cache_verify_checksum=False, verbose="0", write_jack=False, adjust_delta=False,
        enrich_mode="auto", allow_neg_enr=False, clip_nonfinite_vals=False, jack_mode="mean",
    )
    rows = dispatch_h2_batch_fast(args, Logger(suppress=True))
    fast = {phen: float(fit.h2[-1, 0]) for _, phen, fit in rows}
    for fit, name in zip(regular.results, regular.phen_names):
        assert fast[name] == pytest.approx(float(fit.h2[-1, 0]), rel=1e-10)


def test_rg_with_pheno_rg_cov_equals_exact_he(tmp_path):
    s = _setting(tmp_path)
    e11, e22, e12 = _exact(s, s.y[0], s.y[0]), _exact(s, s.y[1], s.y[1]), _exact(s, s.y[0], s.y[1])
    rg_exact = e12[0] / np.sqrt(e11[0] * e22[0])
    res = Sumcore(rg=f"{s.dir / 'plain' / 't0.tsv'},{s.dir / 'plain' / 't1.tsv'}", ldscores=s.ld, njack="chr",
                  chisq_threshold=None, intercept_chisq_thr=None, align_alleles=False,
                  pheno_rg=f"{s.dir / 'ph0.txt'},{s.dir / 'ph1.txt'}",
                  pheno_rg_cov=f"{s.dir / 'covar.txt'},{s.dir / 'covar.txt'}")._run()
    assert float(res["rg_fit"].rg_total[0]) == pytest.approx(rg_exact, rel=1e-6)
    assert float(res["h2_fit1"].h2[-1, 0]) == pytest.approx(e11[0] / e11.sum(), rel=1e-6)
