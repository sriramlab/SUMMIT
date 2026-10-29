"""rg with related pairs in manifest mode (per-pair rel_m12 column) and the long-range LD correction of rg SEs."""
import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.inference.jktraces import load_long_range, long_range_rg, long_range_se, long_range_shares
from summit.relatives.moments import append_rel_m_to_sumstats, compute_relative_pair_totals, extract_rel_m12
from summit.relatives.traces import compute_long_range_sums
from test_relatives_moments import _annot, _exact_sigmas, _family_pairs, _projected, _simulate, _write_fit_inputs


def _rg_inputs(tmp_path, *, partitioned, inbreeding):
    """Two correlated traits with relatives, sumstats with REL_M (and the --inbreeding values), REL_M12 and exact LD."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=360, M=300, npairs=70, ncov=2, missing=0.0)
    rng = np.random.default_rng(21)
    Gs = np.asarray(G, dtype=np.float64)
    Gs = (Gs - Gs.mean(0)) / Gs.std(0)
    g1 = Gs @ rng.normal(0, np.sqrt(0.6 / G.shape[1]), G.shape[1])
    g2 = 0.6 * g1 + Gs @ rng.normal(0, np.sqrt(0.3 / G.shape[1]), G.shape[1])
    y = g1 + 0.5 * y                                              # keeps the shared family effect and covariates
    y2 = g2 + 0.3 * y + rng.standard_normal(y.size) * np.linspace(0.6, 1.2, y.size)
    pd.DataFrame({"FID": iid, "IID": iid, "y": y, "y2": y2}).to_csv(tmp_path / "pheno2.txt", sep=" ", index=False)
    annot, A = _annot(tmp_path, G.shape[1]) if partitioned else (None, None)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "tot"),
                                 pairs=str(tmp_path / "pairs.txt"), covar=cov_path, annot=annot, cross=True,
                                 inbreeding=inbreeding)
    s1, ld, fit_annot = _write_fit_inputs(tmp_path, G, y, covs, iid, A, ["lowmaf", "highmaf"])
    X, y2t, p, n_eff = _projected(G, y2, covs)
    N, M = X.shape
    r = (X.T @ y2t) / n_eff
    pd.DataFrame({"SNP": [f"rs{j}" for j in range(M)], "A1": "A", "A2": "G", "N": N, "BETA": r,
                  "SE": np.sqrt((1 - r * r) / (N - p - 2)), "COV_RANK": p}).to_csv(
        tmp_path / "s2.tsv", sep="\t", index=False, float_format="%.15g")
    append_rel_m_to_sumstats(s1, str(tmp_path / "tot.rel_m.tsv"), "y", str(tmp_path / "a.tsv"), annot=annot)
    append_rel_m_to_sumstats(str(tmp_path / "s2.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y2", str(tmp_path / "b.tsv"),
                             annot=annot)
    extract_rel_m12(str(tmp_path / "tot.rel_m12.tsv"), "y", "y2", str(tmp_path / "m12.tsv"), annot=annot)
    _, y1t, _, _ = _projected(G, y, covs)
    return dict(prefix=prefix, G=G, y=y, y2=y2, covs=covs, iid=iid, cov_path=cov_path, annot=annot, A=A, ld=ld,
                fit_annot=fit_annot, c_ov=float(y1t @ y2t / n_eff))


def _manifest_args(tmp_path, d, njack, extra=()):
    from summit.cli import build_parser
    man = tmp_path / "manifest.tsv"
    pd.DataFrame({"phen1": ["y"], "phen2": ["y2"], "sumstats1": [str(tmp_path / "a.tsv")],
                  "sumstats2": [str(tmp_path / "b.tsv")], "overlap_covariance": [d["c_ov"]],
                  "rel_m12": [str(tmp_path / "m12.tsv")]}).to_csv(man, sep="\t", index=False)
    out = tmp_path / "man_out"
    out.mkdir(exist_ok=True)
    argv = ["--rg", str(man), "--ldscores", d["ld"], "--njack", njack, "--out", str(out), *extra]
    if d["fit_annot"] is not None:
        argv += ["--annot", d["fit_annot"]]
    return build_parser().parse_args(argv), out


def _run_manifest(args):
    from summit.cli import _dispatch_rg_manifest
    from summit.logger import Logger
    _dispatch_rg_manifest(args, Logger(suppress=True))
    return pd.read_csv(f"{args.out}/manifest.results.tsv", sep="\t").iloc[0]


def _single(tmp_path, d, njack, **kw):
    from summit.inference.sumcore import Sumcore
    s = Sumcore(rg=f"{tmp_path / 'a.tsv'},{tmp_path / 'b.tsv'}", rel_m12=str(tmp_path / "m12.tsv"), ldscores=d["ld"],
                annot=d["fit_annot"], njack=njack, chisq_threshold=None, intercept_chisq_thr=None,
                intercept_rg=d["c_ov"], **kw)
    return s, s._run()


@pytest.mark.parametrize("partitioned,inbreeding", [(False, False), (True, False), (False, True), (True, True)])
def test_manifest_rel_m12_matches_single_pair(tmp_path, partitioned, inbreeding):
    """A manifest row with a rel_m12 column gives the same h2, genetic covariance and rg (and SEs) as the single-pair
    --rel-m12 run; with the --inbreeding values both equal the exact estimator without related pairs and self-pairs."""
    d = _rg_inputs(tmp_path, partitioned=partitioned, inbreeding=inbreeding)
    _, res = _single(tmp_path, d, "chr")
    args, _ = _manifest_args(tmp_path, d, "chr")
    row = _run_manifest(args)
    rg_fit = res["rg_fit"]
    assert row.rg_total == pytest.approx(float(rg_fit.rg_total[0]), rel=1e-10)
    assert row.rg_total_se == pytest.approx(float(rg_fit.rg_total[1]), rel=1e-10)
    assert row.gamma_g_total == pytest.approx(float(rg_fit.gamma_total[0]), rel=1e-10)
    assert row.h2_trait1 == pytest.approx(float(res["h2_fit1"].h2[-1, 0]), rel=1e-10)
    assert row.h2_trait2_se == pytest.approx(float(res["h2_fit2"].h2[-1, 1]), rel=1e-10)
    assert str(row.rel_m12_input).endswith("m12.tsv")
    if inbreeding:
        pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), d["iid"])
        s11 = _exact_sigmas(d["G"], d["covs"], pi, pj, d["y"], d["y"], d["A"])
        s22 = _exact_sigmas(d["G"], d["covs"], pi, pj, d["y2"], d["y2"], d["A"])
        s12 = _exact_sigmas(d["G"], d["covs"], pi, pj, d["y"], d["y2"], d["A"])
        rg_exact = s12[:-1].sum() / np.sqrt(s11[:-1].sum() * s22[:-1].sum())
        assert row.rg_total == pytest.approx(rg_exact, rel=1e-6)


def test_manifest_rel_m12_value_and_errors(tmp_path):
    """The rel_m12 column also takes one value; a missing file, a global --rel-m12 and --rg-manifest-fast are refused."""
    from summit.cli import _dispatch_rg_manifest, _normalize_rg_manifest
    from summit.logger import Logger
    d = _rg_inputs(tmp_path, partitioned=False, inbreeding=False)
    m12 = pd.read_csv(tmp_path / "m12.tsv", sep="\t")
    value = float(m12.REL_M12.iloc[0]) if len(m12) == 1 else None
    if value is not None:
        _, res = _single(tmp_path, d, "chr")
        args, _ = _manifest_args(tmp_path, d, "chr")
        man = pd.read_csv(args.rg, sep="\t"); man["rel_m12"] = [repr(value)]
        man.to_csv(args.rg, sep="\t", index=False)
        row = _run_manifest(args)
        assert row.rg_total == pytest.approx(float(res["rg_fit"].rg_total[0]), rel=1e-9)
    args, _ = _manifest_args(tmp_path, d, "chr")
    man = pd.read_csv(args.rg, sep="\t"); man["rel_m12"] = [str(tmp_path / "missing.tsv")]
    man.to_csv(args.rg, sep="\t", index=False)
    with pytest.raises(ValueError, match="neither a file nor a number"):
        _normalize_rg_manifest(args.rg, require_intercept=False)
    args, _ = _manifest_args(tmp_path, d, "chr", extra=["--rel-m12", str(tmp_path / "m12.tsv")])
    with pytest.raises(SystemExit):
        _dispatch_rg_manifest(args, Logger(suppress=True))
    args, _ = _manifest_args(tmp_path, d, "chr", extra=["--rg-manifest-fast"])
    with pytest.raises(SystemExit):
        _dispatch_rg_manifest(args, Logger(suppress=True))


@pytest.mark.parametrize("njack,inbreeding", [("10", False), ("chr", False), ("10", True)])
def test_rg_long_range_se(tmp_path, njack, inbreeding):
    """--long-range-se for rg keeps the estimates, corrects the totals' SEs by the documented rules (h2 per trait;
    the genetic covariance's factor also applied to rg) and gives the same values in manifest mode."""
    d = _rg_inputs(tmp_path, partitioned=False, inbreeding=inbreeding)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=d["cov_path"])
    compute_long_range_sums(d["prefix"], str(tmp_path / "lr"), nblocks=10, probes=300, **kw)
    lr = load_long_range(str(tmp_path / "lr"))
    s0, base = _single(tmp_path, d, njack)
    s1, corr = _single(tmp_path, d, njack, long_range=str(tmp_path / "lr"))
    b, c = base["rg_fit"], corr["rg_fit"]
    assert float(c.rg_total[0]) == pytest.approx(float(b.rg_total[0]), rel=1e-12)
    assert float(c.gamma_total[0]) == pytest.approx(float(b.gamma_total[0]), rel=1e-12)
    info = s1.long_range_info
    design = "chr" if njack == "chr" else "blocks"
    sh = long_range_shares(lr, design=design, nunits=int(b.prepared.jackknife.D.shape[0]), bin_names=["L2_0"],
                           noself=inbreeding)
    h1, h2 = float(base["h2_fit1"].h2[-1, 0]), float(base["h2_fit2"].h2[-1, 0])
    hh1, hh2 = min(max(h1, 0), 1), min(max(h2, 0), 1)
    r = min(max(float(b.rg_total[0]), -1), 1)
    gam = r * np.sqrt(hh1 * hh2)
    e12 = min(max(d["c_ov"] - gam, -1), 1)
    w1 = (1 - hh1) * (1 - hh2) + e12 ** 2
    w2 = hh1 * (1 - hh2) + hh2 * (1 - hh1) + 2 * gam * e12
    w3 = hh1 * hh2 + gam ** 2
    m12 = min(max((w1 + 2 * w2 + 3 * w3) / (w1 + w2 + w3), 1), 3)
    sizes = np.asarray(b.prepared.unit_sizes, dtype=np.float64)
    f = sizes[sizes > 0] / sizes[sizes > 0].sum()
    n12 = np.sqrt(float(base["h2_fit1"].prepared.matched.nsamp) * float(base["h2_fit2"].prepared.matched.nsamp))
    var = float(b.gamma_total[1]) ** 2 / max(1 - sh["rho_diag"] - sh["rho_off"] * m12, 0.2) \
        + 2 * w3 * (1 - (f * f).sum()) / n12
    factor = var / float(b.gamma_total[1]) ** 2
    print(f"[rg-longrange] njack={njack} inbreeding={inbreeding}: rho {sh['rho']:.4f}, m12 {m12:.3f}, "
          f"rg SE {float(b.rg_total[1]):.4f} -> {float(c.rg_total[1]):.4f}")
    assert info["rg"]["mult"] == pytest.approx(m12, rel=1e-12)
    assert info["rg"]["factor"] == pytest.approx(factor, rel=1e-10)
    assert float(c.gamma_total[1]) == pytest.approx(np.sqrt(var), rel=1e-10)
    assert float(c.rg_total[1]) == pytest.approx(float(b.rg_total[1]) * np.sqrt(factor), rel=1e-10)
    assert float(c.rg_total[1]) >= float(b.rg_total[1])
    for t, key in ((0, "h2_fit1"), (1, "h2_fit2")):
        fit = base[key]
        exp = long_range_se(lr, design=design, nunits=int(b.prepared.jackknife.D.shape[0]), bin_names=["L2_0"],
                            h2=float(fit.h2[-1, 0]), se_jk=float(fit.h2[-1, 1]), n=float(fit.prepared.matched.nsamp),
                            unit_sizes=fit.prepared.unit_sizes, noself=inbreeding)
        assert float(corr[key].h2[-1, 1]) == pytest.approx(exp["se"], rel=1e-12)
        assert float(corr[key].h2[-1, 0]) == pytest.approx(float(fit.h2[-1, 0]), rel=1e-12)
        assert info[f"h2_{t + 1}"]["se_jk"] == pytest.approx(float(fit.h2[-1, 1]), rel=1e-12)
    # manifest mode gives the same corrected values and reports the jackknife SEs
    args, _ = _manifest_args(tmp_path, d, njack, extra=["--long-range-se", str(tmp_path / "lr")])
    row = _run_manifest(args)
    assert row.rg_total_se == pytest.approx(float(c.rg_total[1]), rel=1e-10)
    assert row.rg_total_se_jk == pytest.approx(float(b.rg_total[1]), rel=1e-10)
    assert row.jk_factor_rg == pytest.approx(factor, rel=1e-10)
    assert row.long_range_rho == pytest.approx(sh["rho"], rel=1e-10)


def test_rg_rule_without_signal_is_the_h2_rule():
    """With no genetic signal the rg weights reduce to the noise term (m12 = 1, no signal term): the genetic-covariance
    factor equals the h2 factor at h2 = 0, 1 / (1 - rho), the between-group shortfall of the exact jackknife
    (test_between_group_share_is_the_jackknife_shortfall); with rho = 0 the SE is the jackknife SE."""
    L = [[2.0]]
    lr = {"path": "x", "nblocks": 10, "classes": [], "vsum": [1.0], "vsum_all": 1.0,
          "L": {"self": L, "n": L}, "T": {"self": L, "n": L},
          "designs": {"blocks": {"W": {"self": [[1.5]], "n": [[1.6]]}}, "chr": {"W": {"self": [[2.0]], "n": [[2.0]]}}}}
    sizes = np.full(10, 30.0)
    g = long_range_rg(lr, design="blocks", nunits=10, bin_names=["L2_0"], h2_1=0.0, h2_2=0.0, rg=0.0, c=0.0,
                      se_jk_gamma=0.1, n=500.0, unit_sizes=sizes, noself=False)
    h = long_range_se(lr, design="blocks", nunits=10, bin_names=["L2_0"], h2=0.0, se_jk=0.1, n=500.0, unit_sizes=sizes,
                      noself=False)
    assert g["mult"] == 1.0 and g["signal"] == 0.0
    assert g["factor"] == pytest.approx(h["factor"], rel=1e-12)
    assert g["factor"] == pytest.approx(1 / (1 - 0.25), rel=1e-12)              # rho = 1 - 1.5 / 2
    z = long_range_rg(lr, design="chr", nunits=22, bin_names=["L2_0"], h2_1=0.0, h2_2=0.0, rg=0.0, c=0.0,
                      se_jk_gamma=0.1, n=500.0, unit_sizes=sizes, noself=False)
    assert z["rho"] == 0.0 and z["se"] == pytest.approx(0.1, rel=1e-12)
    # same trait: the rg rule is the h2 rule (m12 = 1 + 2 h2, signal 4 h2^2 (1 - sum f^2) / n)
    s = long_range_rg(lr, design="blocks", nunits=10, bin_names=["L2_0"], h2_1=0.4, h2_2=0.4, rg=1.0, c=1.0,
                      se_jk_gamma=0.05, n=500.0, unit_sizes=sizes, noself=False)
    t = long_range_se(lr, design="blocks", nunits=10, bin_names=["L2_0"], h2=0.4, se_jk=0.05, n=500.0,
                      unit_sizes=sizes, noself=False)
    assert s["mult"] == pytest.approx(1.8, rel=1e-12) and s["factor"] == pytest.approx(t["factor"], rel=1e-12)
