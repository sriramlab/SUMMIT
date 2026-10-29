import json

import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.inference.jktraces import jackknife_rescaling, load_jk_traces, load_long_range, long_range_se
from summit.relatives.traces import compute_jackknife_traces, compute_long_range_sums
from test_relatives_traces import _exact_traces, _fit_design, _setup


def _exact_sums(tr, ex, sfx):
    """Class-level L, T and the within-group W of each design from the exact unit traces."""
    ucls = np.asarray(tr["unit_cls"]); ucell = np.asarray(tr["unit_cell"])
    K = int(ucls.max()) + 1
    S = np.zeros((K, ucls.size)); S[ucls, np.arange(ucls.size)] = 1.0
    G1 = ex["G1" + sfx]
    L = S @ G1 @ S.T
    T = S @ ex["Tb" + sfx]
    chr_codes = pd.factorize(pd.Series(np.asarray(tr["cell_chr"]).astype(str)))[0]
    W = {}
    for name, g in (("chr", chr_codes[ucell]), ("blocks", np.asarray(tr["cell_blk"])[ucell])):
        same = g[:, None] == g[None, :]
        W[name] = S @ np.where(same, G1, 0.0) @ S.T
    return L, T, W


@pytest.mark.parametrize("partitioned", [False, True])
def test_long_range_sums_match_exact(tmp_path, partitioned):
    """Probe estimates of the bin x bin LD-score sums (total, moment matrix, within-group) agree with the exact traces,
    and so does the between-group share rho, with and without self-pairs."""
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, partitioned, pairs=False)
    compute_jackknife_traces(prefix, str(tmp_path / "tr"), covar=cov_path, annot=annot, inbreeding=True, nblocks=10,
                             probes=2, seed=1)
    tr = load_jk_traces(str(tmp_path / "tr"))
    ex, _, _ = _exact_traces(tr, G, covs, np.zeros(0, int), np.zeros(0, int), Acls)
    compute_long_range_sums(prefix, str(tmp_path / "lr"), covar=cov_path, annot=annot, nblocks=10, probes=3000, seed=5)
    lr = load_long_range(str(tmp_path / "lr"))
    for sfx, key in (("", "self"), ("n", "n")):
        L, T, W = _exact_sums(tr, ex, sfx)
        scale = np.abs(L).max()
        np.testing.assert_allclose(np.asarray(lr["L"][key]), L, atol=0.03 * scale)
        np.testing.assert_allclose(np.asarray(lr["T"][key]), T, atol=0.03 * np.abs(T).max())
        for d in ("chr", "blocks"):
            np.testing.assert_allclose(np.asarray(lr["designs"][d]["W"][key]), W[d], atol=0.03 * scale)
            if L.shape[0] == 1:
                rho = 1 - W[d][0, 0] / L[0, 0]
            else:
                om = np.linalg.solve(T.T, np.ones(L.shape[0]))
                rho = 1 - (om @ W[d] @ om) / (om @ L @ om)
            print(f"[longrange] partitioned={partitioned} {key} {d}: rho probes {lr['designs'][d]['rho'][key]:.4f} "
                  f"exact {rho:.4f}")
            assert lr["designs"][d]["rho"][key] == pytest.approx(rho, abs=0.02)


@pytest.mark.parametrize("partitioned", [False, True])
def test_between_group_share_is_the_jackknife_shortfall(tmp_path, partitioned):
    """With exact traces and no genetic signal, the expected block-jackknife variance is the within-block part of the
    model variance, so AV / E_model[V_jk] = 1 / (1 - rho)."""
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, partitioned, pairs=False)
    compute_jackknife_traces(prefix, str(tmp_path / "tr"), covar=cov_path, annot=annot, inbreeding=True, nblocks=10,
                             probes=2, seed=1)
    tr = load_jk_traces(str(tmp_path / "tr"))
    ex, _, _ = _exact_traces(tr, G, covs, np.zeros(0, int), np.zeros(0, int), Acls)
    tr.update({k: v for k, v in ex.items() if not k.startswith("A")})
    unit_id, D, sizes = _fit_design(G.shape[1], 10)
    names = ["lowmaf", "highmaf"] if partitioned else ["L2_0"]
    for noself, sfx in ((False, ""), (True, "n")):
        L, T, W = _exact_sums(tr, ex, sfx)
        if partitioned:
            om = np.linalg.solve(T.T, np.ones(2)); rho = 1 - (om @ W["blocks"] @ om) / (om @ L @ om)
        else:
            rho = 1 - W["blocks"][0, 0] / L[0, 0]
        sig = np.zeros(len(names))
        out = jackknife_rescaling(tr, axis_snps=tr["snps"]["SNP"], unit_id=unit_id, D=D, unit_sizes=sizes,
                                  bin_names=names, sigma=sig, sigma_e=1.0, noself=noself)
        print(f"[longrange] partitioned={partitioned} noself={noself}: AV/E[V_jk] {out['factor']:.4f}, "
              f"1/(1-rho) {1 / (1 - rho):.4f}")
        assert out["factor"] == pytest.approx(1 / (1 - rho), rel=0.05)


@pytest.mark.parametrize("njack", ["10", "chr"])
def test_h2_long_range_se(tmp_path, njack):
    """summit --h2 --long-range-se keeps the estimate and the jackknife SE (h2_se_jk) and writes
    h2_se^2 = h2_se_jk^2 / (1 - rho_diag - rho_off (1 + 2 h2)) + 4 h2^2 (1 - sum f^2) / n."""
    from summit.inference.sumrhe import Sumrhe
    from summit.relatives.moments import append_rel_m_to_sumstats, compute_relative_pair_totals
    from test_relatives_moments import _write_fit_inputs
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, False, pairs=True)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "std"), **kw)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "inb"), inbreeding=True, **kw)
    compute_long_range_sums(prefix, str(tmp_path / "lr"), nblocks=10, probes=300, **kw)
    ss, ld, _ = _write_fit_inputs(tmp_path, G, y, covs, iid)
    append_rel_m_to_sumstats(ss, str(tmp_path / "std.rel_m.tsv"), "y", str(tmp_path / "a_std.tsv"))
    append_rel_m_to_sumstats(ss, str(tmp_path / "inb.rel_m.tsv"), "y", str(tmp_path / "a_inb.tsv"))
    lr = load_long_range(str(tmp_path / "lr"))
    for tag, path in (("self", str(tmp_path / "a_std.tsv")), ("noself", str(tmp_path / "a_inb.tsv"))):
        base = Sumrhe(h2_path=path, ldscores=ld, njack=njack, chisq_threshold=None, out=str(tmp_path / f"b_{tag}"))
        base._run()
        s = Sumrhe(h2_path=path, ldscores=ld, njack=njack, chisq_threshold=None, out=str(tmp_path / f"r_{tag}"),
                   long_range=str(tmp_path / "lr"))
        s._run()
        r = pd.read_csv(tmp_path / f"r_{tag}.results.tsv", sep="\t").iloc[0]
        b = pd.read_csv(tmp_path / f"b_{tag}.results.tsv", sep="\t").iloc[0]
        info = s.jk_rescale[0]
        design = "chr" if njack == "chr" else "blocks"
        rs, rn = lr["designs"][design]["rho"]["self"], lr["designs"][design]["rho"]["n"]
        rho = rn if tag == "noself" else rs
        rdiag = 0.0 if tag == "noself" else max(rs - rn, 0.0)
        roff = max(rn, 0.0) if tag == "noself" else max(rs - rdiag, 0.0)
        print(f"[longrange] njack={njack} {tag}: h2 {r.h2:.4f}, SE_jk {r.h2_se_jk:.4f}, rho {r.long_range_rho:.4f}, "
              f"SE {r.h2_se:.4f}")
        assert r.h2 == pytest.approx(b.h2, rel=1e-12)
        assert r.h2_se_jk == pytest.approx(b.h2_se, rel=1e-12)
        assert r.long_range_rho == pytest.approx(rho, rel=1e-9)
        h = min(max(r.h2, 0.0), 1.0)
        assert info["design"] == design and info["mult"] == pytest.approx(1 + 2 * h)
        assert info["rho_diag"] == pytest.approx(rdiag, abs=1e-9) and info["rho_off"] == pytest.approx(roff, abs=1e-9)
        expect = r.h2_se_jk ** 2 / max(1 - rdiag - roff * (1 + 2 * h), 0.2) + info["signal"]
        assert r.h2_se ** 2 == pytest.approx(expect, rel=1e-9)
        assert r.h2_se >= r.h2_se_jk


def test_longrange_cli_and_h2_option_parse(tmp_path):
    """`summit relatives longrange` runs from the command line, and `summit --h2 ... --long-range-se` parses."""
    from summit.relatives.__main__ import main
    from summit.cli import build_parser
    prefix, G, y, covs, iid, cov_path, annot, Acls = _setup(tmp_path, True, pairs=False)
    main(["longrange", "--geno", prefix, "--covar", cov_path, "--annot", annot, "--nblocks", "10", "--probes", "20",
          "--out", str(tmp_path / "lr")])
    lr = load_long_range(str(tmp_path / "lr"))
    assert set(lr["L"]) == {"self", "n"} and set(lr["designs"]) == {"chr", "blocks"}
    assert np.asarray(lr["designs"]["blocks"]["W"]["self"]).shape == (2, 2)
    args = build_parser().parse_args(["--h2", "x.tsv", "--ldscores", "l.gz", "--long-range-se", str(tmp_path / "lr")])
    assert args.long_range_se == str(tmp_path / "lr")
