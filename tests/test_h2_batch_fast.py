from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from summit.inference.h2_batch_fast import dispatch_h2_batch_fast
from summit.inference.h2_cache import trace_axis_digest
from summit.inference.sumrhe import Sumrhe
from summit.logger import Logger


def _write_fixture(root, *, n_per_chr=160, n_traits=5):
    rng = np.random.default_rng(20260711)
    chromosomes = np.repeat(np.arange(1, 4), n_per_chr)
    within = np.tile(np.arange(n_per_chr), 3)
    bp = within * 1000 + 100_000
    snps = np.asarray([f"{chrom}:{pos}" for chrom, pos in zip(chromosomes, bp)])
    n = snps.size

    bin1 = ((within % 3) == 0).astype(float)
    bin2 = ((within % 5) <= 1).astype(float)
    annot = pd.DataFrame(
        {
            "CHR": chromosomes,
            "BP": bp,
            "SNP": snps,
            "CM": np.zeros(n),
            "Base": np.ones(n),
            "bin1": bin1,
            "bin2": bin2,
        }
    )
    annot_path = root / "fixture.annot.tsv"
    annot.to_csv(annot_path, sep="\t", index=False)

    # Positive, nondegenerate LD-score columns are sufficient for testing the
    # estimator's algebra and legacy/fast equivalence.
    ld = pd.DataFrame(
        {
            "CHR": chromosomes,
            "SNP": snps,
            "BP": bp,
            "Base": 1.5 + rng.uniform(0.0, 0.5, n),
            "bin1": 0.4 + 0.7 * bin1 + rng.uniform(0.0, 0.1, n),
            "bin2": 0.3 + 0.8 * bin2 + rng.uniform(0.0, 0.1, n),
        }
    )
    ld_path = root / "fixture.ldscore.tsv"
    ld.to_csv(ld_path, sep="\t", index=False)

    sums = root / "sumstats"
    sums.mkdir()
    for trait in range(n_traits):
        obs_n = 4_000 + trait * 100 + (within % 7)
        se = 0.045 + rng.uniform(0.0, 0.01, n)
        beta = rng.normal(0.0, 0.018 + 0.001 * trait, n)
        if trait == 1:
            beta[[13, n_per_chr + 17]] = [1.2, -1.1]
        frame = pd.DataFrame(
            {
                "SNP": snps,
                "A1": np.where(within % 2 == 0, "A", "C"),
                "A2": np.where(within % 2 == 0, "G", "T"),
                "N": obs_n,
                "BETA": beta,
                "SE": se,
                "Z": beta / se,
            }
        )
        if trait == 2:
            frame = frame.drop(index=[2, 19, n_per_chr + 8, 2 * n_per_chr + 33])
        if trait == 3:
            duplicate = frame.iloc[[10]].copy()
            duplicate["N"] = duplicate["N"] + 250
            duplicate["BETA"] = duplicate["BETA"] * 0.5
            duplicate["Z"] = duplicate["BETA"] / duplicate["SE"]
            frame = pd.concat([frame, duplicate], ignore_index=True)
        frame.to_csv(sums / f"trait_{trait}.tsv", sep="\t", index=False)

    return ld_path, annot_path, sums


def _fast_args(ld_path, annot_path, sums, out, *, njack="chr"):
    return SimpleNamespace(
        trace=None,
        ldscores=str(ld_path),
        bim=None,
        annot=str(annot_path),
        h2=str(sums),
        out=str(out),
        max_chisq="auto",
        chisq_action="drop",
        cov_rank=None,
        njack=njack,
        h2_workers=2,
        h2_batch_size=3,
        h2_fast_reader="stream",
        h2_checkpoint_every=2,
        h2_cache_dir=None,
        h2_cache_mode="readwrite",
        h2_cache_only=False,
        h2_cache_verify_checksum=False,
        verbose="0",
        write_jack=False,
        adjust_delta=False,
        enrich_mode="auto",
        allow_neg_enr=False,
        clip_nonfinite_vals=False,
        jack_mode="mean",
    )


def test_fast_batch_matches_legacy_with_sparse_drops(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path)

    legacy = Sumrhe(
        h2_path=str(sums),
        out=str(tmp_path / "legacy"),
        chisq_threshold="auto",
        log=Logger(suppress=True),
        ldscores=str(ld_path),
        annot=str(annot_path),
        njack="chr",
        chisq_action="drop",
    )
    legacy_results = legacy._run()

    fast_rows = dispatch_h2_batch_fast(
        _fast_args(ld_path, annot_path, sums, tmp_path / "fast"),
        Logger(suppress=True),
    )
    fast_results = [fit for _, _, fit in fast_rows]

    assert len(fast_results) == len(legacy_results) == 5
    for observed, expected in zip(fast_results, legacy_results):
        np.testing.assert_allclose(observed.h2, expected.h2, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.sigmas, expected.sigmas, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.enrich, expected.enrich, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.tau, expected.tau, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.tau_star, expected.tau_star, rtol=2e-11, atol=2e-11)

    legacy_table = pd.read_csv(tmp_path / "legacy.results.tsv", sep="\t")
    fast_table = pd.read_csv(tmp_path / "fast.results.tsv", sep="\t")
    pd.testing.assert_frame_equal(
        fast_table,
        legacy_table,
        check_exact=False,
        rtol=2e-11,
        atol=2e-11,
    )


def test_fast_batch_rejects_post_drop_block_jackknife(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=2)
    args = _fast_args(ld_path, annot_path, sums, tmp_path / "fast", njack="20")
    with pytest.raises(ValueError, match="requires chromosome jackknife"):
        dispatch_h2_batch_fast(args, Logger(suppress=True))


@pytest.mark.parametrize("explicit_cov_rank", [None, "5,5"])
def test_stream_reader_matches_legacy_with_cov_rank_and_off_trace_nmax(
    tmp_path,
    explicit_cov_rank,
):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=2)
    for path in sorted(sums.glob("*.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame["COV_RANK"] = 3
        off_trace = frame.iloc[[0]].copy()
        off_trace["SNP"] = "99:999999"
        off_trace["N"] = 9_500
        frame = pd.concat([frame, off_trace], ignore_index=True)
        frame.to_csv(path, sep="\t", index=False)

    legacy = Sumrhe(
        h2_path=str(sums),
        out=str(tmp_path / "legacy_cov_rank"),
        chisq_threshold="auto",
        log=Logger(suppress=True),
        ldscores=str(ld_path),
        annot=str(annot_path),
        njack="chr",
        chisq_action="drop",
        cov_rank=explicit_cov_rank,
    )._run()

    args = _fast_args(ld_path, annot_path, sums, tmp_path / "fast_cov_rank")
    args.cov_rank = explicit_cov_rank
    fast = [fit for _, _, fit in dispatch_h2_batch_fast(args, Logger(suppress=True))]
    for observed, expected in zip(fast, legacy):
        np.testing.assert_allclose(observed.h2, expected.h2, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.sigmas, expected.sigmas, rtol=2e-11, atol=2e-11)


def test_fast_batch_cache_roundtrip_and_source_invalidation(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=3)
    cache = tmp_path / "cache"

    write_args = _fast_args(ld_path, annot_path, sums, tmp_path / "write")
    write_args.h2_cache_dir = str(cache)
    write_args.h2_cache_verify_checksum = True
    written = dispatch_h2_batch_fast(write_args, Logger(suppress=True))

    metadata = sorted(cache.glob("*/*.json"))
    y_files = sorted(cache.glob("*/*.y.npy"))
    active_files = sorted(cache.glob("*/*.active.packbits.npy"))
    assert len(metadata) == len(y_files) == len(active_files) == 3

    read_args = _fast_args(ld_path, annot_path, sums, tmp_path / "read")
    read_args.h2_cache_dir = str(cache)
    read_args.h2_cache_mode = "read"
    read_args.h2_cache_verify_checksum = True
    read = dispatch_h2_batch_fast(read_args, Logger(suppress=True))

    for (_, _, observed), (_, _, expected) in zip(read, written):
        np.testing.assert_allclose(observed.h2, expected.h2, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(observed.sigmas, expected.sigmas, rtol=0.0, atol=0.0)

    changed = sums / "trait_0.tsv"
    changed.touch()
    invalidated = _fast_args(ld_path, annot_path, sums, tmp_path / "invalidated")
    invalidated.h2_cache_dir = str(cache)
    invalidated.h2_cache_mode = "read"
    with pytest.raises(RuntimeError, match="cache entry is missing or invalid"):
        dispatch_h2_batch_fast(invalidated, Logger(suppress=True))


def test_cache_checksum_detects_array_corruption(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=1)
    cache = tmp_path / "cache"
    write_args = _fast_args(ld_path, annot_path, sums, tmp_path / "write")
    write_args.h2_cache_dir = str(cache)
    dispatch_h2_batch_fast(write_args, Logger(suppress=True))

    y_path = next(cache.glob("*/*.y.npy"))
    y = np.load(y_path, mmap_mode="r+")
    y[0] += 1.0
    y.flush()

    read_args = _fast_args(ld_path, annot_path, sums, tmp_path / "read")
    read_args.h2_cache_dir = str(cache)
    read_args.h2_cache_mode = "read"
    read_args.h2_cache_verify_checksum = True
    with pytest.raises(RuntimeError, match="Cached y checksum failed"):
        dispatch_h2_batch_fast(read_args, Logger(suppress=True))


def test_trace_axis_digest_is_process_independent_for_object_arrays():
    values = np.asarray(["1:101:A:G", "1:200:C:T", "22:999999:G:A"], dtype=object)
    expected = trace_axis_digest(values)
    code = (
        "import numpy as np; "
        "from summit.inference.h2_cache import trace_axis_digest; "
        "print(trace_axis_digest(np.asarray(['1:101:A:G','1:200:C:T','22:999999:G:A'], dtype=object)))"
    )
    observed = [
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        for _ in range(2)
    ]
    assert observed == [expected, expected]


def test_cache_only_builds_entries_without_results(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=2)
    args = _fast_args(ld_path, annot_path, sums, tmp_path / "cache_only")
    args.h2_cache_dir = str(tmp_path / "cache")
    args.h2_cache_only = True
    assert dispatch_h2_batch_fast(args, Logger(suppress=True)) == []
    assert len(list((tmp_path / "cache").glob("*/*.json"))) == 2
    assert not (tmp_path / "cache_only.results.tsv").exists()


def test_chromosome_split_directory_streaming_matches_legacy(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=3)
    split = tmp_path / "split"
    for chrom in range(1, 4):
        destination = split / f"chr{chrom:02d}"
        destination.mkdir(parents=True)
        for source in sorted(sums.glob("*.tsv")):
            frame = pd.read_csv(source, sep="\t")
            frame.loc[frame["SNP"].str.startswith(f"{chrom}:")].to_csv(
                destination / source.name,
                sep="\t",
                index=False,
            )

    split_spec = str(split / "chr@")
    legacy = Sumrhe(
        h2_path=split_spec,
        out=str(tmp_path / "split_legacy"),
        chisq_threshold="auto",
        log=Logger(suppress=True),
        ldscores=str(ld_path),
        annot=str(annot_path),
        njack="chr",
        chisq_action="drop",
    )
    legacy_results = legacy._run()

    args = _fast_args(ld_path, annot_path, split_spec, tmp_path / "split_fast")
    fast_results = [fit for _, _, fit in dispatch_h2_batch_fast(args, Logger(suppress=True))]
    for observed, expected in zip(fast_results, legacy_results):
        np.testing.assert_allclose(observed.h2, expected.h2, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(observed.sigmas, expected.sigmas, rtol=2e-11, atol=2e-11)


def test_fast_batch_cli_dispatch(tmp_path):
    ld_path, annot_path, sums = _write_fixture(tmp_path, n_traits=2)
    out = tmp_path / "cli_fast"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "summit.cli",
            "--h2",
            str(sums),
            "--h2-batch-fast",
            "--h2-workers",
            "2",
            "--h2-batch-size",
            "2",
            "--ldscores",
            str(ld_path),
            "--annot",
            str(annot_path),
            "--out",
            str(out),
            "--njack",
            "chr",
            "--max-chisq",
            "auto",
            "--num-threads",
            "2",
            "--suppress",
        ],
        check=True,
    )
    table = pd.read_csv(str(out) + ".results.tsv", sep="\t")
    assert table.shape[0] == 2
    assert np.isfinite(table[["h2", "h2_se"]].to_numpy()).all()
