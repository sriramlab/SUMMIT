from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_outer_numactl_sentinel_prevents_nested_cli_reexec(monkeypatch):
    from summit.ldscore import gw_ldscore

    monkeypatch.setenv("SUMMIT_NUMACTL_WRAPPED", "1")
    monkeypatch.setattr(gw_ldscore.shutil, "which", lambda name: "/usr/bin/numactl")

    def forbidden_exec(*_):
        raise AssertionError("sealed outer NUMA launch attempted a nested re-exec")

    monkeypatch.setattr(gw_ldscore.os, "execv", forbidden_exec)
    threads = gw_ldscore.apply_env(
        {
            "numa_mode": "interleave",
            "numa_nodes": "all",
            "force_affinity_all": False,
            "num_threads": 1,
            "decode_threads_cap": 1,
        }
    )
    assert isinstance(threads, int) and threads >= 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["SUMMIT_NUMACTL_WRAPPED"] == "1"


def test_gxe_cli_creates_new_log_then_refuses_existing_prefix(tmp_path, monkeypatch):
    from summit import cli

    calls = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    prefix = tmp_path / "run"

    def dispatch(*_):
        calls.append("dispatch")
        (tmp_path / "run.gxe.ref.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_dispatch_ldscore", dispatch)
    argv = [
        "summit",
        "--geno", str(tmp_path / "geno"),
        "--env", str(tmp_path / "env.tsv"),
        "--out", str(prefix),
        "--suppress",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == ["dispatch"]
    log_path = tmp_path / "run.gxe.log"
    assert log_path.is_file()
    assert (log_path.stat().st_mode & 0o777) == 0o600
    original = log_path.read_bytes()

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == ["dispatch"]
    assert log_path.read_bytes() == original

    monkeypatch.setattr(sys, "argv", [*argv, "--gxe-overwrite"])
    cli.main()
    assert calls == ["dispatch", "dispatch"]


def test_failed_gxe_cli_attempt_can_retry_with_existing_log(tmp_path, monkeypatch):
    from summit import cli

    calls = 0
    monkeypatch.setattr(cli, "apply_env", lambda _: None)

    def dispatch(*_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(cli, "_dispatch_ldscore", dispatch)
    prefix = tmp_path / "retry"
    argv = [
        "summit",
        "--geno", str(tmp_path / "geno"),
        "--env", str(tmp_path / "env.tsv"),
        "--out", str(prefix),
        "--suppress",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="injected failure"):
        cli.main()
    assert Path(str(prefix) + ".gxe.log").is_file()

    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == 2


def test_common_cohort_multi_environment_cli_dispatches_once(tmp_path, monkeypatch):
    from summit import cli

    observed = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_dispatch_gxe_multi_reference",
        lambda args, *_: observed.append(args.gxe_env_cols),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi", "--out", str(tmp_path / "multi"),
            "--suppress",
        ],
    )
    cli.main()
    assert observed == ["age,bmi"]


def test_population_reference_cli_dispatches_one_trait_to_scalar_scorer(
    tmp_path, monkeypatch
):
    from summit import cli
    from summit.logger import Logger

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            gwas=Path("height.gxe.gwas.tsv.gz"),
            gwis=Path("height.gxe.gwis.tsv.gz"),
            moments=Path("height.gxe.moments.json"),
        )

    monkeypatch.setattr(cli, "score_phenotype_from_reference", capture)

    def forbidden_wide(**_):
        raise AssertionError("population-reference mode dispatched the wide scorer")

    monkeypatch.setattr(cli, "score_phenotypes_from_reference", forbidden_wide)
    args = cli.build_parser().parse_args(
        [
            "--geno", "study",
            "--env", "age.tsv",
            "--covar", "covariates.tsv",
            "--gxe-pheno", "phenotype.tsv",
            "--gxe-pheno-col", "height",
            "--gxe-score-reference", "reference.gxe.ref.json",
            "--gxe-population-reference",
            "--out", str(tmp_path / "height"),
        ]
    )
    cli._dispatch_gxe_score(args, Logger(suppress=True))

    assert observed["pheno_col"] == "height"
    assert observed["population_transfer"] is True
    assert observed["reference_manifest"] == "reference.gxe.ref.json"


@pytest.mark.parametrize(
    ("extra", "dispatch_name"),
    [
        (["--geno", "geno", "--env", "env", "--_gxe-build-cache"], "_dispatch_gxe_cache"),
        (
            [
                "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
                "--gxe-score-reference", "reference.json",
            ],
            "_dispatch_gxe_score",
        ),
        (
            [
                "--_gxe-merge-shards", "shard-0.json", "shard-1.json",
                "--_gxe-feature-cache", "features.npz",
            ],
            "_dispatch_gxe_merge",
        ),
        (["--gxe-fit-batch", "batch.json"], "_dispatch_gxe_fit_batch"),
    ],
)
def test_gxe_reusable_workflow_modes_dispatch_once(
    tmp_path, monkeypatch, extra, dispatch_name
):
    from summit import cli

    calls = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    for name in (
        "_dispatch_gxe_cache",
        "_dispatch_gxe_score",
        "_dispatch_gxe_merge",
        "_dispatch_gxe_fit_batch",
    ):
        monkeypatch.setattr(
            cli,
            name,
            lambda *args, _name=name: calls.append(_name),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["summit", *extra, "--out", str(tmp_path / dispatch_name), "--suppress"],
    )
    cli.main()
    assert calls == [dispatch_name]


def test_private_gxe_reference_shard_reaches_generator_with_probe_contract(tmp_path, monkeypatch):
    from summit import cli

    observed = {}
    monkeypatch.setattr(cli, "apply_env", lambda _: None)

    def capture(args, *_):
        observed.update(
            shard=args._gxe_reference_shard,
            cache=args._gxe_feature_cache,
            offset=args._gxe_probe_offset,
            probes=args.nvecs,
        )

    monkeypatch.setattr(cli, "_dispatch_ldscore", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "env",
            "--_gxe-reference-shard", "--_gxe-feature-cache", "features.npz",
            "--_gxe-probe-offset", "30", "--nvecs", "10",
            "--out", str(tmp_path / "shard"), "--suppress",
        ],
    )
    cli.main()
    assert observed == {
        "shard": True,
        "cache": "features.npz",
        "offset": 30,
        "probes": 10,
    }


def test_private_gxe_reference_shard_reserves_identity_sidecar(tmp_path, monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    called = False

    def capture(*_):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_dispatch_ldscore", capture)
    prefix = tmp_path / "reserved"
    Path(str(prefix) + ".gxe.shard.identity.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "env",
            "--_gxe-reference-shard", "--_gxe-feature-cache", "features.npz",
            "--nvecs", "10", "--out", str(prefix), "--suppress",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert not called


def test_gxe_reusable_mode_validation_rejects_ambiguous_or_unsafe_calls(tmp_path, monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    cases = [
        [
            "--geno", "geno", "--env", "env", "--_gxe-build-cache",
            "--gxe-score-reference", "reference.json", "--gxe-pheno", "traits.tsv",
        ],
        ["--geno", "geno", "--env", "env", "--_gxe-reference-shard"],
        ["--_gxe-merge-shards", "shard.json"],
        [
            "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
            "--gxe-score-reference", "reference.json", "--gxe-overwrite",
        ],
        ["--gxe-fit", "reference.json", "--gxe-fit-batch", "batch.json"],
        ["--gxe-fit-batch", "batch.json", "--gxe-moments", "moments.json"],
        ["--gxe-fit-batch", "batch.json", "--_gxe-feature-cache", "cache.npz"],
        ["--gxe-fit-batch", "batch.json", "--covar", "covariates.tsv"],
        ["--gxe-fit-batch", "batch.json", "--annot", "annotations.tsv"],
        ["--gxe-fit-batch", "batch.json", "--gxe-kernel-mode", "genie"],
        ["--gxe-fit-batch=batch.json", "--gxe-kernel-mode=standardized"],
        ["--gxe-fit-batch", "batch.json", "--_gxe-feature-c", "cache.npz"],
        ["--gxe-fit-batch", "batch.json", "--gxe-kernel-m", "genie"],
        ["--gxe-fit-batch", "batch.json", "--ann", "annotations.tsv"],
        ["--gxe-fit-batch", "batch.json", "--gxe-overwrite"],
    ]
    for index, extra in enumerate(cases):
        monkeypatch.setattr(
            sys,
            "argv",
            ["summit", *extra, "--out", str(tmp_path / f"bad-{index}"), "--suppress"],
        )
        with pytest.raises(SystemExit):
            cli.main()


@pytest.mark.parametrize(
    "removed_option",
    (
        "--gxe-build-cache",
        "--gxe-feature-cache",
        "--gxe-jackknife-scratch-gib",
        "--gxe-reference-shard",
        "--gxe-probe-offset",
        "--gxe-merge-shards",
    ),
)
def test_gxe_cache_and_shard_controls_are_not_public_cli_options(removed_option):
    from summit import cli

    parser = cli.build_parser()
    help_text = parser.format_help()
    assert removed_option not in help_text
    assert "--_gxe-" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args([removed_option])
