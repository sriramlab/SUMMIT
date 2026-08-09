from __future__ import annotations

import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    ("extra", "dispatch_name"),
    [
        (["--geno", "geno", "--env", "env", "--gxe-build-cache"], "_dispatch_gxe_cache"),
        (
            [
                "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
                "--gxe-score-reference", "reference.json",
            ],
            "_dispatch_gxe_score",
        ),
        (
            [
                "--gxe-merge-shards", "shard-0.json", "shard-1.json",
                "--gxe-feature-cache", "features.npz",
            ],
            "_dispatch_gxe_merge",
        ),
    ],
)
def test_gxe_reusable_workflow_modes_dispatch_once(
    tmp_path, monkeypatch, extra, dispatch_name
):
    from summit import cli

    calls = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    for name in ("_dispatch_gxe_cache", "_dispatch_gxe_score", "_dispatch_gxe_merge"):
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


def test_gxe_reference_shard_reaches_generator_with_probe_contract(tmp_path, monkeypatch):
    from summit import cli

    observed = {}
    monkeypatch.setattr(cli, "apply_env", lambda _: None)

    def capture(args, *_):
        observed.update(
            shard=args.gxe_reference_shard,
            cache=args.gxe_feature_cache,
            offset=args.gxe_probe_offset,
            probes=args.nvecs,
        )

    monkeypatch.setattr(cli, "_dispatch_ldscore", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "env",
            "--gxe-reference-shard", "--gxe-feature-cache", "features.npz",
            "--gxe-probe-offset", "30", "--nvecs", "10",
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


def test_gxe_reference_shard_reserves_identity_sidecar(tmp_path, monkeypatch):
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
            "--gxe-reference-shard", "--gxe-feature-cache", "features.npz",
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
            "--geno", "geno", "--env", "env", "--gxe-build-cache",
            "--gxe-score-reference", "reference.json", "--gxe-pheno", "traits.tsv",
        ],
        ["--geno", "geno", "--env", "env", "--gxe-reference-shard"],
        ["--gxe-merge-shards", "shard.json"],
        [
            "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
            "--gxe-score-reference", "reference.json", "--gxe-overwrite",
        ],
    ]
    for index, extra in enumerate(cases):
        monkeypatch.setattr(
            sys,
            "argv",
            ["summit", *extra, "--out", str(tmp_path / f"bad-{index}"), "--suppress"],
        )
        with pytest.raises(SystemExit):
            cli.main()
