from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import subprocess
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_PATH = ROOT / "scripts" / "gxe" / "hoffman" / "hoffman_deploy.py"
CONFIG_PATH = ROOT / "scripts" / "gxe" / "hoffman" / "deployment_config.json"


def _load_deploy():
    spec = importlib.util.spec_from_file_location(
        "summit_hoffman_deploy_test", DEPLOY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEPLOY = _load_deploy()


def _private_dir(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _private_file(path: Path, text: str = "x") -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def _config_for_scratch(scratch: Path) -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["scratch_root"] = str(scratch)
    return payload


def test_deployment_config_and_wrappers_are_fail_closed():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["schema_version"] == DEPLOY.DEPLOYMENT_CONFIG_SCHEMA_VERSION == 2
    assert DEPLOY.FROZEN_CODE_MANIFEST_SCHEMA_VERSION == 2
    numa = DEPLOY._validate_numa_launch(config, verify_executable=False)
    assert str(numa["executable"]) == "/usr/bin/numactl"
    assert numa["arguments"] == ("--interleave=all",)
    estimator = config["estimator"]
    assert estimator["annotation"] is None
    assert estimator["annotation_contract"] == "all_variants_unit_weight"
    assert estimator["feature_cache_schema_version"] == 2
    assert estimator["reference_shard_schema_version"] == 2
    assert estimator["reference_schema_version"] == 3
    assert estimator["kernel_mode"] == "standardized"
    assert estimator["genotype_scale"] == "sample"
    assert config["monitor_interval_hours"] == 6
    for task in DEPLOY.TASKS:
        resource = DEPLOY._validate_resource(task, config)
        assert resource["total_memory_gib"] == (
            resource["slots"] * resource["h_data_gib_per_slot"]
        )
        wrapper = ROOT / "scripts" / "gxe" / "hoffman" / DEPLOY.WRAPPERS[task]
        text = wrapper.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "umask 077" in text
        assert " -I -B " in text
        assert f"--task {task}" in text
        assert "-tc" not in text
    benchmark = DEPLOY._validate_resource("shard_benchmark", config)
    assert benchmark["slots"] == 8
    assert benchmark["h_data_gib_per_slot"] == 4
    assert benchmark["total_memory_gib"] == 32


def test_generation_args_are_exact_and_annotation_free(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    args = DEPLOY._common_generation_args(
        config,
        tmp_path / "geno",
        {"environment": tmp_path / "env", "covariates": tmp_path / "covar"},
        slots=4,
    )
    assert "--annot" not in args
    assert args[args.index("--gxe-kernel-mode") + 1] == "standardized"
    assert args[args.index("--gxe-genotype-scale") + 1] == "sample"
    assert args[args.index("--njack") + 1] == "100"
    assert args[args.index("--seed") + 1] == "20260808"
    assert args[args.index("--ddof") + 1] == "1"
    assert "--gxe-missing-values=-9,NA,NaN,nan,.,None,null" in args
    assert args[args.index("--num-threads") + 1] == "4"


def test_current_cache_shard_merge_source_contract_is_2_2_3():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    DEPLOY._check_cache_merge_compatibility_sources(ROOT, config["estimator"])


def test_private_path_guard_rejects_hardlinks_symlinks_and_wrong_modes(
    tmp_path: Path,
):
    root = _private_dir(tmp_path / "root")
    good = _private_file(root / "good")
    DEPLOY._require_file(good, "good", private=True)

    hardlink = root / "hardlink"
    os.link(good, hardlink)
    with pytest.raises(PermissionError, match="hard link"):
        DEPLOY._require_file(good, "hardlinked file", private=True)

    other = _private_file(root / "other")
    symlink = root / "symlink"
    symlink.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        DEPLOY._require_file(symlink, "symlink file", private=True)

    wrong = _private_file(root / "wrong")
    wrong.chmod(0o640)
    with pytest.raises(PermissionError, match="0600"):
        DEPLOY._require_file(wrong, "wrong-mode file", private=True)

    outside = _private_file(tmp_path / "outside")
    with pytest.raises(ValueError, match="below scratch"):
        DEPLOY._scratch_path(outside, root, "outside", kind="file")

    nonprivate_parent = root / "nonprivate"
    nonprivate_parent.mkdir(mode=0o750)
    nonprivate_parent.chmod(0o750)
    nested = _private_file(nonprivate_parent / "nested")
    with pytest.raises(PermissionError, match="0700"):
        DEPLOY._verify_record(
            DEPLOY._record(nested),
            "nested input",
            scratch_root=root,
            private=True,
        )


def test_uge_environment_rejects_arrays_and_slot_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    common = {
        "resource": {"slots": 4},
        "frozen": {"python": tmp_path / "env" / "bin" / "python"},
        "tmp": tmp_path / "scratch" / "tmp",
        "numa_launch": {
            "executable_record": {"path": "/usr/bin/numactl"},
            "arguments": ("--interleave=all",),
        },
    }
    monkeypatch.setattr(
        DEPLOY,
        "sys",
        types.SimpleNamespace(
            flags=types.SimpleNamespace(
                isolated=1,
                no_user_site=1,
                dont_write_bytecode=1,
                hash_randomization=1,
            )
        ),
    )
    monkeypatch.setenv("JOB_ID", "1234")
    monkeypatch.setenv("NSLOTS", "4")
    for name, value in DEPLOY._bootstrap_environment(common).items():
        monkeypatch.setenv(name, value)
    assert DEPLOY._runtime_uge_environment(common)["job_id"] == 1234
    monkeypatch.delenv("SUMMIT_NUMACTL_WRAPPED")
    with pytest.raises(RuntimeError, match="sealed job bootstrap"):
        DEPLOY._runtime_uge_environment(common)
    monkeypatch.setenv("SUMMIT_NUMACTL_WRAPPED", "1")
    monkeypatch.setenv("SGE_TASK_ID", "7")
    with pytest.raises(RuntimeError, match="array"):
        DEPLOY._runtime_uge_environment(common)
    monkeypatch.delenv("SGE_TASK_ID")
    monkeypatch.setenv("NSLOTS", "8")
    with pytest.raises(RuntimeError, match="NSLOTS"):
        DEPLOY._runtime_uge_environment(common)


def test_uge_environment_requires_isolated_no_bytecode_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    common = {
        "resource": {"slots": 1},
        "frozen": {"python": tmp_path / "env" / "bin" / "python"},
        "tmp": tmp_path / "scratch" / "tmp",
        "numa_launch": {
            "executable_record": {"path": "/usr/bin/numactl"},
            "arguments": ("--interleave=all",),
        },
    }
    monkeypatch.setenv("JOB_ID", "1234")
    monkeypatch.setenv("NSLOTS", "1")
    for name, value in DEPLOY._bootstrap_environment(common).items():
        monkeypatch.setenv(name, value)
    for isolated, no_user_site, no_bytecode in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        monkeypatch.setattr(
            DEPLOY,
            "sys",
            types.SimpleNamespace(
                flags=types.SimpleNamespace(
                    isolated=isolated,
                    no_user_site=no_user_site,
                    dont_write_bytecode=no_bytecode,
                    hash_randomization=1,
                )
            ),
        )
        with pytest.raises(RuntimeError, match="-I -B"):
            DEPLOY._runtime_uge_environment(common)


def test_embedded_cli_suppresses_numactl_reexec_and_restores_process_state(
    monkeypatch: pytest.MonkeyPatch,
):
    original_argv = ["deployment-runner", "--sealed"]
    monkeypatch.setattr(DEPLOY.sys, "argv", original_argv)
    monkeypatch.setenv("SUMMIT_NUMACTL_WRAPPED", "1")
    observed = {}

    class FakeCli:
        @staticmethod
        def main():
            observed["argv"] = list(DEPLOY.sys.argv)
            observed["sentinel"] = os.environ.get("SUMMIT_NUMACTL_WRAPPED")
            if observed["sentinel"] != "1":
                raise AssertionError("embedded CLI would re-exec through numactl")

    DEPLOY._invoke_summit_cli(FakeCli, ["--gxe-build-cache", "--out", "x"])
    assert observed == {
        "argv": ["summit", "--gxe-build-cache", "--out", "x"],
        "sentinel": "1",
    }
    assert DEPLOY.sys.argv is original_argv
    assert os.environ["SUMMIT_NUMACTL_WRAPPED"] == "1"


def test_distribution_fingerprint_binds_recorded_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_file = tmp_path / "module.py"
    package_file.write_text("value = 1\n", encoding="utf-8")

    class FakeDistribution:
        version = "1.2.3"
        files = [Path("module.py")]

        @staticmethod
        def locate_file(relative: Path) -> Path:
            return tmp_path / relative

    monkeypatch.setattr(
        DEPLOY.importlib_metadata,
        "distribution",
        lambda unused_name: FakeDistribution(),
    )
    before = DEPLOY._distribution_fingerprint("example")
    package_file.write_text("value = 2\n", encoding="utf-8")
    after = DEPLOY._distribution_fingerprint("example")
    assert before["version"] == after["version"] == "1.2.3"
    assert before["content_sha256"] != after["content_sha256"]


def test_renderer_emits_private_nonarray_per_slot_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    scratch = _private_dir(tmp_path / "scratch")
    job_root = _private_dir(scratch / "job")
    spec_path = _private_file(scratch / "spec.json", "{}\n")
    config_path = _private_file(tmp_path / "deployment.json", "{}\n")
    wrapper = tmp_path / "uge_cache.sh"
    wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    wrapper.chmod(0o755)
    numactl = tmp_path / "numactl"
    numactl.write_text("#!/bin/bash\n", encoding="utf-8")
    numactl.chmod(0o755)
    python = Path(os.environ.get("PYTHON", os.sys.executable))
    config = _config_for_scratch(scratch)
    spec = {
        "task": "cache",
        "job_name": "gxe_cache_test",
        "job_root": str(job_root),
    }
    common = {
        "job_name": spec["job_name"],
        "job_root": job_root,
        "frozen": {"python": python, "wrapper": wrapper},
        "resource": config["resources"]["cache"],
        "resource_profile": "cache",
        "scratch_root": scratch,
        "artifacts": job_root / "artifacts",
        "tmp": job_root / "tmp",
        "numa_launch": {
            "executable": numactl,
            "executable_record": DEPLOY._record(numactl),
            "arguments": ("--interleave=all",),
            "environment_sentinel": "SUMMIT_NUMACTL_WRAPPED",
        },
    }
    monkeypatch.setattr(
        DEPLOY, "_load_config", lambda *unused: (config_path, config, "a" * 64)
    )
    monkeypatch.setattr(
        DEPLOY, "_load_spec", lambda *unused: (spec_path, spec, "b" * 64)
    )
    monkeypatch.setattr(DEPLOY, "_validate_common", lambda *unused, **kwargs: common)
    monkeypatch.setattr(
        DEPLOY,
        "_prepare_task",
        lambda *unused, **kwargs: {"outputs": [], "command": [], "details": {}},
    )
    DEPLOY._render_job(argparse.Namespace(config="x", job_spec="y"))
    script = (job_root / "job.sh").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(job_root / "job.sh")], check=True)
    assert "#$ -pe shared 4" in script
    assert "#$ -l h_data=6G,h_rt=48:00:00,highp" in script
    assert "-tc" not in script and "SGE_TASK_ID" not in script
    assert "export OMP_NUM_THREADS=4" in script
    assert "export SUMMIT_NUMACTL_WRAPPED=1" in script
    assert f"exec {numactl} --interleave=all {wrapper}" in script
    assert f"export TMPDIR={job_root / 'tmp'}" in script
    assert stat.S_IMODE((job_root / "job.sh").stat().st_mode) == 0o700
    assert stat.S_IMODE((job_root / "stdout.log").stat().st_mode) == 0o600
    assert stat.S_IMODE((job_root / "tmp").stat().st_mode) == 0o700
    assert capsys.readouterr().out.strip() == f"qsub {job_root / 'job.sh'}"


def test_merge_plan_enforces_contiguous_prefix_and_low_probe_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scratch = _private_dir(tmp_path / "scratch")
    job = _private_dir(scratch / "job")
    cache_path = _private_file(scratch / "cache.npz")
    cache_record = DEPLOY._record(cache_path)
    shard_paths = [_private_file(scratch / f"shard{i}.json") for i in range(10)]
    shard_records = [DEPLOY._record(path) for path in shard_paths]
    config = _config_for_scratch(scratch)
    common = {
        "scratch_root": scratch,
        "artifacts": job / "artifacts",
        "frozen": {"panel_payload": {"groups": {}}},
    }
    monkeypatch.setattr(DEPLOY, "_validate_cache_file", lambda *unused: {})

    def validate_shard(
        record, unused_root, unused_config, unused_cache_sha, unused_cache
    ):
        index = shard_records.index(record)
        return (
            shard_paths[index],
            {
                "randomization": {
                    "probe_offset": 10 * index,
                    "probe_stop": 10 * (index + 1),
                }
            },
            {},
        )

    monkeypatch.setattr(DEPLOY, "_validate_shard_record", validate_shard)

    for count in (1, 10):
        records = shard_records[:count]
        dependencies = [
            {
                "outputs": [record],
                "task_details": {"role": "production"},
            }
            for record in records
        ]
        monkeypatch.setattr(
            DEPLOY,
            "_validate_qacct_dependencies",
            lambda *unused, expected_count=None, deps=dependencies, **kwargs: deps,
        )
        spec = {
            "task": "merge",
            "task_args": {"cache": cache_record, "shards": records},
        }
        plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
        assert plan["details"]["probes"] == count * 10
        assert ("--allow-low-probe-gxe-jackknife" in plan["command"]) is (count < 10)


def test_qacct_parser_requires_one_record():
    raw = """==============================================================
jobnumber    123
failed       0
exit_status  0
slots        4
wallclock    01:02:03
cpu          00:50:00
maxvmem      12.5G
"""
    parsed = DEPLOY._parse_qacct(raw)
    assert parsed["jobnumber"] == "123"
    with pytest.raises(ValueError, match="exactly one"):
        DEPLOY._parse_qacct(raw + raw.replace("123", "124"))


def test_record_qacct_seals_success_and_refuses_overwrite(tmp_path: Path):
    scratch = _private_dir(tmp_path / "scratch")
    job = _private_dir(scratch / "job")
    artifacts = _private_dir(job / "artifacts")
    output = _private_file(artifacts / "result")
    invocation = _private_file(job / "invocation.json", "{}\n")
    attempt = _private_file(job / "attempt.json", "{}\n")
    attempt_lock = _private_file(job / "attempt.lock", "")
    _private_file(job / "stdout.log", "stdout\n")
    _private_file(job / "stderr.log", "stderr\n")
    job_script = job / "job.sh"
    job_script.write_text("#!/bin/bash\n", encoding="utf-8")
    job_script.chmod(0o700)
    spec_file = _private_file(scratch / "job_spec.json", "{}\n")
    code_manifest = _private_file(tmp_path / "code_manifest.json", "{}\n")
    config = _config_for_scratch(scratch)
    config_file = _private_file(
        tmp_path / "deployment.json", json.dumps(config, sort_keys=True) + "\n"
    )
    config_sha = DEPLOY._sha256(config_file)
    receipt = {
        "kind": "summit.gxe.hoffman_process_receipt",
        "schema_version": 1,
        "task": "cache",
        "job_id": 123,
        "deployment_config": DEPLOY._record(config_file),
        "job_spec": DEPLOY._record(spec_file),
        "job_script": DEPLOY._record(job_script),
        "attempt_lock": DEPLOY._record(attempt_lock),
        "attempt": DEPLOY._record(attempt),
        "code_manifest": DEPLOY._record(code_manifest),
        "invocation": DEPLOY._record(invocation),
        "resource_profile": "cache",
        "resource": config["resources"]["cache"],
        "outputs": [DEPLOY._record(output)],
        "task_details": {"group": "age_bp"},
        "qacct_pending": True,
    }
    receipt_path = _private_file(
        job / "process_receipt.json", json.dumps(receipt, sort_keys=True) + "\n"
    )
    qacct = tmp_path / "qacct"
    qacct.write_text(
        "#!/bin/bash\n"
        "cat <<'EOF'\n"
        "==============================================================\n"
        "jobnumber 123\nfailed 0\nexit_status 0\nslots 4\n"
        "wallclock 01:02:03\ncpu 00:50:00\nmaxvmem 12.5G\nEOF\n",
        encoding="utf-8",
    )
    qacct.chmod(0o700)
    args = argparse.Namespace(
        config=str(config_file),
        expected_config_sha256=config_sha,
        receipt=str(receipt_path),
        receipt_sha256=DEPLOY._sha256(receipt_path),
        receipt_bytes=receipt_path.stat().st_size,
        qacct_command=str(qacct),
    )
    DEPLOY._record_qacct(args)
    completed = json.loads((job / "completed_qacct.json").read_text(encoding="utf-8"))
    assert completed["failed"] == 0 and completed["exit_status"] == 0
    assert completed["slots"] == 4
    assert completed["receipt_sha256"] == DEPLOY._sha256(receipt_path)
    assert completed["outputs"] == receipt["outputs"]
    assert completed["qacct_raw"] == DEPLOY._record(job / "qacct.txt")
    assert completed["uge_logs"]["stdout"] == DEPLOY._record(job / "stdout.log")
    with pytest.raises(FileExistsError, match="existing"):
        DEPLOY._record_qacct(args)
