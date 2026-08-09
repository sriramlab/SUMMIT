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
PANEL_PATH = ROOT / "scripts" / "gxe" / "hoffman" / "panel_config.json"


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
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
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
    assert estimator["production_probes"] == 100
    assert estimator["probes_per_shard"] == 50
    assert DEPLOY._production_shard_count(estimator) == 2
    assert config["panel_config_sha256"] == DEPLOY._sha256(PANEL_PATH)
    assert panel["production_estimator"]["probe_shards"] == 2
    assert panel["production_estimator"]["probes_per_shard"] == 50
    assert panel["uge"]["production_b50_shard"] == {
        "slots": 4,
        "h_data_per_slot": "8G",
        "total_memory": "32G",
        "h_rt": "48:00:00",
        "highp": True,
    }
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
    production_shard = DEPLOY._validate_resource("shard", config)
    assert production_shard["slots"] == 4
    assert production_shard["h_data_gib_per_slot"] == 8
    assert production_shard["total_memory_gib"] == 32
    assert production_shard["h_rt"] == "48:00:00"


def test_b50_production_probe_intervals_and_dynamic_index_limit():
    estimator = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["estimator"]
    assert DEPLOY._production_probe_interval(0, estimator) == (0, 50)
    assert DEPLOY._production_probe_interval(1, estimator) == (50, 100)
    with pytest.raises(ValueError, match="0 through 1"):
        DEPLOY._production_probe_interval(2, estimator)
    with pytest.raises(ValueError, match="0 through 1"):
        DEPLOY._production_probe_interval(True, estimator)

    malformed = {**estimator, "probes_per_shard": 60}
    with pytest.raises(ValueError, match="exactly divisible"):
        DEPLOY._production_shard_count(malformed)


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


def _fit_batch_plan_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, dict, dict, list[dict]]:
    scratch = _private_dir(tmp_path / "scratch")
    job = _private_dir(scratch / "job")
    cache = _private_file(scratch / "cache.gxe.cache.npz")
    reference = _private_file(scratch / "B100.gxe.ref.json")
    traits = ["bp_diastolic", "bp_systolic"]
    triplets = []
    for trait in traits:
        triplets.append(
            {
                "trait": trait,
                "moments": DEPLOY._record(
                    _private_file(scratch / f"scores.{trait}.gxe.moments.json")
                ),
                "gwas": DEPLOY._record(
                    _private_file(scratch / f"scores.{trait}.gxe.gwas.tsv.gz")
                ),
                "gwis": DEPLOY._record(
                    _private_file(scratch / f"scores.{trait}.gxe.gwis.tsv.gz")
                ),
            }
        )
    score_log = DEPLOY._record(_private_file(scratch / "scores.gxe.log"))
    config = _config_for_scratch(scratch)
    cache_record = DEPLOY._record(cache)
    reference_record = DEPLOY._record(reference)
    score_spec = _private_file(
        scratch / "score_job_spec.json",
        json.dumps(
            {
                "kind": "summit.gxe.hoffman_job",
                "schema_version": 1,
                "task": "score",
                "task_args": {
                    "cache": cache_record,
                    "reference": reference_record,
                    "traits": traits,
                },
            },
            sort_keys=True,
        )
        + "\n",
    )
    dependency = {
        "job_spec": DEPLOY._record(score_spec),
        "task_details": {
            "group": "age_bp",
            "traits": traits,
            "cache_sha256": cache_record["sha256"],
            "reference_sha256": reference_record["sha256"],
        },
        "outputs": [
            score_log,
            *(entry[key] for entry in triplets for key in ("gwas", "gwis", "moments")),
        ],
    }
    common = {
        "scratch_root": scratch,
        "job_root": job,
        "artifacts": job / "artifacts",
        "frozen": {"panel_payload": {"groups": {"age_bp": {"phenotypes": traits}}}},
    }
    spec = {
        "task": "fit_batch",
        "task_args": {
            "group": "age_bp",
            "cache": cache_record,
            "reference": reference_record,
            "traits": triplets,
        },
    }
    monkeypatch.setattr(
        DEPLOY,
        "_validate_qacct_dependencies",
        lambda *unused, **kwargs: [dependency],
    )
    monkeypatch.setattr(DEPLOY, "_validate_cache_file", lambda *unused: {})
    monkeypatch.setattr(
        DEPLOY,
        "_validate_reference_record",
        lambda *unused, **kwargs: (reference, {"validated": True}, {}),
    )
    monkeypatch.setattr(
        DEPLOY, "_validate_reference_cache_identity", lambda *unused: None
    )
    monkeypatch.setattr(
        DEPLOY,
        "_validate_score_triplet",
        lambda *unused, trait, **kwargs: {"phenotype": trait},
    )
    return spec, common, config, dependency, triplets


def _fit_plan_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, dict, dict, dict]:
    batch_spec, common, config, dependency, triplets = _fit_batch_plan_case(
        tmp_path, monkeypatch
    )
    selected = triplets[0]
    spec = {
        "task": "fit",
        "task_args": {
            "group": batch_spec["task_args"]["group"],
            "trait": selected["trait"],
            "cache": batch_spec["task_args"]["cache"],
            "reference": batch_spec["task_args"]["reference"],
            "moments": selected["moments"],
            "gwas": selected["gwas"],
            "gwis": selected["gwis"],
        },
    }
    return spec, common, config, dependency, selected


def test_single_fit_plan_binds_exact_inputs_and_output_snapshot_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, selected = _fit_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    expected = {
        "reference_manifest": spec["task_args"]["reference"],
        "feature_cache": spec["task_args"]["cache"],
        "phenotype_moments": selected["moments"],
        "gwas": selected["gwas"],
        "gwis": selected["gwis"],
    }
    assert plan["fit_input_records"] == expected
    assert plan["details"]["input_records"] == expected

    common["artifacts"].mkdir(mode=0o700)
    for path in plan["outputs"]:
        _private_file(path)
    observed = []

    def validate(prefix, unused_config, *, expected_input_provenance):
        observed.append((prefix, expected_input_provenance))

    monkeypatch.setattr(DEPLOY, "_validate_fit_outputs", validate)
    DEPLOY._postvalidate_task(spec, common, config, plan, runtime={})
    assert observed == [(common["artifacts"] / "fit", expected)]


@pytest.mark.parametrize("role", ["cache", "reference", "moments", "gwas", "gwis"])
def test_single_fit_rejects_same_hash_alias_not_in_exact_score_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
):
    spec, common, config, _, _ = _fit_plan_case(tmp_path, monkeypatch)
    original = Path(spec["task_args"][role]["path"])
    replacement = _private_file(
        common["scratch_root"] / f"same-{role}-bytes{original.suffix}",
        original.read_text(encoding="utf-8"),
    )
    assert DEPLOY._sha256(replacement) == spec["task_args"][role]["sha256"]
    spec["task_args"][role] = DEPLOY._record(replacement)
    with pytest.raises(ValueError, match="exact"):
        DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)


def test_single_fit_rejects_aba_mutation_from_consumed_snapshot_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, _ = _fit_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    common["artifacts"].mkdir(mode=0o700)
    target = Path(plan["fit_input_records"]["gwas"]["path"])
    original = target.read_bytes()

    class AbaMutatingCli:
        @staticmethod
        def main():
            target.write_text("transient bytes consumed by SUMMIT\n", encoding="utf-8")
            target.chmod(0o600)
            provenance = dict(plan["fit_input_records"])
            provenance["gwas"] = DEPLOY._record(target)
            try:
                _write_fit_output_fixture(
                    common["artifacts"], consumed_input_provenance=provenance
                )
                _private_file(common["artifacts"] / "fit.gxe.log")
            finally:
                target.write_bytes(original)
                target.chmod(0o600)

    DEPLOY._execute_plan(plan, {"cli_module": AbaMutatingCli}, common, task="fit")
    with pytest.raises(ValueError, match="consumed-input provenance differs"):
        DEPLOY._postvalidate_task(spec, common, config, plan, runtime={})


def test_fit_batch_plan_is_complete_ordered_and_manifest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, dependency, triplets = _fit_batch_plan_case(
        tmp_path, monkeypatch
    )
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    assert plan["command"] == [
        "--gxe-fit-batch",
        str(common["job_root"] / DEPLOY.FIT_BATCH_MANIFEST_NAME),
        "--gxe-max-condition",
        "1e12",
        "--out",
        str(common["artifacts"] / "fit_batch"),
    ]
    assert plan["traits"] == ["bp_diastolic", "bp_systolic"]
    assert len(plan["outputs"]) == 5
    assert plan["outputs"][0] == common["artifacts"] / "fit_batch.gxe.log"
    assert {path.name for path in plan["outputs"][1:]} == {
        "bp_diastolic.gxe.results.tsv",
        "bp_diastolic.gxe.fit.json",
        "bp_systolic.gxe.results.tsv",
        "bp_systolic.gxe.fit.json",
    }
    manifest = plan["batch_manifest_payload"]
    assert set(manifest) == {"kind", "schema_version", "reference", "traits"}
    assert manifest["kind"] == "summit.gxe.fit_batch"
    assert [entry["name"] for entry in manifest["traits"]] == [
        "bp_diastolic",
        "bp_systolic",
    ]
    assert all(
        set(entry) == {"name", "moments", "gwas", "gwis", "out"}
        for entry in manifest["traits"]
    )
    assert all(
        Path(entry["out"]).parent == common["artifacts"] for entry in manifest["traits"]
    )
    assert plan["details"]["fit_batch_manifest"] == plan["batch_manifest_record"]
    assert plan["details"]["input_records"] == plan["fit_batch_input_records"]
    assert plan["fit_batch_input_records"]["cache"] == spec["task_args"]["cache"]
    assert (
        plan["fit_batch_input_records"]["reference"] == spec["task_args"]["reference"]
    )
    assert len(dependency["outputs"]) == 3 * len(triplets) + 1

    DEPLOY._atomic_json_noreplace(manifest, plan["batch_manifest"])
    observed = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    assert observed["batch_manifest_record"] == plan["batch_manifest_record"]
    assert stat.S_IMODE(plan["batch_manifest"].stat().st_mode) == 0o600
    altered = dict(manifest)
    altered["reference"] = str(common["scratch_root"] / "other-reference.json")
    plan["batch_manifest"].write_text(
        json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="size/SHA256"):
        DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("extra_arg", "exactly group"),
        ("trait_order", "configured group order"),
        ("entry_shape", "must contain exactly"),
        ("record_shape", "exact path/bytes/sha256"),
        ("missing_score_output", "exact output"),
        ("extra_score_output", "exactly every configured"),
        ("duplicate_score_output", "repeats an output"),
        ("dependency_group", "wide-score provenance"),
        ("dependency_traits", "wide-score provenance"),
        ("dependency_cache", "wide-score provenance"),
        ("dependency_reference", "wide-score provenance"),
        ("cache_same_hash_other_path", "exact record"),
        ("reference_same_hash_other_path", "exact record"),
        ("cache_record_shape", "exact path/bytes/sha256"),
        ("existing_output", "existing task output"),
    ],
)
def test_fit_batch_plan_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
):
    spec, common, config, dependency, triplets = _fit_batch_plan_case(
        tmp_path, monkeypatch
    )
    if failure == "extra_arg":
        spec["task_args"]["unexpected"] = True
    elif failure == "trait_order":
        spec["task_args"]["traits"] = list(reversed(triplets))
    elif failure == "entry_shape":
        spec["task_args"]["traits"][0]["unexpected"] = True
    elif failure == "record_shape":
        spec["task_args"]["traits"][0]["gwas"]["unexpected"] = True
    elif failure == "missing_score_output":
        dependency["outputs"].remove(triplets[0]["gwas"])
    elif failure == "extra_score_output":
        dependency["outputs"].append(
            DEPLOY._record(_private_file(common["scratch_root"] / "unexpected.log"))
        )
    elif failure == "duplicate_score_output":
        dependency["outputs"].append(dependency["outputs"][0])
    elif failure == "dependency_group":
        dependency["task_details"]["group"] = "age_assay"
    elif failure == "dependency_traits":
        dependency["task_details"]["traits"] = list(
            reversed(dependency["task_details"]["traits"])
        )
    elif failure == "dependency_cache":
        dependency["task_details"]["cache_sha256"] = "0" * 64
    elif failure == "dependency_reference":
        dependency["task_details"]["reference_sha256"] = "0" * 64
    elif failure == "cache_same_hash_other_path":
        replacement = _private_file(common["scratch_root"] / "same-cache-bytes.npz")
        assert DEPLOY._sha256(replacement) == spec["task_args"]["cache"]["sha256"]
        spec["task_args"]["cache"] = DEPLOY._record(replacement)
    elif failure == "reference_same_hash_other_path":
        replacement = _private_file(
            common["scratch_root"] / "same-reference-bytes.json"
        )
        assert DEPLOY._sha256(replacement) == spec["task_args"]["reference"]["sha256"]
        spec["task_args"]["reference"] = DEPLOY._record(replacement)
    elif failure == "cache_record_shape":
        spec["task_args"]["cache"]["unexpected"] = True
    elif failure == "existing_output":
        common["artifacts"].mkdir(mode=0o700)
        _private_file(common["artifacts"] / "bp_diastolic.gxe.results.tsv")
    else:  # pragma: no cover - parameter list owns this enum
        raise AssertionError(failure)
    with pytest.raises((ValueError, FileExistsError), match=message):
        DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)


def test_fit_batch_manifest_and_output_transaction_are_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, _ = _fit_batch_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    DEPLOY._atomic_json_noreplace(
        plan["batch_manifest_payload"], plan["batch_manifest"]
    )
    common["artifacts"].mkdir(mode=0o700)
    for path in plan["outputs"]:
        _private_file(path)
    validated = []
    monkeypatch.setattr(
        DEPLOY,
        "_validate_fit_outputs",
        lambda prefix, unused_config, **unused_kwargs: validated.append(prefix),
    )
    records = DEPLOY._postvalidate_task(spec, common, config, plan, runtime={})
    assert validated == [
        common["artifacts"] / "bp_diastolic",
        common["artifacts"] / "bp_systolic",
    ]
    assert records == [DEPLOY._record(path) for path in plan["outputs"]]

    _private_file(common["artifacts"] / "unexpected")
    with pytest.raises(ValueError, match="exact contract"):
        DEPLOY._postvalidate_task(spec, common, config, plan, runtime={})


def test_fit_batch_rehashes_inputs_immediately_around_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, _ = _fit_batch_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    DEPLOY._atomic_json_noreplace(
        plan["batch_manifest_payload"], plan["batch_manifest"]
    )
    target = Path(plan["fit_batch_input_records"]["traits"][0]["gwas"]["path"])
    invoked = []

    class MutatingCli:
        @staticmethod
        def main():
            invoked.append(True)
            target.write_text("changed after preflight\n", encoding="utf-8")
            target.chmod(0o600)

    with pytest.raises(ValueError, match="size/SHA256"):
        DEPLOY._execute_plan(
            plan, {"cli_module": MutatingCli}, common, task="fit_batch"
        )
    assert invoked == [True]


def test_fit_batch_rejects_mutation_before_cli_without_invoking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, _ = _fit_batch_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    DEPLOY._atomic_json_noreplace(
        plan["batch_manifest_payload"], plan["batch_manifest"]
    )
    target = Path(plan["fit_batch_input_records"]["cache"]["path"])
    target.write_text("changed before invocation\n", encoding="utf-8")
    target.chmod(0o600)
    invoked = []

    class ForbiddenCli:
        @staticmethod
        def main():
            invoked.append(True)

    with pytest.raises(ValueError, match="size/SHA256"):
        DEPLOY._execute_plan(
            plan, {"cli_module": ForbiddenCli}, common, task="fit_batch"
        )
    assert invoked == []


def test_fit_batch_rejects_aba_mutation_from_consumed_snapshot_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec, common, config, _, _ = _fit_batch_plan_case(tmp_path, monkeypatch)
    plan = DEPLOY._prepare_task(spec, common, config, artifacts_ready=False)
    DEPLOY._atomic_json_noreplace(
        plan["batch_manifest_payload"], plan["batch_manifest"]
    )
    common["artifacts"].mkdir(mode=0o700)
    first = plan["fit_batch_input_records"]["traits"][0]
    target = Path(first["gwas"]["path"])
    original = target.read_bytes()

    class AbaMutatingCli:
        @staticmethod
        def main():
            target.write_text("transient bytes consumed by SUMMIT\n", encoding="utf-8")
            target.chmod(0o600)
            transient_record = DEPLOY._record(target)
            try:
                for trait_records in plan["fit_batch_input_records"]["traits"]:
                    provenance = {
                        "reference_manifest": plan["fit_batch_input_records"][
                            "reference"
                        ],
                        "feature_cache": plan["fit_batch_input_records"]["cache"],
                        "phenotype_moments": trait_records["moments"],
                        "gwas": trait_records["gwas"],
                        "gwis": trait_records["gwis"],
                    }
                    if trait_records["trait"] == first["trait"]:
                        provenance["gwas"] = transient_record
                    _write_fit_output_fixture(
                        common["artifacts"],
                        prefix_name=trait_records["trait"],
                        consumed_input_provenance=provenance,
                    )
                _private_file(common["artifacts"] / "fit_batch.gxe.log")
            finally:
                target.write_bytes(original)
                target.chmod(0o600)

    DEPLOY._execute_plan(plan, {"cli_module": AbaMutatingCli}, common, task="fit_batch")
    with pytest.raises(ValueError, match="consumed-input provenance differs"):
        DEPLOY._postvalidate_task(spec, common, config, plan, runtime={})


def test_renderer_materializes_private_fit_batch_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    scratch = _private_dir(tmp_path / "scratch")
    job_root = _private_dir(scratch / "job")
    spec_path = _private_file(scratch / "spec.json", "{}\n")
    config_path = _private_file(tmp_path / "deployment.json", "{}\n")
    wrapper = ROOT / "scripts" / "gxe" / "hoffman" / "uge_fit_batch.sh"
    numactl = tmp_path / "numactl"
    numactl.write_text("#!/bin/bash\n", encoding="utf-8")
    numactl.chmod(0o755)
    python = Path(os.environ.get("PYTHON", os.sys.executable))
    config = _config_for_scratch(scratch)
    spec = {
        "task": "fit_batch",
        "job_name": "gxe_fit_batch_test",
        "job_root": str(job_root),
    }
    manifest_path = job_root / DEPLOY.FIT_BATCH_MANIFEST_NAME
    manifest_payload = {
        "kind": "summit.gxe.fit_batch",
        "schema_version": 1,
        "reference": str(scratch / "reference.json"),
        "traits": [
            {
                "name": "Y1",
                "moments": str(scratch / "Y1.moments.json"),
                "gwas": str(scratch / "Y1.gwas.gz"),
                "gwis": str(scratch / "Y1.gwis.gz"),
                "out": str(job_root / "artifacts" / "Y1"),
            }
        ],
    }
    common = {
        "job_name": spec["job_name"],
        "job_root": job_root,
        "frozen": {"python": python, "wrapper": wrapper},
        "resource": config["resources"]["fit_batch"],
        "resource_profile": "fit_batch",
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
    plan = {
        "outputs": [],
        "command": [],
        "details": {},
        "batch_manifest": manifest_path,
        "batch_manifest_payload": manifest_payload,
        "batch_manifest_record": DEPLOY._canonical_json_record(
            manifest_payload, manifest_path
        ),
    }
    monkeypatch.setattr(
        DEPLOY, "_load_config", lambda *unused: (config_path, config, "a" * 64)
    )
    monkeypatch.setattr(
        DEPLOY, "_load_spec", lambda *unused: (spec_path, spec, "b" * 64)
    )
    monkeypatch.setattr(DEPLOY, "_validate_common", lambda *unused, **kwargs: common)
    monkeypatch.setattr(DEPLOY, "_prepare_task", lambda *unused, **kwargs: plan)
    DEPLOY._render_job(argparse.Namespace(config="x", job_spec="y"))
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest_payload
    assert DEPLOY._record(manifest_path) == plan["batch_manifest_record"]
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    script = (job_root / "job.sh").read_text(encoding="utf-8")
    assert "#$ -pe shared 1" in script
    assert f"exec {numactl} --interleave=all {wrapper}" in script
    assert capsys.readouterr().out.strip() == f"qsub {job_root / 'job.sh'}"


def _write_fit_output_fixture(
    tmp_path: Path,
    *,
    prefix_name: str = "fit",
    consumed_input_provenance: dict | None = None,
) -> tuple[Path, dict, dict]:
    prefix = tmp_path / prefix_name
    if consumed_input_provenance is None:
        fixture_inputs = {
            role: DEPLOY._record(
                _private_file(tmp_path / f"{prefix_name}.{role}.input")
            )
            for role in (
                "reference_manifest",
                "feature_cache",
                "phenotype_moments",
                "gwas",
                "gwis",
            )
        }
        consumed_input_provenance = fixture_inputs
    components = ["G:L2_0", "GxE:L2_0", "NxE", "residual"]
    proportions = [0.0, 0.2, -0.3, 1.1]
    standard_errors = [0.0, 0.0, 0.0, 0.05]
    residual_fraction = 0.8
    payload = {
        "kind": "summit.gxe.fit",
        "schema_version": 3,
        "consumed_input_provenance": consumed_input_provenance,
        "component_names": components,
        "rank": 4,
        "condition_number": 2.0,
        "relative_residual": 1.0e-12,
        "nxe_residual_kernel_correlation": 0.1,
        "phenotype_residual_variance_fraction": residual_fraction,
        "kernel_traces": [1.0, 2.0, 3.0, 4.0],
        "coefficients": [0.1, 0.2, 0.3, 0.4],
        "variance_contributions": proportions,
        "proportions": proportions,
        "standard_errors": standard_errors,
        "original_scale_proportions": [
            value * residual_fraction for value in proportions
        ],
        "original_scale_standard_errors": [
            value * residual_fraction for value in standard_errors
        ],
        "jackknife_block_labels": [f"block_{index:03d}" for index in range(100)],
        "jackknife_estimates": [proportions for _ in range(100)],
        "normal_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ],
        "kernel_correlation_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ],
        "rhs": [0.1, 0.2, 0.3, 0.4],
        "singular_values": [2.0, 1.5, 1.0, 0.5],
        "normal_eigenvalues": [0.5, 1.0, 1.5, 2.0],
    }
    _private_file(
        Path(str(prefix) + ".gxe.fit.json"),
        json.dumps(payload, sort_keys=True) + "\n",
    )
    header = [
        "component",
        "coefficient",
        "kernel_trace",
        "variance_contribution",
        "proportion",
        "proportion_se",
        "original_scale_proportion",
        "z",
        "original_scale_se",
    ]
    z_values = ["nan", "inf", "-inf", "22"]
    rows = []
    for index, component in enumerate(components):
        rows.append(
            [
                component,
                str(payload["coefficients"][index]),
                str(payload["kernel_traces"][index]),
                str(payload["variance_contributions"][index]),
                str(proportions[index]),
                str(standard_errors[index]),
                str(payload["original_scale_proportions"][index]),
                z_values[index],
                str(payload["original_scale_standard_errors"][index]),
            ]
        )
    table = "\t".join(header) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"
    _private_file(Path(str(prefix) + ".gxe.results.tsv"), table)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return prefix, config, consumed_input_provenance


def test_fit_output_validation_covers_original_scale_and_zero_se_z(tmp_path: Path):
    prefix, config, provenance = _write_fit_output_fixture(tmp_path)
    DEPLOY._validate_fit_outputs(prefix, config, expected_input_provenance=provenance)


@pytest.mark.parametrize("null_role", [None, "feature_cache"])
def test_fit_output_validation_rejects_null_consumed_input_provenance(
    tmp_path: Path, null_role: str | None
):
    prefix, config, provenance = _write_fit_output_fixture(tmp_path)
    json_path = Path(str(prefix) + ".gxe.fit.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if null_role is None:
        payload["consumed_input_provenance"] = None
    else:
        payload["consumed_input_provenance"][null_role] = None
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    json_path.chmod(0o600)
    with pytest.raises(
        ValueError, match="consumed-input|exact path/bytes/sha256 record"
    ):
        DEPLOY._validate_fit_outputs(
            prefix, config, expected_input_provenance=provenance
        )


def test_fit_output_validation_rejects_consumed_input_provenance_corruption(
    tmp_path: Path,
):
    prefix, config, provenance = _write_fit_output_fixture(tmp_path)
    json_path = Path(str(prefix) + ".gxe.fit.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["consumed_input_provenance"]["gwas"] = {
        **provenance["gwas"],
        "sha256": "0" * 64,
    }
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    json_path.chmod(0o600)
    with pytest.raises(ValueError, match="consumed-input provenance differs"):
        DEPLOY._validate_fit_outputs(
            prefix, config, expected_input_provenance=provenance
        )


@pytest.mark.parametrize(
    ("column", "row", "replacement", "message"),
    [
        ("original_scale_proportion", 1, "0.17", "original_scale_proportion"),
        ("original_scale_se", 2, "0.025", "original_scale_se"),
        ("z", 3, "9", "zero-SE NaN/infinity"),
        ("z", 1, "0", "zero-SE NaN/infinity"),
        ("z", 2, "0", "zero-SE NaN/infinity"),
        ("z", 0, "0", "zero-SE NaN/infinity"),
        ("z", 0, "corrupt", "nonnumeric"),
    ],
)
def test_fit_output_validation_rejects_derived_column_corruption(
    tmp_path: Path,
    column: str,
    row: int,
    replacement: str,
    message: str,
):
    prefix, config, provenance = _write_fit_output_fixture(tmp_path)
    table_path = Path(str(prefix) + ".gxe.results.tsv")
    lines = table_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    fields = lines[row + 1].split("\t")
    fields[header.index(column)] = replacement
    lines[row + 1] = "\t".join(fields)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table_path.chmod(0o600)
    with pytest.raises(ValueError, match=message):
        DEPLOY._validate_fit_outputs(
            prefix, config, expected_input_provenance=provenance
        )


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
    shard_paths = [_private_file(scratch / f"shard{i}.json") for i in range(2)]
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
                    "probe_offset": 50 * index,
                    "probe_stop": 50 * (index + 1),
                }
            },
            {},
        )

    monkeypatch.setattr(DEPLOY, "_validate_shard_record", validate_shard)

    for count in (1, 2):
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
        assert plan["details"]["probes"] == count * 50
        assert ("--allow-low-probe-gxe-jackknife" in plan["command"]) is (count == 1)
        assert any(
            path.name == f"B{count * 50:03d}.gxe.ref.json" for path in plan["outputs"]
        )

    with pytest.raises(ValueError, match="one through 2"):
        DEPLOY._prepare_task(
            {
                "task": "merge",
                "task_args": {
                    "cache": cache_record,
                    "shards": [*shard_records, shard_records[0]],
                },
            },
            common,
            config,
            artifacts_ready=False,
        )


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


def test_single_fit_qacct_preserves_exact_input_binding(tmp_path: Path):
    scratch = _private_dir(tmp_path / "scratch")
    job = _private_dir(scratch / "job")
    artifacts = _private_dir(job / "artifacts")
    output = _private_file(artifacts / "fit.gxe.log")
    inputs = {
        "reference_manifest": DEPLOY._record(
            _private_file(scratch / "B100.gxe.ref.json")
        ),
        "feature_cache": DEPLOY._record(_private_file(scratch / "cache.gxe.cache.npz")),
        "phenotype_moments": DEPLOY._record(_private_file(scratch / "Y1.moments.json")),
        "gwas": DEPLOY._record(_private_file(scratch / "Y1.gwas.tsv.gz")),
        "gwis": DEPLOY._record(_private_file(scratch / "Y1.gwis.tsv.gz")),
    }
    invocation = _private_file(
        job / "invocation.json",
        json.dumps({"fit_input_records": inputs}, sort_keys=True) + "\n",
    )
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
        "task": "fit",
        "job_id": 455,
        "deployment_config": DEPLOY._record(config_file),
        "job_spec": DEPLOY._record(spec_file),
        "job_script": DEPLOY._record(job_script),
        "attempt_lock": DEPLOY._record(attempt_lock),
        "attempt": DEPLOY._record(attempt),
        "code_manifest": DEPLOY._record(code_manifest),
        "invocation": DEPLOY._record(invocation),
        "resource_profile": "fit",
        "resource": config["resources"]["fit"],
        "outputs": [DEPLOY._record(output)],
        "task_details": {
            "input_records": {
                **inputs,
                "gwas": {**inputs["gwas"], "sha256": "0" * 64},
            }
        },
        "fit_input_records": inputs,
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
        "jobnumber 455\nfailed 0\nexit_status 0\nslots 1\n"
        "wallclock 00:00:10\ncpu 00:00:09\nmaxvmem 512M\nEOF\n",
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
    with pytest.raises(ValueError, match="do not bind the same exact inputs"):
        DEPLOY._record_qacct(args)

    receipt["task_details"]["input_records"] = inputs
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o600)
    args.receipt_sha256 = DEPLOY._sha256(receipt_path)
    args.receipt_bytes = receipt_path.stat().st_size
    DEPLOY._record_qacct(args)
    completed = json.loads((job / "completed_qacct.json").read_text(encoding="utf-8"))
    assert completed["task"] == "fit"
    assert completed["slots"] == 1
    assert completed["fit_input_records"] == inputs


def test_fit_batch_qacct_preserves_exact_manifest_binding(tmp_path: Path):
    scratch = _private_dir(tmp_path / "scratch")
    job = _private_dir(scratch / "job")
    artifacts = _private_dir(job / "artifacts")
    output = _private_file(artifacts / "fit_batch.gxe.log")
    cache = _private_file(scratch / "cache.gxe.cache.npz")
    reference = _private_file(scratch / "B100.gxe.ref.json")
    moments = _private_file(scratch / "Y1.moments.json")
    gwas = _private_file(scratch / "Y1.gwas.tsv.gz")
    gwis = _private_file(scratch / "Y1.gwis.tsv.gz")
    input_records = {
        "cache": DEPLOY._record(cache),
        "reference": DEPLOY._record(reference),
        "traits": [
            {
                "trait": "Y1",
                "moments": DEPLOY._record(moments),
                "gwas": DEPLOY._record(gwas),
                "gwis": DEPLOY._record(gwis),
            }
        ],
    }
    manifest = _private_file(
        job / DEPLOY.FIT_BATCH_MANIFEST_NAME,
        json.dumps(
            {
                "kind": "summit.gxe.fit_batch",
                "schema_version": 1,
                "reference": str(reference),
                "traits": [
                    {
                        "name": "Y1",
                        "moments": str(moments),
                        "gwas": str(gwas),
                        "gwis": str(gwis),
                        "out": str(artifacts / "Y1"),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
    )
    manifest_record = DEPLOY._record(manifest)
    invocation = _private_file(
        job / "invocation.json",
        json.dumps(
            {
                "fit_batch_manifest": manifest_record,
                "fit_batch_input_records": input_records,
            },
            sort_keys=True,
        )
        + "\n",
    )
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
        "task": "fit_batch",
        "job_id": 456,
        "deployment_config": DEPLOY._record(config_file),
        "job_spec": DEPLOY._record(spec_file),
        "job_script": DEPLOY._record(job_script),
        "attempt_lock": DEPLOY._record(attempt_lock),
        "attempt": DEPLOY._record(attempt),
        "code_manifest": DEPLOY._record(code_manifest),
        "invocation": DEPLOY._record(invocation),
        "resource_profile": "fit_batch",
        "resource": config["resources"]["fit_batch"],
        "outputs": [DEPLOY._record(output)],
        "task_details": {
            "fit_batch_manifest": {**manifest_record, "sha256": "0" * 64},
            "input_records": input_records,
        },
        "fit_batch_manifest": manifest_record,
        "fit_batch_input_records": input_records,
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
        "jobnumber 456\nfailed 0\nexit_status 0\nslots 1\n"
        "wallclock 00:00:10\ncpu 00:00:09\nmaxvmem 512M\nEOF\n",
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
    with pytest.raises(ValueError, match="do not bind the same manifest"):
        DEPLOY._record_qacct(args)
    receipt["task_details"]["fit_batch_manifest"] = manifest_record
    receipt["task_details"]["input_records"] = {
        **input_records,
        "cache": {**input_records["cache"], "sha256": "0" * 64},
    }
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o600)
    args.receipt_sha256 = DEPLOY._sha256(receipt_path)
    args.receipt_bytes = receipt_path.stat().st_size
    with pytest.raises(ValueError, match="do not bind the same exact inputs"):
        DEPLOY._record_qacct(args)
    receipt["task_details"]["input_records"] = input_records
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o600)
    args.receipt_sha256 = DEPLOY._sha256(receipt_path)
    args.receipt_bytes = receipt_path.stat().st_size
    DEPLOY._record_qacct(args)
    completed = json.loads((job / "completed_qacct.json").read_text(encoding="utf-8"))
    assert completed["task"] == "fit_batch"
    assert completed["slots"] == 1
    assert completed["fit_batch_manifest"] == manifest_record
    assert completed["fit_batch_input_records"] == input_records
