#!/usr/bin/env python3
"""Benchmark the native contextual reference+trait workflow in fresh processes.

The controller is intentionally import-light while workers are live.  It
prepares one sealed deterministic input bundle, launches every plan/replicate
through ``taskset`` in a fresh ``python -S`` process, samples Linux process
resources externally, and only then imports the Stage 6 record adapter.  The
harness measures the contextual reference and trait estimand; it does not reuse
the older GxE layout microbenchmark estimand.

Large runs fail closed.  A single invocation accepts one exact M, applies
checked memory/descriptor/byte budgets before input preparation, and requires a
compatible bounded pilot before M exceeds the configured pilot ceiling.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "context_native_stage6_benchmark_v1"
SCHEMA_VERSION = 1
BUILD_SPEC_SCHEMA = "context_native_stage6_build_spec_v1"
PLAN_SPEC_SCHEMA = "context_native_stage6_plan_spec_v1"
CASE_SCHEMA = "context_native_stage6_case_v1"
CALIBRATION_SCHEMA = "context_native_stage6_runtime_calibration_v1"
WORKER_SCHEMA = "context_native_stage6_worker_result_v1"
PREPARED_CASE_SCHEMA = "context_native_stage6_prepared_case_v1"
FAULT_QUALIFICATION_SCHEMA = "context_native_stage5_fault_qualification_v1"
CACHE_STATE_POLICY = "outer_full_hash_then_warm_cache_v1"
FAULT_COVERAGE_CONTRACT = "contextual_stage5_every_operation_r7_t4_v1"
NATIVE_BUILD_PROVENANCE_SCHEMA = "context_native_stage6_build_provenance_v1"

# These fields describe the compiled extension rather than process-start or
# per-plan runtime policy.  The native module content hash remains the primary
# identity, while the explicit fields make the provenance self-describing and
# fail closed if reported compile metadata changes.  The complete build_info,
# including BLAS/OpenMP runtime settings, remains in every worker result.
IMMUTABLE_NATIVE_BUILD_INFO_KEYS = (
    "api_version",
    "backend_name",
    "backend_version",
    "source_commit",
    "source_tree_sha256",
    "compiler_id",
    "compiler_version",
    "cxx_standard",
    "build_type",
    "sanitizer_mode",
    "address_sanitizer_enabled",
    "undefined_behavior_sanitizer_enabled",
    "optimization",
    "architecture_tuning",
    "compiler_flags",
    "compiler_flags_scope",
    "openmp_enabled",
    "native_optimization_enabled",
    "platform",
    "gemm_integrity_enabled",
    "gemm_checksum_enabled",
    "gemm_integrity_minimum_vendor_flops",
    "private_openblas_archive_sha256",
    "private_blas_backend",
    "private_blas_archive_sha256",
    "private_blas_source_commit",
    "private_blas_source_tree_sha256",
    "private_blas_config_family",
    "private_blas_header_sha256",
    "private_blas_cblas_header_sha256",
)

FAULT_QUALIFICATION_KEYS = (
    "schema",
    "source_commit",
    "source_tree_sha256",
    "native_module_sha256",
    "integrity_policy",
    "integrity_backend",
    "fault_coverage_contract",
    "covered_operations",
    "recoverable_fault_modes",
    "terminal_fault_modes",
    "release_fault_matrix_passed",
    "focused_test_count",
    "full_regression_test_count",
    "asan_ubsan_status",
    "ubsan_status",
)
RECOVERABLE_FAULT_MODES = (
    "one_shot",
    "nan",
    "inf",
    "canary",
    "repeated",
    "repair_corruption",
    "force_fallback",
)
TERMINAL_FAULT_MODES = (
    "fallback_corruption",
    "fallback_failure",
    "operand_mutation",
    "runtime_mutation",
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = ROOT / "src"

DEFAULT_SAMPLE_INTERVAL_MS = 100
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0
DEFAULT_SWEEP_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_WORKSPACE_BYTES = 8 * 1024**3
DEFAULT_MAX_LOGICAL_VARIANT_VISITS = 100_000_000
DEFAULT_MAX_LOGICAL_BED_BYTES = 8 * 1024**3
DEFAULT_PILOT_VARIANTS = 8_192
DEFAULT_MAX_EXTRAPOLATION_RATIO = 4.0
DEFAULT_ESTIMATE_SAFETY_FACTOR = 2.0
DEFAULT_MAX_SYNTHETIC_CELLS = 20_000_000
DEFAULT_RSS_MARGIN_BYTES = 256 * 1024**2
DEFAULT_RSS_MARGIN_FRACTION = 0.25

MAX_FAILURE_TAIL = 8_000
TERM_WAIT_SECONDS = 2.0
KILL_WAIT_SECONDS = 2.0

BUILD_KEYS = (
    "schema",
    "label",
    "install_prefix",
    "native_module",
    "python_executable",
    "dependency_paths",
    "environment",
    "expected_native_sha256",
    "expected_package_sha256",
    "expected_python_runtime_sha256",
    "expected_dependency_runtime_sha256",
    "expected_source_commit",
    "expected_source_tree_sha256",
    "qualification_evidence_path",
    "qualification_evidence_sha256",
    "declared_parent_source_commit",
)
PLAN_SPEC_KEYS = (
    "schema",
    "plan_id",
    "build_label",
    "comparison_mode",
    "plan",
)
ALLOWED_BUILD_ENVIRONMENT = frozenset(
    {
        "ASAN_OPTIONS",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "OMP_DYNAMIC",
        "OPENBLAS_VERBOSE",
        "UBSAN_OPTIONS",
    }
)
_ASAN_PRELOAD_BASENAME_PREFIXES = ("libasan.so", "libclang_rt.asan")
COMPARISON_MODES = frozenset({"exact_trace_v1", "science_tolerance_v1"})

REFERENCE_ARRAYS = (
    "gram",
    "same_person",
    "group_gram_unnormalized_num",
    "annotation_masses",
    "group_annotation_masses",
    "group_variant_counts",
)
TRAIT_ARRAYS = (
    "genetic_rhs",
    "genetic_traces",
    "genetic_residual",
    "residual_rhs",
    "residual_traces",
    "residual_gram",
    "group_rhs_unnormalized_num",
    "group_trace_unnormalized_num",
    "group_genetic_residual_num",
    "annotation_masses",
    "group_annotation_masses",
    "group_variant_counts",
)
FIT_ARRAYS = (
    "raw_coefficients",
    "raw_loo_coefficients",
    "raw_jackknife_covariance",
    "raw_standard_errors",
    "raw_solve_retained_directions",
    "raw_solve_null_space",
)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not strict canonical JSON") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_build_provenance_sha256(
    build_info: Mapping[str, Any],
    native_module_sha256: str,
    installed_package_sha256: str,
) -> str:
    """Bind immutable compile identity without folding in runtime policy."""
    if not isinstance(build_info, Mapping):
        raise ValueError("native build_info must be a mapping")
    if not _is_sha256(native_module_sha256):
        raise ValueError("native module SHA-256 is invalid")
    if not _is_sha256(installed_package_sha256):
        raise ValueError("installed package SHA-256 is invalid")
    missing = [key for key in IMMUTABLE_NATIVE_BUILD_INFO_KEYS if key not in build_info]
    if missing:
        raise ValueError(f"native build_info missing immutable keys {missing!r}")
    immutable_build_info = {
        key: build_info[key] for key in IMMUTABLE_NATIVE_BUILD_INFO_KEYS
    }
    return _canonical_sha256(
        {
            "schema": NATIVE_BUILD_PROVENANCE_SCHEMA,
            "native_module_sha256": native_module_sha256,
            "installed_package_sha256": installed_package_sha256,
            "immutable_build_info": immutable_build_info,
        }
    )


def _package_sha256(prefix: Path) -> str:
    """Hash installed regular files, excluding interpreter-created caches."""
    digest = hashlib.sha256()
    files = [
        path
        for path in prefix.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    for path in sorted(files, key=lambda item: item.relative_to(prefix).as_posix()):
        relative = path.relative_to(prefix).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _runtime_dependency_identity() -> dict[str, Any]:
    import importlib.metadata

    import bed_reader
    import numpy
    import scipy

    packages: dict[str, Any] = {}
    for name, module in (
        ("bed_reader", bed_reader),
        ("numpy", numpy),
        ("scipy", scipy),
    ):
        module_path = Path(module.__file__).resolve(strict=True)
        package_root = module_path.parent
        distribution_name = "bed-reader" if name == "bed_reader" else name
        packages[name] = {
            "version": importlib.metadata.version(distribution_name),
            "module_path": str(module_path),
            "module_sha256": _file_sha256(module_path),
            "package_content_sha256": _package_sha256(package_root),
        }
    python_identity = {
        "schema": "context_native_stage6_python_runtime_v1",
        "implementation": platform.python_implementation(),
        "version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "executable_path": str(Path(sys.executable).resolve(strict=True)),
        "executable_sha256": _file_sha256(Path(sys.executable).resolve(strict=True)),
    }
    dependency_identity = {
        "schema": "context_native_stage6_dependency_runtime_v1",
        "packages": packages,
    }
    return {
        "python": python_identity,
        "python_runtime_sha256": _canonical_sha256(python_identity),
        "dependencies": dependency_identity,
        "dependency_runtime_sha256": _canonical_sha256(dependency_identity),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(label: str, value: Mapping[str, Any], expected: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} has missing keys {sorted(wanted - actual)!r} and "
            f"extra keys {sorted(actual - wanted)!r}"
        )


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _write_json_no_replace(path: Path, value: Any) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSON artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_internal_json(path: Path, value: Any) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short internal JSON write")
            view = view[written:]
    finally:
        os.close(descriptor)


def _regular_file(path: Path, *, executable: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"symbolic links are not accepted: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected regular file: {path}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"expected executable file: {path}")
    return resolved


def _regular_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"symbolic links are not accepted: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"expected directory: {path}")
    return resolved


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise argparse.ArgumentTypeError("CPU list must be nonempty")
    cpus: list[int] = []
    try:
        for token in value.split(","):
            if not token or token.strip() != token:
                raise ValueError
            if "-" in token:
                pieces = token.split("-")
                if len(pieces) != 2:
                    raise ValueError
                first, last = (int(piece) for piece in pieces)
                if first < 0 or last < first:
                    raise ValueError
                cpus.extend(range(first, last + 1))
            else:
                cpu = int(token)
                if cpu < 0:
                    raise ValueError
                cpus.append(cpu)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid CPU list {value!r}") from error
    if not cpus or cpus != sorted(set(cpus)):
        raise argparse.ArgumentTypeError(
            "CPU list must be unique and strictly increasing"
        )
    return tuple(cpus)


def _compress_cpu_list(cpus: Sequence[int]) -> str:
    values = tuple(cpus)
    if not values:
        raise ValueError("cannot compress an empty CPU list")
    ranges: list[str] = []
    begin = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(begin) if begin == previous else f"{begin}-{previous}")
        begin = previous = value
    ranges.append(str(begin) if begin == previous else f"{begin}-{previous}")
    return ",".join(ranges)


@dataclass(frozen=True)
class BuildSpec:
    label: str
    install_prefix: Path
    native_module: Path
    python_executable: Path
    dependency_paths: tuple[Path, ...]
    environment: Mapping[str, str]
    expected_native_sha256: str
    expected_package_sha256: str
    expected_python_runtime_sha256: str
    expected_dependency_runtime_sha256: str
    expected_source_commit: str
    expected_source_tree_sha256: str
    qualification_evidence_path: Path | None
    qualification_evidence_sha256: str | None
    declared_parent_source_commit: str | None
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "BuildSpec":
        source = _regular_file(path)
        value = _strict_json_load(source)
        _exact_keys("build spec", value, BUILD_KEYS)
        if value["schema"] != BUILD_SPEC_SCHEMA:
            raise ValueError("unsupported build spec schema")
        label = value["label"]
        if not isinstance(label, str) or not label:
            raise ValueError("build label must be nonempty")
        install = _regular_directory(Path(value["install_prefix"]))
        native = _regular_file(Path(value["native_module"]))
        if not native.is_relative_to(install):
            raise ValueError("native module must be inside install_prefix")
        python = Path(value["python_executable"]).expanduser().resolve(strict=True)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("python_executable must resolve to an executable file")
        raw_dependencies = value["dependency_paths"]
        if not isinstance(raw_dependencies, list):
            raise ValueError("dependency_paths must be a list")
        dependencies = tuple(
            _regular_directory(Path(item)) for item in raw_dependencies
        )
        raw_environment = value["environment"]
        if not isinstance(raw_environment, Mapping):
            raise ValueError("build environment must be a mapping")
        if not set(raw_environment) <= ALLOWED_BUILD_ENVIRONMENT:
            raise ValueError("build environment contains a forbidden key")
        environment: dict[str, str] = {}
        for key, item in raw_environment.items():
            if not isinstance(item, str):
                raise ValueError("build environment values must be strings")
            environment[key] = item
        for name in (
            "expected_native_sha256",
            "expected_package_sha256",
            "expected_python_runtime_sha256",
            "expected_dependency_runtime_sha256",
            "expected_source_tree_sha256",
        ):
            if not _is_sha256(value[name]):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not _is_commit(value["expected_source_commit"]):
            raise ValueError("expected_source_commit must be a 40-character commit")
        qualification = value["qualification_evidence_sha256"]
        if qualification is not None and not _is_sha256(qualification):
            raise ValueError("qualification_evidence_sha256 is invalid")
        qualification_path_value = value["qualification_evidence_path"]
        if (qualification is None) != (qualification_path_value is None):
            raise ValueError(
                "qualification evidence path and SHA-256 must be both present or absent"
            )
        qualification_path = (
            None
            if qualification_path_value is None
            else _regular_file(Path(qualification_path_value))
        )
        if (
            qualification_path is not None
            and _file_sha256(qualification_path) != qualification
        ):
            raise ValueError("qualification evidence content hash mismatch")
        parent = value["declared_parent_source_commit"]
        if parent is not None and not _is_commit(parent):
            raise ValueError("declared_parent_source_commit is invalid")
        if _file_sha256(native) != value["expected_native_sha256"]:
            raise ValueError(f"native module hash mismatch for build {label!r}")
        if _package_sha256(install) != value["expected_package_sha256"]:
            raise ValueError(f"installed package hash mismatch for build {label!r}")
        return cls(
            label=label,
            install_prefix=install,
            native_module=native,
            python_executable=python,
            dependency_paths=dependencies,
            environment=environment,
            expected_native_sha256=value["expected_native_sha256"],
            expected_package_sha256=value["expected_package_sha256"],
            expected_python_runtime_sha256=value["expected_python_runtime_sha256"],
            expected_dependency_runtime_sha256=value[
                "expected_dependency_runtime_sha256"
            ],
            expected_source_commit=value["expected_source_commit"],
            expected_source_tree_sha256=value["expected_source_tree_sha256"],
            qualification_evidence_path=qualification_path,
            qualification_evidence_sha256=qualification,
            declared_parent_source_commit=parent,
            source_path=source,
        )

    def to_worker_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "install_prefix": str(self.install_prefix),
            "native_module": str(self.native_module),
            "python_executable": str(self.python_executable),
            "dependency_paths": [str(path) for path in self.dependency_paths],
            "environment": dict(self.environment),
            "expected_native_sha256": self.expected_native_sha256,
            "expected_package_sha256": self.expected_package_sha256,
            "expected_python_runtime_sha256": self.expected_python_runtime_sha256,
            "expected_dependency_runtime_sha256": (
                self.expected_dependency_runtime_sha256
            ),
            "expected_source_commit": self.expected_source_commit,
            "expected_source_tree_sha256": self.expected_source_tree_sha256,
            "qualification_evidence_path": (
                str(self.qualification_evidence_path)
                if self.qualification_evidence_path is not None
                else None
            ),
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "declared_parent_source_commit": self.declared_parent_source_commit,
        }


@dataclass(frozen=True)
class PlanSpec:
    plan_id: str
    build_label: str
    comparison_mode: str
    plan: Mapping[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "PlanSpec":
        source = _regular_file(path)
        value = _strict_json_load(source)
        _exact_keys("plan spec", value, PLAN_SPEC_KEYS)
        if value["schema"] != PLAN_SPEC_SCHEMA:
            raise ValueError("unsupported plan spec schema")
        for name in ("plan_id", "build_label"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(f"{name} must be nonempty")
        if value["comparison_mode"] not in COMPARISON_MODES:
            raise ValueError("unsupported comparison_mode")
        if not isinstance(value["plan"], Mapping):
            raise ValueError("plan must be a mapping")
        return cls(
            plan_id=value["plan_id"],
            build_label=value["build_label"],
            comparison_mode=value["comparison_mode"],
            plan=dict(value["plan"]),
            source_path=source,
        )


def _planner_module() -> Any:
    source = str(SOURCE_PACKAGE)
    if source not in sys.path:
        sys.path.insert(0, source)
    from summit.context import performance_v1

    return performance_v1


def _affinity_sha256(cpus: Sequence[int]) -> str:
    return _canonical_sha256(
        {
            "schema": "context_native_stage6_affinity_v1",
            "cpus": list(cpus),
            "numa_policy": "unbound_first_touch_v1",
            "output_numa_node": -1,
        }
    )


def _instantiate_plan(plan_spec: PlanSpec, cpus: Sequence[int]) -> Any:
    planner = _planner_module()
    values = dict(plan_spec.plan)
    values.pop("schema", None)
    if values.get("affinity_identity_sha256") == "auto":
        values["affinity_identity_sha256"] = _affinity_sha256(cpus)
    if values.get("affinity_cpu_count") == "auto":
        values["affinity_cpu_count"] = len(cpus)
    plan = planner.ContextualTablaPlanV1(**values)
    if plan.affinity_cpu_count != len(cpus):
        raise ValueError(
            f"plan {plan_spec.plan_id!r} affinity_cpu_count disagrees with --cpu-list"
        )
    if plan.affinity_identity_sha256 != _affinity_sha256(cpus):
        raise ValueError(
            f"plan {plan_spec.plan_id!r} affinity identity disagrees with --cpu-list"
        )
    return plan


def _validate_executable_plan(plan: Any, *, group_layout: str) -> None:
    blocks = {
        plan.source_variant_block,
        plan.target_variant_block,
        plan.grouped_variant_block,
        plan.same_person_variant_block,
        plan.trait_variant_block,
    }
    reasons: list[str] = []
    if len(blocks) != 1:
        reasons.append("current native executor has one shared variant block")
    if plan.source_coordinate_tile != plan.context_tile:
        reasons.append(
            "current native executor has no independent source-coordinate tile"
        )
    if not plan.resident_source_scores or not plan.resident_actions:
        reasons.append(
            "current native executor requires resident source scores/actions"
        )
    if plan.process_count != 1:
        reasons.append("current harness worker executes one native process")
    if plan.multiplication_backend != "protected_dense_gemm_v1":
        reasons.append("current native executor exposes only protected dense GEMM")
    if plan.integrity_policy != "full_scalar_witness_v1":
        reasons.append("checksum execution is not present in the frozen native build")
    if plan.numa_policy != "unbound_first_touch_v1" or plan.output_numa_node != -1:
        reasons.append("current native NUMA policy is unbound first touch/output -1")
    if plan.huge_page_policy != "disabled_v1":
        reasons.append("explicit huge-page policy is not implemented")
    if plan.telemetry_capacity_bytes != 0:
        reasons.append("native telemetry capacity is event-count based, not bytes")
    if not plan.allocation_reuse or not plan.deterministic_reductions:
        reasons.append(
            "current execution requires allocation reuse/deterministic reductions"
        )
    if plan.numeric_policy != "fp64_v1":
        reasons.append("current execution is fp64 only")
    expected_order = {
        "contiguous": "contiguous_sealed_v1",
        "interleaved": "indexed_group_major_v1",
    }[group_layout]
    if plan.group_execution_order != expected_order:
        reasons.append("plan group execution order disagrees with case group layout")
    if reasons:
        raise ValueError("unexecutable Stage 6 plan: " + "; ".join(reasons))


def _dimensions(args: argparse.Namespace) -> Any:
    planner = _planner_module()
    pairs = args.q * (args.q + 1) // 2
    return planner.ContextualTablaDimensionsV1(
        n_samples=args.retained_samples,
        n_variants=args.retained_variants,
        context_count=args.q,
        annotation_count=args.k,
        pair_count=pairs,
        component_count=args.k * pairs,
        group_count=args.groups,
        sample_probe_count=args.sample_probes,
        variant_probe_count=args.variant_probes,
        trait_count=args.traits,
        residual_basis_count=args.residual_bases,
        fixed_rank=args.fixed_rank,
    )


def _calibration_key(
    *,
    build: BuildSpec,
    plan: Any,
    dimensions: Any,
    args: argparse.Namespace,
    host_identity: Mapping[str, Any],
) -> str:
    dimension_values = dimensions.to_dict()
    dimension_values["n_variants"] = "variable"
    return _canonical_sha256(
        {
            "schema": "context_native_stage6_calibration_compatibility_v1",
            "host_identity": dict(host_identity),
            "build_source_commit": build.expected_source_commit,
            "build_source_tree_sha256": build.expected_source_tree_sha256,
            "native_module_sha256": build.expected_native_sha256,
            "installed_package_sha256": build.expected_package_sha256,
            "python_runtime_sha256": build.expected_python_runtime_sha256,
            "dependency_runtime_sha256": build.expected_dependency_runtime_sha256,
            "plan": plan.to_dict(),
            "dimensions_except_m": dimension_values,
            "input_kind": args.input_kind,
            "annotation_mode": args.annotation_mode,
            "group_layout": args.group_layout,
        }
    )


def _load_calibrations(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        value = _strict_json_load(_regular_file(path))
        records: list[Any]
        if isinstance(value, Mapping) and value.get("schema") == CALIBRATION_SCHEMA:
            records = [value]
        elif isinstance(value, Mapping) and value.get("schema") == SCHEMA:
            records = list(value.get("runtime_calibrations", []))
        else:
            raise ValueError(f"unsupported calibration record: {path}")
        for record in records:
            expected = {
                "schema",
                "compatibility_key",
                "n_variants",
                "logical_variant_visits",
                "median_worker_wall_seconds",
            }
            _exact_keys("runtime calibration", record, tuple(expected))
            if record["schema"] != CALIBRATION_SCHEMA:
                raise ValueError("invalid runtime calibration schema")
            key = record["compatibility_key"]
            if not _is_sha256(key):
                raise ValueError("invalid runtime calibration compatibility key")
            for name in ("n_variants", "logical_variant_visits"):
                if (
                    isinstance(record[name], bool)
                    or not isinstance(record[name], int)
                    or record[name] <= 0
                ):
                    raise ValueError(f"invalid runtime calibration {name}")
            wall = record["median_worker_wall_seconds"]
            if (
                isinstance(wall, bool)
                or not isinstance(wall, (int, float))
                or not math.isfinite(wall)
                or wall <= 0
            ):
                raise ValueError("invalid runtime calibration wall time")
            result.setdefault(key, []).append(dict(record))
    return result


def _runtime_preflight(
    *,
    args: argparse.Namespace,
    builds: Mapping[str, BuildSpec],
    plans: Sequence[tuple[PlanSpec, Any]],
    dimensions: Any,
    calibrations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    planner = _planner_module()
    host_identity = _host_identity()
    records: list[dict[str, Any]] = []
    total_estimated = 0.0
    for spec, plan in plans:
        estimate = planner.estimate_contextual_tabla_plan_v1(
            dimensions,
            plan,
            workspace_cap_bytes=args.max_workspace_bytes,
        )
        if not estimate.admitted:
            raise ValueError(f"plan {spec.plan_id!r} exceeds --max-workspace-bytes")
        passes = sum(
            int(value) for value in estimate.logical_descriptor_passes.values()
        )
        logical_visits = passes * dimensions.n_variants
        logical_bytes = logical_visits * ((dimensions.n_samples + 3) // 4)
        if logical_visits > args.max_logical_variant_visits:
            raise ValueError(
                f"plan {spec.plan_id!r} exceeds --max-logical-variant-visits"
            )
        if logical_bytes > args.max_logical_bed_bytes:
            raise ValueError(f"plan {spec.plan_id!r} exceeds --max-logical-bed-bytes")
        key = _calibration_key(
            build=builds[spec.build_label],
            plan=plan,
            dimensions=dimensions,
            args=args,
            host_identity=host_identity,
        )
        projected: float | None = None
        pilot_used: Mapping[str, Any] | None = None
        if dimensions.n_variants > args.pilot_variant_ceiling:
            compatible = sorted(
                calibrations.get(key, ()),
                key=lambda item: int(item["n_variants"]),
                reverse=True,
            )
            compatible = [
                item
                for item in compatible
                if int(item["n_variants"]) < dimensions.n_variants
            ]
            if not compatible:
                raise ValueError(
                    f"plan {spec.plan_id!r} requires a compatible runtime pilot"
                )
            pilot_used = compatible[0]
            ratio = dimensions.n_variants / int(pilot_used["n_variants"])
            if ratio > args.max_extrapolation_ratio:
                raise ValueError(
                    f"plan {spec.plan_id!r} M extrapolation {ratio:.3g} exceeds cap"
                )
            visit_ratio = logical_visits / int(pilot_used["logical_variant_visits"])
            projected = (
                float(pilot_used["median_worker_wall_seconds"])
                * max(ratio, visit_ratio)
                * args.estimate_safety_factor
            )
            if projected > args.worker_timeout_seconds:
                raise ValueError(
                    f"plan {spec.plan_id!r} projected worker time exceeds timeout"
                )
            total_estimated += projected * (args.warmups + args.repeats)
        records.append(
            {
                "plan_id": spec.plan_id,
                "estimate": estimate.to_dict(),
                "logical_variant_visits": logical_visits,
                "logical_bed_bytes": logical_bytes,
                "calibration_compatibility_key": key,
                "pilot_used": pilot_used,
                "projected_worker_wall_seconds_upper": projected,
            }
        )
    if total_estimated and total_estimated > args.sweep_timeout_seconds:
        raise ValueError("projected sweep time exceeds --sweep-timeout-seconds")
    return records


def _parse_size_lines(path: Path, wanted: Iterable[str]) -> dict[str, int]:
    keys = set(wanted)
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].rstrip(":") in keys and parts[2] == "kB":
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return values


def _parse_proc_io(path: Path) -> dict[str, int]:
    allowed = {
        "rchar",
        "wchar",
        "syscr",
        "syscw",
        "read_bytes",
        "write_bytes",
        "cancelled_write_bytes",
    }
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in allowed:
            values[name] = int(raw.strip())
    return values


def _parse_proc_stat(path: Path) -> dict[str, int]:
    line = path.read_text(encoding="utf-8").strip()
    closing = line.rfind(")")
    if closing < 0:
        raise ValueError(f"malformed proc stat: {path}")
    # Fields after comm begin at kernel field 3 (state).
    fields = line[closing + 2 :].split()
    if len(fields) < 22:
        raise ValueError(f"short proc stat: {path}")
    return {
        "minor_page_faults": int(fields[7]),
        "major_page_faults": int(fields[9]),
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "thread_count": int(fields[17]),
        "virtual_memory_bytes": int(fields[20]),
        "resident_pages": int(fields[21]),
    }


def _parse_proc_status(path: Path) -> dict[str, Any]:
    wanted_sizes = {"VmPeak", "VmSize", "VmHWM", "VmRSS", "RssAnon", "RssFile"}
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        value = raw.strip()
        if key in wanted_sizes:
            parts = value.split()
            if len(parts) == 2 and parts[1] == "kB":
                result[f"{key}_bytes"] = int(parts[0]) * 1024
        elif key in {
            "Threads",
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        }:
            result[key] = int(value)
        elif key in {"Cpus_allowed_list", "Mems_allowed_list"}:
            result[key] = value
    return result


def _parse_smaps_rollup(path: Path) -> dict[str, int]:
    wanted = {
        "Rss",
        "Pss",
        "Private_Clean",
        "Private_Dirty",
        "Anonymous",
        "AnonHugePages",
        "FilePmdMapped",
        "Shared_Hugetlb",
        "Private_Hugetlb",
        "Swap",
    }
    return _parse_size_lines(path, wanted)


def _parse_numa_maps(path: Path) -> dict[str, Any]:
    nodes: Counter[int] = Counter()
    mapped_pages = 0
    huge_pages = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        for token in line.split():
            if token.startswith("N") and "=" in token:
                name, raw = token.split("=", 1)
                if name[1:].isdigit() and raw.isdigit():
                    count = int(raw)
                    nodes[int(name[1:])] += count
                    mapped_pages += count
            elif token.startswith("kernelpagesize_kB="):
                raw = token.split("=", 1)[1]
                if raw.isdigit() and int(raw) > 4:
                    huge_pages += 1
    return {
        "pages_by_node": {str(node): count for node, count in sorted(nodes.items())},
        "mapped_pages": mapped_pages,
        "mappings_with_kernel_page_gt_4k": huge_pages,
    }


def _read_proc_snapshot(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    base = proc_root / str(pid)
    result: dict[str, Any] = {
        "controller_monotonic_ns": time.monotonic_ns(),
        "pid": pid,
        "available": {},
    }
    readers = {
        "io": (_parse_proc_io, base / "io"),
        "stat": (_parse_proc_stat, base / "stat"),
        "status": (_parse_proc_status, base / "status"),
        "smaps_rollup": (_parse_smaps_rollup, base / "smaps_rollup"),
        "numa_maps": (_parse_numa_maps, base / "numa_maps"),
    }
    for name, (reader, path) in readers.items():
        try:
            result[name] = reader(path)
            result["available"][name] = True
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            OSError,
            ValueError,
        ):
            result[name] = None
            result["available"][name] = False
    try:
        children_path = base / "task" / str(pid) / "children"
        raw_children = children_path.read_text(encoding="utf-8").strip()
        result["children"] = [int(value) for value in raw_children.split()]
        result["available"]["children"] = True
    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
        OSError,
        ValueError,
    ):
        result["children"] = None
        result["available"]["children"] = False
    return result


def _proc_monitor_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rss_values: list[int] = []
    pss_values: list[int] = []
    hwm_values: list[int] = []
    children: set[int] = set()
    for sample in samples:
        smaps = sample.get("smaps_rollup")
        status = sample.get("status")
        if isinstance(smaps, Mapping):
            if isinstance(smaps.get("Rss"), int):
                rss_values.append(int(smaps["Rss"]))
            if isinstance(smaps.get("Pss"), int):
                pss_values.append(int(smaps["Pss"]))
        if isinstance(status, Mapping) and isinstance(status.get("VmHWM_bytes"), int):
            hwm_values.append(int(status["VmHWM_bytes"]))
        raw_children = sample.get("children")
        if isinstance(raw_children, list):
            children.update(int(value) for value in raw_children)
    return {
        "scope": "whole_fresh_worker_process",
        "sample_count": len(samples),
        "first_snapshot": dict(samples[0]) if samples else None,
        "last_snapshot": dict(samples[-1]) if samples else None,
        "sampled_peak_rss_bytes": max(rss_values, default=None),
        "sampled_peak_pss_bytes": max(pss_values, default=None),
        "observed_vmhwm_bytes": max(hwm_values, default=None),
        "observed_child_pids": sorted(children),
        "numa_remote_bytes_measured": False,
        "numa_remote_bytes_unavailable_reason": (
            "numa_maps records placement pages, not remote-memory traffic"
        ),
    }


def _host_identity() -> dict[str, Any]:
    cpu_model = "unknown"
    physical: set[tuple[str, str]] = set()
    sockets: set[str] = set()
    current: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines() + [
            ""
        ]:
            if not line:
                if "model name" in current:
                    cpu_model = current["model name"]
                if "physical id" in current and "core id" in current:
                    physical.add((current["physical id"], current["core id"]))
                    sockets.add(current["physical id"])
                current = {}
                continue
            key, separator, value = line.partition(":")
            if separator:
                current[key.strip()] = value.strip()
    except OSError:
        pass
    memory_bytes = 1
    try:
        memory = _parse_size_lines(Path("/proc/meminfo"), {"MemTotal"})
        memory_bytes = max(1, memory.get("MemTotal", 1))
    except OSError:
        pass
    try:
        numa_nodes = len(
            [path for path in Path("/sys/devices/system/node").glob("node[0-9]*")]
        )
    except OSError:
        numa_nodes = 1
    logical = os.cpu_count() or 1
    return {
        "host_name": socket.gethostname(),
        "machine": platform.machine() or "unknown",
        "kernel_release": platform.release() or "unknown",
        "cpu_model": cpu_model,
        "physical_core_count": len(physical) or logical,
        "logical_cpu_count": logical,
        "socket_count": len(sockets) or 1,
        "numa_node_count": max(1, numa_nodes),
        "memory_bytes": memory_bytes,
    }


def _rusage_snapshot() -> dict[str, Any]:
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": float(value.ru_utime),
        "system_seconds": float(value.ru_stime),
        "maximum_rss_kib": int(value.ru_maxrss),
        "minor_page_faults": int(value.ru_minflt),
        "major_page_faults": int(value.ru_majflt),
        "input_block_operations": int(value.ru_inblock),
        "output_block_operations": int(value.ru_oublock),
        "voluntary_context_switches": int(value.ru_nvcsw),
        "involuntary_context_switches": int(value.ru_nivcsw),
    }


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=TERM_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        raise RuntimeError(f"worker {process.pid} remained live after SIGKILL")


def _run_monitored(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    sample_interval_ms: int,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
        start_new_session=True,
    )
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    try:
        while process.poll() is None:
            samples.append(_read_proc_snapshot(process.pid))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            time.sleep(min(sample_interval_ms / 1000.0, remaining))
        if timed_out:
            _terminate_process(process)
        stdout, stderr = process.communicate(timeout=0.1)
    except BaseException:
        _terminate_process(process)
        raise
    ended_ns = time.monotonic_ns()
    result = {
        "command": list(command),
        "pid": process.pid,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "controller_wall_seconds": (ended_ns - started_ns) / 1.0e9,
        "stdout_tail": stdout[-MAX_FAILURE_TAIL:],
        "stderr_tail": stderr[-MAX_FAILURE_TAIL:],
        "process_resources": _proc_monitor_summary(samples),
    }
    if timed_out:
        raise TimeoutError(f"worker {process.pid} exceeded {timeout_seconds} seconds")
    if process.returncode != 0:
        raise RuntimeError(
            f"worker {process.pid} failed with exit {process.returncode}: "
            f"{stderr[-MAX_FAILURE_TAIL:]}"
        )
    return result


def _activate_worker_paths(build: Mapping[str, Any]) -> None:
    paths = [str(build["install_prefix"]), *map(str, build["dependency_paths"])]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _array_sha256_stream_header(dtype: Any, shape: Sequence[int]) -> Any:
    import numpy as np

    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    digest = hashlib.sha256()
    digest.update(canonical_dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(tuple(shape), dtype="<i8").tobytes())
    return digest


def _file_identity(path: Path, *, hash_content: bool) -> dict[str, Any]:
    resolved = _regular_file(path)
    status = resolved.stat()
    return {
        "path": str(resolved),
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "size_bytes": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
        "sha256": _file_sha256(resolved) if hash_content else None,
    }


def _verify_case_file_identities(
    case: Mapping[str, Any],
    *,
    require_full_hashes: bool,
    verify_content_hashes: bool = True,
) -> None:
    files = case.get("files")
    if not isinstance(files, Mapping) or set(files) != {"bed", "bim", "fam"}:
        raise ValueError("prepared case has an invalid PLINK file-identity map")
    for extension in ("bed", "bim", "fam"):
        expected = files[extension]
        if not isinstance(expected, Mapping):
            raise ValueError(f"prepared {extension} identity is invalid")
        hash_content = expected.get("sha256") is not None
        if require_full_hashes and not hash_content:
            raise ValueError(f"production publication requires a {extension} hash")
        observed = _file_identity(
            Path(expected["path"]),
            hash_content=hash_content and verify_content_hashes,
        )
        stat_keys = ("path", "device", "inode", "size_bytes", "mtime_ns")
        if any(observed[key] != expected.get(key) for key in stat_keys):
            raise RuntimeError(f"prepared {extension} input identity changed")
        if verify_content_hashes and observed["sha256"] != expected.get("sha256"):
            raise RuntimeError(f"prepared {extension} input content changed")


def _verify_case_science_identity(case: Mapping[str, Any]) -> str:
    identity = case.get("science_identity")
    digest = case.get("case_identity_sha256")
    if not isinstance(identity, Mapping) or not _is_sha256(digest):
        raise ValueError("prepared case science identity is malformed")
    observed = _canonical_sha256(identity)
    if observed != digest:
        raise RuntimeError("prepared case science identity digest mismatch")
    return observed


def _philox_keys(count: int, *, seed: int, domain: int) -> Any:
    import numpy as np

    keys = np.empty((count, 2), dtype=np.uint64, order="C")
    for index in range(count):
        mixed = (seed + domain * 1_000_003 + index * 104_729) & ((1 << 63) - 1)
        keys[index] = np.asarray(
            np.random.Philox(mixed).state["state"]["key"], dtype=np.uint64
        )
    return keys


def _prepare_case(request_path: Path, result_path: Path) -> None:
    request = _strict_json_load(request_path)
    build = request["build"]
    _activate_worker_paths(build)
    import numpy as np
    from bed_reader import open_bed, to_bed
    from summit.context import array_sha256

    workspace = Path(request["workspace"]).resolve(strict=True)
    kind = request["input_kind"]
    n = int(request["n"])
    m = int(request["m"])
    q = int(request["q"])
    k = int(request["k"])
    groups = int(request["groups"])
    traits = int(request["traits"])
    residual_count = int(request["residual_bases"])
    fixed_rank = int(request["fixed_rank"])
    seed = int(request["seed"])
    if m < k * groups:
        raise ValueError("M must cover every annotation in every deletion group")

    if kind == "synthetic":
        if n * m > int(request["max_synthetic_cells"]):
            raise ValueError("synthetic input exceeds --max-synthetic-cells")
        rng = np.random.default_rng(seed)
        genotype = rng.integers(0, 3, size=(n, m), dtype=np.int8).astype(np.float64)
        missing_mask = rng.random((n, m)) < 0.002
        genotype[missing_mask] = np.nan
        prefix = workspace / "synthetic"
        properties = {
            "fid": [f"F{index}" for index in range(n)],
            "iid": [f"I{index}" for index in range(n)],
            "chromosome": [str(1 + index // max(1, m // 2)) for index in range(m)],
            "sid": [f"s6v{index}" for index in range(m)],
            "bp_position": [1000 + index for index in range(m)],
            "allele_1": ["A" if index % 2 == 0 else "C" for index in range(m)],
            "allele_2": ["G" if index % 2 == 0 else "T" for index in range(m)],
        }
        to_bed(str(prefix) + ".bed", genotype, properties=properties, count_A1=True)
        del genotype, missing_mask
    elif kind == "real-prefix":
        prefix = Path(request["geno_prefix"])
        if prefix.suffix in {".bed", ".bim", ".fam"}:
            prefix = prefix.with_suffix("")
        prefix = prefix.expanduser().resolve()
    else:
        raise ValueError("unsupported input kind")

    paths = [Path(str(prefix) + extension) for extension in (".bed", ".bim", ".fam")]
    for path in paths:
        _regular_file(path)
    bed = open_bed(paths[0], count_A1=True, num_threads=1)
    if n > bed.iid_count or m > bed.sid_count:
        raise ValueError("retained prefix exceeds PLINK dimensions")
    samples = np.arange(n, dtype=np.int64)
    variants = np.arange(m, dtype=np.int64)
    missing_counts = np.zeros(m, dtype=np.int64)
    sums = np.zeros(m, dtype=np.float64)
    observed = np.zeros(m, dtype=np.int64)
    missing_digest = _array_sha256_stream_header(np.uint8, (m, n))
    genotype_digest = _array_sha256_stream_header(np.float64, (m, n))
    preparation_block = int(request["preparation_variant_block"])
    for begin in range(0, m, preparation_block):
        end = min(m, begin + preparation_block)
        block = bed.read(
            index=(slice(0, n), slice(begin, end)),
            dtype="float64",
            order="F",
            num_threads=1,
        )
        missing = np.isnan(block)
        missing_counts[begin:end] = np.sum(missing, axis=0, dtype=np.int64)
        sums[begin:end] = np.nansum(block, axis=0, dtype=np.float64)
        observed[begin:end] = n - missing_counts[begin:end]
        missing_digest.update(
            np.ascontiguousarray(missing.T, dtype=np.uint8).tobytes(order="C")
        )
        genotype_digest.update(
            np.ascontiguousarray(block.T, dtype="<f8").tobytes(order="C")
        )
    if np.any(observed <= 0):
        raise ValueError("benchmark selection contains an all-missing variant")
    affine_mean = sums / observed
    affine_inverse_scale = np.ones(m, dtype=np.float64)

    coordinate = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    fixed_columns = [np.ones(n, dtype=np.float64)]
    for degree in range(1, fixed_rank):
        fixed_columns.append(coordinate**degree)
    fixed_basis = np.asfortranarray(
        np.linalg.qr(np.column_stack(fixed_columns), mode="reduced")[0]
    )
    context_candidates = (
        np.ones(n, dtype=np.float64),
        coordinate,
        np.sin(1.7 * coordinate) + 0.2 * coordinate,
        np.cos(0.8 * coordinate) - 0.15 * coordinate,
    )
    context_basis = np.asfortranarray(np.column_stack(context_candidates[:q]))

    indices = np.arange(m, dtype=np.int64)
    if request["group_layout"] == "interleaved":
        group_index = indices % groups
        annotation_index = (indices // groups) % k
    elif request["group_layout"] == "contiguous":
        group_index = np.minimum(groups - 1, indices * groups // m)
        annotation_index = indices % k
    else:
        raise ValueError("unsupported group layout")
    if request["annotation_mode"] == "strict_disjoint_binary_v1":
        annotations = np.zeros((m, k), dtype=np.float64, order="F")
        annotations[indices, annotation_index] = 1.0
    elif request["annotation_mode"] == "generic_nonnegative_weights_v1":
        annotations = np.empty((m, k), dtype=np.float64, order="F")
        for annotation in range(k):
            annotations[:, annotation] = 0.25 + (
                (indices + 3 * annotation) % (k + 5)
            ) / (k + 5)
    else:
        raise ValueError("unsupported annotation mode")

    rng = np.random.default_rng(seed + 41)
    phenotypes = np.asfortranarray(rng.normal(size=(n, traits)))
    residual_columns = [np.ones(n, dtype=np.float64)]
    for degree in range(1, residual_count):
        residual_columns.append(coordinate**degree)
    residual_basis = np.asfortranarray(np.column_stack(residual_columns))
    sample_keys = _philox_keys(int(request["sample_probes"]), seed=seed, domain=1)
    variant_keys = _philox_keys(int(request["variant_probes"]), seed=seed, domain=2)

    bundle = workspace / "case_arrays.npz"
    np.savez(
        bundle,
        retained_sample_indices=samples,
        retained_variant_indices=variants,
        affine_mean=affine_mean,
        affine_inverse_scale=affine_inverse_scale,
        expected_missing_counts=missing_counts,
        fixed_basis=fixed_basis,
        context_basis=context_basis,
        annotation_weights=annotations,
        group_index=np.ascontiguousarray(group_index, dtype=np.int64),
        phenotype_batch=phenotypes,
        residual_basis=residual_basis,
        sample_philox_keys=sample_keys,
        variant_philox_keys=variant_keys,
    )
    hash_large = bool(request["hash_large_inputs"])
    maximum_hash_bytes = int(request["max_input_hash_bytes"])
    file_identities = {
        path.suffix[1:]: _file_identity(
            path,
            hash_content=hash_large or path.stat().st_size <= maximum_hash_bytes,
        )
        for path in paths
    }
    array_identities = {
        name: array_sha256(value)
        for name, value in {
            "retained_sample_indices": samples,
            "retained_variant_indices": variants,
            "affine_mean": affine_mean,
            "affine_inverse_scale": affine_inverse_scale,
            "fixed_basis": fixed_basis,
            "context_basis": context_basis,
            "annotation_weights": annotations,
            "group_index": group_index,
            "phenotype_batch": phenotypes,
            "residual_basis": residual_basis,
            "sample_philox_keys": sample_keys,
            "variant_philox_keys": variant_keys,
        }.items()
    }
    array_identities["missingness"] = missing_digest.hexdigest()
    array_identities["selected_genotype"] = genotype_digest.hexdigest()
    selected_variant_metadata_sha256 = _canonical_sha256(
        {
            "sid": list(map(str, bed.sid[:m])),
            "allele_1": list(map(str, bed.allele_1[:m])),
            "allele_2": list(map(str, bed.allele_2[:m])),
        }
    )
    selected_sample_metadata_sha256 = _canonical_sha256(
        {
            "fid": list(map(str, bed.fid[:n])),
            "iid": list(map(str, bed.iid[:n])),
        }
    )
    metadata = {
        "schema": PREPARED_CASE_SCHEMA,
        "input_kind": kind,
        "prefix": str(prefix),
        "dimensions": {
            "n": n,
            "m": m,
            "q": q,
            "k": k,
            "groups": groups,
            "traits": traits,
            "residual_bases": residual_count,
            "fixed_rank": fixed_rank,
            "sample_probes": int(request["sample_probes"]),
            "variant_probes": int(request["variant_probes"]),
        },
        "annotation_mode": request["annotation_mode"],
        "group_layout": request["group_layout"],
        "seed": seed,
        "bundle": str(bundle),
        "bundle_sha256": _file_sha256(bundle),
        "files": file_identities,
        "full_input_hash_complete": all(
            value["sha256"] is not None for value in file_identities.values()
        ),
        "arrays": array_identities,
    }
    metadata["science_identity"] = {
        "schema": "context_native_stage6_science_identity_v1",
        "input_kind": kind,
        "dimensions": metadata["dimensions"],
        "annotation_mode": request["annotation_mode"],
        "group_layout": request["group_layout"],
        "seed": seed,
        "selected_variant_metadata_sha256": selected_variant_metadata_sha256,
        "selected_sample_metadata_sha256": selected_sample_metadata_sha256,
        "full_file_content_sha256": {
            extension: file_identities[extension]["sha256"]
            for extension in ("bed", "bim", "fam")
        },
        "arrays": array_identities,
    }
    metadata["case_identity_sha256"] = _canonical_sha256(metadata["science_identity"])
    _write_internal_json(result_path, metadata)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    try:
        return [_jsonable(item) for item in value]
    except TypeError as error:
        raise ValueError(f"unsupported JSON evidence type {type(value)!r}") from error


def _performance_report_required(report: Mapping[str, Any], mode: str) -> None:
    required = {
        "schema",
        "schema_version",
        "metadata_only",
        "contextual_native_api_version",
        "contextual_backend_version",
        "contextual_backend",
        "contextual_execution_backend",
        "contextual_build_id",
        "source_tree_sha256",
        "execution_plan_sha256",
        "lifecycle",
        "execution_mode",
        "units",
        "dimensions",
        "selected_plan",
        "runtime_policy",
        "admission_bytes",
        "phase_order",
        "phases",
        "category_order",
        "categories",
        "operation_order",
        "operations",
        "accounting",
        "totals",
        "resource_capabilities",
        "invariants",
        "categories_are_nonoverlapping",
        "categories_cover_entire_run",
    }
    if set(report) != required:
        missing = sorted(required - set(report))
        extra = sorted(set(report) - required)
        raise ValueError(
            "native performance report top-level keys mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    if report["schema"] != "contextual_native_performance_report_v1":
        raise ValueError("native performance report schema mismatch")
    if report["schema_version"] != 1 or report["metadata_only"] is not True:
        raise ValueError("native performance report envelope mismatch")
    if report["execution_mode"] != mode:
        raise ValueError("native performance report execution mode mismatch")
    invariants = report["invariants"]
    expected_invariants = {
        "phase_wall_within_run": True,
        "phase_process_cpu_within_run": True,
        "category_wall_within_active_phases": True,
        "category_process_cpu_within_active_phases": True,
        "descriptor_accounting_exact": True,
        "protected_calls_match_semantic_ledger": True,
        "scientific_state_unchanged_verified": True,
        "report_contains_scientific_ndarray": False,
        "instrumentation_changes_execution_plan": False,
    }
    if not isinstance(invariants, Mapping) or dict(invariants) != expected_invariants:
        raise ValueError("native performance report invariant mapping mismatch")
    if report["categories_are_nonoverlapping"] is not True:
        raise ValueError("native performance categories are not nonoverlapping")


def _protected_trace(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    try:
        events = manifest["execution"]["telemetry"]["events"]
    except (KeyError, TypeError) as error:
        raise ValueError("artifact manifest lacks protected telemetry") from error
    trace: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_class") != "protected_call":
            continue
        trace.append(
            {
                "semantic_anchor": event.get("semantic_anchor"),
                "operation": event.get("operation", event.get("semantic_operation")),
                "rows": event.get("rows"),
                "columns": event.get("columns"),
                "reduction": event.get("reduction"),
                "left_stride": event.get("left_stride"),
                "right_stride": event.get("right_stride"),
                "output_stride": event.get("output_stride"),
                "transpose_left": event.get("transpose_left"),
                "operand_fingerprint_fnv64": event.get("operand_fingerprint_fnv64"),
                "witness_fingerprint_fnv64": event.get("witness_fingerprint_fnv64"),
                "accepted_output_fingerprint_fnv64": event.get(
                    "accepted_output_fingerprint_fnv64"
                ),
            }
        )
    if not trace:
        raise ValueError("artifact contains no protected-call trace")
    return trace, _canonical_sha256(trace)


def _save_science_capsule(path: Path, reference: Any, trait: Any, fit: Any) -> None:
    import numpy as np

    arrays: dict[str, Any] = {}
    for name in REFERENCE_ARRAYS:
        arrays[f"reference.{name}"] = np.asarray(getattr(reference, name))
    for name in TRAIT_ARRAYS:
        arrays[f"trait.{name}"] = np.asarray(getattr(trait, name))
    for name in FIT_ARRAYS:
        arrays[f"fit.{name}"] = np.asarray(getattr(fit, name))
    arrays["fit.raw_rank"] = np.asarray([fit.raw_rank], dtype=np.int64)
    np.savez(path, **arrays)


def _compare_capsules(actual_path: Path, baseline_path: Path | None) -> dict[str, Any]:
    import numpy as np

    with np.load(actual_path, allow_pickle=False) as actual_file:
        actual = {name: np.asarray(actual_file[name]) for name in actual_file.files}
    baseline = actual
    if baseline_path is not None:
        with np.load(baseline_path, allow_pickle=False) as baseline_file:
            baseline = {
                name: np.asarray(baseline_file[name]) for name in baseline_file.files
            }
    if set(actual) != set(baseline):
        raise ValueError("science capsule key mismatch")
    discrepancies: dict[str, float] = {}
    exact = True
    within = True
    deletion_within = True
    deletion_exact = True
    rank_identical = True
    for name in sorted(actual):
        left = baseline[name]
        right = actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"science capsule schema mismatch for {name}")
        equal = np.array_equal(left, right)
        exact = exact and bool(equal)
        if left.dtype.kind in "iu":
            max_abs = float(
                np.max(
                    np.abs(left.astype(np.int64) - right.astype(np.int64)), initial=0
                )
            )
            max_relative = max_abs
            tolerance_ratio = max_abs
            accepted = equal
        else:
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            max_abs = float(np.max(difference, initial=0.0))
            scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
            max_relative = float(np.max(difference / scale, initial=0.0))
            tolerance = 1.0e-9 + 1.0e-11 * np.abs(left.astype(np.float64))
            tolerance_ratio = float(np.max(difference / tolerance, initial=0.0))
            accepted = bool(np.all(difference <= tolerance))
        discrepancies[f"{name}.max_abs"] = max_abs
        discrepancies[f"{name}.max_relative"] = max_relative
        discrepancies[f"{name}.tolerance_ratio"] = tolerance_ratio
        within = within and accepted
        if name.startswith("reference.group_") or name.startswith("trait.group_"):
            deletion_within = deletion_within and accepted
            deletion_exact = deletion_exact and equal
        if name.startswith("fit.raw_rank") or name.startswith("fit.raw_solve_"):
            rank_identical = rank_identical and equal
    return {
        "fixed_probe_discrepancies": discrepancies,
        "all_arrays_exact": exact,
        "all_arrays_within_tolerance": within,
        "every_deletion_within_tolerance": deletion_within,
        "every_deletion_exact": deletion_exact,
        "rank_estimability_identical": rank_identical,
    }


def _comparison_passes(
    comparison: Mapping[str, Any], comparison_mode: str, trace_exact: bool
) -> bool:
    if comparison_mode == "exact_trace_v1":
        return bool(comparison["all_arrays_exact"]) and trace_exact
    if comparison_mode == "science_tolerance_v1":
        return bool(comparison["all_arrays_within_tolerance"])
    raise ValueError(f"unsupported comparison mode {comparison_mode!r}")


def _open_descriptors(prefix: Path) -> list[int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    return [
        os.open(str(prefix) + extension, flags)
        for extension in (".bed", ".bim", ".fam")
    ]


def _worker_plan(plan_value: Mapping[str, Any]) -> Any:
    from summit.context.performance_v1 import ContextualTablaPlanV1

    values = dict(plan_value)
    values.pop("schema", None)
    return ContextualTablaPlanV1(**values)


def _integrity_identity_sha256(plan: Any) -> str:
    return _canonical_sha256(
        {
            "integrity_policy": plan.integrity_policy,
            "integrity_backend": plan.integrity_backend,
            "fault_coverage_contract": FAULT_COVERAGE_CONTRACT,
        }
    )


def _execute_worker(request_path: Path, result_path: Path) -> None:
    request = _strict_json_load(request_path)
    build = request["build"]
    if request.get("cache_state_policy") != CACHE_STATE_POLICY:
        raise RuntimeError("worker cache-state policy mismatch")
    for key, value in build["environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"worker environment {key} disagrees with build spec")
    expected_cpus = tuple(int(value) for value in request["cpus"])
    actual_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if actual_cpus != expected_cpus:
        raise RuntimeError("worker taskset affinity differs from requested CPU list")
    installed_package_sha256 = _package_sha256(Path(build["install_prefix"]))
    if installed_package_sha256 != build["expected_package_sha256"]:
        raise RuntimeError("installed package changed before worker execution")
    qualification_path = build.get("qualification_evidence_path")
    qualification_sha256 = build.get("qualification_evidence_sha256")
    if qualification_path is not None and (
        qualification_sha256 is None
        or _file_sha256(Path(qualification_path)) != qualification_sha256
    ):
        raise RuntimeError("fault qualification evidence changed")
    _activate_worker_paths(build)

    import numpy as np
    from bed_reader import open_bed
    from summit import gxeldcore
    from summit.context import (
        ContextualReferencePublicationIdentityV1,
        GenotypeScalePlanV1,
        GenotypeScalePolicy,
        array_sha256,
        canonical_sha256,
        contextual_variant_order_allele_sha256_v1,
        load_contextual_reference_v1,
        run_contextual_reference_v1,
        write_contextual_reference_v1,
    )
    from summit.context.fit_v1 import (
        fit_contextual_model_v1,
        load_contextual_fit_v1,
        write_contextual_fit_v1,
    )
    from summit.context.trait_v1 import (
        ContextualTraitPublicationIdentityV1,
        load_contextual_trait_v1,
        run_contextual_trait_v1,
        write_contextual_trait_v1,
    )

    runtime_identity = _runtime_dependency_identity()
    if (
        runtime_identity["python_runtime_sha256"]
        != build["expected_python_runtime_sha256"]
        or runtime_identity["dependency_runtime_sha256"]
        != build["expected_dependency_runtime_sha256"]
    ):
        raise RuntimeError("worker Python/dependency runtime identity mismatch")

    native_path = Path(gxeldcore.__file__).resolve()
    if native_path != Path(build["native_module"]).resolve():
        raise RuntimeError("loaded native module path differs from build spec")
    native_module_sha256 = _file_sha256(native_path)
    if native_module_sha256 != build["expected_native_sha256"]:
        raise RuntimeError("loaded native module changed")
    build_info = _jsonable(dict(gxeldcore.build_info()))
    if (
        build_info.get("source_commit") != build["expected_source_commit"]
        or build_info.get("source_tree_sha256") != build["expected_source_tree_sha256"]
    ):
        raise RuntimeError("native build provenance differs from build spec")

    case = _strict_json_load(Path(request["case_metadata"]))
    if case["schema"] != PREPARED_CASE_SCHEMA:
        raise ValueError("prepared case schema mismatch")
    _verify_case_science_identity(case)
    _verify_case_file_identities(
        case, require_full_hashes=False, verify_content_hashes=False
    )
    bundle_path = Path(case["bundle"])
    if _file_sha256(bundle_path) != case["bundle_sha256"]:
        raise RuntimeError("prepared case bundle changed")
    with np.load(bundle_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    prefix = Path(case["prefix"])
    bed = open_bed(str(prefix) + ".bed", count_A1=True, num_threads=1)
    retained_samples = arrays["retained_sample_indices"]
    retained_variants = arrays["retained_variant_indices"]
    m = retained_variants.size
    expected_ids = list(map(str, bed.sid[:m]))
    counted_alleles = list(map(str, bed.allele_1[:m]))
    other_alleles = list(map(str, bed.allele_2[:m]))
    counted_a1 = np.ones(m, dtype=np.uint8)
    annotation_names = [
        f"annotation-{index}" for index in range(arrays["annotation_weights"].shape[1])
    ]
    group_names = [f"group-{index}" for index in range(case["dimensions"]["groups"])]
    trait_names = [f"trait-{index}" for index in range(case["dimensions"]["traits"])]
    residual_names = [
        f"residual-{index}" for index in range(case["dimensions"]["residual_bases"])
    ]
    order_sha = array_sha256(retained_variants)
    mean_sha = array_sha256(arrays["affine_mean"])
    inverse_sha = array_sha256(arrays["affine_inverse_scale"])
    scale_plan = GenotypeScalePlanV1(
        policy=GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        retained_variant_order_sha256=order_sha,
        allele_orientation="bim_a1_counted_v1",
        allele_coding="plink_bed_snp_major_diploid_hardcall_v1",
        centering_source="provided_v1",
        centering_formula="provided_variant_affine_mean_v1",
        scaling_formula="dosage_minus_mean_times_inverse_scale_v1",
        missing_imputation="sealed_mean_v1",
        ploidy_policy="diploid_v1",
        affine_mean_sha256=mean_sha,
        affine_inverse_scale_sha256=inverse_sha,
    )
    fixed_basis_sha = array_sha256(arrays["fixed_basis"])
    context_basis_sha = array_sha256(arrays["context_basis"])
    annotation_sha = array_sha256(arrays["annotation_weights"])
    group_sha = array_sha256(arrays["group_index"])
    sample_map_sha = array_sha256(retained_samples)
    sample_order_sha = canonical_sha256(
        {"retained_sample_rows": retained_samples.tolist()}
    )
    fixed_spec_sha = canonical_sha256({"fixed_effect_basis": fixed_basis_sha})
    basis_spec_sha = canonical_sha256(
        {"context_basis": "deterministic_stage6_benchmark_v1"}
    )
    basis_calibration_sha = canonical_sha256({"evaluated_phi": context_basis_sha})
    variant_allele_sha = contextual_variant_order_allele_sha256_v1(
        retained_variants,
        expected_ids,
        counted_alleles,
        other_alleles,
        counted_a1,
    )
    sample_probe_sha = array_sha256(arrays["sample_philox_keys"])
    variant_probe_sha = array_sha256(arrays["variant_philox_keys"])
    publication = ContextualReferencePublicationIdentityV1(
        sample_order_sha256=sample_order_sha,
        variant_order_allele_sha256=variant_allele_sha,
        fixed_effect_spec_sha256=fixed_spec_sha,
        basis_specification_sha256=basis_spec_sha,
        basis_calibration_sha256=basis_calibration_sha,
        retained_sample_map_sha256=sample_map_sha,
        retained_variant_order_sha256=order_sha,
        fixed_basis_sha256=fixed_basis_sha,
        evaluated_phi_sha256=context_basis_sha,
        genotype_scale_plan_sha256=scale_plan.digest,
        missingness_sha256=case["arrays"]["missingness"],
        annotation_map_sha256=annotation_sha,
        annotation_names=tuple(annotation_names),
        group_map_sha256=group_sha,
        group_ids=tuple(group_names),
        sample_probe_policy="numpy_philox_per_probe_key_v1",
        sample_probe_identity_sha256=sample_probe_sha,
        variant_probe_policy="numpy_philox_variant_per_probe_key_v1",
        variant_probe_identity_sha256=variant_probe_sha,
    )
    plan = _worker_plan(request["plan"])
    shared_block = plan.source_variant_block
    reference_options = {
        "annotation_mode": plan.annotation_mode,
        "retained_variant_order_sha256": order_sha,
        "affine_mean_sha256": mean_sha,
        "affine_inverse_scale_sha256": inverse_sha,
        "missingness_sha256": case["arrays"]["missingness"],
        "scale_plan_sha256": scale_plan.digest,
        "centering_source": "provided_v1",
        "sample_probe_count": case["dimensions"]["sample_probes"],
        "variant_block": shared_block,
        "sample_probe_resident": plan.sample_probe_resident_count,
        "sample_probe_tile": plan.sample_probe_tile,
        "action_tile": plan.action_tile,
        "annotation_tile": plan.annotation_tile,
        "context_tile": plan.context_tile,
        "decode_threads": plan.decode_threads,
        "blas_threads": plan.blas_threads,
        "workspace_cap_bytes": int(request["max_workspace_bytes"]),
        "variant_probes": None,
        "variant_philox_keys": arrays["variant_philox_keys"],
        "variant_probe_count": case["dimensions"]["variant_probes"],
        "variant_probe_tile": plan.variant_probe_tile,
        "group_tile": plan.group_tile,
        "grouped_algorithm": plan.grouped_attribution_algorithm,
        "direct_grouped_scaling": (
            "action_scaled_v1"
            if plan.direct_grouped_scaling_policy == "not_applicable"
            else plan.direct_grouped_scaling_policy
        ),
        "enable_grouped_differential": False,
        "numa_policy": plan.numa_policy,
        "output_numa_node": plan.output_numa_node,
    }
    if plan.telemetry_capacity_bytes:
        reference_options["telemetry_capacity"] = plan.telemetry_capacity_bytes

    output_directory = Path(request["output_directory"])
    output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    workflow_rusage_before = _rusage_snapshot()
    workflow_started_ns = time.perf_counter_ns()
    workflow_cpu_started_ns = time.process_time_ns()

    descriptors = _open_descriptors(prefix)
    try:
        reference_executor = gxeldcore.ContextualReferenceExecutorV1(
            descriptors[0],
            descriptors[1],
            descriptors[2],
            retained_samples,
            retained_variants,
            expected_ids,
            counted_alleles,
            other_alleles,
            counted_a1,
            arrays["affine_mean"],
            arrays["affine_inverse_scale"],
            arrays["expected_missing_counts"],
            arrays["fixed_basis"],
            arrays["context_basis"],
            arrays["annotation_weights"],
            arrays["group_index"],
            None,
            arrays["sample_philox_keys"],
            annotation_names,
            group_names,
            **reference_options,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    reference_preflight = _jsonable(dict(reference_executor.preflight()))
    reference_started_ns = time.perf_counter_ns()
    reference_cpu_started_ns = time.process_time_ns()
    reference = run_contextual_reference_v1(reference_executor, publication)
    reference_performance = _jsonable(dict(reference_executor.performance_report()))
    reference_wall_seconds = (time.perf_counter_ns() - reference_started_ns) / 1.0e9
    reference_cpu_seconds = (time.process_time_ns() - reference_cpu_started_ns) / 1.0e9
    _performance_report_required(reference_performance, "reference_v1")

    started = time.perf_counter_ns()
    reference_path = write_contextual_reference_v1(
        reference, output_directory / "reference"
    )
    reference_write_seconds = (time.perf_counter_ns() - started) / 1.0e9
    started = time.perf_counter_ns()
    loaded_reference = load_contextual_reference_v1(reference_path)
    reference_load_seconds = (time.perf_counter_ns() - started) / 1.0e9

    trait_publication = ContextualTraitPublicationIdentityV1(
        sample_order_sha256=sample_order_sha,
        variant_order_allele_sha256=variant_allele_sha,
        fixed_effect_spec_sha256=fixed_spec_sha,
        basis_specification_sha256=basis_spec_sha,
        basis_calibration_sha256=basis_calibration_sha,
        compatible_reference_identity_sha256=loaded_reference.manifest_sha256,
        retained_sample_map_sha256=sample_map_sha,
        retained_variant_order_sha256=order_sha,
        fixed_basis_sha256=fixed_basis_sha,
        evaluated_phi_sha256=context_basis_sha,
        genotype_scale_plan_sha256=scale_plan.digest,
        missingness_sha256=case["arrays"]["missingness"],
        annotation_map_sha256=annotation_sha,
        annotation_names=tuple(annotation_names),
        group_map_sha256=group_sha,
        group_ids=tuple(group_names),
        phenotype_batch_sha256=array_sha256(arrays["phenotype_batch"]),
        residual_basis_sha256=array_sha256(arrays["residual_basis"]),
        trait_ids=tuple(trait_names),
        residual_names=tuple(residual_names),
    )
    trait_options = {
        "annotation_mode": plan.annotation_mode,
        "retained_variant_order_sha256": order_sha,
        "affine_mean_sha256": mean_sha,
        "affine_inverse_scale_sha256": inverse_sha,
        "missingness_sha256": case["arrays"]["missingness"],
        "scale_plan_sha256": scale_plan.digest,
        "centering_source": "provided_v1",
        "variant_block": plan.trait_variant_block,
        "trait_feature_tile": plan.trait_variant_block,
        "decode_threads": plan.decode_threads,
        "blas_threads": plan.blas_threads,
        "workspace_cap_bytes": int(request["max_workspace_bytes"]),
        "numa_policy": plan.numa_policy,
        "output_numa_node": plan.output_numa_node,
    }
    if plan.telemetry_capacity_bytes:
        trait_options["telemetry_capacity"] = plan.telemetry_capacity_bytes
    descriptors = _open_descriptors(prefix)
    try:
        trait_executor = gxeldcore.ContextualTraitExecutorV1(
            descriptors[0],
            descriptors[1],
            descriptors[2],
            retained_samples,
            retained_variants,
            expected_ids,
            counted_alleles,
            other_alleles,
            counted_a1,
            arrays["affine_mean"],
            arrays["affine_inverse_scale"],
            arrays["expected_missing_counts"],
            arrays["fixed_basis"],
            arrays["context_basis"],
            arrays["annotation_weights"],
            arrays["group_index"],
            arrays["phenotype_batch"],
            arrays["residual_basis"],
            annotation_names,
            group_names,
            trait_names,
            residual_names,
            **trait_options,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    trait_preflight = _jsonable(dict(trait_executor.preflight()))
    trait_started_ns = time.perf_counter_ns()
    trait_cpu_started_ns = time.process_time_ns()
    trait = run_contextual_trait_v1(trait_executor, trait_publication)
    trait_performance = _jsonable(dict(trait_executor.performance_report()))
    trait_wall_seconds = (time.perf_counter_ns() - trait_started_ns) / 1.0e9
    trait_cpu_seconds = (time.process_time_ns() - trait_cpu_started_ns) / 1.0e9
    _performance_report_required(trait_performance, "trait_v1")

    started = time.perf_counter_ns()
    trait_path = write_contextual_trait_v1(trait, output_directory / "trait")
    trait_write_seconds = (time.perf_counter_ns() - started) / 1.0e9
    started = time.perf_counter_ns()
    loaded_trait = load_contextual_trait_v1(trait_path)
    trait_load_seconds = (time.perf_counter_ns() - started) / 1.0e9

    started = time.perf_counter_ns()
    fit = fit_contextual_model_v1(
        loaded_reference,
        loaded_trait,
        trait_selector=0,
        project_psd=False,
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1.0e9
    started = time.perf_counter_ns()
    fit_path = write_contextual_fit_v1(fit, output_directory / "fit")
    fit_write_seconds = (time.perf_counter_ns() - started) / 1.0e9
    started = time.perf_counter_ns()
    loaded_fit = load_contextual_fit_v1(fit_path)
    fit_load_seconds = (time.perf_counter_ns() - started) / 1.0e9

    workflow_wall_seconds = (time.perf_counter_ns() - workflow_started_ns) / 1.0e9
    workflow_cpu_seconds = (time.process_time_ns() - workflow_cpu_started_ns) / 1.0e9
    workflow_rusage_after = _rusage_snapshot()
    reference_trace, reference_trace_sha = _protected_trace(reference.manifest)
    trait_trace, trait_trace_sha = _protected_trace(trait.manifest)
    combined_trace_sha = _canonical_sha256(
        {"reference": reference_trace, "trait": trait_trace}
    )
    capsule_path = Path(request["capsule_path"])
    _save_science_capsule(capsule_path, reference, trait, loaded_fit)
    comparison = _compare_capsules(
        capsule_path,
        Path(request["baseline_capsule"]) if request["baseline_capsule"] else None,
    )
    trace_exact = True
    if request["baseline_trace_sha256"] is not None:
        trace_exact = combined_trace_sha == request["baseline_trace_sha256"]
    comparison["protected_trace_exact"] = trace_exact
    comparison["comparison_mode"] = request["comparison_mode"]
    comparison["comparison_passed"] = _comparison_passes(
        comparison, request["comparison_mode"], trace_exact
    )

    rank_identity_sha256 = canonical_sha256(
        {
            "raw_rank": loaded_fit.raw_rank,
            "retained_directions_sha256": array_sha256(
                loaded_fit.raw_solve_retained_directions
            ),
            "null_space_sha256": array_sha256(loaded_fit.raw_solve_null_space),
        }
    )
    integrity_identity_sha256 = _integrity_identity_sha256(plan)
    backend_identity = {
        "native_backend": str(reference_performance["contextual_backend"]),
        "native_execution_backend": str(
            reference_performance["contextual_execution_backend"]
        ),
        "blas_vendor": str(build_info.get("blas_vendor", "unknown")),
        "blas_version": str(build_info.get("blas_runtime_config", "unknown")),
        "openmp_runtime": (
            "enabled" if build_info.get("openmp_enabled") is True else "disabled"
        ),
        "numeric_policy": plan.numeric_policy,
    }
    build_identity = {
        "source_commit": str(build_info["source_commit"]),
        "source_tree_sha256": str(build_info["source_tree_sha256"]),
        "native_build_provenance_sha256": _native_build_provenance_sha256(
            build_info, native_module_sha256, installed_package_sha256
        ),
        "python_runtime_sha256": runtime_identity["python_runtime_sha256"],
        "dependency_runtime_sha256": runtime_identity["dependency_runtime_sha256"],
        "compiler_id": str(build_info.get("compiler_id", "unknown")),
        "compiler_version": str(build_info.get("compiler_version", "unknown")),
        "build_type": str(build_info.get("build_type", "unknown")),
        "sanitizer_mode": str(build_info.get("sanitizer_mode", "unknown")),
        "effective_optimization": str(build_info.get("optimization", "unknown")),
        "architecture_tuning": str(build_info.get("architecture_tuning", "unknown")),
    }
    result = {
        "schema": WORKER_SCHEMA,
        "plan_id": request["plan_id"],
        "build_label": build["label"],
        "cache_state_policy": CACHE_STATE_POLICY,
        "pid": os.getpid(),
        "affinity": list(actual_cpus),
        "build_info": build_info,
        "runtime_dependency_identity": runtime_identity,
        "backend_identity": backend_identity,
        "build_identity": build_identity,
        "science_identity_sha256": case["case_identity_sha256"],
        "integrity_identity_sha256": integrity_identity_sha256,
        "rank_identity_sha256": rank_identity_sha256,
        "reference": {
            "adapter_wall_seconds": reference_wall_seconds,
            "adapter_cpu_seconds": reference_cpu_seconds,
            "preflight": reference_preflight,
            "performance_report": reference_performance,
            "manifest_sha256": reference.manifest_sha256,
            "protected_trace_sha256": reference_trace_sha,
        },
        "trait": {
            "adapter_wall_seconds": trait_wall_seconds,
            "adapter_cpu_seconds": trait_cpu_seconds,
            "preflight": trait_preflight,
            "performance_report": trait_performance,
            "manifest_sha256": trait.manifest_sha256,
            "protected_trace_sha256": trait_trace_sha,
        },
        "combined_protected_trace_sha256": combined_trace_sha,
        "workflow": {
            "wall_seconds": workflow_wall_seconds,
            "cpu_seconds": workflow_cpu_seconds,
            "rusage_before": workflow_rusage_before,
            "rusage_after": workflow_rusage_after,
        },
        "artifact_timings_seconds": {
            "reference_write": reference_write_seconds,
            "reference_load": reference_load_seconds,
            "trait_write": trait_write_seconds,
            "trait_load": trait_load_seconds,
            "fit": fit_seconds,
            "fit_write": fit_write_seconds,
            "fit_load": fit_load_seconds,
        },
        "artifact_round_trip_passed": True,
        "comparison": comparison,
        "capsule_path": str(capsule_path),
        "capsule_sha256": _file_sha256(capsule_path),
        "full_input_hash_complete": case["full_input_hash_complete"],
    }
    _verify_case_file_identities(
        case, require_full_hashes=False, verify_content_hashes=False
    )
    if (
        _package_sha256(Path(build["install_prefix"]))
        != build["expected_package_sha256"]
    ):
        raise RuntimeError("installed package changed during worker execution")
    final_runtime_identity = _runtime_dependency_identity()
    if (
        final_runtime_identity["python_runtime_sha256"]
        != build["expected_python_runtime_sha256"]
        or final_runtime_identity["dependency_runtime_sha256"]
        != build["expected_dependency_runtime_sha256"]
    ):
        raise RuntimeError("Python/dependency runtime changed during worker execution")
    _write_internal_json(result_path, result)


def _median(values: Sequence[int | float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty benchmark sample")
    result = float(statistics.median(values))
    if not math.isfinite(result):
        raise ValueError("benchmark sample contains a non-finite value")
    return result


def _constant(values: Sequence[Any], label: str) -> Any:
    if not values:
        raise ValueError(f"cannot validate empty {label}")
    encoded = [_canonical_json(value) for value in values]
    if len(set(encoded)) != 1:
        raise ValueError(f"replicates disagree on {label}")
    return values[0]


def _combined_reports(worker: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return (
        worker["reference"]["performance_report"],
        worker["trait"]["performance_report"],
    )


def _phase_sample(worker: Mapping[str, Any], field: str) -> dict[str, float]:
    planner = _planner_module()
    return {
        phase: sum(
            float(report["phases"][phase][field])
            for report in _combined_reports(worker)
        )
        / 1.0e9
        for phase in planner.PERFORMANCE_PHASE_KEYS_V1
    }


def _category_sample(worker: Mapping[str, Any], field: str) -> dict[str, float]:
    planner = _planner_module()
    return {
        category: sum(
            float(report["categories"][category][field])
            for report in _combined_reports(worker)
        )
        / 1.0e9
        for category in planner.PERFORMANCE_CATEGORY_KEYS_V1
    }


def _descriptor_sample(worker: Mapping[str, Any], suffix: str) -> dict[str, int]:
    reference = worker["reference"]["preflight"]
    trait = worker["trait"]["preflight"]
    result = {
        "source": int(reference[f"source_{suffix}"]),
        "action": int(reference[f"action_{suffix}"]),
        "group": int(reference[f"group_{suffix}"]),
        "same_person": int(reference[f"same_person_{suffix}"]),
        "trait": int(trait[f"trait_{suffix}"]),
    }
    reference_total = sum(result[key] for key in result if key != "trait")
    trait_total = result["trait"]
    accounting_field = {
        "descriptor_passes": "observed_descriptor_passes",
        "decoded_blocks": "observed_decoded_blocks",
    }[suffix]
    reports = _combined_reports(worker)
    if reference_total != int(reports[0]["accounting"][accounting_field]):
        raise ValueError(f"reference {suffix} partition disagrees with native total")
    if trait_total != int(reports[1]["accounting"][accounting_field]):
        raise ValueError(f"trait {suffix} partition disagrees with native total")
    return result


def _protected_sample(
    worker: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, Any], dict[str, float]]:
    planner = _planner_module()
    counts: dict[str, int] = {}
    shapes: dict[str, Any] = {}
    throughputs: dict[str, float] = {}
    histogram_keys = tuple(planner.PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1)
    for operation in planner.PROTECTED_OPERATION_KEYS_V1:
        histogram: Counter[tuple[Any, ...]] = Counter()
        calls = 0
        flop_count = 0.0
        primary_wall_ns = 0
        for report in _combined_reports(worker):
            item = report["operations"][operation]
            calls += int(item["calls"])
            flop_count += float(item["logical_flop_count"])
            primary_wall_ns += int(item["primary_wall_ns"])
            for entry in item["shape_histogram"]:
                identity = tuple(entry[key] for key in histogram_keys[:-1])
                histogram[identity] += int(entry["calls"])
        entries = [
            {
                **dict(zip(histogram_keys[:-1], identity, strict=True)),
                "calls": histogram[identity],
            }
            for identity in sorted(histogram)
        ]
        counts[operation] = calls
        if entries:
            shapes[operation] = {
                "minimum_rows": min(entry["rows"] for entry in entries),
                "maximum_rows": max(entry["rows"] for entry in entries),
                "minimum_columns": min(entry["columns"] for entry in entries),
                "maximum_columns": max(entry["columns"] for entry in entries),
                "minimum_reduction": min(entry["reduction"] for entry in entries),
                "maximum_reduction": max(entry["reduction"] for entry in entries),
                "shape_histogram": entries,
            }
        else:
            shapes[operation] = {
                "minimum_rows": 0,
                "maximum_rows": 0,
                "minimum_columns": 0,
                "maximum_columns": 0,
                "minimum_reduction": 0,
                "maximum_reduction": 0,
                "shape_histogram": [],
            }
        if sum(entry["calls"] for entry in entries) != calls:
            raise ValueError(f"protected shape ledger mismatch for {operation}")
        throughputs[operation] = (
            flop_count / primary_wall_ns if primary_wall_ns else 0.0
        )
    return counts, shapes, throughputs


def _worker_peak_rss(worker: Mapping[str, Any]) -> int:
    values: list[int] = []
    monitor = worker.get("controller_monitor", {}).get("process_resources", {})
    for key in ("sampled_peak_rss_bytes", "observed_vmhwm_bytes"):
        if isinstance(monitor.get(key), int):
            values.append(int(monitor[key]))
    rusage = worker["workflow"]["rusage_after"]
    values.append(int(rusage["maximum_rss_kib"]) * 1024)
    for report in _combined_reports(worker):
        values.append(int(report["totals"]["process_peak_rss_bytes"]))
    return max(values)


def _random_indexed_visits(
    workers: Sequence[Mapping[str, Any]], group_execution_order: str
) -> int:
    if group_execution_order != "indexed_group_major_v1":
        return 0
    return int(
        _constant(
            [
                int(
                    worker["reference"]["performance_report"]["accounting"][
                        "observed_group_execution_variant_visits"
                    ]
                )
                for worker in workers
            ],
            "indexed group execution visits",
        )
    )


def _global_duplicate_decodes(workers: Sequence[Mapping[str, Any]]) -> int:
    values: list[int] = []
    for worker in workers:
        reports = _combined_reports(worker)
        variants = _constant(
            [int(report["dimensions"]["variants"]) for report in reports],
            "reference/trait retained variant count",
        )
        visits = sum(
            int(report["accounting"]["observed_variant_record_visits"])
            for report in reports
        )
        if visits < variants:
            raise ValueError("physical record visits are smaller than unique variants")
        values.append(visits - variants)
    return int(_constant(values, "global duplicate decodes"))


def _make_metrics(workers: Sequence[Mapping[str, Any]], plan: Any) -> Mapping[str, Any]:
    planner = _planner_module()
    Evidence = planner.ContextualPerformanceEvidenceV1

    def evidence(key: str, value: Any, source: str, kind: str = "measured") -> Any:
        return Evidence(
            evidence_kind=kind,
            value=value,
            unit=planner.PERFORMANCE_METRIC_UNITS_V1[key],
            source=source,
        )

    phase_wall_samples = [_phase_sample(worker, "wall_ns") for worker in workers]
    phase_cpu_samples = [_phase_sample(worker, "process_cpu_ns") for worker in workers]
    category_wall_samples = [_category_sample(worker, "wall_ns") for worker in workers]
    category_cpu_samples = [
        _category_sample(worker, "process_cpu_ns") for worker in workers
    ]
    phase_wall = {
        key: _median([sample[key] for sample in phase_wall_samples])
        for key in planner.PERFORMANCE_PHASE_KEYS_V1
    }
    phase_cpu = {
        key: _median([sample[key] for sample in phase_cpu_samples])
        for key in planner.PERFORMANCE_PHASE_KEYS_V1
    }
    category_wall = {
        key: _median([sample[key] for sample in category_wall_samples])
        for key in planner.PERFORMANCE_CATEGORY_KEYS_V1
    }
    category_cpu = {
        key: _median([sample[key] for sample in category_cpu_samples])
        for key in planner.PERFORMANCE_CATEGORY_KEYS_V1
    }
    descriptor_passes = _constant(
        [_descriptor_sample(worker, "descriptor_passes") for worker in workers],
        "logical descriptor-pass maps",
    )
    decoded_blocks = _constant(
        [_descriptor_sample(worker, "decoded_blocks") for worker in workers],
        "decoded-block maps",
    )
    protected = [_protected_sample(worker) for worker in workers]
    protected_counts = _constant(
        [sample[0] for sample in protected], "protected call counts"
    )
    protected_shapes = _constant(
        [sample[1] for sample in protected], "protected shape histograms"
    )
    protected_throughput = {
        key: _median([sample[2][key] for sample in protected])
        for key in planner.PROTECTED_OPERATION_KEYS_V1
    }
    physical_visits = _constant(
        [
            sum(
                int(report["accounting"]["observed_variant_record_visits"])
                for report in _combined_reports(worker)
            )
            for worker in workers
        ],
        "physical record visits",
    )
    physical_bytes = _constant(
        [
            sum(
                int(report["accounting"]["observed_physical_record_bytes_visited"])
                for report in _combined_reports(worker)
            )
            for worker in workers
        ],
        "logical BED bytes visited",
    )
    random_visits = _random_indexed_visits(workers, plan.group_execution_order)
    duplicate_decodes = _global_duplicate_decodes(workers)
    walls = [float(worker["workflow"]["wall_seconds"]) for worker in workers]
    decode_seconds = [sample["decode"] for sample in category_wall_samples]
    visits_per_second = [
        physical_visits / value if value > 0.0 else 0.0 for value in decode_seconds
    ]
    decoded_gb_per_second = [
        physical_bytes / 1.0e9 / value if value > 0.0 else 0.0
        for value in decode_seconds
    ]
    overhead_ns = [
        sum(
            int(report["totals"]["integrity_overhead_wall_ns"])
            for report in _combined_reports(worker)
        )
        for worker in workers
    ]
    protected_total_ns = [
        sum(
            int(report["totals"]["protected_total_wall_ns"])
            for report in _combined_reports(worker)
        )
        for worker in workers
    ]
    overhead_fractions = [
        overhead / total if total else 0.0
        for overhead, total in zip(overhead_ns, protected_total_ns, strict=True)
    ]
    artifact_seconds = [
        sum(float(value) for value in worker["artifact_timings_seconds"].values())
        for worker in workers
    ]
    admitted_peak = _constant(
        [
            max(
                int(report["admission_bytes"]["required_workspace"])
                for report in _combined_reports(worker)
            )
            for worker in workers
        ],
        "admitted peak bytes",
    )
    values = {
        "end_to_end_wall_seconds": evidence(
            "end_to_end_wall_seconds",
            _median(walls),
            "fresh_worker_reference_trait_artifact_fit_median_v1",
        ),
        "phase_wall_seconds": evidence(
            "phase_wall_seconds", phase_wall, "native_phase_wall_ns_median_v1"
        ),
        "phase_cpu_seconds": evidence(
            "phase_cpu_seconds", phase_cpu, "native_phase_process_cpu_ns_median_v1"
        ),
        "category_wall_seconds": evidence(
            "category_wall_seconds",
            category_wall,
            "native_nonoverlapping_category_wall_ns_median_v1",
        ),
        "category_cpu_seconds": evidence(
            "category_cpu_seconds",
            category_cpu,
            "native_nonoverlapping_category_process_cpu_ns_median_v1",
        ),
        "logical_descriptor_passes": evidence(
            "logical_descriptor_passes",
            descriptor_passes,
            "native_preflight_partition_and_observed_total_v1",
        ),
        "decoded_blocks": evidence(
            "decoded_blocks",
            decoded_blocks,
            "native_preflight_partition_and_observed_total_v1",
        ),
        "physical_record_visits": evidence(
            "physical_record_visits",
            physical_visits,
            "native_observed_variant_record_visits_v1",
        ),
        "physical_read_bytes": evidence(
            "physical_read_bytes",
            None,
            planner.MMAP_PHYSICAL_READ_BYTES_UNAVAILABLE_SOURCE_V1,
            "unavailable",
        ),
        "random_indexed_visits": evidence(
            "random_indexed_visits",
            random_visits,
            "native_observed_group_execution_variant_visits_v1",
        ),
        "duplicate_decodes": evidence(
            "duplicate_decodes",
            duplicate_decodes,
            "combined_physical_visits_minus_one_global_variant_set_v1",
        ),
        "variants_per_second": evidence(
            "variants_per_second",
            _median(visits_per_second),
            "native_record_visits_over_decode_category_wall_v1",
        ),
        "decoded_gb_per_second": evidence(
            "decoded_gb_per_second",
            _median(decoded_gb_per_second),
            "native_logical_bed_bytes_over_decode_category_wall_v1",
        ),
        "protected_call_shapes": evidence(
            "protected_call_shapes",
            protected_shapes,
            "native_exact_protected_shape_histograms_v1",
        ),
        "protected_call_count": evidence(
            "protected_call_count",
            protected_counts,
            "native_exact_semantic_protected_call_ledger_v1",
        ),
        "protected_effective_gflops": evidence(
            "protected_effective_gflops",
            protected_throughput,
            "native_logical_flops_over_primary_wall_ns_median_v1",
        ),
        "peak_rss_bytes": evidence(
            "peak_rss_bytes",
            max(_worker_peak_rss(worker) for worker in workers),
            "whole_worker_proc_rusage_and_native_hwm_max_v1",
        ),
        "admitted_peak_bytes": evidence(
            "admitted_peak_bytes",
            admitted_peak,
            "native_sequential_reference_trait_admission_max_v1",
        ),
        "numa_remote_bytes": evidence(
            "numa_remote_bytes",
            None,
            "numa_remote_traffic_counter_unavailable_v1",
            "unavailable",
        ),
        "integrity_overhead_seconds": evidence(
            "integrity_overhead_seconds",
            _median([value / 1.0e9 for value in overhead_ns]),
            "native_explicit_integrity_overhead_wall_ns_median_v1",
        ),
        "integrity_overhead_fraction": evidence(
            "integrity_overhead_fraction",
            _median(overhead_fractions),
            "native_explicit_integrity_overhead_and_protected_total_median_v1",
        ),
        "artifact_write_load_fit_seconds": evidence(
            "artifact_write_load_fit_seconds",
            _median(artifact_seconds),
            "official_reference_trait_fit_artifact_round_trip_median_v1",
        ),
    }
    if set(values) != set(planner.PERFORMANCE_METRIC_KEYS_V1):
        raise RuntimeError("harness metric map disagrees with frozen planner")
    return values


def _has_content_bound_qualification(build: BuildSpec, plan: Any) -> bool:
    if (
        build.qualification_evidence_path is None
        or build.qualification_evidence_sha256 is None
        or _file_sha256(build.qualification_evidence_path)
        != build.qualification_evidence_sha256
    ):
        return False
    try:
        value = _strict_json_load(build.qualification_evidence_path)
        _exact_keys("fault qualification", value, FAULT_QUALIFICATION_KEYS)
    except (OSError, TypeError, ValueError):
        return False
    planner = _planner_module()
    if value["schema"] != FAULT_QUALIFICATION_SCHEMA:
        return False
    bindings = {
        "source_commit": build.expected_source_commit,
        "source_tree_sha256": build.expected_source_tree_sha256,
        "native_module_sha256": build.expected_native_sha256,
        "integrity_policy": plan.integrity_policy,
        "integrity_backend": plan.integrity_backend,
        "fault_coverage_contract": FAULT_COVERAGE_CONTRACT,
    }
    if any(value[key] != expected for key, expected in bindings.items()):
        return False
    for key in (
        "covered_operations",
        "recoverable_fault_modes",
        "terminal_fault_modes",
    ):
        if not isinstance(value[key], list):
            return False
    if tuple(value["covered_operations"]) != tuple(planner.PROTECTED_OPERATION_KEYS_V1):
        return False
    if tuple(value["recoverable_fault_modes"]) != RECOVERABLE_FAULT_MODES:
        return False
    if tuple(value["terminal_fault_modes"]) != TERMINAL_FAULT_MODES:
        return False
    if value["release_fault_matrix_passed"] is not True:
        return False
    for key in ("focused_test_count", "full_regression_test_count"):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] <= 0
        ):
            return False
    return value["asan_ubsan_status"] == "passed" and value["ubsan_status"] == "passed"


def _aggregate_record(
    *,
    run_id: str,
    workers: Sequence[Mapping[str, Any]],
    plan: Any,
    dimensions: Any,
    host_identity: Mapping[str, Any],
    production_requested: bool,
    build: BuildSpec,
    repeat_count: int,
    baseline_record: Any | None,
    baseline_build_label: str,
) -> tuple[Any, dict[str, Any]]:
    planner = _planner_module()
    _constant(
        [worker["cache_state_policy"] for worker in workers], "cache-state policy"
    )
    if workers[0]["cache_state_policy"] != CACHE_STATE_POLICY:
        raise ValueError("replicate cache-state policy is unsupported")
    backend_identity = _constant(
        [worker["backend_identity"] for worker in workers], "backend identity"
    )
    build_identity = _constant(
        [worker["build_identity"] for worker in workers], "build identity"
    )
    _constant(
        [worker["runtime_dependency_identity"] for worker in workers],
        "runtime dependency identity",
    )
    science_identity = _constant(
        [worker["science_identity_sha256"] for worker in workers], "science identity"
    )
    integrity_identity = _constant(
        [worker["integrity_identity_sha256"] for worker in workers],
        "integrity identity",
    )
    rank_identity = _constant(
        [worker["rank_identity_sha256"] for worker in workers], "rank identity"
    )
    discrepancy_keys = set(workers[0]["comparison"]["fixed_probe_discrepancies"])
    if any(
        set(worker["comparison"]["fixed_probe_discrepancies"]) != discrepancy_keys
        for worker in workers
    ):
        raise ValueError("fixed-probe discrepancy keys differ across replicates")
    discrepancies = {
        key: max(
            float(worker["comparison"]["fixed_probe_discrepancies"][key])
            for worker in workers
        )
        for key in sorted(discrepancy_keys)
    }
    metrics = _make_metrics(workers, plan)
    peak = int(metrics["peak_rss_bytes"].value)
    admitted = int(metrics["admitted_peak_bytes"].value)
    memory_margin = max(
        DEFAULT_RSS_MARGIN_BYTES, int(math.ceil(admitted * DEFAULT_RSS_MARGIN_FRACTION))
    )
    comparisons_passed = all(
        bool(worker["comparison"]["comparison_passed"]) for worker in workers
    )
    descriptor_explained = all(
        bool(report["accounting"]["descriptor_accounting_exact"])
        for worker in workers
        for report in _combined_reports(worker)
    )
    trace_exact = all(
        bool(worker["comparison"]["protected_trace_exact"]) for worker in workers
    )
    qualification_content_bound = _has_content_bound_qualification(build, plan)
    acceptance = planner.ContextualTablaAcceptanceV1(
        fixed_probe_evidence_kind="measured",
        fixed_probe_discrepancies=discrepancies,
        every_deletion_science_unchanged=comparisons_passed,
        rank_estimability_identical=all(
            bool(worker["comparison"]["rank_estimability_identical"])
            for worker in workers
        ),
        integrity_fault_coverage_identical=qualification_content_bound,
        descriptor_passes_explained=descriptor_explained,
        admitted_measured_memory_agrees=peak <= admitted + memory_margin,
        reference_trait_end_to_end_complete=True,
        artifact_round_trip_passed=all(
            bool(worker["artifact_round_trip_passed"]) for worker in workers
        ),
    )
    rejection_reasons: list[str] = []
    if not comparisons_passed:
        rejection_reasons.append("fixed_probe_science_comparison_failed")
    requires_exact_trace = all(
        worker["comparison"]["comparison_mode"] == "exact_trace_v1"
        for worker in workers
    )
    if requires_exact_trace and not trace_exact:
        rejection_reasons.append("protected_trace_comparison_failed")
    if not qualification_content_bound:
        rejection_reasons.append("content_bound_fault_qualification_missing")
    if not descriptor_explained:
        rejection_reasons.append("descriptor_accounting_failed")
    if peak > admitted + memory_margin:
        rejection_reasons.append("measured_rss_exceeds_admission_margin")
    if not acceptance.all_gates_passed:
        rejection_reasons.append("full_acceptance_evidence_failed")
    terminal_status = "accepted" if not rejection_reasons else "rejected"
    production_reasons: list[str] = []
    if not production_requested:
        production_reasons.append("production_qualification_not_requested")
    if repeat_count < 3:
        production_reasons.append("fewer_than_three_measured_replicates")
    if not qualification_content_bound:
        production_reasons.append("build_qualification_evidence_missing")
    if not all(bool(worker["full_input_hash_complete"]) for worker in workers):
        production_reasons.append("full_input_content_hash_missing")
    if terminal_status != "accepted":
        production_reasons.append("terminal_status_not_accepted")
    production_qualified = not production_reasons
    relationship: dict[str, Any] = {
        "comparison_baseline_run_id": None,
        "comparison_baseline_record_sha256": None,
        "declared_parent_source_commit": None,
    }
    if baseline_record is not None and build.label != baseline_build_label:
        relationship = {
            "comparison_baseline_run_id": baseline_record.run_id,
            "comparison_baseline_record_sha256": baseline_record.record_sha256,
            "declared_parent_source_commit": build.declared_parent_source_commit,
        }
    cache_key = planner.contextual_tabla_cache_key_v1(
        host_identity=host_identity,
        backend_identity=backend_identity,
        build_identity=build_identity,
        dimensions=dimensions,
    )
    record = planner.ContextualTablaBenchmarkRecordV1(
        run_id=run_id,
        host_identity=host_identity,
        backend_identity=backend_identity,
        build_identity=build_identity,
        cache_key=cache_key,
        dimensions=dimensions,
        plan=plan,
        science_identity_sha256=science_identity,
        integrity_identity_sha256=integrity_identity,
        rank_identity_sha256=rank_identity,
        metrics=metrics,
        acceptance=acceptance,
        terminal_status=terminal_status,
        production_qualified=production_qualified,
        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        comparison_baseline_run_id=relationship["comparison_baseline_run_id"],
        comparison_baseline_record_sha256=relationship[
            "comparison_baseline_record_sha256"
        ],
        declared_parent_source_commit=relationship["declared_parent_source_commit"],
    )
    return record, {
        "production_qualification_requested": production_requested,
        "production_nonqualification_reasons": production_reasons,
        "rss_admission_margin_bytes": memory_margin,
        "observed_peak_rss_bytes": peak,
        "admitted_peak_bytes": admitted,
    }


def _child_environment(
    build: BuildSpec, workspace: Path, plan: Any | None = None
) -> dict[str, str]:
    environment: dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "TZ"):
        if key in os.environ:
            environment[key] = os.environ[key]
    environment.update(build.environment)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["TMPDIR"] = str(workspace)
    if plan is not None:
        environment["OMP_NUM_THREADS"] = str(plan.decode_threads)
        environment["OPENBLAS_NUM_THREADS"] = str(plan.blas_threads)
        environment.setdefault("OMP_DYNAMIC", "FALSE")
    return environment


def _preload_entries(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        entry
        for colon_group in value.split(":")
        for entry in colon_group.split()
        if entry
    )


def _is_asan_preload(entry: str) -> bool:
    basename = entry.rsplit("/", maxsplit=1)[-1]
    return basename.startswith(_ASAN_PRELOAD_BASENAME_PREFIXES)


def _qualification_build_preload(
    inherited_preload: str | None, configured_preload: str | None
) -> str:
    """Create one explicit, sealed preload for an ASan qualification child.

    Only inherited ASan runtimes are retained.  Arbitrary controller preloads
    are deliberately excluded, ASan order is preserved ahead of explicitly
    configured build libraries, and exact duplicate entries are removed.  An
    ASan runtime present only in the configured suffix is rejected because the
    helper cannot safely infer or silently repair its required loader order.
    """

    inherited_asan = tuple(
        entry
        for entry in _preload_entries(inherited_preload)
        if _is_asan_preload(entry)
    )
    configured = _preload_entries(configured_preload)
    inherited_asan_set = set(inherited_asan)
    unexpected_configured_asan = tuple(
        entry
        for entry in configured
        if _is_asan_preload(entry) and entry not in inherited_asan_set
    )
    if unexpected_configured_asan:
        raise ValueError(
            "configured LD_PRELOAD contains an ASan runtime not inherited "
            "from the qualification controller"
        )
    merged = tuple(dict.fromkeys((*inherited_asan, *configured)))
    if not merged:
        raise ValueError("qualification LD_PRELOAD must not be empty")
    return ":".join(merged)


def _redact_workspace(value: Any, workspace: Path) -> Any:
    prefix = str(workspace)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_workspace(item, workspace) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_workspace(item, workspace) for item in value]
    if isinstance(value, tuple):
        return [_redact_workspace(item, workspace) for item in value]
    if isinstance(value, str):
        return value.replace(prefix, "$WORKSPACE")
    return value


def _run_child_json(
    *,
    mode: str,
    request: Mapping[str, Any],
    request_path: Path,
    result_path: Path,
    build: BuildSpec,
    cpus: Sequence[int],
    workspace: Path,
    timeout_seconds: float,
    sample_interval_ms: int,
    plan: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _write_internal_json(request_path, request)
    taskset_path = shutil.which("taskset")
    if taskset_path is None:
        raise RuntimeError("taskset is required for fresh-worker affinity")
    command = [
        str(Path(taskset_path).resolve(strict=True)),
        "-c",
        _compress_cpu_list(cpus),
        str(build.python_executable),
        "-S",
        str(Path(__file__).resolve()),
        mode,
        "--_request",
        str(request_path),
        "--_result",
        str(result_path),
    ]
    monitor = _run_monitored(
        command,
        environment=_child_environment(build, workspace, plan),
        timeout_seconds=timeout_seconds,
        sample_interval_ms=sample_interval_ms,
    )
    result = _strict_json_load(result_path)
    return result, monitor


def _remaining_child_timeout(
    *, sweep_started: float, sweep_timeout: float, worker_timeout: float
) -> float:
    remaining = sweep_timeout - (time.monotonic() - sweep_started)
    if remaining <= 0.0:
        raise TimeoutError("Stage 6 sweep exhausted its wall-time budget")
    return min(worker_timeout, remaining)


def _validate_public_args(args: argparse.Namespace) -> None:
    for name in (
        "retained_samples",
        "retained_variants",
        "q",
        "k",
        "groups",
        "sample_probes",
        "variant_probes",
        "traits",
        "residual_bases",
        "fixed_rank",
        "repeats",
        "preparation_variant_block",
        "sample_interval_ms",
        "max_workspace_bytes",
        "max_logical_variant_visits",
        "max_logical_bed_bytes",
        "pilot_variant_ceiling",
        "max_synthetic_cells",
        "max_input_hash_bytes",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be nonnegative")
    for name in (
        "worker_timeout_seconds",
        "sweep_timeout_seconds",
        "max_extrapolation_ratio",
        "estimate_safety_factor",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.fixed_rank >= args.retained_samples:
        raise ValueError("--fixed-rank must be smaller than retained samples")
    if args.sample_probes < 2 or args.variant_probes < 2:
        raise ValueError("both probe counts must be at least two")
    if args.retained_variants < args.k * args.groups:
        raise ValueError("M must cover every annotation/group combination")
    if args.input_kind == "real-prefix" and args.geno_prefix is None:
        raise ValueError("--geno-prefix is required for --input-kind real-prefix")
    if args.input_kind == "synthetic" and args.geno_prefix is not None:
        raise ValueError("--geno-prefix is forbidden for synthetic input")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace benchmark output {output}")
    _regular_directory(output.parent)
    work_root = args.work_root.expanduser().resolve()
    _regular_directory(work_root)


def _load_configuration(
    args: argparse.Namespace,
) -> tuple[
    dict[str, BuildSpec],
    list[tuple[PlanSpec, Any]],
    Any,
    tuple[int, ...],
    list[dict[str, Any]],
]:
    builds: dict[str, BuildSpec] = {}
    for path in args.build_spec:
        build = BuildSpec.load(path)
        if build.label in builds:
            raise ValueError(f"duplicate build label {build.label!r}")
        builds[build.label] = build
    plan_paths = [args.baseline_plan, *args.candidate_plan]
    specs = [PlanSpec.load(path) for path in plan_paths]
    if len({spec.plan_id for spec in specs}) != len(specs):
        raise ValueError("plan_id values must be unique")
    cpus = tuple(args.cpu_list)
    allowed = set(os.sched_getaffinity(0))
    if not set(cpus) <= allowed:
        raise ValueError("--cpu-list contains CPUs outside this process affinity")
    plans: list[tuple[PlanSpec, Any]] = []
    for index, spec in enumerate(specs):
        if spec.build_label not in builds:
            raise ValueError(f"plan {spec.plan_id!r} names an unknown build")
        plan = _instantiate_plan(spec, cpus)
        _validate_executable_plan(plan, group_layout=args.group_layout)
        if plan.annotation_mode != args.annotation_mode:
            raise ValueError(
                f"plan {spec.plan_id!r} annotation mode disagrees with the case"
            )
        if index == 0 and spec.comparison_mode != "exact_trace_v1":
            raise ValueError("baseline plan must use exact_trace_v1")
        plans.append((spec, plan))
    baseline_build = builds[specs[0].build_label]
    for spec, _ in plans[1:]:
        if spec.build_label != specs[0].build_label:
            candidate_build = builds[spec.build_label]
            if (
                candidate_build.declared_parent_source_commit
                != baseline_build.expected_source_commit
            ):
                raise ValueError(
                    f"cross-build plan {spec.plan_id!r} lacks the exact baseline "
                    "parent-commit declaration"
                )
    dimensions = _dimensions(args)
    calibrations = _load_calibrations(args.calibration)
    preflight = _runtime_preflight(
        args=args,
        builds=builds,
        plans=plans,
        dimensions=dimensions,
        calibrations=calibrations,
    )
    return builds, plans, dimensions, cpus, preflight


def _revalidate_publication_inputs(
    *,
    builds: Mapping[str, BuildSpec],
    build_spec_hashes: Mapping[str, str],
    case: Mapping[str, Any],
    require_full_input_hashes: bool,
) -> None:
    _verify_case_file_identities(case, require_full_hashes=require_full_input_hashes)
    for label, build in builds.items():
        if _file_sha256(build.source_path) != build_spec_hashes[label]:
            raise RuntimeError(f"build spec {label!r} changed before publication")
        if _file_sha256(build.native_module) != build.expected_native_sha256:
            raise RuntimeError(f"native module {label!r} changed before publication")
        if _package_sha256(build.install_prefix) != build.expected_package_sha256:
            raise RuntimeError(
                f"installed package {label!r} changed before publication"
            )
        if build.qualification_evidence_path is not None and (
            build.qualification_evidence_sha256 is None
            or _file_sha256(build.qualification_evidence_path)
            != build.qualification_evidence_sha256
        ):
            raise RuntimeError(
                f"qualification evidence {label!r} changed before publication"
            )


def _controller(args: argparse.Namespace) -> dict[str, Any]:
    _validate_public_args(args)
    builds, plans, dimensions, cpus, preflight = _load_configuration(args)
    build_spec_hashes = {
        label: _file_sha256(build.source_path) for label, build in builds.items()
    }
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    sweep_id = _canonical_sha256(
        {
            "schema": "context_native_stage6_sweep_identity_v1",
            "generated_at": generated_at,
            "argv": sys.argv[1:],
            "plans": [plan.plan_sha256 for _, plan in plans],
        }
    )
    base_output = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "sweep_id": sweep_id,
        "harness_sha256": _file_sha256(Path(__file__).resolve()),
        "execution_policy": {
            "fresh_process_per_replicate": True,
            "cpu_affinity": list(cpus),
            "cpu_affinity_identity_sha256": _affinity_sha256(cpus),
            "external_resource_scope": "whole_fresh_worker_process",
            "baseline_integrity_policy": plans[0][1].integrity_policy,
            "baseline_integrity_backend": plans[0][1].integrity_backend,
            "large_run_policy": "checked_budget_and_compatible_bounded_pilot_v1",
            "physical_read_bytes_policy": (
                "mmap_physical_read_bytes_explicitly_unavailable_v1"
            ),
            "cache_state_policy": CACHE_STATE_POLICY,
        },
        "host_identity": _host_identity(),
        "dimensions": dimensions.to_dict(),
        "build_specs": [
            {
                **build.to_worker_dict(),
                "spec_path": str(build.source_path),
                "spec_sha256": build_spec_hashes[build.label],
            }
            for build in builds.values()
        ],
        "plans": [
            {
                "plan_id": spec.plan_id,
                "build_label": spec.build_label,
                "comparison_mode": spec.comparison_mode,
                "plan": plan.to_dict(),
                "plan_sha256": plan.plan_sha256,
                "spec_path": str(spec.source_path),
                "spec_sha256": _file_sha256(spec.source_path),
            }
            for spec, plan in plans
        ],
        "preflight": preflight,
    }
    if args.dry_run:
        return {
            **base_output,
            "terminal_status": "dry_run_only",
            "case": None,
            "benchmark_runs": [],
            "selection": None,
            "runtime_calibrations": [],
        }

    sweep_started = time.monotonic()
    work_root = args.work_root.expanduser().resolve()
    with tempfile.TemporaryDirectory(
        prefix="summit-context-stage6-", dir=work_root
    ) as temporary:
        workspace = Path(temporary)
        baseline_spec, baseline_plan = plans[0]
        baseline_build = builds[baseline_spec.build_label]
        prepare_request = {
            "schema": CASE_SCHEMA,
            "build": baseline_build.to_worker_dict(),
            "workspace": str(workspace),
            "input_kind": args.input_kind,
            "geno_prefix": str(args.geno_prefix) if args.geno_prefix else None,
            "n": args.retained_samples,
            "m": args.retained_variants,
            "q": args.q,
            "k": args.k,
            "groups": args.groups,
            "traits": args.traits,
            "residual_bases": args.residual_bases,
            "fixed_rank": args.fixed_rank,
            "sample_probes": args.sample_probes,
            "variant_probes": args.variant_probes,
            "annotation_mode": args.annotation_mode,
            "group_layout": args.group_layout,
            "seed": args.seed,
            "max_synthetic_cells": args.max_synthetic_cells,
            "max_input_hash_bytes": args.max_input_hash_bytes,
            "hash_large_inputs": args.hash_large_inputs,
            "preparation_variant_block": args.preparation_variant_block,
        }
        case_path = workspace / "prepared_case.json"
        case, prepare_monitor = _run_child_json(
            mode="--_prepare",
            request=prepare_request,
            request_path=workspace / "prepare_request.json",
            result_path=case_path,
            build=baseline_build,
            cpus=(cpus[0],),
            workspace=workspace,
            timeout_seconds=_remaining_child_timeout(
                sweep_started=sweep_started,
                sweep_timeout=args.sweep_timeout_seconds,
                worker_timeout=args.worker_timeout_seconds,
            ),
            sample_interval_ms=args.sample_interval_ms,
            plan=None,
        )
        _verify_case_science_identity(case)
        _verify_case_file_identities(
            case,
            require_full_hashes=args.production_qualified,
            verify_content_hashes=True,
        )
        baseline_capsule: Path | None = None
        baseline_trace: str | None = None
        baseline_record: Any | None = None
        benchmark_runs: list[dict[str, Any]] = []
        records: list[Any] = []
        runtime_calibrations: list[dict[str, Any]] = []
        for plan_index, ((spec, plan), preflight_item) in enumerate(
            zip(plans, preflight, strict=True)
        ):
            build = builds[spec.build_label]
            measured_workers: list[dict[str, Any]] = []
            warmup_summaries: list[dict[str, Any]] = []
            for replicate in range(args.warmups + args.repeats):
                phase = "warmup" if replicate < args.warmups else "measured"
                ordinal = (
                    replicate + 1 if phase == "warmup" else replicate - args.warmups + 1
                )
                stem = f"p{plan_index:02d}-{phase}-{ordinal:03d}"
                capsule_path = workspace / f"{stem}-science.npz"
                request = {
                    "schema": "context_native_stage6_worker_request_v1",
                    "cache_state_policy": CACHE_STATE_POLICY,
                    "plan_id": spec.plan_id,
                    "comparison_mode": spec.comparison_mode,
                    "build": build.to_worker_dict(),
                    "plan": plan.to_dict(),
                    "cpus": list(cpus),
                    "case_metadata": str(case_path),
                    "max_workspace_bytes": args.max_workspace_bytes,
                    "output_directory": str(workspace / f"{stem}-artifacts"),
                    "capsule_path": str(capsule_path),
                    "baseline_capsule": (
                        str(baseline_capsule) if baseline_capsule is not None else None
                    ),
                    "baseline_trace_sha256": baseline_trace,
                }
                worker, monitor = _run_child_json(
                    mode="--_worker",
                    request=request,
                    request_path=workspace / f"{stem}-request.json",
                    result_path=workspace / f"{stem}-result.json",
                    build=build,
                    cpus=cpus,
                    workspace=workspace,
                    timeout_seconds=_remaining_child_timeout(
                        sweep_started=sweep_started,
                        sweep_timeout=args.sweep_timeout_seconds,
                        worker_timeout=args.worker_timeout_seconds,
                    ),
                    sample_interval_ms=args.sample_interval_ms,
                    plan=plan,
                )
                worker["controller_monitor"] = monitor
                if baseline_capsule is None:
                    if plan_index != 0:
                        raise RuntimeError("candidate executed before baseline capsule")
                    baseline_capsule = capsule_path
                    baseline_trace = worker["combined_protected_trace_sha256"]
                if phase == "measured":
                    measured_workers.append(worker)
                else:
                    warmup_summaries.append(
                        {
                            "worker_wall_seconds": worker["workflow"]["wall_seconds"],
                            "protected_trace_sha256": worker[
                                "combined_protected_trace_sha256"
                            ],
                            "comparison_passed": worker["comparison"][
                                "comparison_passed"
                            ],
                            "controller_monitor": monitor,
                        }
                    )
            run_id = f"{spec.plan_id}-{sweep_id[:16]}"
            record, qualification = _aggregate_record(
                run_id=run_id,
                workers=measured_workers,
                plan=plan,
                dimensions=dimensions,
                host_identity=base_output["host_identity"],
                production_requested=args.production_qualified,
                build=build,
                repeat_count=args.repeats,
                baseline_record=baseline_record,
                baseline_build_label=baseline_spec.build_label,
            )
            if plan_index == 0:
                baseline_record = record
            records.append(record)
            runtime_calibration = {
                "schema": CALIBRATION_SCHEMA,
                "compatibility_key": preflight_item["calibration_compatibility_key"],
                "n_variants": dimensions.n_variants,
                "logical_variant_visits": preflight_item["logical_variant_visits"],
                "median_worker_wall_seconds": _median(
                    [
                        worker["controller_monitor"]["controller_wall_seconds"]
                        for worker in measured_workers
                    ]
                ),
            }
            runtime_calibrations.append(runtime_calibration)
            benchmark_runs.append(
                {
                    "plan_id": spec.plan_id,
                    "record": record.to_dict(),
                    "record_sha256": record.record_sha256,
                    "qualification_policy": qualification,
                    "warmups": warmup_summaries,
                    "replicates": measured_workers,
                }
            )
        assert baseline_record is not None
        _revalidate_publication_inputs(
            builds=builds,
            build_spec_hashes=build_spec_hashes,
            case=case,
            require_full_input_hashes=args.production_qualified,
        )
        baseline_ineligible = baseline_record.selection_ineligibility_reasons()
        if baseline_ineligible:
            selection: Mapping[str, Any] = {
                "status": "unavailable",
                "reasons": list(baseline_ineligible),
            }
        else:
            selected = _planner_module().select_contextual_tabla_plan_v1(
                baseline_record, records[1:]
            )
            selection = {"status": "complete", "value": selected.to_dict()}
        output = {
            **base_output,
            "terminal_status": "complete",
            "case": case,
            "preparation_monitor": prepare_monitor,
            "benchmark_runs": benchmark_runs,
            "baseline_run_id": baseline_record.run_id,
            "selection": selection,
            "runtime_calibrations": runtime_calibrations,
        }
        return _redact_workspace(output, workspace)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded fresh-process contextual Stage 6 benchmark harness"
    )
    parser.add_argument("--build-spec", type=Path, action="append", required=True)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, action="append", default=[])
    parser.add_argument(
        "--input-kind", choices=("synthetic", "real-prefix"), default="synthetic"
    )
    parser.add_argument("--geno-prefix", type=Path)
    parser.add_argument("--retained-samples", type=int, default=64)
    parser.add_argument("--retained-variants", type=int, default=96)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--sample-probes", type=int, default=4)
    parser.add_argument("--variant-probes", type=int, default=4)
    parser.add_argument("--traits", type=int, default=1)
    parser.add_argument("--residual-bases", type=int, default=2)
    parser.add_argument("--fixed-rank", type=int, default=2)
    parser.add_argument(
        "--annotation-mode",
        choices=("strict_disjoint_binary_v1", "generic_nonnegative_weights_v1"),
        default="strict_disjoint_binary_v1",
    )
    parser.add_argument(
        "--group-layout", choices=("contiguous", "interleaved"), default="contiguous"
    )
    parser.add_argument("--seed", type=int, default=604_2026)
    parser.add_argument("--cpu-list", type=_parse_cpu_list, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--sweep-timeout-seconds", type=float, default=DEFAULT_SWEEP_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--sample-interval-ms", type=int, default=DEFAULT_SAMPLE_INTERVAL_MS
    )
    parser.add_argument(
        "--max-workspace-bytes", type=int, default=DEFAULT_MAX_WORKSPACE_BYTES
    )
    parser.add_argument(
        "--max-logical-variant-visits",
        type=int,
        default=DEFAULT_MAX_LOGICAL_VARIANT_VISITS,
    )
    parser.add_argument(
        "--max-logical-bed-bytes", type=int, default=DEFAULT_MAX_LOGICAL_BED_BYTES
    )
    parser.add_argument(
        "--pilot-variant-ceiling", type=int, default=DEFAULT_PILOT_VARIANTS
    )
    parser.add_argument(
        "--max-extrapolation-ratio",
        type=float,
        default=DEFAULT_MAX_EXTRAPOLATION_RATIO,
    )
    parser.add_argument(
        "--estimate-safety-factor",
        type=float,
        default=DEFAULT_ESTIMATE_SAFETY_FACTOR,
    )
    parser.add_argument(
        "--max-synthetic-cells", type=int, default=DEFAULT_MAX_SYNTHETIC_CELLS
    )
    parser.add_argument("--max-input-hash-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--hash-large-inputs", action="store_true")
    parser.add_argument("--preparation-variant-block", type=int, default=4096)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--production-qualified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--work-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"--_prepare", "--_worker"}:
        mode = arguments.pop(0)
        hidden = argparse.ArgumentParser(add_help=False)
        hidden.add_argument("--_request", type=Path, required=True)
        hidden.add_argument("--_result", type=Path, required=True)
        parsed = hidden.parse_args(arguments)
        if mode == "--_prepare":
            _prepare_case(parsed._request, parsed._result)
        else:
            _execute_worker(parsed._request, parsed._result)
        return 0
    args = _parser().parse_args(arguments)
    output_path = args.output.expanduser().resolve()
    result = _controller(args)
    _write_json_no_replace(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
