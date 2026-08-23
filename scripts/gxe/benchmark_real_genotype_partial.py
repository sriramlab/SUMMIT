#!/usr/bin/env python3
"""Run a bounded, disposable real-genotype GxE throughput benchmark.

The controller copies a literal prefix of a variant-major PLINK BED trio into
``TemporaryDirectory``, invokes the existing multi-environment direct CLI in a
fresh ``python -S`` process, extracts the final performance telemetry, and
removes both the genotype subset and all reference artifacts before publishing
one compact JSON report.  It never launches more than 32 probes; the B=1024
result is a phase-rate projection only.

Example (four production-width blocks, three environments, B=32)::

    python scripts/gxe/benchmark_real_genotype_partial.py \
      --install-prefix /path/to/fresh/install \
      --native-module /path/to/fresh/install/summit/gxeldcore.cpython-312-x86_64-linux-gnu.so \
      --cpu-list 0-31 \
      --output /new/path/gxe_real_prefix_4blocks_b32.json

Use ``--dry-run`` to validate the PLINK subset, CPU placement, import paths,
and exact CLI command without importing SUMMIT or constructing references.
The rejected API-8 source/target orientations are diagnostic-only.  This
real-data harness therefore exposes only the established ``current`` layout
through the accepted private upstream-BLIS API-9/backend-1.6 runtime.  The
single environment group is always launched through the fresh-exec explicit
OpenMP-placement route; private OpenBLAS is not an acceptance candidate here.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import itertools
import json
import math
import numbers
import os
import platform
import resource
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "summit.gxe.real_genotype_partial_benchmark"
SCHEMA_VERSION = 2
OPENMP_PLACEMENT_SCHEMA = "summit.openmp_placement_attestation.v1"
NUMA_BOUND_DECODE_SCHEMA = "summit.numa_bound_bed_decode.v1"
NUMA_BOUND_BUFFER_SCHEMA = "summit.numa_bound_anonymous_buffer.v1"
NUMA_PAGE_QUERY_CHUNK_LIMIT = 65536
NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA = (
    "summit.native_integrity_snapshot_numa.v1"
)
NATIVE_INTEGRITY_SNAPSHOT_NUMA_ROLE = (
    "native_integrity_snapshot_of_logical_b"
)
NATIVE_INTEGRITY_SNAPSHOT_NUMA_POLICY_VALUE = 32770
NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT = 65536
NATIVE_GEMM_OUTPUT_NUMA_SCHEMA = "summit.native_gemm_output_numa.v1"
NATIVE_GEMM_OUTPUT_NUMA_ROLE = "protected_gemm_output"
NATIVE_GEMM_OUTPUT_NUMA_POLICY_VALUE = 32770
NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT = 65536
NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY = 16384
EXPLICIT_OPENMP_MEMORY_SCOPE = "selected-socket"
EXPECTED_NATIVE_API_VERSION = 9
EXPECTED_NATIVE_BACKEND_VERSION = "1.9"
EXPECTED_PRIVATE_BACKEND = "blis"
EXPECTED_PRIVATE_BLAS_NAME = "upstream_blis"
EXPECTED_BLIS_CONFIG_FAMILY = "zen"
EXPECTED_BLIS_RUNTIME_CONFIG = "BLIS 2.0 config=zen"
BLIS_WAY_NAMES = ("jc", "pc", "ic", "jr", "ir")
OPENMP_AFFINITY_CONFLICTS = (
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "KMP_HW_SUBSET",
    "KMP_PLACE_THREADS",
    "OMP_NESTED",
)
BLIS_AUTOMATIC_CONFLICTS = (
    "BLIS_NT",
    "BLIS_TI",
    "BLIS_THREAD_IMPL",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
)
MIN_BLOCKS = 4
MAX_BLOCKS = 16
MAX_BLOCK_WIDTH = 6144
MAX_PROBES = 32
MAX_TIMEOUT_SECONDS = 30 * 60
MAX_COMPARISON_CHILD_SECONDS = 10 * 60
MAX_COMPARISON_SECONDS = 20 * 60
TUNING_BLOCK_WIDTHS = (2000, 3072)
MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS = 4 * 60
DEFAULT_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS = 9 * 60
MAX_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS = 10 * 60
MAX_BLIS_REPLICATE_CHILD_SECONDS = 4 * 60
DEFAULT_BLIS_REPLICATE_CONTROLLER_SECONDS = 9 * 60
MAX_BLIS_REPLICATE_CONTROLLER_SECONDS = 10 * 60
MAX_BLIS_T1_REFERENCE_CHILD_SECONDS = 7 * 60
MAX_BLIS_T32_CANDIDATE_CHILD_SECONDS = 4 * 60
DEFAULT_BLIS_T1_COMPARISON_CONTROLLER_SECONDS = 9 * 60
MAX_BLIS_T1_COMPARISON_CONTROLLER_SECONDS = 10 * 60
PROJECTED_PROBES = 1024
FP64_LAYOUTS = ("current", "source-tt-target-current")
SUPPORTED_FP64_LAYOUTS = ("current",)
SCORE_FAMILIES = ("xx", "xw", "wx", "ww")
DEFAULT_COMPARISON_RTOL = 5.0e-12
DEFAULT_COMPARISON_ATOL = 5.0e-12
MATERIAL_SPEEDUP_THRESHOLD = 1.05
RESOURCE_DIAGNOSTIC_ALLOWLIST = (
    "max_native_source_projection_leakage",
    "max_source_projection_leakage",
    "native_gemm_integrity_enabled",
    "native_repaired_gemm_output_columns",
    "native_retried_gemm_input_mutations",
)
NUMA_ADDRESS_SELECTION_POLICY = "evenly_spaced_fully_contained_page_bases"
DEFAULT_GENOTYPE = Path(
    "/home/bronsonj/UKBB/geno/EUR_300k/"
    "UKBB_EUR_300k_unrel_3rd.no_mhc_imp"
)
DEFAULT_ENVIRONMENT = Path(
    "/home/bronsonj/SUMMIT_gxe_real_40x5_20260811/"
    "population_multi5_45d1bdf_b32_20260814/inputs/"
    "reference_common.env.tsv"
)
DEFAULT_COVARIATES = Path(
    "/home/bronsonj/SUMMIT_gxe_real_40x5_20260811/"
    "population_multi5_45d1bdf_b32_20260814/inputs/"
    "common_fixed.covar.tsv"
)
BED_MAGIC = b"\x6c\x1b\x01"


_CLI_BOOTSTRAP = r"""
import hashlib
import os
from pathlib import Path
import sys

expected_prefix = Path(
    os.environ.pop("SUMMIT_GXE_EXPECTED_INSTALL_PREFIX")
).resolve()
expected_package_manifest_sha256 = os.environ.pop(
    "SUMMIT_GXE_EXPECTED_PACKAGE_MANIFEST_SHA256"
)
expected = Path(os.environ.pop("SUMMIT_GXE_EXPECTED_NATIVE_MODULE")).resolve()
expected_sha256 = os.environ.pop("SUMMIT_GXE_EXPECTED_NATIVE_SHA256")
expected_source_commit = os.environ.pop("SUMMIT_GXE_EXPECTED_SOURCE_COMMIT")
expected_source_tree_sha256 = os.environ.pop(
    "SUMMIT_GXE_EXPECTED_SOURCE_TREE_SHA256"
)
expected_archive_sha256 = os.environ.pop(
    "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_ARCHIVE_SHA256"
)
expected_private_source_commit = os.environ.pop(
    "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_COMMIT"
)
expected_private_source_tree_sha256 = os.environ.pop(
    "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_TREE_SHA256"
)
expected_threads = int(os.environ.pop("SUMMIT_GXE_EXPECTED_BLAS_THREADS"))
require_integrity = os.environ.pop("SUMMIT_GXE_REQUIRE_INTEGRITY") == "1"
observed_package = (expected_prefix / "summit").resolve()
if not observed_package.is_dir():
    raise RuntimeError(
        f"expected summit package directory does not exist: {observed_package}"
    )
records = []
for path in sorted(observed_package.rglob("*")):
    relative = path.relative_to(observed_package)
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        raise RuntimeError(
            f"accepted summit package contains executable bytecode/cache: {relative}"
        )
    if not path.is_file():
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append((str(path.relative_to(expected_prefix)), path.stat().st_size, digest.hexdigest()))
manifest = hashlib.sha256()
for relative, size, file_hash in records:
    manifest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
if manifest.hexdigest() != expected_package_manifest_sha256:
    raise RuntimeError("installed summit package manifest changed before CLI entry")
import summit
loaded_package = Path(summit.__file__).resolve().parent
if loaded_package != observed_package:
    raise RuntimeError(
        f"loaded summit package {loaded_package}, expected {observed_package}"
    )
from summit._early_numa import preconfigure_numa_from_argv
early_numa_attestation = preconfigure_numa_from_argv(sys.argv[1:])
if os.environ.pop("SUMMIT_GXE_REQUIRE_EARLY_NUMA") == "1":
    if not isinstance(early_numa_attestation, dict):
        raise RuntimeError(
            "benchmark child lacks a verified pre-import NUMA attestation"
        )
# summit.cli captures the taskset mask at module entry, before importing
# gxeldcore can initialize libgomp and narrow the master thread to place zero.
# Importing the verified CLI performs no scientific reference construction;
# cli.main remains below every native identity/build gate.
sys.argv = ["summit", *sys.argv[1:]]
from summit import cli as summit_cli
observed_cli = Path(summit_cli.__file__).resolve()
if observed_cli.parent != observed_package:
    raise RuntimeError(f"loaded summit.cli outside the expected package: {observed_cli}")
from summit import gxeldcore
observed = Path(gxeldcore.__file__).resolve()
if observed != expected:
    raise RuntimeError(f"loaded native module {observed}, expected {expected}")
digest = hashlib.sha256(observed.read_bytes()).hexdigest()
if digest != expected_sha256:
    raise RuntimeError("native module bytes changed before CLI entry")
build = dict(gxeldcore.build_info())
required_build = {
    "api_version": 9,
    "backend_version": "1.9",
    "source_commit": expected_source_commit,
    "source_tree_sha256": expected_source_tree_sha256,
    "blas_vendor": "BLIS",
    "blas_runtime_isolation": "private_static",
    "gemm_execution_mode": "serialized_fixed_private_blis",
    "private_openblas_archive_sha256": "none",
    "private_blas_backend": "upstream_blis",
    "private_blas_archive_sha256": expected_archive_sha256,
    "private_blas_source_commit": expected_private_source_commit,
    "private_blas_source_tree_sha256": expected_private_source_tree_sha256,
    "private_blas_config_family": "zen",
    "blas_runtime_config": "BLIS 2.0 config=zen",
    "blas_runtime_corename": "zen",
    "blas_runtime_threads": expected_threads,
    "blas_runtime_threading_layer": "pthreads",
    "blas_runtime_worker_affinity_policy": (
        "inherit_authenticated_selected_cpu_set_per_call"
    ),
    "blas_runtime_thread_strategy": "automatic",
    "blas_runtime_owner_thread_enforced": True,
    "blas_runtime_owner_thread_configured": True,
    "blas_runtime_environment_immutable": True,
    "blas_runtime_environment_contract": "blis_process_start_v1",
    "blas_runtime_tls_enabled": True,
    "gemm_vendor_entry_outer_openmp_guard": True,
    "openmp_effective_capacity_policy": "bound_places_else_sched_affinity_v1",
    "openmp_placement_contract_supported": True,
    "openmp_placement_contract_schema": "summit.openmp_placement_attestation.v1",
    "openmp_placement_contract_configured": False,
    "openmp_placement_contract_immutable": True,
    "openmp_placement_probe_vendor_calls": 0,
    "openmp_placement_contract_evidence": None,
    "native_integrity_snapshot_numa_contract_supported": True,
    "native_integrity_snapshot_numa_contract_schema":
        "summit.native_integrity_snapshot_numa.v1",
    "native_integrity_snapshot_numa_query_chunk_page_limit": 65536,
    "native_gemm_output_numa_contract_supported": True,
    "native_gemm_output_numa_contract_schema":
        "summit.native_gemm_output_numa.v1",
    "native_gemm_output_numa_query_chunk_page_limit": 65536,
    "native_gemm_output_numa_evidence_capacity": 16384,
}
mismatches = {
    name: {"expected": value, "observed": build.get(name)}
    for name, value in required_build.items()
    if build.get(name) != value
}
if mismatches:
    raise RuntimeError(f"native private-BLIS build contract mismatch: {mismatches}")
if type(build.get("api_version")) is not int:
    raise RuntimeError("native API version is not a built-in integer")
for name in (
    "blas_runtime_owner_thread_enforced",
    "blas_runtime_owner_thread_configured",
    "blas_runtime_environment_immutable",
    "blas_runtime_tls_enabled",
    "gemm_vendor_entry_outer_openmp_guard",
    "openmp_placement_contract_supported",
    "openmp_placement_contract_configured",
    "openmp_placement_contract_immutable",
    "native_integrity_snapshot_numa_contract_supported",
    "native_gemm_output_numa_contract_supported",
):
    if type(build.get(name)) is not bool:
        raise RuntimeError(f"native build field {name} is not a built-in boolean")
ways = build.get("blas_runtime_thread_ways")
if (
    not isinstance(ways, dict)
    or set(ways) != {"jc", "pc", "ic", "jr", "ir"}
    or any(type(ways.get(name)) is not int or ways[name] != 1 for name in ways)
):
    raise RuntimeError("native automatic BLIS loop-way contract is not exact")
for name in ("private_blas_header_sha256", "private_blas_cblas_header_sha256"):
    value = build.get(name)
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"native build lacks canonical {name}")
if require_integrity and build.get("gemm_integrity_enabled") is not True:
    raise RuntimeError("development benchmark requires GEMM integrity checking")
if require_integrity and (
    type(build.get("gemm_integrity_minimum_vendor_flops")) is not int
    or build["gemm_integrity_minimum_vendor_flops"] <= 0
    or type(build.get(
        "native_integrity_snapshot_numa_query_chunk_page_limit"
    )) is not int
    or build["native_integrity_snapshot_numa_query_chunk_page_limit"] <= 0
    or type(build.get(
        "native_gemm_output_numa_query_chunk_page_limit"
    )) is not int
    or build["native_gemm_output_numa_query_chunk_page_limit"] <= 0
    or type(build.get("native_gemm_output_numa_evidence_capacity")) is not int
    or build["native_gemm_output_numa_evidence_capacity"] <= 0
):
    raise RuntimeError(
        "development benchmark lacks an exact integrity/output NUMA threshold, "
        "chunk, and capacity contract"
    )
if not callable(getattr(gxeldcore, "consume_gemm_telemetry", None)):
    raise RuntimeError("native module lacks vendor GEMM telemetry")
for output_api in (
    "consume_native_gemm_output_numa_evidence",
    "native_gemm_output_numa_evidence_status",
    "reset_native_gemm_output_numa_evidence",
):
    if not callable(getattr(gxeldcore, output_api, None)):
        raise RuntimeError(
            f"native module lacks protected-output NUMA API {output_api}"
        )
summit_cli.main()
"""


@dataclass(frozen=True)
class PlinkMetadata:
    prefix: str
    samples: int
    variants: int
    bytes_per_variant: int
    expected_bed_bytes: int
    actual_bed_bytes: int


@dataclass(frozen=True)
class CpuRecord:
    cpu: int
    core: int
    socket: int
    node: int | None


def _comparison_block_widths(value: str) -> tuple[int, int]:
    try:
        widths = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--compare-block-widths requires comma-separated integers"
        ) from exc
    if widths != TUNING_BLOCK_WIDTHS:
        expected = ",".join(map(str, TUNING_BLOCK_WIDTHS))
        raise argparse.ArgumentTypeError(
            f"the current tuning plan requires --compare-block-widths {expected}"
        )
    return widths


def _canonical_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geno-prefix", type=Path, default=DEFAULT_GENOTYPE)
    parser.add_argument("--environment-file", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--covariate-file", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--environment-columns", default="age,sex,bmi")
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument(
        "--expected-backend", choices=(EXPECTED_PRIVATE_BACKEND,), required=True,
    )
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--private-archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-private-source-commit", required=True)
    parser.add_argument("--expected-private-source-tree-sha256", required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--dependency-path", type=Path, action="append", default=None,
        help="Explicit dependency directory for python -S; repeat as needed.",
    )
    parser.add_argument("--cpu-list", default="auto")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-width", type=int, default=2000)
    parser.add_argument(
        "--subset-variants", type=int, default=None,
        help=(
            "Exact literal-prefix size for --compare-block-widths; permits a "
            "terminal partial block and is invalid outside that mode."
        ),
    )
    parser.add_argument(
        "--compare-block-widths", type=_comparison_block_widths, default=None,
        metavar="2000,3072",
        help=(
            "Run fresh sequential current-layout children at the two block "
            "widths in the current tuning plan."
        ),
    )
    parser.add_argument(
        "--compare-blis-replicates",
        action="store_true",
        help=(
            "Run two fresh sequential, otherwise identical private-BLIS "
            "children for artifact-level determinism evidence; this mode is "
            "not acceptance-eligible."
        ),
    )
    parser.add_argument(
        "--compare-blis-t1-reference",
        action="store_true",
        help=(
            "Compare a fresh one-thread private-BLIS numerical reference with "
            "the fresh 32-thread private-BLIS selection candidate."
        ),
    )
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float64"), default="float32",
        help="Requested sketch/storage dtype passed literally to --dtype.",
    )
    parser.add_argument(
        "--full-precision-layout", choices=SUPPORTED_FP64_LAYOUTS, default="current",
        help="Single-run full-precision GEMM layout; default preserves current routing.",
    )
    parser.add_argument(
        "--compare-fp64-layouts", action="store_true",
        help=(
            "Rejected compatibility option: the API-8 source/target candidates "
            "failed their integrity gates and cannot run on real data."
        ),
    )
    parser.add_argument("--workspace-gib", type=float, default=16.0)
    parser.add_argument("--target-memory-gib", type=float, default=16.0)
    parser.add_argument(
        "--timeout-seconds", type=float,
        default=MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS,
    )
    parser.add_argument(
        "--controller-timeout-seconds", type=float,
        default=DEFAULT_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS,
        help="Hard wall cap for the complete two-child comparison controller.",
    )
    parser.add_argument(
        "--reference-timeout-seconds",
        type=float,
        default=MAX_BLIS_T1_REFERENCE_CHILD_SECONDS,
        help="Hard child cap for the BLIS T1 numerical-reference run.",
    )
    parser.add_argument(
        "--comparison-rtol", type=float, default=DEFAULT_COMPARISON_RTOL,
    )
    parser.add_argument(
        "--comparison-atol", type=float, default=DEFAULT_COMPARISON_ATOL,
    )
    # Retain the internal attribute for fail-closed runtime checks, but expose no
    # CLI escape hatch: API-9 private-BLIS real-data evidence always requires
    # integrity telemetry and zero repairs.
    parser.set_defaults(allow_integrity_disabled=False)
    parser.add_argument(
        "--temporary-parent", type=Path, default=None,
        help="Existing directory in which to create the automatically removed workspace.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _normalize_prefix(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.with_suffix("") if expanded.suffix in {".bed", ".bim", ".fam"} else expanded


def _regular_input(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _executable_input(path: Path) -> Path:
    """Resolve an interpreter symlink, then require a regular executable file."""
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"Python executable is not a regular file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ValueError(f"Python executable is not executable: {resolved}")
    return resolved


def _count_nonempty_records(path: Path, label: str) -> int:
    count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} contains an empty record at line {line_number}")
            count += 1
    if count == 0:
        raise ValueError(f"{label} contains no records: {path}")
    return count


def _validate_plink_prefix(prefix: Path) -> PlinkMetadata:
    prefix = _normalize_prefix(prefix)
    bed = _regular_input(Path(f"{prefix}.bed"), "BED")
    bim = _regular_input(Path(f"{prefix}.bim"), "BIM")
    fam = _regular_input(Path(f"{prefix}.fam"), "FAM")
    samples = _count_nonempty_records(fam, "FAM")
    variants = _count_nonempty_records(bim, "BIM")
    bytes_per_variant = (samples + 3) // 4
    expected = 3 + variants * bytes_per_variant
    actual = bed.stat().st_size
    with bed.open("rb") as handle:
        magic = handle.read(3)
    if magic != BED_MAGIC:
        raise ValueError(
            f"BED must be PLINK 1 variant-major format; observed header {magic.hex()}"
        )
    if actual != expected:
        raise ValueError(
            f"BED length mismatch: observed {actual} bytes, expected {expected} "
            f"for N={samples}, M={variants}"
        )
    return PlinkMetadata(
        prefix=str(prefix.resolve()), samples=samples, variants=variants,
        bytes_per_variant=bytes_per_variant, expected_bed_bytes=expected,
        actual_bed_bytes=actual,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_package_identity(install: Path) -> dict[str, Any]:
    package = install / "summit"
    if not package.is_dir():
        raise ValueError(f"install prefix lacks the summit package: {install}")
    records: list[tuple[str, int, str]] = []
    for path in sorted(package.rglob("*")):
        relative_to_package = path.relative_to(package)
        if "__pycache__" in relative_to_package.parts or path.suffix == ".pyc":
            raise RuntimeError(
                "accepted summit install contains executable bytecode/cache: "
                f"{relative_to_package}"
            )
        if not path.is_file():
            continue
        records.append(
            (str(path.relative_to(install)), path.stat().st_size, _sha256(path))
        )
    if not records:
        raise ValueError(f"installed summit package is empty: {package}")
    manifest = hashlib.sha256()
    for relative, size, file_hash in records:
        manifest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
    return {
        "install_prefix": str(install),
        "package_path": str(package),
        "file_count": len(records),
        "bytes": sum(size for _, size, _ in records),
        "manifest_sha256": manifest.hexdigest(),
    }


def _copy_exact_bytes(source, destination, count: int) -> None:
    remaining = int(count)
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise ValueError("source BED ended while copying the validated prefix")
        destination.write(block)
        remaining -= len(block)


def _create_prefix_subset(
    source: PlinkMetadata, destination_prefix: Path, variants: int
) -> tuple[PlinkMetadata, dict[str, dict[str, Any]]]:
    if variants <= 0 or variants > source.variants:
        raise ValueError(
            f"requested subset M={variants} outside source range [1, {source.variants}]"
        )
    source_prefix = Path(source.prefix)
    destination_prefix.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    consumed_bed_bytes = 3 + variants * source.bytes_per_variant
    with Path(f"{source_prefix}.bed").open("rb") as src, Path(
        f"{destination_prefix}.bed"
    ).open("xb") as dst:
        _copy_exact_bytes(src, dst, consumed_bed_bytes)
    with Path(f"{source_prefix}.bim").open("rb") as src, Path(
        f"{destination_prefix}.bim"
    ).open("xb") as dst:
        for index in range(variants):
            line = src.readline()
            if not line:
                raise ValueError(f"source BIM ended at record {index}")
            dst.write(line)
    shutil.copyfile(Path(f"{source_prefix}.fam"), Path(f"{destination_prefix}.fam"))
    for extension in (".bed", ".bim", ".fam"):
        os.chmod(Path(f"{destination_prefix}{extension}"), 0o600)
    observed = _validate_plink_prefix(destination_prefix)
    if observed.samples != source.samples or observed.variants != variants:
        raise RuntimeError("validated PLINK subset dimensions changed during staging")
    identities = {}
    for extension in (".bed", ".bim", ".fam"):
        target = Path(f"{destination_prefix}{extension}")
        identities[extension[1:]] = {
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    return observed, identities


def _parse_integer_ranges(value: str) -> list[int]:
    selected: set[int] = set()
    for component in str(value).split(","):
        component = component.strip()
        if not component:
            raise ValueError("CPU list contains an empty component")
        if "-" in component:
            left, right = component.split("-", 1)
            start, stop = int(left), int(right)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range: {component}")
            selected.update(range(start, stop + 1))
        else:
            cpu = int(component)
            if cpu < 0:
                raise ValueError("CPU identifiers must be nonnegative")
            selected.add(cpu)
    if not selected:
        raise ValueError("CPU list is empty")
    return sorted(selected)


def _format_integer_ranges(values: Sequence[int]) -> str:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        raise ValueError("cannot format an empty CPU list")
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _affinity_cpu_set(value: Any) -> set[int] | None:
    """Normalize native range strings and Python integer sequences."""
    if value is None:
        return None
    if isinstance(value, str):
        return set(_parse_integer_ranges(value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            return {int(cpu) for cpu in value}
        except (TypeError, ValueError) as exc:
            raise ValueError("GEMM affinity sequence contains a non-integer") from exc
    raise ValueError(f"unsupported GEMM affinity representation: {type(value).__name__}")


def _validate_openmp_placement_attestation(
    raw: Mapping[str, Any],
    *,
    cpu_records: Sequence[CpuRecord],
    threads: int,
) -> dict[str, Any]:
    """Require the exact API-9 ordered singleton-place worker evidence."""
    required_keys = {
        "schema", "schema_version", "verified", "immutable",
        "requested_threads", "expected_cpu_ids", "omp_dynamic",
        "omp_thread_limit", "omp_max_active_levels", "omp_proc_bind",
        "omp_binding_active", "omp_num_places", "effective_openmp_capacity",
        "place_cpu_ids", "team_size", "exact_singleton_places",
        "exact_team_coverage", "workers", "vendor_calls",
    }
    if not isinstance(raw, Mapping) or set(raw) != required_keys:
        raise RuntimeError("OpenMP placement attestation has a noncanonical schema")
    record = dict(raw)
    expected_cpus = [item.cpu for item in cpu_records]
    expected_scalars = {
        "schema": OPENMP_PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": threads,
        "expected_cpu_ids": expected_cpus,
        "omp_dynamic": False,
        "omp_thread_limit": threads,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": threads,
        "effective_openmp_capacity": threads,
        "place_cpu_ids": [[cpu] for cpu in expected_cpus],
        "team_size": threads,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "vendor_calls": 0,
    }
    mismatches = {
        name: {"expected": value, "observed": record.get(name)}
        for name, value in expected_scalars.items()
        if record.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"OpenMP placement attestation mismatch: {mismatches}")
    integer_fields = (
        "schema_version", "requested_threads", "omp_thread_limit",
        "omp_max_active_levels", "omp_num_places",
        "effective_openmp_capacity", "team_size", "vendor_calls",
    )
    boolean_fields = (
        "verified", "immutable", "omp_dynamic", "omp_binding_active",
        "exact_singleton_places", "exact_team_coverage",
    )
    if any(type(record.get(name)) is not int for name in integer_fields):
        raise RuntimeError("OpenMP placement integer fields are not built-in integers")
    if any(type(record.get(name)) is not bool for name in boolean_fields):
        raise RuntimeError("OpenMP placement boolean fields are not built-in booleans")
    worker_keys = {
        "thread_num", "place_num", "place_cpu_ids",
        "sched_affinity_cpu_ids", "current_cpu", "verified",
    }
    workers = record.get("workers")
    if not isinstance(workers, list) or len(workers) != threads:
        raise RuntimeError("OpenMP placement worker count is incomplete")
    for index, (cpu, worker) in enumerate(zip(expected_cpus, workers, strict=True)):
        expected_worker = {
            "thread_num": index,
            "place_num": index,
            "place_cpu_ids": [cpu],
            "sched_affinity_cpu_ids": [cpu],
            "current_cpu": cpu,
            "verified": True,
        }
        if (
            not isinstance(worker, Mapping)
            or set(worker) != worker_keys
            or dict(worker) != expected_worker
            or type(worker.get("thread_num")) is not int
            or type(worker.get("place_num")) is not int
            or type(worker.get("current_cpu")) is not int
            or type(worker.get("verified")) is not bool
        ):
            raise RuntimeError(f"OpenMP placement worker {index} is noncanonical")
    return record


def _validate_blis_compile_options(
    raw: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    placement: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("reference backend provenance lacks compile options")
    options = dict(raw)
    required = {
        "api_version": EXPECTED_NATIVE_API_VERSION,
        "blas_vendor": "BLIS",
        "blas_runtime_config": EXPECTED_BLIS_RUNTIME_CONFIG,
        "blas_runtime_corename": EXPECTED_BLIS_CONFIG_FAMILY,
        "blas_runtime_isolation": "private_static",
        "blas_runtime_threads": args.threads,
        "blas_runtime_threading_layer": "pthreads",
        "blas_runtime_worker_affinity_policy": (
            "inherit_authenticated_selected_cpu_set_per_call"
        ),
        "blas_runtime_thread_strategy": "automatic",
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_owner_thread_configured": True,
        "blas_runtime_environment_immutable": True,
        "blas_runtime_environment_contract": "blis_process_start_v1",
        "blas_runtime_tls_enabled": True,
        "gemm_execution_mode": "serialized_fixed_private_blis",
        "private_openblas_archive_sha256": "none",
        "private_blas_backend": EXPECTED_PRIVATE_BLAS_NAME,
        "private_blas_archive_sha256": args.expected_archive_sha256,
        "private_blas_source_commit": args.expected_private_source_commit,
        "private_blas_source_tree_sha256": (
            args.expected_private_source_tree_sha256
        ),
        "private_blas_config_family": EXPECTED_BLIS_CONFIG_FAMILY,
        "gemm_vendor_entry_outer_openmp_guard": True,
        "openmp_effective_capacity_policy": (
            "bound_places_else_sched_affinity_v1"
        ),
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": OPENMP_PLACEMENT_SCHEMA,
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
        "openmp_placement_contract_evidence": dict(placement),
        "native_integrity_snapshot_numa_contract_supported": True,
        "native_integrity_snapshot_numa_contract_schema": (
            NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
        ),
        "native_integrity_snapshot_numa_query_chunk_page_limit": (
            NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_schema": (
            NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
        ),
        "native_gemm_output_numa_query_chunk_page_limit": (
            NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_evidence_capacity": (
            NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ),
    }
    mismatches = {
        name: {"expected": value, "observed": options.get(name)}
        for name, value in required.items()
        if options.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"reference private-BLIS compile contract mismatch: {mismatches}")
    if (
        type(options.get("gemm_integrity_enabled")) is not bool
        or (
            not args.allow_integrity_disabled
            and options.get("gemm_integrity_enabled") is not True
        )
    ):
        raise RuntimeError("reference private-BLIS integrity contract is not accepted")
    ways = options.get("blas_runtime_thread_ways")
    if (
        not isinstance(ways, Mapping)
        or set(ways) != set(BLIS_WAY_NAMES)
        or any(type(ways.get(name)) is not int or ways[name] != 1 for name in BLIS_WAY_NAMES)
    ):
        raise RuntimeError("reference automatic BLIS loop-way contract is not exact")
    for name in ("private_blas_header_sha256", "private_blas_cblas_header_sha256"):
        if not _canonical_sha256(options.get(name)):
            raise RuntimeError(f"reference backend provenance lacks canonical {name}")
    loaded = options.get("loaded_blas_runtime")
    expected_loaded = {
        "internal_api": "blis",
        "version": "2.0",
        "path": "private-static gxeldcore image",
        "num_threads": args.threads,
        "isolation": "private_static",
        "threading_layer": "openmp",
    }
    if not isinstance(loaded, Mapping) or any(
        loaded.get(name) != value for name, value in expected_loaded.items()
    ):
        raise RuntimeError("reference loaded private-BLIS runtime contract is not exact")
    if (
        type(options.get("api_version")) is not int
        or type(options.get("blas_runtime_threads")) is not int
        or type(options.get("blas_runtime_tls_enabled")) is not bool
        or type(options.get("openmp_placement_probe_vendor_calls")) is not int
        or type(options.get("gemm_integrity_minimum_vendor_flops")) is not int
        or options["gemm_integrity_minimum_vendor_flops"] <= 0
        or type(options.get(
            "native_integrity_snapshot_numa_contract_supported"
        )) is not bool
        or type(options.get(
            "native_integrity_snapshot_numa_query_chunk_page_limit"
        )) is not int
        or options[
            "native_integrity_snapshot_numa_query_chunk_page_limit"
        ] <= 0
        or type(options.get(
            "native_gemm_output_numa_contract_supported"
        )) is not bool
        or type(options.get(
            "native_gemm_output_numa_query_chunk_page_limit"
        )) is not int
        or options["native_gemm_output_numa_query_chunk_page_limit"] <= 0
        or type(options.get(
            "native_gemm_output_numa_evidence_capacity"
        )) is not int
        or options["native_gemm_output_numa_evidence_capacity"] <= 0
    ):
        raise RuntimeError("reference private-BLIS contract has noncanonical JSON types")
    return options


def _cpu_record(cpu: int) -> CpuRecord:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        core = int((topology / "core_id").read_text(encoding="utf-8").strip())
        socket = int(
            (topology / "physical_package_id").read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot resolve physical topology for CPU {cpu}") from exc
    nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    node = int(nodes[0].name[4:]) if len(nodes) == 1 else None
    return CpuRecord(cpu=cpu, core=core, socket=socket, node=node)


def _validate_cpu_records(records: Sequence[CpuRecord], threads: int) -> None:
    if len(records) != threads:
        raise ValueError(
            f"CPU list contains {len(records)} logical CPUs but --threads is {threads}"
        )
    if len({record.socket for record in records}) != 1:
        raise ValueError("accepted partial benchmarks must use exactly one CPU socket")
    if any(
        type(record.node) is not int or record.node < 0 for record in records
    ):
        raise ValueError("every selected CPU must have exactly one NUMA node")
    physical = {(record.socket, record.core) for record in records}
    if len(physical) != len(records):
        raise ValueError("CPU list contains SMT siblings for the same physical core")


def _resolve_cpu_records(cpu_list: str, threads: int) -> list[CpuRecord]:
    allowed = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    )
    if cpu_list != "auto":
        selected = _parse_integer_ranges(cpu_list)
        unavailable = sorted(set(selected) - allowed)
        if unavailable:
            raise ValueError(f"requested CPUs are outside current affinity: {unavailable}")
        records = [_cpu_record(cpu) for cpu in selected]
        _validate_cpu_records(records, threads)
        return records
    by_socket: dict[int, list[CpuRecord]] = {}
    seen: set[tuple[int, int]] = set()
    for cpu in sorted(allowed):
        record = _cpu_record(cpu)
        identity = (record.socket, record.core)
        if identity in seen:
            continue
        seen.add(identity)
        by_socket.setdefault(record.socket, []).append(record)
    candidates = [records[:threads] for records in by_socket.values() if len(records) >= threads]
    if not candidates:
        raise ValueError(
            f"no allowed socket supplies {threads} distinct physical CPU cores"
        )
    records = sorted(candidates, key=lambda item: (item[0].socket, item[0].cpu))[0]
    _validate_cpu_records(records, threads)
    return records


def _full_socket_numa_nodes(
    records: Sequence[CpuRecord],
    *,
    topology_records: Sequence[CpuRecord] | None = None,
) -> list[int]:
    """Resolve and verify every NUMA node belonging to one selected socket."""
    if not records or len({record.socket for record in records}) != 1:
        raise RuntimeError(
            "full-socket memory scope requires one nonempty selected socket"
        )
    if topology_records is None:
        try:
            online = _parse_integer_ranges(
                Path("/sys/devices/system/cpu/online")
                .read_text(encoding="utf-8")
                .strip()
            )
            topology_records = [_cpu_record(cpu) for cpu in online]
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "cannot resolve the online CPU topology for full-socket memory scope"
            ) from exc
    topology = list(topology_records)
    if not topology:
        raise RuntimeError("online CPU topology is empty")
    by_cpu: dict[int, CpuRecord] = {}
    node_sockets: dict[int, set[int]] = {}
    for record in topology:
        if record.cpu in by_cpu:
            raise RuntimeError("online CPU topology contains a duplicate CPU")
        if type(record.node) is not int or record.node < 0:
            raise RuntimeError("online CPU topology contains an unresolved NUMA node")
        by_cpu[record.cpu] = record
        node_sockets.setdefault(record.node, set()).add(record.socket)
    for record in records:
        if by_cpu.get(record.cpu) != record:
            raise RuntimeError(
                "selected CPU identity differs from the verified online topology"
            )
    target_socket = records[0].socket
    nodes = sorted(
        {
            record.node
            for record in topology
            if record.socket == target_socket and record.node is not None
        }
    )
    if not nodes:
        raise RuntimeError("selected socket has no verified NUMA nodes")
    if any(node_sockets[node] != {target_socket} for node in nodes):
        raise RuntimeError("a NUMA node spans multiple physical sockets")
    selected_nodes = {record.node for record in records}
    if not selected_nodes.issubset(set(nodes)):
        raise RuntimeError("selected CPU NUMA nodes escape the verified socket")
    return nodes


def _expected_memory_nodes(
    cpu_records: Sequence[CpuRecord],
    explicit_nodes: Sequence[int] | None,
) -> list[int]:
    selected_cpu_nodes = sorted(
        {record.node for record in cpu_records if record.node is not None}
    )
    if not selected_cpu_nodes:
        raise RuntimeError("selected CPUs have no resolved NUMA nodes")
    if explicit_nodes is None:
        return selected_cpu_nodes
    nodes = list(explicit_nodes)
    if (
        not nodes
        or any(type(node) is not int or node < 0 for node in nodes)
        or nodes != sorted(set(nodes))
        or not set(selected_cpu_nodes).issubset(nodes)
    ):
        raise RuntimeError(
            "explicit memory nodes are not an exact ordered superset of selected CPU nodes"
        )
    return nodes


def _file_identity(path: Path) -> dict[str, Any]:
    observed = path.stat()
    return {
        "path": str(path), "device": observed.st_dev, "inode": observed.st_ino,
        "bytes": observed.st_size, "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _same_file_identity(path: Path, identity: dict[str, Any]) -> bool:
    return _file_identity(path) == identity


def _assert_staged_subset_integrity(
    paths: Sequence[Path], identities: Mapping[str, dict[str, Any]],
    hashes: Mapping[str, Mapping[str, Any]],
) -> None:
    for path in paths:
        identity_equal = _same_file_identity(path, identities[str(path)])
        role = path.suffix[1:]
        expected = hashes.get(role)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"staged subset lacks expected {role} identity")
        observed_hash = _sha256(path)
        if not identity_equal:
            raise RuntimeError(f"literal comparison subset identity changed: {path}")
        if (
            path.stat().st_size != int(expected.get("bytes", -1))
            or observed_hash != expected.get("sha256")
        ):
            raise RuntimeError(f"literal comparison subset hash changed: {path}")


def _block_schedule(subset_variants: int, block_width: int) -> dict[str, Any]:
    if subset_variants <= 0 or block_width <= 0:
        raise ValueError("block schedules require positive variants and width")
    block_count = math.ceil(subset_variants / block_width)
    terminal_width = subset_variants - (block_count - 1) * block_width
    return {
        "subset_variants": subset_variants,
        "block_width": block_width,
        "block_count": block_count,
        "full_width_block_count": (
            block_count if terminal_width == block_width else block_count - 1
        ),
        "terminal_block_width": terminal_width,
        "has_terminal_partial_block": terminal_width != block_width,
    }


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "expected_native_sha256",
        "expected_package_manifest_sha256",
        "expected_archive_sha256",
        "expected_source_tree_sha256",
        "expected_private_source_tree_sha256",
    ):
        if not _canonical_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be 64 lowercase hex")
    for name in ("expected_source_commit", "expected_private_source_commit"):
        if not _canonical_commit(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be 40 lowercase hex")
    if args.expected_backend != EXPECTED_PRIVATE_BACKEND:
        parser.error("real-genotype acceptance permits only private BLIS")
    if not MIN_BLOCKS <= args.blocks <= MAX_BLOCKS:
        parser.error(f"--blocks must be in [{MIN_BLOCKS}, {MAX_BLOCKS}]")
    if not 1 <= args.block_width <= MAX_BLOCK_WIDTH:
        parser.error(f"--block-width must be in [1, {MAX_BLOCK_WIDTH}]")
    if not 1 <= args.probes <= MAX_PROBES:
        parser.error(f"--probes must be in [1, {MAX_PROBES}]; B=1024 is projection-only")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.compare_fp64_layouts:
        parser.error(
            "--compare-fp64-layouts is disabled: both API-8 orientation "
            "candidates failed the integrity/reproducibility gate"
        )
    block_width_comparison = args.compare_block_widths is not None
    blis_replicate_comparison = bool(args.compare_blis_replicates)
    blis_t1_reference_comparison = bool(args.compare_blis_t1_reference)
    if block_width_comparison and args.compare_fp64_layouts:
        parser.error(
            "--compare-block-widths and --compare-fp64-layouts are mutually exclusive"
        )
    if blis_replicate_comparison and (
        block_width_comparison
        or args.compare_fp64_layouts
        or blis_t1_reference_comparison
    ):
        parser.error(
            "--compare-blis-replicates is mutually exclusive with other comparisons"
        )
    if blis_t1_reference_comparison and (
        block_width_comparison or args.compare_fp64_layouts
    ):
        parser.error(
            "--compare-blis-t1-reference is mutually exclusive with other comparisons"
        )
    if block_width_comparison and args.subset_variants is None:
        parser.error("--compare-block-widths requires --subset-variants")
    if not block_width_comparison and args.subset_variants is not None:
        parser.error("--subset-variants requires --compare-block-widths")
    if block_width_comparison:
        if args.subset_variants <= 0:
            parser.error("--subset-variants must be positive")
        if args.full_precision_layout != "current":
            parser.error("--compare-block-widths permits only the current layout")
        schedules = [
            _block_schedule(args.subset_variants, width)
            for width in args.compare_block_widths
        ]
        invalid = [
            schedule for schedule in schedules
            if not MIN_BLOCKS <= schedule["block_count"] <= MAX_BLOCKS
        ]
        if invalid:
            counts = ", ".join(
                f"K={item['block_width']}: {item['block_count']} blocks"
                for item in schedules
            )
            parser.error(
                "--subset-variants must yield 4-16 blocks for every candidate; "
                + counts
            )
    if not 0.0 < args.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout-seconds must be in (0, {MAX_TIMEOUT_SECONDS}]")
    if not 0.0 < args.controller_timeout_seconds <= MAX_COMPARISON_SECONDS:
        parser.error(
            "--controller-timeout-seconds must be in "
            f"(0, {MAX_COMPARISON_SECONDS}]"
        )
    if not 0.0 < args.reference_timeout_seconds <= MAX_BLIS_T1_REFERENCE_CHILD_SECONDS:
        parser.error(
            "--reference-timeout-seconds must be in "
            f"(0, {MAX_BLIS_T1_REFERENCE_CHILD_SECONDS}]"
        )
    if args.compare_fp64_layouts and args.timeout_seconds > MAX_COMPARISON_CHILD_SECONDS:
        parser.error(
            "--compare-fp64-layouts caps each child at "
            f"{MAX_COMPARISON_CHILD_SECONDS} seconds"
        )
    if (
        block_width_comparison
        and args.timeout_seconds > MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS
    ):
        parser.error(
            "--compare-block-widths caps each child at "
            f"{MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS} seconds"
        )
    if (
        blis_replicate_comparison
        and args.timeout_seconds > MAX_BLIS_REPLICATE_CHILD_SECONDS
    ):
        parser.error(
            "--compare-blis-replicates caps each child at "
            f"{MAX_BLIS_REPLICATE_CHILD_SECONDS} seconds"
        )
    if (
        blis_t1_reference_comparison
        and args.timeout_seconds > MAX_BLIS_T32_CANDIDATE_CHILD_SECONDS
    ):
        parser.error(
            "--compare-blis-t1-reference caps the T32 candidate at "
            f"{MAX_BLIS_T32_CANDIDATE_CHILD_SECONDS} seconds"
        )
    if (
        block_width_comparison
        and args.controller_timeout_seconds
        > MAX_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS
    ):
        parser.error(
            "--compare-block-widths caps the controller at "
            f"{MAX_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS} seconds"
        )
    if (
        blis_replicate_comparison
        and args.controller_timeout_seconds
        > MAX_BLIS_REPLICATE_CONTROLLER_SECONDS
    ):
        parser.error(
            "--compare-blis-replicates caps the controller at "
            f"{MAX_BLIS_REPLICATE_CONTROLLER_SECONDS} seconds"
        )
    if (
        blis_t1_reference_comparison
        and args.controller_timeout_seconds
        > MAX_BLIS_T1_COMPARISON_CONTROLLER_SECONDS
    ):
        parser.error(
            "--compare-blis-t1-reference caps the controller at "
            f"{MAX_BLIS_T1_COMPARISON_CONTROLLER_SECONDS} seconds"
        )
    if args.allow_integrity_disabled:
        parser.error(
            "API-9 private-BLIS real-genotype runs require GEMM integrity checks"
        )
    if blis_t1_reference_comparison and (
        args.threads != 32
        or args.blocks != 4
        or args.block_width != 2000
        or args.probes != 32
    ):
        parser.error(
            "--compare-blis-t1-reference requires exactly 32 candidate threads, "
            "four full K=2000 genotype blocks (the first 8000 variants), and B=32"
        )
    if blis_t1_reference_comparison and (
        args.cpu_list == "auto"
        or _parse_integer_ranges(args.cpu_list) != list(range(32))
    ):
        parser.error(
            "--compare-blis-t1-reference requires candidate CPUs 0-31; "
            "the T1 reference uses CPU0"
        )
    if blis_t1_reference_comparison and (
        args.comparison_rtol != DEFAULT_COMPARISON_RTOL
        or args.comparison_atol != DEFAULT_COMPARISON_ATOL
    ):
        parser.error(
            "--compare-blis-t1-reference requires the predeclared "
            "rtol=atol=5e-12 accuracy gate"
        )
    if args.full_precision_layout != "current" and args.storage_dtype != "float64":
        parser.error(
            "--full-precision-layout source-tt-target-current requires "
            "--storage-dtype float64"
        )
    if args.compare_fp64_layouts and args.storage_dtype != "float64":
        parser.error("--compare-fp64-layouts requires --storage-dtype float64")
    if args.compare_fp64_layouts and args.full_precision_layout != "current":
        parser.error(
            "--compare-fp64-layouts controls both layouts; leave "
            "--full-precision-layout at its current default"
        )
    for name, value in (
        ("--comparison-rtol", args.comparison_rtol),
        ("--comparison-atol", args.comparison_atol),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.workspace_gib) or not 0.0 < args.workspace_gib <= 32.0:
        parser.error("--workspace-gib must be finite and in (0, 32]")
    if not math.isfinite(args.target_memory_gib) or not 0.0 < args.target_memory_gib <= 32.0:
        parser.error("--target-memory-gib must be finite and in (0, 32]")
    columns = [item.strip() for item in args.environment_columns.split(",")]
    if not 2 <= len(columns) <= 3 or any(not item for item in columns):
        parser.error("--environment-columns must name two or three nonempty columns")
    if len(set(columns)) != len(columns):
        parser.error("--environment-columns must be unique")
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        parser.error(f"refusing existing output: {output}")
    if not output.parent.resolve().is_dir():
        parser.error(f"output parent must already exist: {output.parent}")
    if args.temporary_parent is not None and not args.temporary_parent.expanduser().resolve().is_dir():
        parser.error("--temporary-parent must be an existing directory")


def _native_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, list[Path]]:
    install = args.install_prefix.expanduser().resolve(strict=True)
    if not install.is_dir():
        raise ValueError(f"install prefix is not a directory: {install}")
    native = _regular_input(args.native_module, "native module")
    package = (install / "summit").resolve()
    if native.parent != package or native.suffix != ".so":
        raise ValueError(
            "--native-module must be a .so directly inside --install-prefix/summit"
        )
    python = _executable_input(args.python_executable)
    archive = _regular_input(args.private_archive, "private BLIS archive")
    dependencies = args.dependency_path
    if dependencies is None:
        dependencies = [Path(sysconfig.get_paths()["purelib"])]
    resolved_dependencies = []
    for dependency in dependencies:
        resolved = dependency.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"dependency path is not a directory: {resolved}")
        resolved_dependencies.append(resolved)
    if _sha256(native) != args.expected_native_sha256:
        raise ValueError(
            "native module SHA-256 differs from --expected-native-sha256"
        )
    if _sha256(archive) != args.expected_archive_sha256:
        raise ValueError(
            "private BLIS archive SHA-256 differs from --expected-archive-sha256"
        )
    package_identity = _installed_package_identity(install)
    if (
        package_identity["manifest_sha256"]
        != args.expected_package_manifest_sha256
    ):
        raise ValueError(
            "installed package differs from --expected-package-manifest-sha256"
        )
    return install, native, archive, python, resolved_dependencies


def _command(
    args: argparse.Namespace,
    *, subset_prefix: Path,
    output_prefix: Path,
    python: Path,
    cpu_records: Sequence[CpuRecord],
    full_precision_layout: str | None = None,
    block_width: int | None = None,
    explicit_memory_nodes: Sequence[int] | None = None,
) -> list[str]:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for an accepted benchmark")
    nodes = _expected_memory_nodes(cpu_records, explicit_memory_nodes)
    layout = (
        args.full_precision_layout
        if full_precision_layout is None else str(full_precision_layout)
    )
    if layout not in FP64_LAYOUTS:
        raise ValueError(f"unsupported full-precision layout: {layout}")
    selected_block_width = args.block_width if block_width is None else int(block_width)
    if not 1 <= selected_block_width <= MAX_BLOCK_WIDTH:
        raise ValueError(
            f"command block width must be in [1, {MAX_BLOCK_WIDTH}]"
        )
    command = [
        taskset, "-c", _format_integer_ranges([record.cpu for record in cpu_records]),
        str(python), "-S", "-c", _CLI_BOOTSTRAP,
        "--geno", str(subset_prefix),
        "--env", str(args.environment_file.expanduser().resolve()),
        "--gxe-env-cols", args.environment_columns,
        "--gxe-parallel-environment-groups", "1",
        "--gxe-explicit-openmp-placement",
    ]
    if explicit_memory_nodes is not None:
        command.extend(
            [
                "--gxe-explicit-openmp-memory-scope",
                EXPLICIT_OPENMP_MEMORY_SCOPE,
            ]
        )
    command.extend([
        "--covar", str(args.covariate_file.expanduser().resolve()),
        "--gxe-kernel-mode", "standardized",
        "--gxe-genotype-scale", "sample",
        "--gxe-native-backend", "direct",
        "--gxe-fp64-layout", layout,
        "--gxe-native-workspace-gib", str(args.workspace_gib),
        "--gxe-native-target-panel-columns", "64",
        "--rand-dist", "rademacher", "--seed", str(args.seed),
        "--dtype", args.storage_dtype, "--ddof", "1", "--impute-method", "mean",
        "--step_size", str(selected_block_width),
        "--target-xz-mem", str(args.target_memory_gib),
        "--num-threads", str(args.threads), "--force_affinity_all", "false",
        "--numa-mode", "membind", "--numa-nodes", _format_integer_ranges(nodes),
        "--nvecs", str(args.probes),
        "--suppress", "--out", str(output_prefix),
    ])
    return command


def _child_environment(
    args: argparse.Namespace,
    install: Path,
    native: Path,
    dependencies: Sequence[Path],
    cpu_records: Sequence[CpuRecord],
) -> tuple[dict[str, str], dict[str, str]]:
    environment = os.environ.copy()
    selected_paths = [install, *dependencies]
    package_identity = _installed_package_identity(install)
    ordered_cpus = [record.cpu for record in cpu_records]
    if len(ordered_cpus) != args.threads:
        raise RuntimeError("child environment CPU/thread contract is incomplete")
    places = ",".join(f"{{{cpu}}}" for cpu in ordered_cpus)
    settings = {
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join(map(str, selected_paths)),
        "SUMMIT_DECODE_THREADS": str(min(args.threads, 32)),
        "OMP_NUM_THREADS": str(args.threads),
        "OMP_THREAD_LIMIT": str(args.threads),
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_DYNAMIC": "FALSE", "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "0", "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": places,
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "OPENBLAS_THREAD_TIMEOUT": "1", "MKL_NUM_THREADS": str(args.threads),
        "MKL_DYNAMIC": "FALSE", "NUMEXPR_NUM_THREADS": str(args.threads),
        "VECLIB_MAXIMUM_THREADS": str(args.threads),
        "BLIS_NUM_THREADS": str(args.threads),
        "SUMMIT_GXE_EXPECTED_NATIVE_MODULE": str(native),
        "SUMMIT_GXE_EXPECTED_NATIVE_SHA256": _sha256(native),
        "SUMMIT_GXE_EXPECTED_SOURCE_COMMIT": args.expected_source_commit,
        "SUMMIT_GXE_EXPECTED_SOURCE_TREE_SHA256": (
            args.expected_source_tree_sha256
        ),
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_ARCHIVE_SHA256": (
            args.expected_archive_sha256
        ),
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_COMMIT": (
            args.expected_private_source_commit
        ),
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_TREE_SHA256": (
            args.expected_private_source_tree_sha256
        ),
        "SUMMIT_GXE_EXPECTED_BLAS_THREADS": str(args.threads),
        "SUMMIT_GXE_EXPECTED_INSTALL_PREFIX": str(install),
        "SUMMIT_GXE_EXPECTED_PACKAGE_MANIFEST_SHA256": (
            package_identity["manifest_sha256"]
        ),
        "SUMMIT_GXE_REQUIRE_INTEGRITY": "0" if args.allow_integrity_disabled else "1",
        "SUMMIT_GXE_REQUIRE_EARLY_NUMA": "1",
    }
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    for inherited in (
        *OPENMP_AFFINITY_CONFLICTS, *BLIS_AUTOMATIC_CONFLICTS,
        "SUMMIT_NUMACTL_WRAPPED", "SUMMIT_NUMA_POLICY_APPLIED",
        "SUMMIT_NUMA_POLICY_PROVENANCE",
    ):
        environment.pop(inherited, None)
    environment.update(settings)
    return environment, settings


def _run_bounded(
    command: Sequence[str], environment: dict[str, str], timeout_seconds: float
) -> tuple[subprocess.CompletedProcess[str], float, float, int]:
    started = time.perf_counter()
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    process = subprocess.Popen(
        list(command), env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise TimeoutError(
            f"partial real-genotype benchmark exceeded {timeout_seconds:.1f} seconds"
        )
    elapsed = time.perf_counter() - started
    child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    completed = subprocess.CompletedProcess(
        list(command), process.returncode, stdout=stdout, stderr=stderr
    )
    # RUSAGE_CHILDREN ru_maxrss is process-lifetime state for the controller.
    # It is cumulative across sequential children and cannot be attributed to
    # one layout.  Each batch manifest's in-process peak is authoritative for
    # per-layout RSS.
    maxrss_kib = max(child_usage_before, child_usage_after)
    return completed, elapsed, maxrss_kib / 1024**2, process.pid


def _redact_temporary(value: str, temporary_root: Path) -> str:
    return value.replace(str(temporary_root), "<temporary_directory>")


def _bounded_text_record(value: str, temporary_root: Path) -> dict[str, Any]:
    encoded = value.encode("utf-8", errors="replace")
    tail = encoded[-16_384:].decode("utf-8", errors="replace")
    return {
        "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
        "tail": _redact_temporary(tail, temporary_root), "tail_limit_bytes": 16_384,
    }


def _gemm_summaries(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("phase"), record.get("operation"), record.get("layout"),
            record.get("transpose_a"), record.get("transpose_b"),
            int(record.get("m", 0)), int(record.get("n", 0)), int(record.get("k", 0)),
        )
        target = grouped.setdefault(
            key,
            {
                "phase": key[0], "operation": key[1], "layout": key[2],
                "transpose_a": key[3], "transpose_b": key[4],
                "m": key[5], "n": key[6], "k": key[7], "calls": 0,
                "wall_seconds": 0.0, "process_cpu_seconds": 0.0, "flops": 0,
            },
        )
        target["calls"] += 1
        target["wall_seconds"] += float(record.get("wall_seconds", 0.0))
        target["process_cpu_seconds"] += float(record.get("process_cpu_seconds", 0.0))
        target["flops"] += int(record.get("flops", record.get("flop_count", 0)))
    result = []
    for target in grouped.values():
        wall = target["wall_seconds"]
        target["achieved_gflops"] = target["flops"] / wall / 1e9 if wall else 0.0
        target["average_active_cores"] = (
            target["process_cpu_seconds"] / wall if wall else 0.0
        )
        result.append(target)
    return sorted(
        result,
        key=lambda item: (
            str(item["phase"]), str(item["operation"]), item["m"], item["n"], item["k"]
        ),
    )


_PHASE_SCALING_CLASS = {
    "feature_plan_construction": "fixed",
    "genotype_read_decode_standardization": "genotype_pass_variant",
    "feature_moments": "variant",
    "random_generation": "tile_variant",
    "source_weight_packing": "tile_variant",
    "source_gemm": "tile_variant",
    "source_context_correction": "tile_variant",
    "feature_finalization_fp64": "variant",
    "projection_context_correction": "tile",
    "fp64_same_person_accumulation": "tile",
    "source_to_target_layout_conversion": "tile",
    "target_operand_construction": "tile",
    "target_gemm": "tile_variant",
    "fp64_score_reductions": "tile_variant",
    "native_binary_integrity_hashing": "fixed",
    "fp64_same_person_finalization": "fixed",
    "fp64_score_normalization": "variant",
    "shared_input_identity_hashing": "variant",
    "output_bundle_end_to_end": "variant",
    "output_identity_hashing": "variant",
}


def _schedule_passes(environment_tiles: int, probe_tiles: int) -> int:
    combinations = environment_tiles * probe_tiles
    return 2 if combinations == 1 else 1 + 2 * combinations


def _projections(
    *, elapsed: float, subset_variants: int, full_variants: int,
    block_width: int, measured_probes: int, manifest: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    full_scale = full_variants / subset_variants
    measured_schedule = _block_schedule(subset_variants, block_width)
    full_blocks = math.ceil(full_variants / block_width)
    phase_totals = performance.get("phase_totals", {})
    unknown_phases = sorted(set(phase_totals) - set(_PHASE_SCALING_CLASS))
    if unknown_phases:
        raise RuntimeError(
            "partial benchmark projection has no scaling rule for phases: "
            + ", ".join(unknown_phases)
        )
    measured_passes = int(manifest.get("shared_genotype_passes", 0))
    environment_tiles = len(manifest.get("environment_tiles", []))
    measured_probe_tiles = manifest.get("randomization", {}).get("probe_tiles", [])
    if environment_tiles <= 0 or not measured_probe_tiles:
        raise RuntimeError("completed manifest lacks its measured execution tiles")
    measured_tile_widths = []
    next_probe = 0
    for tile in measured_probe_tiles:
        if not isinstance(tile, (list, tuple)) or len(tile) != 2:
            raise RuntimeError("completed manifest contains a malformed probe tile")
        start, width = map(int, tile)
        if start != next_probe or width <= 0:
            raise RuntimeError(
                "completed manifest probe tiles are not positive and contiguous"
            )
        measured_tile_widths.append(width)
        next_probe += width
    if next_probe != measured_probes:
        raise RuntimeError("completed manifest probe tiles do not cover measured probes")
    measured_probe_tile_count = len(measured_probe_tiles)
    measured_tile_combinations = environment_tiles * measured_probe_tile_count
    expected_measured_passes = _schedule_passes(
        environment_tiles, measured_probe_tile_count
    )
    if measured_passes != expected_measured_passes:
        raise RuntimeError(
            f"manifest records {measured_passes} passes, expected "
            f"{expected_measured_passes} from its execution tiles"
        )
    # This Stage-1 projection deliberately reuses the measured probe-tile
    # width.  It is a baseline schedule, not the later memory-planner choice.
    projected_tile_width = max(measured_tile_widths)
    projected_probe_tiles = math.ceil(PROJECTED_PROBES / projected_tile_width)
    projected_tile_combinations = environment_tiles * projected_probe_tiles
    projected_passes = _schedule_passes(
        environment_tiles, projected_probe_tiles
    )
    tile_ratio = projected_tile_combinations / measured_tile_combinations
    pass_ratio = projected_passes / measured_passes
    phase_model = []
    attributed_wall = 0.0
    b32_attributed_wall = 0.0
    b1024_attributed_wall = 0.0
    for name, observed in sorted(phase_totals.items()):
        wall = float(observed.get("wall_seconds", 0.0))
        if not math.isfinite(wall) or wall < 0.0:
            raise RuntimeError(f"invalid wall time for phase {name}: {wall}")
        scaling_class = _PHASE_SCALING_CLASS[name]
        if scaling_class == "fixed":
            b32_multiplier = 1.0
            b1024_multiplier = 1.0
        elif scaling_class == "variant":
            b32_multiplier = full_scale
            b1024_multiplier = full_scale
        elif scaling_class == "tile":
            b32_multiplier = 1.0
            b1024_multiplier = tile_ratio
        elif scaling_class == "tile_variant":
            b32_multiplier = full_scale
            b1024_multiplier = full_scale * tile_ratio
        elif scaling_class == "genotype_pass_variant":
            b32_multiplier = full_scale
            b1024_multiplier = full_scale * pass_ratio
        else:  # pragma: no cover - the table above is closed by construction.
            raise AssertionError(f"unknown scaling class: {scaling_class}")
        attributed_wall += wall
        b32_attributed_wall += wall * b32_multiplier
        b1024_attributed_wall += wall * b1024_multiplier
        phase_model.append(
            {
                "phase": name, "measured_wall_seconds": wall,
                "scaling_class": scaling_class,
                "full_m_b32_multiplier": b32_multiplier,
                "full_m_b1024_multiplier": b1024_multiplier,
                "full_m_b32_projected_seconds": wall * b32_multiplier,
                "full_m_b1024_projected_seconds": wall * b1024_multiplier,
            }
        )
    overlap_tolerance = max(0.5, 0.05 * elapsed)
    if attributed_wall > elapsed + overlap_tolerance:
        raise RuntimeError(
            "named phase wall times materially overlap or exceed child wall time: "
            f"phases={attributed_wall:.6f}s child={elapsed:.6f}s"
        )
    unattributed_wall = max(0.0, elapsed - attributed_wall)
    # The conservative headline scales unattributed setup/remainder with M but
    # never with B.  A fixed-residual lower bound is retained beside it.
    residual_lower = unattributed_wall
    residual_conservative = unattributed_wall * full_scale
    b32_lower = b32_attributed_wall + residual_lower
    b32_seconds = b32_attributed_wall + residual_conservative
    b1024_lower = b1024_attributed_wall + residual_lower
    b1024_seconds = b1024_attributed_wall + residual_conservative
    return {
        "method": (
            "explicit phase scaling from the disposable prefix; the B1024 "
            "baseline reuses the measured probe-tile width and exact current "
            "pass formula"
        ),
        "projections_are_not_observed_runs": True,
        "projection_role": (
            "Stage-1 same-tile baseline only; replace with the accepted "
            "Stage-3 memory-planner schedule for final recommendations"
        ),
        "measured_subset": {
            "seconds": elapsed,
            "block_width": block_width,
            "block_count": measured_schedule["block_count"],
            "terminal_block_width": measured_schedule["terminal_block_width"],
            "has_terminal_partial_block": measured_schedule[
                "has_terminal_partial_block"
            ],
            "variants_per_second": subset_variants / elapsed,
            "blocks_per_second": measured_schedule["block_count"] / elapsed,
            "attributed_phase_seconds": attributed_wall,
            "unattributed_seconds": unattributed_wall,
            "unattributed_fraction": unattributed_wall / elapsed,
        },
        "full_m_b32": {
            "variants": full_variants, "probes": measured_probes,
            "block_width": block_width, "projected_block_count": full_blocks,
            "projected_seconds": b32_seconds,
            "projected_minutes": b32_seconds / 60.0,
            "projected_seconds_with_fixed_unattributed_residual": b32_lower,
            "projected_variants_per_second": full_variants / b32_seconds,
            "projected_blocks_per_second": full_blocks / b32_seconds,
        },
        "full_m_b1024_same_tile_schedule": {
            "variants": full_variants, "probes": PROJECTED_PROBES,
            "block_width": block_width, "projected_block_count": full_blocks,
            "measured_probe_tile_widths": measured_tile_widths,
            "projected_probe_tile_width": projected_tile_width,
            "projected_probe_tiles": projected_probe_tiles,
            "environment_tiles": environment_tiles,
            "measured_tile_combinations": measured_tile_combinations,
            "projected_tile_combinations": projected_tile_combinations,
            "measured_genotype_passes": measured_passes,
            "projected_genotype_passes": projected_passes,
            "projected_seconds": b1024_seconds,
            "projected_minutes": b1024_seconds / 60.0,
            "projected_seconds_with_fixed_unattributed_residual": b1024_lower,
            "projected_variants_per_second": full_variants / b1024_seconds,
            "projected_blocks_per_second": full_blocks / b1024_seconds,
        },
        "phase_schedule_model": phase_model,
    }


def _artifact_identities(
    manifest_path: Path, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Verify and retain every disposable scientific artifact identity."""
    records = []
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise RuntimeError("batch manifest lacks reference artifact declarations")
    for item in references:
        if not isinstance(item, Mapping):
            raise RuntimeError("batch manifest contains a malformed reference declaration")
        unresolved = manifest_path.parent / str(item.get("reference", ""))
        if unresolved.is_symlink():
            raise RuntimeError(f"reference manifest must not be a symlink: {unresolved}")
        path = unresolved.resolve()
        if not path.is_file():
            raise RuntimeError(f"reference manifest is missing: {path}")
        observed = _sha256(path)
        if observed != item.get("sha256"):
            raise RuntimeError(f"reference hash mismatch before temporary cleanup: {path}")
        reference = json.loads(path.read_text(encoding="utf-8"))
        environment = str(item.get("environment", ""))
        if (
            reference.get("kind") != "summit.gxe.reference"
            or int(reference.get("schema_version", -1)) != 3
            or reference.get("environment") != environment
        ):
            raise RuntimeError(f"unexpected reference schema/environment in {path}")
        records.append(
            {
                "role": "reference_manifest", "environment": environment,
                "name": path.name, "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
        files = reference.get("files")
        hashes = reference.get("artifact_sha256")
        if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
            raise RuntimeError(f"reference manifest lacks artifact files: {path}")
        if set(files) != set(hashes):
            raise RuntimeError(
                f"reference file/hash declarations differ in {path}"
            )
        required_roles = {*SCORE_FAMILIES, "diagonal"}
        missing_roles = sorted(required_roles.difference(files))
        if missing_roles:
            raise RuntimeError(
                f"reference manifest lacks required artifacts: {missing_roles}"
            )
        ordered_roles = [*SCORE_FAMILIES, "diagonal"]
        if "jackknife" in files:
            ordered_roles.append("jackknife")
        ordered_roles.extend(sorted(set(files).difference(ordered_roles)))
        for role in ordered_roles:
            artifact = _declared_artifact(path, reference, role)
            artifact_sha256 = _sha256(artifact)
            if artifact_sha256 != hashes[role]:
                raise RuntimeError(
                    f"declared {role!r} artifact changed during identity capture"
                )
            record_role = (
                "score"
                if role in SCORE_FAMILIES
                else role
                if role in {"diagonal", "jackknife"}
                else "declared_artifact"
            )
            records.append(
                {
                    "role": record_role,
                    "declared_role": role,
                    **({"family": role} if role in SCORE_FAMILIES else {}),
                    "environment": environment, "name": artifact.name,
                    "declared_relative_path": str(files[role]),
                    "bytes": artifact.stat().st_size, "sha256": artifact_sha256,
                }
            )
    records.append(
        {
            "role": "batch_manifest", "name": manifest_path.name,
            "bytes": manifest_path.stat().st_size, "sha256": _sha256(manifest_path),
        }
    )
    return records


def _canonical_layout(value: str) -> str:
    layout = str(value).strip().lower().replace("-", "_")
    if layout not in {"current", "source_tt_target_current"}:
        raise ValueError(f"unsupported full-precision layout: {value!r}")
    return layout


def _integer_pair(record: Mapping[str, Any], name: str, phase: str) -> tuple[int, int]:
    raw = record.get(name)
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != 2
    ):
        raise RuntimeError(f"{phase} vendor telemetry has malformed {name}")
    try:
        left, right = (int(raw[0]), int(raw[1]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{phase} vendor telemetry has non-integer {name}"
        ) from exc
    return left, right


def _positive_finite_metric(
    record: Mapping[str, Any], name: str, phase: str
) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{phase} vendor telemetry lacks finite positive {name}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            f"{phase} vendor telemetry {name} must be finite and positive"
        )
    return value


def _validate_current_vendor_acceptance(
    records: Sequence[dict[str, Any]], *, samples: int, variants: int,
    block_width: int, probes: int, annotation_bins: int, threads: int,
    environment_count: int, environment_tiles: Sequence[Any],
    probe_tiles: Sequence[Any],
    require_private_blis: bool = False,
    native_sha256: str | None = None,
    source_commit: str | None = None,
    source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    for label, value in (
        ("samples", samples), ("variants", variants),
        ("block_width", block_width), ("probes", probes),
        ("annotation_bins", annotation_bins), ("threads", threads),
        ("environment_count", environment_count),
    ):
        if int(value) <= 0:
            raise RuntimeError(f"current vendor acceptance requires positive {label}")
    if not isinstance(environment_tiles, Sequence) or not environment_tiles:
        raise RuntimeError("batch manifest lacks its environment-tile schedule")
    normalized_environment_tiles = []
    next_environment = 0
    for raw in environment_tiles:
        start, stop = _integer_pair(
            {"environment_tile": raw}, "environment_tile", "batch manifest"
        )
        if start != next_environment or stop <= start or stop > environment_count:
            raise RuntimeError(
                "batch environment tiles must be positive, contiguous, and complete"
            )
        normalized_environment_tiles.append((start, stop))
        next_environment = stop
    if next_environment != environment_count:
        raise RuntimeError(
            "batch environment tiles do not cover every requested environment"
        )
    if not isinstance(probe_tiles, Sequence) or not probe_tiles:
        raise RuntimeError("batch manifest lacks its probe-tile schedule")
    normalized_probe_tiles = []
    next_probe = 0
    for raw in probe_tiles:
        start, count = _integer_pair(
            {"probe_tile": raw}, "probe_tile", "batch manifest"
        )
        if start != next_probe or count <= 0 or start + count > probes:
            raise RuntimeError(
                "batch probe tiles must be positive, contiguous, and complete"
            )
        normalized_probe_tiles.append((start, count))
        next_probe = start + count
    if next_probe != probes:
        raise RuntimeError("batch probe tiles do not cover every measured probe")
    genotype_blocks = [
        (index, start, min(start + block_width, variants))
        for index, start in enumerate(range(0, variants, block_width))
    ]
    expected_contexts = sorted(
        (
            environment_start, environment_stop,
            probe_start, probe_count,
            genotype_index, genotype_start, genotype_stop,
        )
        for environment_start, environment_stop in normalized_environment_tiles
        for probe_start, probe_count in normalized_probe_tiles
        for genotype_index, genotype_start, genotype_stop in genotype_blocks
    )
    threshold = 0.75 * threads
    by_phase: dict[str, list[dict[str, Any]]] = {}
    contexts: dict[str, list[tuple[int, ...]]] = {}
    for phase in ("source_gemm", "target_gemm"):
        vendor = [
            record for record in records
            if record.get("phase") == phase
            and record.get("telemetry_scope") == "vendor_call"
        ]
        if not vendor:
            raise RuntimeError(f"completed batch has no vendor telemetry for {phase}")
        summaries = []
        phase_contexts = []
        for record in vendor:
            if record.get("completed") is not True:
                raise RuntimeError(f"{phase} vendor telemetry is not completed")
            wall = _positive_finite_metric(record, "wall_seconds", phase)
            process_cpu = _positive_finite_metric(
                record, "process_cpu_seconds", phase
            )
            achieved_gflops = _positive_finite_metric(
                record, "achieved_gflops", phase
            )
            reported_gflops = _positive_finite_metric(
                record, "gflops_per_second", phase
            )
            active_cores = _positive_finite_metric(
                record, "active_core_equivalents", phase
            )
            if active_cores < threshold:
                raise RuntimeError(
                    f"{phase} active_core_equivalents {active_cores:.6g} is below "
                    f"the required 75% threshold {threshold:.6g}"
                )
            if require_private_blis:
                exact_backend = {
                    "backend": "gxeldcore_direct",
                    "backend_version": EXPECTED_NATIVE_BACKEND_VERSION,
                    "backend_build_sha256": native_sha256,
                    "native_source_commit": source_commit,
                    "native_source_tree_sha256": source_tree_sha256,
                    "blas_backend": "BLIS",
                    "blas_backend_config": EXPECTED_BLIS_RUNTIME_CONFIG,
                    "blas_backend_corename": EXPECTED_BLIS_CONFIG_FAMILY,
                }
                mismatches = {
                    name: {"expected": value, "observed": record.get(name)}
                    for name, value in exact_backend.items()
                    if record.get(name) != value
                }
                if mismatches:
                    raise RuntimeError(
                        f"{phase} vendor private-BLIS identity mismatch: {mismatches}"
                    )
            for name in (
                "configured_threads", "requested_threads",
                "requested_blas_threads", "backend_threads",
            ):
                try:
                    observed_threads = int(record[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{phase} vendor telemetry lacks integer {name}"
                    ) from exc
                if observed_threads != threads:
                    raise RuntimeError(
                        f"{phase} vendor telemetry {name}={observed_threads}, "
                        f"expected {threads}"
                    )

            environment_start, environment_stop = _integer_pair(
                record, "environment_tile", phase
            )
            if environment_start < 0 or environment_stop <= environment_start:
                raise RuntimeError(f"{phase} environment_tile is not positive")
            environment_names = record.get("environment_names")
            environment_tile_size = environment_stop - environment_start
            if (
                not isinstance(environment_names, Sequence)
                or isinstance(environment_names, (str, bytes, bytearray))
                or len(environment_names) != environment_tile_size
                or any(not isinstance(name, str) or not name for name in environment_names)
            ):
                raise RuntimeError(
                    f"{phase} environment names do not match environment_tile"
                )
            probe_start, probe_count = _integer_pair(record, "probe_tile", phase)
            if (
                probe_start < 0 or probe_count <= 0
                or probe_start + probe_count > probes
            ):
                raise RuntimeError(f"{phase} probe_tile is outside measured probes")
            genotype_start, genotype_stop = _integer_pair(
                record, "genotype_block", phase
            )
            try:
                genotype_index = int(record["genotype_block_index"])
                reported_block_width = int(record["genotype_block_width"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{phase} vendor telemetry lacks genotype block context"
                ) from exc
            expected_start = genotype_index * block_width
            expected_stop = min(expected_start + block_width, variants)
            genotype_width = genotype_stop - genotype_start
            if (
                genotype_index < 0
                or genotype_start != expected_start
                or genotype_stop != expected_stop
                or genotype_width <= 0
                or reported_block_width != genotype_width
            ):
                raise RuntimeError(
                    f"{phase} genotype block context is not an exact full or "
                    "terminal controller block"
                )
            semantic_p = (
                2 * environment_tile_size * annotation_bins * probe_count
            )
            expected_dimensions = (
                (samples, semantic_p, genotype_width)
                if phase == "source_gemm"
                else (genotype_width, 2 * semantic_p, samples)
            )
            try:
                observed_dimensions = tuple(
                    int(record[name]) for name in ("m", "n", "k")
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{phase} vendor telemetry lacks semantic dimensions"
                ) from exc
            if observed_dimensions != expected_dimensions:
                raise RuntimeError(
                    f"{phase} semantic dimensions {observed_dimensions!r} do not "
                    f"match {expected_dimensions!r} from N, tile context, "
                    "annotation bins, and genotype block"
                )
            phase_contexts.append(
                (
                    environment_start, environment_stop,
                    probe_start, probe_count,
                    genotype_index, genotype_start, genotype_stop,
                )
            )
            summaries.append(
                {
                    "wall_seconds": wall,
                    "process_cpu_seconds": process_cpu,
                    "achieved_gflops": achieved_gflops,
                    "gflops_per_second": reported_gflops,
                    "active_core_equivalents": active_cores,
                    "semantic_p": semantic_p,
                    "genotype_block_width": genotype_width,
                    "terminal_block": genotype_width < block_width,
                    "dimensions": observed_dimensions,
                }
            )
        by_phase[phase] = summaries
        contexts[phase] = sorted(phase_contexts)
    for phase in ("source_gemm", "target_gemm"):
        if contexts[phase] != expected_contexts:
            raise RuntimeError(
                f"{phase} vendor contexts do not exactly match the complete "
                "environment-tile x probe-tile x genotype-block schedule once each"
            )
    return {
        "schema_version": 1,
        "required_for_acceptance": True,
        "gate_passed": True,
        "full_precision_layout": "current",
        "private_blis_identity_required": bool(require_private_blis),
        "analysis_complete_case_samples": samples,
        "requested_threads": threads,
        "minimum_active_core_equivalents_required": threshold,
        "annotation_bin_count": annotation_bins,
        "environment_tile_count": len(normalized_environment_tiles),
        "probe_tile_count": len(normalized_probe_tiles),
        "genotype_block_count": len(genotype_blocks),
        "expected_cartesian_context_count": len(expected_contexts),
        "complete_cartesian_context_gate_passed": True,
        "source_record_count": len(by_phase["source_gemm"]),
        "target_record_count": len(by_phase["target_gemm"]),
        "minimum_active_core_equivalents": min(
            item["active_core_equivalents"]
            for phase in by_phase.values() for item in phase
        ),
        "terminal_block_context_count": sum(
            item["terminal_block"] for item in by_phase["source_gemm"]
        ),
        "semantic_p_values": sorted(
            {item["semantic_p"] for item in by_phase["source_gemm"]}
        ),
        "genotype_block_widths": sorted(
            {item["genotype_block_width"] for item in by_phase["source_gemm"]}
        ),
        "source_semantic_dimensions": [
            list(shape) for shape in sorted(
                {item["dimensions"] for item in by_phase["source_gemm"]}
            )
        ],
        "target_semantic_dimensions": [
            list(shape) for shape in sorted(
                {item["dimensions"] for item in by_phase["target_gemm"]}
            )
        ],
    }


def _validate_layout_records(
    records: Sequence[dict[str, Any]], expected_layout: str
) -> None:
    specifications = {
        "current": {
            "source_gemm": {
                "literal": ("dgemm_nn", "column_major", "N", "N"),
                "leading_dimensions": ("m", "k", "m"),
            },
            "target_gemm": {
                "literal": ("dgemm_tn", "column_major", "T", "N"),
                "leading_dimensions": ("k", "k", "m"),
            },
        },
        "source_tt_target_current": {
            "source_gemm": {
                "literal": ("dgemm_tt", "column_major", "T", "T"),
                "leading_dimensions": ("k", "n", "m"),
            },
            "target_gemm": {
                "literal": ("dgemm_tn", "column_major", "T", "N"),
                "leading_dimensions": ("k", "k", "m"),
            },
        },
    }[_canonical_layout(expected_layout)]
    for phase, specification in specifications.items():
        vendor = [
            record for record in records
            if record.get("phase") == phase
            and record.get("telemetry_scope") == "vendor_call"
        ]
        if not vendor:
            raise RuntimeError(f"completed batch has no vendor telemetry for {phase}")
        for record in vendor:
            observed_literal = (
                record.get("operation"), record.get("layout"),
                record.get("transpose_a"), record.get("transpose_b"),
            )
            if observed_literal != specification["literal"]:
                raise RuntimeError(
                    f"{phase} vendor telemetry {observed_literal!r} does not match "
                    f"the requested {_canonical_layout(expected_layout)!r} layout"
                )
            try:
                dimensions = {
                    name: int(record[name]) for name in ("m", "n", "k")
                }
                observed_ld = tuple(
                    int(record[name]) for name in ("lda", "ldb", "ldc")
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{phase} vendor telemetry lacks integer dimensions/LDs"
                ) from exc
            if any(value <= 0 for value in dimensions.values()):
                raise RuntimeError(f"{phase} vendor dimensions must be positive")
            expected_ld = tuple(
                dimensions[name] for name in specification["leading_dimensions"]
            )
            if observed_ld != expected_ld:
                raise RuntimeError(
                    f"{phase} vendor leading dimensions {observed_ld!r} do not "
                    f"match literal {expected_ld!r}"
                )


def _numa_record_context(
    record: Mapping[str, Any], selected_nodes: set[int], *,
    operand: str | None = None,
    operand_sample: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "sequence": record.get("sequence"),
        "phase": record.get("phase"),
        "operation": record.get("operation"),
        **{name: record.get(name) for name in ("m", "n", "k", "lda", "ldb", "ldc")},
        "selected_numa_nodes": sorted(selected_nodes),
    }
    if operand is not None:
        context["operand"] = operand
    if isinstance(operand_sample, Mapping):
        context["query_status"] = operand_sample.get("query_status")
        context["histogram"] = operand_sample.get("node_histogram")
        if "ordered_samples" in operand_sample:
            context["ordered_samples"] = operand_sample["ordered_samples"]
    return context


def _raise_numa_rejection(reason: str, context: Mapping[str, Any]) -> None:
    raise RuntimeError(
        f"{reason}; numa_context="
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )


def _require_json_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an exact JSON integer")
    return value


def _early_numa_attestation_status(
    payload: Mapping[str, Any], performance: Mapping[str, Any],
    selected_nodes: set[int], *, expected_child_pid: int,
) -> dict[str, Any]:
    expected_child_pid = _require_json_int(
        expected_child_pid, field="controller expected_child_pid"
    )
    if expected_child_pid <= 0:
        raise RuntimeError("controller expected_child_pid must be positive")
    expected_policy = "libnuma:membind:" + ",".join(
        str(node) for node in sorted(selected_nodes)
    )
    expected_request = _format_integer_ranges(sorted(selected_nodes))
    observed = []
    for label, container in (
        ("batch_manifest", payload), ("performance_telemetry", performance)
    ):
        attestation = container.get("early_numa_attestation")
        if not isinstance(attestation, Mapping) or not attestation:
            raise RuntimeError(
                f"{label} lacks the required structured early NUMA attestation"
            )
        effective_nodes = attestation.get("effective_nodes")
        if not isinstance(effective_nodes, list):
            raise RuntimeError(
                f"{label} early NUMA attestation is malformed"
            )
        try:
            effective = [
                _require_json_int(
                    node,
                    field=f"{label}.early_numa_attestation.effective_nodes",
                )
                for node in effective_nodes
            ]
            pid = _require_json_int(
                attestation.get("pid"),
                field=f"{label}.early_numa_attestation.pid",
            )
            task_count = _require_json_int(
                attestation.get("task_count_at_application"),
                field=(
                    f"{label}.early_numa_attestation."
                    "task_count_at_application"
                ),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{label} early NUMA attestation is malformed"
            ) from exc
        required = {
            "schema": "summit.numa_policy_attestation.v1",
            "mode": "membind",
            "source": "libnuma",
            "applied_policy": expected_policy,
            "requested_nodes": expected_request,
        }
        for key, value in required.items():
            if attestation.get(key) != value:
                raise RuntimeError(
                    f"{label} early NUMA attestation has unexpected {key}: "
                    f"{attestation.get(key)!r}"
                )
        if (
            attestation.get("applied_before_numeric_import") is not True
            or attestation.get("verified") is not True
            or attestation.get("static_nodes") is not True
        ):
            raise RuntimeError(
                f"{label} early NUMA attestation is not verified before "
                "numeric import with an exact static binding"
            )
        if (
            effective != sorted(selected_nodes)
            or len(effective) != len(set(effective))
        ):
            raise RuntimeError(
                f"{label} early NUMA effective nodes do not match the selected nodes"
            )
        if task_count != 1 or pid != expected_child_pid:
            raise RuntimeError(
                f"{label} early NUMA attestation was not created by the "
                "expected single-task child"
            )
        observed.append(dict(attestation))
    if observed[0] != observed[1]:
        raise RuntimeError(
            "batch and performance early NUMA attestations do not match exactly"
        )
    return {
        "schema_version": 1,
        "required_for_acceptance": True,
        "complete": True,
        "effective_nodes": sorted(selected_nodes),
        "applied_policy": expected_policy,
        "task_count_at_application": 1,
        "child_pid": expected_child_pid,
        "attestation": observed[0],
    }


def _compact_numa_bound_decode_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    compact = dict(report)
    compact.pop("records", None)
    compact["records_included"] = False
    return compact


def _validate_numa_bound_decode_report(
    report: Mapping[str, Any],
    *,
    selected_nodes: Sequence[int],
    samples: int,
    variants: int,
    block_width: int,
    passes: int,
) -> dict[str, Any]:
    """Independently validate the complete prebound BED-decode evidence."""
    raw_nodes = list(selected_nodes)
    expected_nodes = sorted(raw_nodes) if all(
        type(node) is int for node in raw_nodes
    ) else []
    if (
        not isinstance(report, Mapping)
        or not expected_nodes
        or raw_nodes != expected_nodes
        or len(expected_nodes) != len(set(expected_nodes))
        or any(node < 0 for node in expected_nodes)
        or type(samples) is not int
        or samples <= 0
        or type(variants) is not int
        or variants <= 0
        or type(block_width) is not int
        or block_width <= 0
        or type(passes) is not int
        or passes <= 0
    ):
        raise RuntimeError("NUMA-bound BED decode controller inputs are malformed")
    blocks = [
        [start, min(start + block_width, variants)]
        for start in range(0, variants, block_width)
    ]
    expected_records = blocks * passes
    records = report.get("records")
    fixed = {
        "schema": NUMA_BOUND_DECODE_SCHEMA,
        "schema_version": 1,
        "required": True,
        "bounded": True,
        "decoder": "bed_reader.read_f64_into_bound_mapping",
        "memory_order": "F",
        "verification_stage": "post_standardization_pre_return",
        "selected_nodes": expected_nodes,
        "sample_count": samples,
        "num_variants": variants,
        "float64_itemsize": 8,
        "genotype_blocks": blocks,
        "shared_genotype_passes": passes,
        "expected_block_read_count": len(expected_records),
        "observed_block_read_count": len(expected_records),
        "complete_page_query_records": len(expected_records),
        "records_included": True,
        "complete": True,
    }
    report_nodes = report.get("selected_nodes")
    report_blocks = report.get("genotype_blocks")
    if (
        not isinstance(report_nodes, list)
        or any(type(node) is not int for node in report_nodes)
        or not isinstance(report_blocks, list)
        or any(
            not isinstance(block, list)
            or len(block) != 2
            or any(type(endpoint) is not int for endpoint in block)
            for block in report_blocks
        )
    ):
        raise RuntimeError("NUMA-bound BED decode summary uses invalid JSON types")
    if any(report.get(name) != value for name, value in fixed.items()):
        raise RuntimeError(
            "NUMA-bound BED decode summary disagrees with the controller plan"
        )
    if any(
        type(report.get(name)) is not int
        for name in (
            "schema_version",
            "sample_count",
            "num_variants",
            "float64_itemsize",
            "shared_genotype_passes",
            "expected_block_read_count",
            "observed_block_read_count",
            "complete_page_query_records",
        )
    ) or any(
        type(report.get(name)) is not bool
        for name in ("required", "bounded", "records_included", "complete")
    ):
        raise RuntimeError("NUMA-bound BED decode summary uses invalid JSON types")
    if not isinstance(records, list) or len(records) != len(expected_records):
        raise RuntimeError("NUMA-bound BED decode records are absent or incomplete")

    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise RuntimeError("the operating system returned an invalid page size")
    total_payload_bytes = 0
    total_mapping_bytes = 0
    maximum_payload_bytes = 0
    maximum_mapping_bytes = 0
    normalized_histogram: dict[int, int] = {}
    for index, (record, block) in enumerate(
        zip(records, expected_records, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} is malformed"
            )
        start, stop = block
        payload_bytes = samples * (stop - start) * 8
        mapping_bytes = ((payload_bytes + page_size - 1) // page_size) * page_size
        page_count = mapping_bytes // page_size
        record_block = record.get("genotype_block")
        if (
            not isinstance(record_block, list)
            or len(record_block) != 2
            or any(type(endpoint) is not int for endpoint in record_block)
            or record_block != block
            or record.get("memory_order") != "F"
            or record.get("decoder")
            != "bed_reader.read_f64_into_bound_mapping"
            or record.get("verification_stage")
            != "post_standardization_pre_return"
            or record.get("bound_mapping_preserved_after_standardization")
            is not True
        ):
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} changed its block contract"
            )
        allocation = record.get("allocation")
        verification = record.get("verification")
        if not isinstance(allocation, Mapping) or not isinstance(
            verification, Mapping
        ):
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} lacks range evidence"
            )
        common = {
            "schema": NUMA_BOUND_BUFFER_SCHEMA,
            "schema_version": 1,
            "byte_count": payload_bytes,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "page_count": page_count,
            "selected_nodes": expected_nodes,
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
        allocation_nodes = allocation.get("selected_nodes")
        verification_nodes = verification.get("selected_nodes")
        if (
            any(
                type(allocation.get(name)) is not int
                for name in (
                    "schema_version",
                    "byte_count",
                    "mapping_bytes",
                    "page_size",
                    "page_count",
                )
            )
            or any(
                type(allocation.get(name)) is not bool
                for name in (
                    "page_aligned_mapping",
                    "bound_before_first_touch",
                    "live_owner_policy_verified",
                    "range_policy_verified",
                    "post_decode_complete_page_query",
                    "page_migration_requested",
                    "placement_repair_performed",
                )
            )
            or any(
                type(verification.get(name)) is not int
                for name in (
                    "schema_version",
                    "byte_count",
                    "mapping_bytes",
                    "page_size",
                    "page_count",
                    "queried_pages",
                    "resolved_pages",
                    "query_chunk_page_limit",
                    "query_chunks",
                )
            )
            or any(
                type(verification.get(name)) is not bool
                for name in (
                    "page_aligned_mapping",
                    "bound_before_first_touch",
                    "live_owner_policy_verified",
                    "range_policy_verified",
                    "post_decode_complete_page_query",
                    "post_decode_strict_policy_verified",
                    "page_migration_requested",
                    "placement_repair_performed",
                    "complete",
                )
            )
            or not isinstance(allocation_nodes, list)
            or not isinstance(verification_nodes, list)
            or any(type(node) is not int for node in allocation_nodes)
            or any(type(node) is not int for node in verification_nodes)
            or any(
                allocation.get(name) != value
                for name, value in common.items()
            )
            or allocation.get("post_decode_complete_page_query") is not False
            or any(
                verification.get(name) != value
                for name, value in common.items()
            )
            or verification.get("post_decode_complete_page_query") is not True
            or verification.get("post_decode_strict_policy_verified") is not True
            or verification.get("queried_pages") != page_count
            or verification.get("resolved_pages") != page_count
            or verification.get("query_chunk_page_limit")
            != NUMA_PAGE_QUERY_CHUNK_LIMIT
            or verification.get("query_chunks")
            != (page_count + NUMA_PAGE_QUERY_CHUNK_LIMIT - 1)
            // NUMA_PAGE_QUERY_CHUNK_LIMIT
            or verification.get("ordered_status_encoding")
            != f"native_32bit_signed_{sys.byteorder}"
            or not _canonical_sha256(
                verification.get("ordered_status_sha256")
            )
            or verification.get("complete") is not True
        ):
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} failed its exact page contract"
            )
        histogram = verification.get("node_histogram")
        if not isinstance(histogram, Mapping) or not histogram:
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} lacks a page histogram"
            )
        record_total = 0
        for node_text, count in histogram.items():
            try:
                node = int(node_text)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"NUMA-bound BED decode record {index} has a malformed node"
                ) from exc
            if (
                not isinstance(node_text, str)
                or str(node) != node_text
                or node not in expected_nodes
                or type(count) is not int
                or count <= 0
            ):
                raise RuntimeError(
                    f"NUMA-bound BED decode record {index} has a malformed histogram"
                )
            record_total += count
            normalized_histogram[node] = (
                normalized_histogram.get(node, 0) + count
            )
        if record_total != page_count:
            raise RuntimeError(
                f"NUMA-bound BED decode record {index} has incomplete page coverage"
            )
        total_payload_bytes += payload_bytes
        total_mapping_bytes += mapping_bytes
        maximum_payload_bytes = max(maximum_payload_bytes, payload_bytes)
        maximum_mapping_bytes = max(maximum_mapping_bytes, mapping_bytes)

    totals = {
        "max_payload_bytes_per_block": maximum_payload_bytes,
        "max_mapping_bytes_per_block": maximum_mapping_bytes,
        "total_payload_bytes_across_reads": total_payload_bytes,
        "total_mapping_bytes_across_reads": total_mapping_bytes,
    }
    if any(
        type(report.get(name)) is not int for name in totals
    ) or any(report.get(name) != value for name, value in totals.items()):
        raise RuntimeError("NUMA-bound BED decode byte totals are inconsistent")
    return {
        "schema": NUMA_BOUND_DECODE_SCHEMA,
        "required_for_acceptance": True,
        "complete": True,
        "selected_nodes": expected_nodes,
        "genotype_blocks": blocks,
        "shared_genotype_passes": passes,
        "expected_block_read_count": len(expected_records),
        "observed_block_read_count": len(records),
        "queried_pages": sum(normalized_histogram.values()),
        "observed_node_histogram": {
            str(node): count
            for node, count in sorted(normalized_histogram.items())
        },
        "page_migration_requested": False,
        "placement_repair_performed": False,
    }


def _gemm_operand_byte_count(
    record: Mapping[str, Any], operand_name: str
) -> int:
    dimensions = {
        name: _require_json_int(record.get(name), field=f"GEMM record {name}")
        for name in ("m", "n", "k", "lda", "ldb", "ldc")
    }
    if any(value <= 0 for value in dimensions.values()):
        raise RuntimeError("GEMM record dimensions and leading dimensions must be positive")
    m, n, k = (dimensions[name] for name in ("m", "n", "k"))
    transpose_a = record.get("transpose_a")
    transpose_b = record.get("transpose_b")
    if transpose_a not in {"N", "T"} or transpose_b not in {"N", "T"}:
        raise RuntimeError("GEMM record has an unsupported transpose flag")
    if operand_name == "a":
        rows, columns = (m, k) if transpose_a == "N" else (k, m)
        leading_dimension = dimensions["lda"]
    elif operand_name == "b":
        rows, columns = (k, n) if transpose_b == "N" else (n, k)
        leading_dimension = dimensions["ldb"]
    elif operand_name == "c":
        rows, columns = m, n
        leading_dimension = dimensions["ldc"]
    else:
        raise RuntimeError(f"unknown GEMM operand {operand_name!r}")
    layout = record.get("layout")
    if layout == "column_major":
        major_dimension, minor_dimension = columns, rows
    elif layout == "row_major":
        major_dimension, minor_dimension = rows, columns
    else:
        raise RuntimeError(f"GEMM record has unsupported layout {layout!r}")
    if leading_dimension < minor_dimension:
        raise RuntimeError(
            f"GEMM operand {operand_name} leading dimension is smaller than "
            "its stored minor dimension"
        )
    span_elements = (major_dimension - 1) * leading_dimension + minor_dimension
    return span_elements * 8


def _canonical_positive_histogram(
    value: object, *, field: str, allow_empty: bool
) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a JSON object")
    result: dict[int, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, str) or not raw_key.isdigit():
            raise RuntimeError(f"{field} has a non-canonical node/errno key")
        key = int(raw_key)
        if str(key) != raw_key or key in result:
            raise RuntimeError(f"{field} has a non-canonical or duplicate key")
        count = _require_json_int(raw_count, field=f"{field}[{raw_key!r}]")
        if count <= 0:
            raise RuntimeError(f"{field} counts must be positive")
        result[key] = count
    if not allow_empty and not result:
        raise RuntimeError(f"{field} must not be empty")
    return result


def _require_numa_json_int(
    value: object, *, field: str, context: Mapping[str, Any]
) -> int:
    try:
        return _require_json_int(value, field=field)
    except RuntimeError as exc:
        try:
            _raise_numa_rejection(f"{field} is malformed", context)
        except RuntimeError as rejection:
            raise rejection from exc


def _validate_operand_numa_evidence(
    record: Mapping[str, Any], operand_name: str,
    raw_operand: Mapping[str, Any], *, page_size: int, sample_limit: int,
    selected_nodes: set[int],
) -> tuple[str, dict[int, int]]:
    operand_context = _numa_record_context(
        record, selected_nodes, operand=operand_name,
        operand_sample=raw_operand,
    )
    try:
        byte_count = _gemm_operand_byte_count(record, operand_name)
    except RuntimeError as exc:
        try:
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} storage span is malformed",
                operand_context,
            )
        except RuntimeError as rejection:
            raise rejection from exc
    exact_fields = {
        field: _require_numa_json_int(
            raw_operand.get(field), field=f"operand {operand_name} {field}",
            context=operand_context,
        )
        for field in (
            "storage_span_pages", "fully_contained_pages",
            "operand_byte_count", "operand_start_address_page_offset",
            "operand_end_exclusive_address_page_offset",
            "selected_sample_pages", "resolved_sample_pages",
            "page_query_error_pages",
        )
    }
    start_offset = exact_fields["operand_start_address_page_offset"]
    if start_offset < 0 or start_offset >= page_size:
        _raise_numa_rejection(
            f"hot vendor operand {operand_name} has an invalid start-page offset",
            operand_context,
        )
    storage_pages = (start_offset + byte_count - 1) // page_size + 1
    end_offset = (start_offset + byte_count) % page_size
    first_full_offset = 0 if start_offset == 0 else page_size - start_offset
    full_pages = (
        (byte_count - first_full_offset) // page_size
        if first_full_offset <= byte_count else 0
    )
    selected = min(full_pages, sample_limit)
    expected_fields = {
        "operand_byte_count": byte_count,
        "storage_span_pages": storage_pages,
        "fully_contained_pages": full_pages,
        "operand_end_exclusive_address_page_offset": end_offset,
        "selected_sample_pages": selected,
    }
    for field, expected in expected_fields.items():
        if exact_fields[field] != expected:
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} {field} does not match "
                "the literal GEMM storage span",
                operand_context,
            )
    if selected <= 0:
        _raise_numa_rejection(
            f"hot vendor operand {operand_name} has no fully contained sample page",
            operand_context,
        )
    ordered_samples = raw_operand.get("ordered_samples")
    if not isinstance(ordered_samples, list) or len(ordered_samples) != selected:
        _raise_numa_rejection(
            f"hot vendor operand {operand_name} ordered sample count is malformed",
            operand_context,
        )
    status = raw_operand.get("query_status")
    if not isinstance(status, str):
        _raise_numa_rejection(
            f"hot vendor operand {operand_name} query status is malformed",
            operand_context,
        )
    ordered_nodes: list[int] = []
    for ordinal, item in enumerate(ordered_samples):
        if not isinstance(item, Mapping):
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} ordered sample is malformed",
                operand_context,
            )
        expected_page_index = (
            0 if selected == 1
            else ordinal * (full_pages - 1) // (selected - 1)
        )
        expected_byte_offset = first_full_offset + expected_page_index * page_size
        for field, expected in (
            ("sample_ordinal", ordinal),
            ("full_page_index", expected_page_index),
            ("byte_offset_from_operand_start", expected_byte_offset),
        ):
            observed = _require_numa_json_int(
                item.get(field),
                field=f"operand {operand_name} ordered_samples[{ordinal}].{field}",
                context=operand_context,
            )
            if observed != expected:
                _raise_numa_rejection(
                    f"hot vendor operand {operand_name} ordered sample {field} "
                    "does not match the exact page-selection schedule",
                    operand_context,
                )
        if (
            (start_offset + expected_byte_offset) % page_size != 0
            or expected_byte_offset < 0
            or expected_byte_offset + page_size > byte_count
        ):
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} selected a non-contained page",
                operand_context,
            )
        if status == "queried":
            node = _require_numa_json_int(
                item.get("numa_node"),
                field=f"operand {operand_name} ordered_samples[{ordinal}].numa_node",
                context=operand_context,
            )
            raw_status = _require_numa_json_int(
                item.get("raw_move_pages_status"),
                field=(
                    f"operand {operand_name} ordered_samples[{ordinal}]."
                    "raw_move_pages_status"
                ),
                context=operand_context,
            )
            if (
                item.get("status_kind") != "numa_node"
                or item.get("page_query_errno") is not None
                or raw_status != node
                or node < 0
            ):
                _raise_numa_rejection(
                    f"hot vendor operand {operand_name} ordered query result is "
                    "inconsistent",
                    operand_context,
                )
            ordered_nodes.append(node)
        elif status in {"permission_denied", "unsupported"}:
            if (
                item.get("status_kind") != "unavailable"
                or item.get("raw_move_pages_status") is not None
                or item.get("numa_node") is not None
                or item.get("page_query_errno") is not None
            ):
                _raise_numa_rejection(
                    f"hot vendor operand {operand_name} unavailable ordered "
                    "sample is inconsistent",
                    operand_context,
                )
        else:
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} NUMA query is {status!r}",
                operand_context,
            )
    try:
        histogram = _canonical_positive_histogram(
            raw_operand.get("node_histogram"),
            field=f"operand {operand_name} node_histogram",
            allow_empty=status != "queried",
        )
        error_histogram = _canonical_positive_histogram(
            raw_operand.get("page_error_errno_histogram"),
            field=f"operand {operand_name} page_error_errno_histogram",
            allow_empty=True,
        )
    except RuntimeError as exc:
        try:
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} histogram is malformed",
                operand_context,
            )
        except RuntimeError as rejection:
            raise rejection from exc
    resolved = exact_fields["resolved_sample_pages"]
    errors = exact_fields["page_query_error_pages"]
    if status == "queried":
        ordered_histogram: dict[int, int] = {}
        for node in ordered_nodes:
            ordered_histogram[node] = ordered_histogram.get(node, 0) + 1
        if (
            resolved != selected or errors != 0 or error_histogram
            or histogram != ordered_histogram
        ):
            _raise_numa_rejection(
                f"hot vendor operand {operand_name} query counts/histograms do "
                "not match ordered evidence",
                operand_context,
            )
        remote = sorted(set(histogram).difference(selected_nodes))
        if remote:
            _raise_numa_rejection(
                "hot vendor operand pages are outside selected NUMA nodes: "
                f"{remote}", operand_context,
            )
    elif resolved != 0 or errors != 0 or histogram or error_histogram:
        _raise_numa_rejection(
            f"hot vendor operand {operand_name} unavailable query reports results",
            operand_context,
        )
    return status, histogram


def _hot_vendor_numa_summary(
    records: Sequence[dict[str, Any]], selected_nodes: set[int]
) -> dict[str, Any]:
    if not selected_nodes:
        raise RuntimeError("selected CPUs have no resolvable NUMA nodes")
    hot = [
        record for record in records
        if record.get("telemetry_scope") == "vendor_call"
        and record.get("phase") in {"source_gemm", "target_gemm"}
    ]
    if not hot:
        raise RuntimeError("completed batch has no hot vendor NUMA telemetry")
    record_summaries = []
    rejection_contexts = []
    all_queried_local = True
    observed_nodes: set[int] = set()
    for index, record in enumerate(hot):
        numa = record.get("operand_numa_page_samples")
        if not isinstance(numa, Mapping):
            _raise_numa_rejection(
                "hot vendor record lacks the move_pages NUMA schema",
                _numa_record_context(record, selected_nodes),
            )
        try:
            numa_schema_version = _require_json_int(
                numa.get("schema_version"), field="NUMA schema_version"
            )
            address_schema_version = _require_json_int(
                numa.get("address_selection_schema_version"),
                field="NUMA address_selection_schema_version",
            )
            sample_limit = _require_json_int(
                numa.get("sample_limit_per_operand"),
                field="NUMA sample_limit_per_operand",
            )
            page_size = _require_json_int(
                numa.get("system_page_size"), field="NUMA system_page_size"
            )
        except RuntimeError as exc:
            try:
                _raise_numa_rejection(
                    "hot vendor record has malformed NUMA integer metadata",
                    _numa_record_context(record, selected_nodes),
                )
            except RuntimeError as rejection:
                raise rejection from exc
        if (
            numa_schema_version != 1
            or numa.get("sampling_method") != "move_pages_query_no_migration"
            or numa.get("sampling_timing")
            != "after_vendor_call_outside_timed_interval"
            or address_schema_version != 1
            or numa.get("address_selection_policy")
            != NUMA_ADDRESS_SELECTION_POLICY
            or sample_limit != 8
            or page_size != os.sysconf("SC_PAGE_SIZE")
            or numa.get("selected_addresses_are_page_bases") is not True
            or numa.get("partial_boundary_pages_included") is not False
            or numa.get("first_and_last_fully_contained_pages_selected") is not True
            or numa.get("virtual_addresses_exposed") is not False
            or numa.get("operand_byte_range_semantics")
            != "[start_address,end_exclusive_address)"
            or numa.get("address_evidence") != (
                "ordered_samples_with_operand_relative_byte_offsets_and_"
                "full_page_indices"
            )
        ):
            _raise_numa_rejection(
                "hot vendor record lacks the move_pages NUMA schema",
                _numa_record_context(record, selected_nodes),
            )
        operands = numa.get("operands")
        if not isinstance(operands, Mapping) or set(operands) != {"a", "b", "c"}:
            _raise_numa_rejection(
                "hot vendor record lacks A/B/C NUMA page samples",
                _numa_record_context(record, selected_nodes),
            )
        statuses = {}
        operand_summaries = {}
        record_nodes: set[int] = set()
        for operand_name in ("a", "b", "c"):
            raw_operand = operands[operand_name]
            if not isinstance(raw_operand, Mapping):
                _raise_numa_rejection(
                    "hot vendor NUMA operand sample is malformed",
                    _numa_record_context(
                        record, selected_nodes, operand=operand_name
                    ),
                )
            operand_context = _numa_record_context(
                record, selected_nodes, operand=operand_name,
                operand_sample=raw_operand,
            )
            status, histogram = _validate_operand_numa_evidence(
                record, operand_name, raw_operand,
                page_size=page_size, sample_limit=sample_limit,
                selected_nodes=selected_nodes,
            )
            statuses[operand_name] = status
            operand_summaries[operand_name] = operand_context
            if status in {"permission_denied", "unsupported"}:
                all_queried_local = False
                rejection_contexts.append(operand_context)
                continue
            record_nodes.update(histogram)
            observed_nodes.update(histogram)
        if all(status == "queried" for status in statuses.values()):
            try:
                syscall_result = _require_json_int(
                    numa.get("syscall_result"), field="NUMA syscall_result"
                )
                syscall_errno = _require_json_int(
                    numa.get("syscall_errno"), field="NUMA syscall_errno"
                )
            except RuntimeError as exc:
                try:
                    _raise_numa_rejection(
                        "successful move_pages telemetry has malformed syscall status",
                        _numa_record_context(record, selected_nodes),
                    )
                except RuntimeError as rejection:
                    raise rejection from exc
            if syscall_result != 0 or syscall_errno != 0:
                _raise_numa_rejection(
                    "successful move_pages telemetry reports syscall errors",
                    _numa_record_context(record, selected_nodes),
                )
        record_summaries.append(
            {
                "record_index": index, "sequence": record.get("sequence"),
                "phase": record.get("phase"),
                "operation": record.get("operation"),
                **{
                    name: record.get(name)
                    for name in ("m", "n", "k", "lda", "ldb", "ldc")
                },
                "selected_numa_nodes": sorted(selected_nodes),
                "operand_statuses": statuses,
                "operands": operand_summaries,
                "observed_nodes": sorted(record_nodes),
            }
        )
    return {
        "schema_version": 1,
        "required_for_acceptance": True,
        "sampling_method": "move_pages_query_no_migration",
        "selected_numa_nodes": sorted(selected_nodes),
        "observed_numa_nodes": sorted(observed_nodes),
        "hot_vendor_record_count": len(hot),
        "all_hot_operands_queried_resolved_local": all_queried_local,
        "rejection_contexts": rejection_contexts,
        "records": record_summaries,
    }


def _native_integrity_snapshot_numa_summary(
    records: Sequence[dict[str, Any]],
    selected_nodes: set[int],
    *,
    integrity_enabled: bool,
    integrity_minimum_vendor_flops: int,
    query_chunk_page_limit: int,
) -> dict[str, Any]:
    """Validate complete pre-vendor placement of each checked source-NN B copy."""
    if (
        type(integrity_enabled) is not bool
        or integrity_enabled is not True
        or type(integrity_minimum_vendor_flops) is not int
        or integrity_minimum_vendor_flops <= 0
        or type(query_chunk_page_limit) is not int
        or query_chunk_page_limit <= 0
        or query_chunk_page_limit
        != NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        or not selected_nodes
        or any(type(node) is not int or node < 0 for node in selected_nodes)
    ):
        raise RuntimeError(
            "native integrity snapshot NUMA controller inputs are malformed"
        )
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise RuntimeError("the operating system returned an invalid page size")
    base_keys = {
        "schema", "schema_version", "contract_required", "operand_role",
        "integrity_check_shape_eligible", "integrity_check_executed",
        "snapshot_available", "complete",
    }
    complete_keys = base_keys | {
        "logical_byte_count", "mapping_bytes", "page_size",
        "mapping_page_count", "selected_nodes", "policy_mode",
        "policy_mode_value", "anonymous_private_mapping",
        "page_aligned_mapping", "bound_before_first_touch",
        "live_owner_policy_verified", "pre_touch_range_policy_verified",
        "sealed_read_only_before_vendor", "pre_vendor_complete_page_query",
        "queried_pages", "resolved_pages", "query_chunks",
        "query_chunk_page_limit", "node_histogram",
        "ordered_status_sha256", "ordered_status_encoding",
        "pre_vendor_strict_policy_verified", "strict_policy_check",
        "page_query_method", "page_migration_requested",
        "placement_repair_performed", "complete",
    }
    hot_vendor = [
        (record_index, record) for record_index, record in enumerate(records)
        if record.get("telemetry_scope") == "vendor_call"
        and record.get("phase") in {"source_gemm", "target_gemm"}
    ]
    if not hot_vendor:
        raise RuntimeError(
            "completed batch has no hot vendor integrity snapshot telemetry"
        )
    checked_record_summaries = []
    non_applicable_records = 0
    logical_bytes_total = 0
    mapping_bytes_total = 0
    mapping_pages_total = 0
    query_chunks_total = 0
    aggregate_histogram: dict[int, int] = {}
    for hot_vendor_index, (record_index, record) in enumerate(hot_vendor):
        context = {
            "record_index": record_index,
            "hot_vendor_index": hot_vendor_index,
            "sequence": record.get("sequence"),
            "phase": record.get("phase"),
            "operation": record.get("operation"),
            **{name: record.get(name) for name in ("m", "n", "k")},
            "selected_numa_nodes": sorted(selected_nodes),
        }
        evidence = record.get("native_integrity_snapshot_numa")
        if not isinstance(evidence, Mapping):
            raise RuntimeError(
                "hot vendor record lacks native integrity snapshot NUMA evidence; "
                "snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        try:
            dimensions = {
                name: _require_json_int(
                    record.get(name), field=f"snapshot record {name}"
                )
                for name in ("m", "n", "k")
            }
            schema_version = _require_json_int(
                evidence.get("schema_version"),
                field="native integrity snapshot schema_version",
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "hot vendor native integrity snapshot dimensions/schema are malformed; "
                "snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            ) from exc
        if any(value <= 0 for value in dimensions.values()):
            raise RuntimeError(
                "hot vendor native integrity snapshot dimensions must be positive; "
                "snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        expected_eligible = bool(
            record.get("phase") == "source_gemm"
            and record.get("operation") == "dgemm_nn"
            and 2 * dimensions["m"] * dimensions["n"] * dimensions["k"]
            >= integrity_minimum_vendor_flops
        )
        fixed_base = {
            "schema": NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
            "schema_version": 1,
            "contract_required": True,
            "operand_role": NATIVE_INTEGRITY_SNAPSHOT_NUMA_ROLE,
            "integrity_check_shape_eligible": expected_eligible,
            "integrity_check_executed": expected_eligible,
            "snapshot_available": expected_eligible,
            "complete": expected_eligible,
        }
        if (
            schema_version != 1
            or any(evidence.get(name) != value for name, value in fixed_base.items())
            or any(
                type(evidence.get(name)) is not bool
                for name in (
                    "contract_required", "integrity_check_shape_eligible",
                    "integrity_check_executed", "snapshot_available", "complete",
                )
            )
        ):
            raise RuntimeError(
                "hot vendor native integrity snapshot discriminator is not exact; "
                "snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        if not expected_eligible:
            if set(evidence) != base_keys:
                raise RuntimeError(
                    "nonchecked hot vendor record has noncanonical native integrity "
                    "snapshot evidence; snapshot_context="
                    + json.dumps(context, sort_keys=True, separators=(",", ":"))
                )
            non_applicable_records += 1
            continue
        if set(evidence) != complete_keys:
            raise RuntimeError(
                "checked source NN vendor record has a noncanonical native integrity "
                "snapshot schema; snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        exact_integers = {
            name: _require_json_int(
                evidence.get(name), field=f"native integrity snapshot {name}"
            )
            for name in (
                "logical_byte_count", "mapping_bytes", "page_size",
                "mapping_page_count", "policy_mode_value", "queried_pages",
                "resolved_pages", "query_chunks", "query_chunk_page_limit",
            )
        }
        logical_bytes = dimensions["k"] * dimensions["n"] * 8
        mapping_bytes = (
            (logical_bytes + page_size - 1) // page_size
        ) * page_size
        mapping_pages = mapping_bytes // page_size
        query_chunks = (
            mapping_pages + query_chunk_page_limit - 1
        ) // query_chunk_page_limit
        expected_integers = {
            "logical_byte_count": logical_bytes,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "mapping_page_count": mapping_pages,
            "policy_mode_value": NATIVE_INTEGRITY_SNAPSHOT_NUMA_POLICY_VALUE,
            "queried_pages": mapping_pages,
            "resolved_pages": mapping_pages,
            "query_chunks": query_chunks,
            "query_chunk_page_limit": query_chunk_page_limit,
        }
        raw_nodes = evidence.get("selected_nodes")
        if (
            exact_integers != expected_integers
            or not isinstance(raw_nodes, list)
            or any(type(node) is not int for node in raw_nodes)
            or raw_nodes != sorted(selected_nodes)
        ):
            raise RuntimeError(
                "checked source NN native integrity snapshot range/NUMA contract "
                "does not match its GEMM shape; snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        fixed_complete = {
            "policy_mode": "bind_static_nodes",
            "anonymous_private_mapping": True,
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "pre_touch_range_policy_verified": True,
            "sealed_read_only_before_vendor": True,
            "pre_vendor_complete_page_query": True,
            "ordered_status_encoding": "signed_int32_little_endian",
            "pre_vendor_strict_policy_verified": True,
            "strict_policy_check": "MPOL_MF_STRICT_without_MPOL_MF_MOVE",
            "page_query_method": "move_pages_query_no_migration",
            "page_migration_requested": False,
            "placement_repair_performed": False,
            "complete": True,
        }
        if (
            any(evidence.get(name) != value for name, value in fixed_complete.items())
            or any(
                type(evidence.get(name)) is not bool
                for name, value in fixed_complete.items()
                if type(value) is bool
            )
            or not _canonical_sha256(evidence.get("ordered_status_sha256"))
        ):
            raise RuntimeError(
                "checked source NN native integrity snapshot is not dedicated, "
                "prebound, sealed, strict-no-move, and complete; snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        histogram = _canonical_positive_histogram(
            evidence.get("node_histogram"),
            field="native integrity snapshot node_histogram",
            allow_empty=False,
        )
        if (
            sum(histogram.values()) != mapping_pages
            or not set(histogram).issubset(selected_nodes)
        ):
            raise RuntimeError(
                "checked source NN native integrity snapshot pages are incomplete "
                "or outside selected NUMA nodes; snapshot_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        sequence = _require_json_int(
            record.get("sequence"), field="integrity snapshot record sequence"
        )
        native_sequence = _require_json_int(
            record.get("native_sequence"),
            field="native integrity snapshot native_sequence",
        )
        if sequence <= 0 or native_sequence <= 0:
            raise RuntimeError(
                "checked source NN integrity snapshot sequences must be positive"
            )
        retained = {
            "record_index": record_index,
            "hot_vendor_index": hot_vendor_index,
            "sequence": sequence,
            "native_sequence": native_sequence,
            "phase": record.get("phase"),
            "operation": record.get("operation"),
            **{name: dimensions[name] for name in ("m", "n", "k")},
            **{
                name: record.get(name)
                for name in (
                    "environment_tile", "environment_names", "probe_tile",
                    "genotype_block", "genotype_block_index",
                    "genotype_block_width",
                )
            },
            "native_integrity_snapshot_numa": dict(evidence),
        }
        try:
            retained = json.loads(json.dumps(retained, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "checked source NN native integrity snapshot evidence is not JSON-safe"
            ) from exc
        checked_record_summaries.append(retained)
        logical_bytes_total += logical_bytes
        mapping_bytes_total += mapping_bytes
        mapping_pages_total += mapping_pages
        query_chunks_total += query_chunks
        for node, count in histogram.items():
            aggregate_histogram[node] = aggregate_histogram.get(node, 0) + count
    return {
        "schema": NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
        "schema_version": 1,
        "required_for_acceptance": True,
        "complete": True,
        "operand_role": NATIVE_INTEGRITY_SNAPSHOT_NUMA_ROLE,
        "integrity_minimum_vendor_flops": integrity_minimum_vendor_flops,
        "query_chunk_page_limit": query_chunk_page_limit,
        "selected_numa_nodes": sorted(selected_nodes),
        "hot_vendor_record_count": len(hot_vendor),
        "checked_source_nn_vendor_record_count": len(checked_record_summaries),
        "nonchecked_hot_vendor_record_count": non_applicable_records,
        "non_vendor_record_count_ignored": sum(
            record.get("telemetry_scope") != "vendor_call" for record in records
        ),
        "logical_byte_count_total": logical_bytes_total,
        "mapping_bytes_total": mapping_bytes_total,
        "mapping_page_count_total": mapping_pages_total,
        "query_chunk_count_total": query_chunks_total,
        "node_histogram": {
            str(node): aggregate_histogram[node]
            for node in sorted(aggregate_histogram)
        },
        "every_checked_source_nn_snapshot_fully_resolved_local": True,
        "every_checked_source_nn_snapshot_strict_no_move": True,
        "page_migration_requested": False,
        "placement_repair_performed": False,
        "record_evidence_included": True,
        "records": checked_record_summaries,
    }


def _native_gemm_output_numa_summary(
    records: Sequence[dict[str, Any]],
    output_records: Any,
    status: Any,
    selected_nodes: set[int],
    *,
    query_chunk_page_limit: int,
    evidence_capacity: int,
) -> dict[str, Any]:
    """Validate every contracted protected-output mapping and its call join."""
    if (
        not selected_nodes
        or any(type(node) is not int or node < 0 for node in selected_nodes)
        or type(query_chunk_page_limit) is not int
        or query_chunk_page_limit != NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        or type(evidence_capacity) is not int
        or evidence_capacity != NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        or not isinstance(output_records, list)
        or not output_records
        or len(output_records) > evidence_capacity
        or not isinstance(status, Mapping)
    ):
        raise RuntimeError("native GEMM output NUMA controller inputs are malformed")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise RuntimeError("the operating system returned an invalid page size")
    status_keys = {
        "schema_version", "capacity", "buffered_records", "captured_records",
        "dropped_records", "next_call_id", "query_chunk_page_limit",
        "attempted_calls", "verified_calls", "legacy_calls", "failed_calls",
    }
    if set(status) != status_keys:
        raise RuntimeError("native GEMM output NUMA evidence status is noncanonical")
    exact_status = {
        name: _require_json_int(
            status.get(name), field=f"native GEMM output status {name}"
        )
        for name in status_keys
    }
    record_count = len(output_records)
    if (
        exact_status["schema_version"] != 1
        or exact_status["capacity"] != evidence_capacity
        or exact_status["buffered_records"] != 0
        or exact_status["captured_records"] != record_count
        or exact_status["dropped_records"] != 0
        or exact_status["query_chunk_page_limit"] != query_chunk_page_limit
        or exact_status["attempted_calls"] != record_count
        or exact_status["verified_calls"] != record_count
        or exact_status["legacy_calls"] != 0
        or exact_status["failed_calls"] != 0
    ):
        raise RuntimeError(
            "native GEMM output NUMA evidence status is incomplete or lossy"
        )
    complete_keys = {
        "schema", "schema_version", "applicable", "call_id",
        "contract_required", "complete", "operand_role", "logical_rows",
        "logical_columns", "logical_byte_count", "storage_layout",
        "mapping_bytes", "page_size", "mapping_page_count", "selected_nodes",
        "policy_mode", "policy_mode_value", "allocation_mode",
        "anonymous_private_mapping", "page_aligned_mapping", "writable_output",
        "bound_before_first_touch", "pre_touch_live_owner_policy_verified",
        "pre_touch_range_policy_verified",
        "post_repair_live_owner_policy_verified",
        "post_repair_range_policy_verified",
        "post_repair_complete_page_query", "queried_pages", "resolved_pages",
        "query_chunks", "query_chunk_page_limit", "node_histogram",
        "ordered_status_sha256", "ordered_status_encoding",
        "post_repair_strict_policy_verified", "verification_boundary",
        "strict_policy_check", "page_query_method", "page_migration_requested",
        "placement_repair_performed", "sealed_read_only",
    }
    bool_fields = {
        "applicable": True,
        "contract_required": True,
        "complete": True,
        "anonymous_private_mapping": True,
        "page_aligned_mapping": True,
        "writable_output": True,
        "bound_before_first_touch": True,
        "pre_touch_live_owner_policy_verified": True,
        "pre_touch_range_policy_verified": True,
        "post_repair_live_owner_policy_verified": True,
        "post_repair_range_policy_verified": True,
        "post_repair_complete_page_query": True,
        "post_repair_strict_policy_verified": True,
        "page_migration_requested": False,
        "placement_repair_performed": False,
        "sealed_read_only": False,
    }
    fixed_fields = {
        "schema": NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "schema_version": 1,
        "operand_role": NATIVE_GEMM_OUTPUT_NUMA_ROLE,
        "policy_mode": "bind_static_nodes",
        "policy_mode_value": NATIVE_GEMM_OUTPUT_NUMA_POLICY_VALUE,
        "allocation_mode": "mmap_private_anonymous",
        "ordered_status_encoding": "signed_int32_little_endian",
        "verification_boundary": (
            "after_partitioned_or_integrity_repair_before_python_return"
        ),
        "strict_policy_check": "MPOL_MF_STRICT_without_MPOL_MF_MOVE",
        "page_query_method": "move_pages_query_no_migration",
    }

    def validate_evidence(raw: Any, *, context: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) != complete_keys:
            raise RuntimeError(
                "native GEMM output NUMA evidence has a noncanonical schema; "
                "output_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        evidence = dict(raw)
        integers = {
            name: _require_json_int(
                evidence.get(name), field=f"native GEMM output {name}"
            )
            for name in (
                "schema_version", "call_id", "logical_rows", "logical_columns",
                "logical_byte_count", "mapping_bytes", "page_size",
                "mapping_page_count", "policy_mode_value", "queried_pages",
                "resolved_pages", "query_chunks", "query_chunk_page_limit",
            )
        }
        rows = integers["logical_rows"]
        columns = integers["logical_columns"]
        logical_bytes = rows * columns * 8
        mapping_bytes = (
            (logical_bytes + page_size - 1) // page_size
        ) * page_size
        mapping_pages = mapping_bytes // page_size
        query_chunks = (
            mapping_pages + query_chunk_page_limit - 1
        ) // query_chunk_page_limit
        expected_integers = {
            "schema_version": 1,
            "logical_byte_count": logical_bytes,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "mapping_page_count": mapping_pages,
            "policy_mode_value": NATIVE_GEMM_OUTPUT_NUMA_POLICY_VALUE,
            "queried_pages": mapping_pages,
            "resolved_pages": mapping_pages,
            "query_chunks": query_chunks,
            "query_chunk_page_limit": query_chunk_page_limit,
        }
        raw_nodes = evidence.get("selected_nodes")
        if (
            integers["call_id"] <= 0
            or rows <= 0
            or columns <= 0
            or any(integers[name] != value for name, value in expected_integers.items())
            or not isinstance(raw_nodes, list)
            or any(type(node) is not int for node in raw_nodes)
            or raw_nodes != sorted(selected_nodes)
            or evidence.get("storage_layout") not in {"column_major", "row_major"}
            or any(evidence.get(name) != value for name, value in fixed_fields.items())
            or any(
                type(evidence.get(name)) is not bool
                or evidence.get(name) is not value
                for name, value in bool_fields.items()
            )
            or not _canonical_sha256(evidence.get("ordered_status_sha256"))
        ):
            raise RuntimeError(
                "native GEMM output NUMA evidence is not exact, prebound, "
                "exhaustively verified, and strict-no-move; output_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        histogram = _canonical_positive_histogram(
            evidence.get("node_histogram"),
            field="native GEMM output node_histogram",
            allow_empty=False,
        )
        if (
            sum(histogram.values()) != mapping_pages
            or not set(histogram).issubset(selected_nodes)
        ):
            raise RuntimeError(
                "native GEMM output pages are incomplete or outside selected NUMA "
                "nodes; output_context="
                + json.dumps(context, sort_keys=True, separators=(",", ":"))
            )
        return evidence

    by_call_id: dict[int, dict[str, Any]] = {}
    logical_bytes_total = 0
    mapping_bytes_total = 0
    mapping_pages_total = 0
    query_chunks_total = 0
    single_node_digest_recomputed_count = 0
    aggregate_histogram: dict[int, int] = {}
    for index, raw in enumerate(output_records):
        evidence = validate_evidence(raw, context={"output_record_index": index})
        call_id = evidence["call_id"]
        if call_id in by_call_id:
            raise RuntimeError("native GEMM output call IDs are not one-to-one")
        by_call_id[call_id] = evidence
        logical_bytes_total += evidence["logical_byte_count"]
        mapping_bytes_total += evidence["mapping_bytes"]
        mapping_pages_total += evidence["mapping_page_count"]
        query_chunks_total += evidence["query_chunks"]
        histogram = _canonical_positive_histogram(
            evidence["node_histogram"],
            field="native GEMM output node_histogram",
            allow_empty=False,
        )
        if len(histogram) == 1:
            node = next(iter(histogram))
            encoded = int(node).to_bytes(4, "little", signed=True)
            expected_digest = hashlib.sha256(
                encoded * evidence["mapping_page_count"]
            ).hexdigest()
            if evidence["ordered_status_sha256"] != expected_digest:
                raise RuntimeError(
                    "single-node native GEMM output ordered-status digest does "
                    "not match its exhaustive page histogram"
                )
            single_node_digest_recomputed_count += 1
        for node, count in histogram.items():
            aggregate_histogram[node] = aggregate_histogram.get(node, 0) + count
    observed_call_ids = list(by_call_id)
    expected_first_call_id = exact_status["next_call_id"] - record_count
    expected_call_ids = list(range(
        expected_first_call_id, exact_status["next_call_id"]
    ))
    if expected_first_call_id <= 0 or observed_call_ids != expected_call_ids:
        raise RuntimeError(
            "native GEMM output evidence call IDs are not the exact contiguous "
            "queue order ending at next_call_id"
        )

    joined_indices: dict[int, list[int]] = {call_id: [] for call_id in by_call_id}
    retained_context: dict[int, dict[str, Any]] = {}
    hot_call_ids: set[int] = set()
    hot_record_count = 0
    for record_index, record in enumerate(records):
        raw = record.get("native_gemm_output_numa")
        is_hot = (
            record.get("phase") in {"source_gemm", "target_gemm"}
            and record.get("telemetry_scope")
            in {"vendor_call", "deterministic_tiled_call_boundary"}
        )
        if not isinstance(raw, Mapping) or raw.get("applicable") is not True:
            if is_hot:
                raise RuntimeError(
                    "hot source/target GEMM record lacks applicable protected-output "
                    "NUMA evidence"
                )
            continue
        context = {
            "record_index": record_index,
            "sequence": record.get("sequence"),
            "native_sequence": record.get("native_sequence"),
            "phase": record.get("phase"),
            "operation": record.get("operation"),
            **{name: record.get(name) for name in ("m", "n", "k")},
        }
        evidence = validate_evidence(raw, context=context)
        call_id = evidence["call_id"]
        if call_id not in by_call_id or evidence != by_call_id[call_id]:
            raise RuntimeError(
                "GEMM telemetry output evidence does not join exactly to its "
                "one-per-call drained record"
            )
        joined_indices[call_id].append(record_index)
        retained_context.setdefault(call_id, context)
        try:
            m = _require_json_int(record.get("m"), field="output GEMM m")
            n = _require_json_int(record.get("n"), field="output GEMM n")
            k = _require_json_int(record.get("k"), field="output GEMM k")
        except RuntimeError as exc:
            raise RuntimeError(
                "applicable protected-output GEMM dimensions are malformed"
            ) from exc
        ordinary_shape = bool(
            evidence["logical_rows"] == m
            and evidence["logical_columns"] == n
            and evidence["storage_layout"] == record.get("layout")
        )
        tt_row_major_alias = False
        if record.get("operation") == "dgemm_tt":
            try:
                lda = _require_json_int(record.get("lda"), field="TT output GEMM lda")
                ldb = _require_json_int(record.get("ldb"), field="TT output GEMM ldb")
                ldc = _require_json_int(record.get("ldc"), field="TT output GEMM ldc")
            except RuntimeError as exc:
                raise RuntimeError(
                    "TT row-major output GEMM leading dimensions are malformed"
                ) from exc
            tt_row_major_alias = bool(
                record.get("layout") == "column_major"
                and record.get("transpose_a") == "T"
                and record.get("transpose_b") == "T"
                and m > 0
                and n > 0
                and k > 0
                and lda == k
                and ldb == n
                and ldc == m
                and evidence["logical_rows"] == n
                and evidence["logical_columns"] == m
                and evidence["storage_layout"] == "row_major"
            )
            ordinary_shape = False
        if (
            evidence["logical_byte_count"] != m * n * 8
            or not (ordinary_shape or tt_row_major_alias)
        ):
            raise RuntimeError(
                "applicable executor output mapping does not match the exact "
                "logical C shape and storage layout"
            )
        if is_hot:
            hot_record_count += 1
            hot_call_ids.add(call_id)
    unjoined = sorted(
        call_id for call_id, indices in joined_indices.items() if not indices
    )
    if unjoined:
        raise RuntimeError(
            f"native GEMM output evidence has unjoined call IDs: {unjoined[:16]}"
        )
    if not hot_call_ids:
        raise RuntimeError("completed batch has no hot protected-output NUMA evidence")
    retained = []
    for call_id, evidence in by_call_id.items():
        context = retained_context[call_id]
        item = {
            "call_id": call_id,
            "gemm_record_indices": joined_indices[call_id],
            "hot_source_or_target": call_id in hot_call_ids,
            **context,
            "native_gemm_output_numa": dict(evidence),
        }
        try:
            retained.append(json.loads(json.dumps(item, allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "native GEMM output NUMA evidence is not JSON-safe"
            ) from exc
    return {
        "schema": NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "schema_version": 1,
        "required_for_acceptance": True,
        "complete": True,
        "operand_role": NATIVE_GEMM_OUTPUT_NUMA_ROLE,
        "query_chunk_page_limit": query_chunk_page_limit,
        "evidence_capacity": evidence_capacity,
        "selected_numa_nodes": sorted(selected_nodes),
        "protected_output_call_count": record_count,
        "hot_source_target_record_count": hot_record_count,
        "hot_source_target_call_count": len(hot_call_ids),
        "non_hot_output_call_count": record_count - len(hot_call_ids),
        "logical_byte_count_total": logical_bytes_total,
        "mapping_bytes_total": mapping_bytes_total,
        "mapping_page_count_total": mapping_pages_total,
        "query_chunk_count_total": query_chunks_total,
        "node_histogram": {
            str(node): aggregate_histogram[node]
            for node in sorted(aggregate_histogram)
        },
        "one_evidence_record_per_call_id": True,
        "exact_contiguous_call_id_queue_order": True,
        "first_call_id": expected_first_call_id,
        "next_call_id": exact_status["next_call_id"],
        "every_evidence_record_joined_to_gemm_telemetry": True,
        "every_hot_source_target_output_fully_resolved_local": True,
        "every_output_verified_after_repair_before_python_return": True,
        "every_output_strict_no_move": True,
        "ordered_status_digest_evidence": (
            "canonical native SHA-256 retained for every call; controller "
            "recomputed only for single-node histograms"
        ),
        "single_node_ordered_status_digest_recomputed_count": (
            single_node_digest_recomputed_count
        ),
        "page_migration_requested": False,
        "placement_repair_performed": False,
        "record_evidence_included": True,
        "status": dict(status),
        "records": retained,
    }


def _resolve_single_explicit_blis_group(
    canonical_path: Path,
    canonical: Mapping[str, Any],
    *,
    subset: PlinkMetadata,
    args: argparse.Namespace,
    native_sha256: str,
    cpu_records: Sequence[CpuRecord],
    outer_child_pid: int,
    expected_numa_nodes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Verify the one-group canonical index and return its full group payload."""
    columns = [item.strip() for item in args.environment_columns.split(",")]
    expected_canonical = {
        "kind": "summit.gxe.multi_environment_reference_batch",
        "schema_version": 1,
        "execution": "parallel_isolated_environment_groups",
        "requested_backend": "direct",
        "full_precision_layout": "current",
        "protected_native_gemm": True,
        "arithmetic_dtype": "float64",
        "requested_storage_dtype": args.storage_dtype,
        "gemm_backend": "gxeldcore_direct",
        "gemm_backend_build_sha256": native_sha256,
        "native_source_commit": args.expected_source_commit,
        "native_source_tree_sha256": args.expected_source_tree_sha256,
        "native_blas_runtime_isolation": "private_static",
        "repaired_gemm_output_columns": 0,
        "source_panel_memory_order": "F",
        "target_panel_memory_order": "F",
        "target_genotype_memory_order": "F",
        "source_to_target_layout_transition": "none",
        "num_environments": len(columns),
        "num_variants": subset.variants,
        "cpu_placement_complete": True,
    }
    mismatches = {
        name: {"expected": value, "observed": canonical.get(name)}
        for name, value in expected_canonical.items()
        if canonical.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"canonical explicit-placement batch mismatch: {mismatches}")
    if (
        type(canonical.get("native_gemm_integrity_enabled")) is not bool
        or (
            not args.allow_integrity_disabled
            and canonical.get("native_gemm_integrity_enabled") is not True
        )
    ):
        raise RuntimeError("canonical explicit-placement integrity contract changed")
    groups = canonical.get("environment_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("explicit single-group batch must declare exactly one group")
    group_record = groups[0]
    if not isinstance(group_record, Mapping):
        raise RuntimeError("explicit group declaration is malformed")
    relative = group_record.get("manifest")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("explicit group declaration lacks its manifest path")
    unresolved = canonical_path.parent / relative
    if unresolved.is_symlink():
        raise RuntimeError("explicit group manifest must not be a symbolic link")
    group_path = unresolved.resolve()
    try:
        group_path.relative_to(canonical_path.parent.resolve())
    except ValueError as exc:
        raise RuntimeError("explicit group manifest escapes its output directory") from exc
    if not group_path.is_file() or not stat.S_ISREG(group_path.stat().st_mode):
        raise RuntimeError("explicit group manifest is missing or not regular")
    group_sha256 = _sha256(group_path)
    if group_sha256 != group_record.get("sha256"):
        raise RuntimeError("explicit group manifest hash differs from canonical index")
    group = json.loads(group_path.read_text(encoding="utf-8"))
    invariant_keys = (
        "kind", "schema_version", "requested_backend", "full_precision_layout",
        "protected_native_gemm", "arithmetic_dtype", "requested_storage_dtype",
        "gemm_backend", "gemm_backend_build_sha256", "native_source_commit",
        "native_source_tree_sha256", "native_gemm_integrity_enabled",
        "native_blas_runtime_isolation", "source_panel_memory_order",
        "target_panel_memory_order", "target_genotype_memory_order",
        "source_to_target_layout_transition", "common_complete_case_samples",
        "num_variants", "randomization", "references",
    )
    disagreements = [
        name for name in invariant_keys if canonical.get(name) != group.get(name)
    ]
    if disagreements:
        raise RuntimeError(
            "canonical and explicit group manifests disagree on: "
            + ", ".join(disagreements)
        )
    if group.get("execution") != "shared_in_memory_decoded_blocks":
        raise RuntimeError("explicit worker did not use the shared current-layout executor")
    if group_record.get("environments") != columns:
        raise RuntimeError("explicit group environment order differs from the request")
    if int(group_record.get("repaired_gemm_output_columns", -1)) != 0:
        raise RuntimeError("explicit group declaration reports a GEMM repair")
    performance = group.get("performance_telemetry")
    if not isinstance(performance, Mapping):
        raise RuntimeError("explicit group lacks full performance telemetry")
    placement = _validate_openmp_placement_attestation(
        group.get("cpu_placement"), cpu_records=cpu_records, threads=args.threads
    )
    if group.get("cpu_placement_complete") is not True:
        raise RuntimeError("explicit group placement is not marked complete")
    if canonical.get("cpu_placements") != [placement]:
        raise RuntimeError("canonical batch placement differs from its explicit group")
    if (
        performance.get("cpu_placement_complete") is not True
        or performance.get("cpu_placement") != placement
    ):
        raise RuntimeError("performance placement differs from its explicit group")
    summary = group_record.get("performance_telemetry_summary")
    if not isinstance(summary, Mapping) or any(
        performance.get(name) != value for name, value in summary.items()
    ):
        raise RuntimeError("canonical performance summary differs from its group")
    early = group.get("early_numa_attestation")
    if not isinstance(early, Mapping) or performance.get(
        "early_numa_attestation"
    ) != early:
        raise RuntimeError("explicit group NUMA attestations disagree")
    raw_worker_pid = early.get("pid")
    if (
        type(raw_worker_pid) is not int
        or raw_worker_pid <= 0
        or raw_worker_pid == outer_child_pid
    ):
        raise RuntimeError("explicit group lacks its distinct fresh-exec worker PID")
    if group_record.get("early_numa_attestation") != early:
        raise RuntimeError("canonical group record NUMA attestation differs")
    if canonical.get("early_numa_attestations") != [dict(early)]:
        raise RuntimeError("canonical NUMA attestation list differs from its group")
    selected_nodes = _expected_memory_nodes(cpu_records, expected_numa_nodes)
    if early.get("effective_nodes") != selected_nodes:
        raise RuntimeError(
            "explicit group NUMA nodes differ from selected memory nodes"
        )
    decode_summary = group.get("numa_bound_bed_decode")
    decode_report = performance.get("numa_bound_bed_decode")
    if not isinstance(decode_summary, Mapping) or not isinstance(
        decode_report, Mapping
    ):
        raise RuntimeError("explicit group lacks NUMA-bound BED decode evidence")
    rebuilt_decode_summary = _compact_numa_bound_decode_report(decode_report)
    if (
        group.get("numa_bound_bed_decode_required") is not True
        or group.get("numa_bound_bed_decode_complete") is not True
        or performance.get("numa_bound_bed_decode_required") is not True
        or performance.get("numa_bound_bed_decode_complete") is not True
        or dict(decode_summary) != rebuilt_decode_summary
        or group_record.get("numa_bound_bed_decode_summary")
        != rebuilt_decode_summary
        or canonical.get("numa_bound_bed_decode_required") is not True
        or canonical.get("numa_bound_bed_decode_complete") is not True
        or canonical.get("numa_bound_bed_decode_groups")
        != [rebuilt_decode_summary]
    ):
        raise RuntimeError(
            "canonical and explicit group NUMA-bound BED decode evidence disagrees"
        )
    references = _reference_manifests(canonical_path, canonical, columns)
    integrity_contracts = []
    for environment in columns:
        _path, reference = references[environment]
        integrity_contracts.append(
            _validate_reference_blis_provenance(
                reference, args=args, native_sha256=native_sha256,
                placement=placement,
            )
        )
    if any(
        contract != integrity_contracts[0]
        for contract in integrity_contracts[1:]
    ):
        raise RuntimeError(
            "environment references disagree on the native integrity snapshot contract"
        )
    return {
        "canonical_path": canonical_path,
        "canonical_payload": dict(canonical),
        "group_path": group_path,
        "group_sha256": group_sha256,
        "group_payload": group,
        "placement": placement,
        "worker_pid": raw_worker_pid,
        **integrity_contracts[0],
    }


def _validate_completed_manifest(
    payload: dict[str, Any], *, subset: PlinkMetadata, args: argparse.Namespace,
    native_sha256: str, cpu_records: Sequence[CpuRecord],
    full_precision_layout: str | None = None,
    block_width: int | None = None,
    annotation_bin_count: int | None = None,
    reference_sample_count: int | None = None,
    expected_child_pid: int | None = None,
    require_explicit_blis_placement: bool = False,
    expected_placement: Mapping[str, Any] | None = None,
    expected_numa_nodes: Sequence[int] | None = None,
    integrity_minimum_vendor_flops: int | None = None,
    native_integrity_snapshot_numa_query_chunk_page_limit: int | None = None,
    native_gemm_output_numa_query_chunk_page_limit: int | None = None,
    native_gemm_output_numa_evidence_capacity: int | None = None,
) -> dict[str, Any]:
    if expected_child_pid is None:
        raise RuntimeError(
            "completed-manifest validation requires the exact spawned child PID"
        )
    columns = [item.strip() for item in args.environment_columns.split(",")]
    expected_layout = _canonical_layout(
        args.full_precision_layout
        if full_precision_layout is None else full_precision_layout
    )
    expected = {
        "kind": "summit.gxe.multi_environment_reference_batch",
        "execution": "shared_in_memory_decoded_blocks",
        "requested_backend": "direct",
        "full_precision_layout": expected_layout,
        "protected_native_gemm": True,
        "requested_storage_dtype": args.storage_dtype,
        "num_environments": len(columns), "num_variants": subset.variants,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"unexpected batch manifest {key}: {payload.get(key)!r}")
    raw_complete_cases = payload.get("common_complete_case_samples")
    if isinstance(raw_complete_cases, bool) or not isinstance(
        raw_complete_cases, int
    ):
        raise RuntimeError(
            "batch common_complete_case_samples must be a positive integer"
        )
    analysis_samples = int(raw_complete_cases)
    if analysis_samples <= 0 or analysis_samples > subset.samples:
        raise RuntimeError(
            "batch common_complete_case_samples must be positive and no larger "
            "than the validated FAM sample count"
        )
    if reference_sample_count is None or reference_sample_count != analysis_samples:
        raise RuntimeError(
            "batch common_complete_case_samples does not match hashed reference "
            "n_samples metadata"
        )
    if payload.get("arithmetic_dtype") != "float64":
        raise RuntimeError("partial benchmark did not retain float64 GEMM arithmetic")
    randomization = payload.get("randomization", {})
    if randomization.get("num_vectors") != args.probes:
        raise RuntimeError("batch manifest probe count does not match the controller")
    if randomization.get("seed") != args.seed:
        raise RuntimeError("batch manifest seed does not match the controller")
    if randomization.get("distribution") != "rademacher":
        raise RuntimeError("batch manifest randomization distribution changed")
    if int(randomization.get("probe_offset", -1)) != 0:
        raise RuntimeError("batch manifest probe offset changed")
    if payload.get("gemm_backend_build_sha256") != native_sha256:
        raise RuntimeError("batch manifest native hash does not match the explicit module")
    if require_explicit_blis_placement and (
        payload.get("gemm_backend") != "gxeldcore_direct"
        or payload.get("native_source_commit") != args.expected_source_commit
        or payload.get("native_source_tree_sha256")
        != args.expected_source_tree_sha256
    ):
        raise RuntimeError("batch manifest private-BLIS native identity changed")
    if payload.get("native_blas_runtime_isolation") != "private_static":
        raise RuntimeError("completed batch does not retain private-static BLAS isolation")
    if not args.allow_integrity_disabled and payload.get("native_gemm_integrity_enabled") is not True:
        raise RuntimeError("completed development run did not retain GEMM integrity checks")
    if int(payload.get("repaired_gemm_output_columns", -1)) != 0:
        raise RuntimeError("accepted real-genotype comparison requires zero GEMM repairs")
    expected_transition = (
        {
            "source_panel_memory_order": "C",
            "target_panel_memory_order": "F",
            "target_genotype_memory_order": "F",
            "source_to_target_layout_transition": (
                "explicit_c_to_f_before_protected_pair_sealing"
            ),
        }
        if expected_layout == "source_tt_target_current"
        else {
            "source_panel_memory_order": "F",
            "target_panel_memory_order": "F",
            "target_genotype_memory_order": "F",
            "source_to_target_layout_transition": "none",
        }
    )
    for key, value in expected_transition.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"completed batch reports an unexpected layout transition field {key}"
            )
    copy_count = int(payload.get("source_to_target_layout_copy_count", 0))
    if (expected_layout == "source_tt_target_current" and copy_count <= 0) or (
        expected_layout == "current" and copy_count != 0
    ):
        raise RuntimeError("completed batch reports an invalid layout-copy count")
    performance = payload.get("performance_telemetry")
    if not isinstance(performance, dict):
        raise RuntimeError("completed batch lacks performance telemetry")
    if int(performance.get("requested_blas_threads", -1)) != args.threads:
        raise RuntimeError("performance telemetry reports an unexpected BLAS thread count")
    if performance.get("arithmetic_dtype") != "float64":
        raise RuntimeError("performance telemetry reports non-float64 arithmetic")
    if performance.get("requested_storage_dtype") != args.storage_dtype:
        raise RuntimeError("performance telemetry reports an unexpected storage dtype")
    if performance.get("full_precision_layout") != expected_layout:
        raise RuntimeError("performance telemetry reports an unexpected FP64 layout")
    if performance.get("backend_build_sha256") != native_sha256:
        raise RuntimeError("performance telemetry reports an unexpected native hash")
    if int(performance.get("repaired_gemm_output_columns", -1)) != 0:
        raise RuntimeError("accepted performance telemetry requires zero GEMM repairs")
    expected_calling_cpu: int | None = None
    expected_calling_affinity: set[int] | None = None
    full_team_cpu_ids: list[int] | None = None
    if require_explicit_blis_placement:
        if expected_placement is None:
            raise RuntimeError("explicit BLIS validation lacks placement evidence")
        placement = _validate_openmp_placement_attestation(
            payload.get("cpu_placement"),
            cpu_records=cpu_records,
            threads=args.threads,
        )
        if (
            payload.get("cpu_placement_complete") is not True
            or performance.get("cpu_placement_complete") is not True
            or performance.get("cpu_placement") != placement
            or placement != dict(expected_placement)
        ):
            raise RuntimeError(
                "group/performance OpenMP placement evidence is inconsistent"
            )
        calling_worker = placement["workers"][0]
        expected_calling_cpu = calling_worker["current_cpu"]
        expected_calling_affinity = set(
            calling_worker["sched_affinity_cpu_ids"]
        )
        full_team_cpu_ids = list(placement["expected_cpu_ids"])
    if performance.get("capture_boundary") != (
        "post_output_artifact_publication_pre_batch_manifest"
    ):
        raise RuntimeError("performance telemetry was not captured at the final boundary")
    for field in (
        "telemetry_complete", "vendor_call_telemetry_complete",
        "hot_gemm_telemetry_complete", "phase_telemetry_complete",
    ):
        if performance.get(field) is not True:
            raise RuntimeError(f"completed batch telemetry field {field} is not true")
    records = performance.get("gemm_records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("completed batch contains no GEMM records")
    _validate_layout_records(records, expected_layout)
    if expected_layout == "current":
        if annotation_bin_count is None:
            raise RuntimeError(
                "current-layout acceptance requires independently derived "
                "annotation-bin metadata"
            )
        performance["controller_current_vendor_acceptance"] = (
            _validate_current_vendor_acceptance(
                records,
                samples=analysis_samples,
                variants=subset.variants,
                block_width=(
                    args.block_width if block_width is None else block_width
                ),
                probes=args.probes,
                annotation_bins=annotation_bin_count,
                threads=args.threads,
                environment_count=len(columns),
                environment_tiles=payload.get("environment_tiles", []),
                probe_tiles=randomization.get("probe_tiles", []),
                require_private_blis=require_explicit_blis_placement,
                native_sha256=native_sha256,
                source_commit=(
                    args.expected_source_commit
                    if require_explicit_blis_placement else None
                ),
                source_tree_sha256=(
                    args.expected_source_tree_sha256
                    if require_explicit_blis_placement else None
                ),
            )
        )
    optimized_status = performance.get("optimized_fp64_layout_telemetry")
    if expected_layout == "source_tt_target_current":
        if performance.get("optimized_fp64_layout_zero_repair") is not True:
            raise RuntimeError("optimized layout did not pass its zero-repair gate")
        if performance.get("optimized_fp64_layout_telemetry_complete") is not True:
            raise RuntimeError("optimized layout telemetry contract is incomplete")
        if not isinstance(optimized_status, dict) or optimized_status.get("complete") is not True:
            raise RuntimeError("optimized layout telemetry status is incomplete")
    selected = {record.cpu for record in cpu_records}
    selected_nodes = _expected_memory_nodes(cpu_records, expected_numa_nodes)
    performance["controller_early_numa_attestation_requirement"] = (
        _early_numa_attestation_status(
            payload, performance, set(selected_nodes),
            expected_child_pid=expected_child_pid,
        )
    )
    if require_explicit_blis_placement:
        environment_tiles = payload.get("environment_tiles")
        probe_tiles = randomization.get("probe_tiles")
        if (
            not isinstance(environment_tiles, list)
            or not environment_tiles
            or not isinstance(probe_tiles, list)
            or not probe_tiles
        ):
            raise RuntimeError(
                "completed batch lacks its NUMA-bound decode tile schedule"
            )
        expected_passes = _schedule_passes(
            len(environment_tiles), len(probe_tiles)
        )
        if payload.get("shared_genotype_passes") != expected_passes:
            raise RuntimeError(
                "completed batch shared genotype passes disagree with its tile schedule"
            )
        decode_summary = payload.get("numa_bound_bed_decode")
        decode_report = performance.get("numa_bound_bed_decode")
        if (
            payload.get("numa_bound_bed_decode_required") is not True
            or payload.get("numa_bound_bed_decode_complete") is not True
            or performance.get("numa_bound_bed_decode_required") is not True
            or performance.get("numa_bound_bed_decode_complete") is not True
            or not isinstance(decode_summary, Mapping)
            or not isinstance(decode_report, Mapping)
            or dict(decode_summary)
            != _compact_numa_bound_decode_report(decode_report)
        ):
            raise RuntimeError(
                "completed batch lacks matching NUMA-bound BED decode evidence"
            )
        performance["controller_numa_bound_bed_decode_validation"] = (
            _validate_numa_bound_decode_report(
                decode_report,
                selected_nodes=selected_nodes,
                samples=analysis_samples,
                variants=subset.variants,
                block_width=(
                    args.block_width if block_width is None else block_width
                ),
                passes=expected_passes,
            )
        )
        performance["controller_native_integrity_snapshot_numa_validation"] = (
            _native_integrity_snapshot_numa_summary(
                records,
                set(selected_nodes),
                integrity_enabled=(
                    payload.get("native_gemm_integrity_enabled") is True
                ),
                integrity_minimum_vendor_flops=(
                    integrity_minimum_vendor_flops
                    if integrity_minimum_vendor_flops is not None else -1
                ),
                query_chunk_page_limit=(
                    native_integrity_snapshot_numa_query_chunk_page_limit
                    if native_integrity_snapshot_numa_query_chunk_page_limit
                    is not None else -1
                ),
            )
        )
        if (
            performance.get("native_gemm_output_numa_contract_supported") is not True
            or performance.get("native_gemm_output_numa_contract_required") is not True
            or performance.get("native_gemm_output_numa_evidence_available") is not True
            or performance.get("native_gemm_output_numa_evidence_complete") is not True
            or type(performance.get("native_gemm_output_numa_record_count")) is not int
            or type(performance.get(
                "dropped_native_gemm_output_numa_records"
            )) is not int
            or performance.get("dropped_native_gemm_output_numa_records") != 0
        ):
            raise RuntimeError(
                "completed batch lacks complete, lossless native GEMM output NUMA evidence"
            )
        output_numa_records = performance.get(
            "native_gemm_output_numa_records"
        )
        if (
            not isinstance(output_numa_records, list)
            or performance["native_gemm_output_numa_record_count"]
            != len(output_numa_records)
        ):
            raise RuntimeError(
                "completed batch native GEMM output NUMA record count is inconsistent"
            )
        performance["controller_native_gemm_output_numa_validation"] = (
            _native_gemm_output_numa_summary(
                records,
                output_numa_records,
                performance.get("native_gemm_output_numa_evidence_status"),
                set(selected_nodes),
                query_chunk_page_limit=(
                    native_gemm_output_numa_query_chunk_page_limit
                    if native_gemm_output_numa_query_chunk_page_limit is not None
                    else -1
                ),
                evidence_capacity=(
                    native_gemm_output_numa_evidence_capacity
                    if native_gemm_output_numa_evidence_capacity is not None
                    else -1
                ),
            )
        )
    numa_summary = _hot_vendor_numa_summary(records, set(selected_nodes))
    performance["controller_hot_vendor_numa_validation"] = numa_summary
    if not numa_summary["all_hot_operands_queried_resolved_local"]:
        raise RuntimeError(
            "accepted real-genotype comparison requires all hot vendor "
            "move_pages samples to be queried, resolved, and local; "
            "numa_context="
            + json.dumps(
                numa_summary["rejection_contexts"],
                sort_keys=True, separators=(",", ":"),
            )
        )
    expected_numa_policy = (
        "libnuma:membind:" + ",".join(str(node) for node in selected_nodes)
    )
    expected_early_attestation = performance[
        "controller_early_numa_attestation_requirement"
    ]["attestation"]
    vendor_call_count = 0
    for record in records:
        inside_outer_openmp = record.get(
            "call_site_inside_openmp", record.get("omp_in_parallel")
        )
        if inside_outer_openmp is not None and bool(inside_outer_openmp):
            raise RuntimeError("vendor GEMM was entered from an active outer OpenMP region")
        requested_threads = record.get(
            "requested_blas_threads", record.get("requested_threads")
        )
        if requested_threads is not None and int(requested_threads) != args.threads:
            raise RuntimeError("GEMM record reports an unexpected BLAS thread count")
        affinity_raw = record.get("affinity_core_list")
        affinity = _affinity_cpu_set(affinity_raw)
        if require_explicit_blis_placement:
            telemetry_scope = record.get("telemetry_scope")
            if telemetry_scope == "vendor_call":
                alias_is_canonical = affinity_raw == str(expected_calling_cpu)
            elif telemetry_scope in {
                "deterministic_tiled_call_boundary",
                "protected_call_boundary",
            }:
                alias_is_canonical = (
                    type(affinity_raw) is list
                    and affinity_raw == [expected_calling_cpu]
                    and all(type(cpu) is int for cpu in affinity_raw)
                )
                if any(
                    field in record
                    for field in (
                        "cpu_affinity_list", "cpu_affinity_count",
                        "entry_cpu", "exit_cpu",
                    )
                ):
                    raise RuntimeError(
                        "non-vendor GEMM telemetry unexpectedly contains native "
                        "vendor-entry placement fields"
                    )
            else:
                raise RuntimeError(
                    f"explicit BLIS GEMM telemetry has an unexpected scope: "
                    f"{telemetry_scope!r}"
                )
            if (
                expected_calling_cpu is None
                or expected_calling_affinity is None
                or not alias_is_canonical
                or affinity != expected_calling_affinity
            ):
                raise RuntimeError(
                    "GEMM telemetry does not report the exact bound calling-"
                    "thread placement; expected_cpu="
                    f"{expected_calling_cpu}, expected_affinity="
                    f"{sorted(expected_calling_affinity or set())}, "
                    f"observed_affinity={sorted(affinity or set())}"
                )
            if telemetry_scope == "vendor_call":
                vendor_call_count += 1
                native_affinity_raw = record.get("cpu_affinity_list")
                affinity_count = record.get("cpu_affinity_count")
                entry_cpu = record.get("entry_cpu")
                exit_cpu = record.get("exit_cpu")
                if (
                    type(native_affinity_raw) is not str
                    or native_affinity_raw != str(expected_calling_cpu)
                    or affinity_raw != native_affinity_raw
                    or type(affinity_count) is not int
                    or type(entry_cpu) is not int
                    or type(exit_cpu) is not int
                    or affinity_count != 1
                    or entry_cpu != expected_calling_cpu
                    or exit_cpu != expected_calling_cpu
                ):
                    raise RuntimeError(
                        "vendor GEMM telemetry does not report the exact native "
                        "bound calling-thread placement; expected_cpu="
                        f"{expected_calling_cpu}, observed_entry={entry_cpu!r}, "
                        f"observed_exit={exit_cpu!r}, "
                        f"observed_affinity={native_affinity_raw!r}"
                    )
        elif affinity != selected:
            raise RuntimeError(
                "GEMM telemetry does not report the exact selected CPU list"
            )
        numa_placement = record.get("numa_node_placement")
        process_policy = (
            numa_placement.get("process_policy")
            if isinstance(numa_placement, Mapping) else None
        )
        if (
            not isinstance(process_policy, Mapping)
            or process_policy.get("applied_policy") != expected_numa_policy
            or process_policy.get("policy_provenance")
            != "pre_numeric_import"
            or process_policy.get("early_numa_attestation")
            != expected_early_attestation
        ):
            raise RuntimeError(
                "GEMM telemetry does not report the verified early NUMA policy"
            )
    if require_explicit_blis_placement:
        performance["controller_gemm_calling_thread_affinity_validation"] = {
            "complete": True,
            "scope": (
                "affinity_core_list field validated on every GEMM record; "
                "raw native vendor-entry affinity/count and entry/exit CPU "
                "validated on vendor_call records; full OpenMP team placement "
                "validated separately"
            ),
            "record_count": len(records),
            "vendor_record_count": vendor_call_count,
            "non_vendor_record_count": len(records) - vendor_call_count,
            "expected_calling_cpu": expected_calling_cpu,
            "expected_calling_affinity_cpu_ids": sorted(
                expected_calling_affinity or set()
            ),
            "full_team_cpu_ids": full_team_cpu_ids,
            "full_team_evidence_field": "cpu_placement",
        }
    return performance


@dataclass
class _ErrorAccumulator:
    rtol: float
    atol: float
    count: int = 0
    reference_square_sum: float = 0.0
    difference_square_sum: float = 0.0
    maximum_absolute_error: float = 0.0
    maximum_normalized_error: float = 0.0
    maximum_absolute_error_label: str | None = None
    maximum_normalized_error_label: str | None = None
    maximum_normalized_error_reference_value: float | None = None
    maximum_normalized_error_candidate_value: float | None = None
    maximum_normalized_error_signed_difference: float | None = None
    maximum_normalized_error_tolerance: float | None = None

    def add(self, reference: float, candidate: float, label: str) -> None:
        reference = float(reference)
        candidate = float(candidate)
        if not math.isfinite(reference) or not math.isfinite(candidate):
            raise RuntimeError(f"non-finite numerical comparison value at {label}")
        difference = abs(candidate - reference)
        tolerance = self.atol + self.rtol * abs(reference)
        normalized_error = difference / tolerance
        self.count += 1
        self.reference_square_sum += reference * reference
        self.difference_square_sum += difference * difference
        if (
            self.maximum_absolute_error_label is None
            or difference > self.maximum_absolute_error
        ):
            self.maximum_absolute_error = difference
            self.maximum_absolute_error_label = label
        if (
            self.maximum_normalized_error_label is None
            or normalized_error > self.maximum_normalized_error
        ):
            self.maximum_normalized_error = normalized_error
            self.maximum_normalized_error_label = label
            self.maximum_normalized_error_reference_value = reference
            self.maximum_normalized_error_candidate_value = candidate
            self.maximum_normalized_error_signed_difference = candidate - reference
            self.maximum_normalized_error_tolerance = tolerance

    def record(self) -> dict[str, Any]:
        reference_norm = math.sqrt(self.reference_square_sum)
        difference_norm = math.sqrt(self.difference_square_sum)
        denominator = (
            reference_norm
            if reference_norm > 0.0
            else self.atol * math.sqrt(max(self.count, 1))
        )
        return {
            "value_count": self.count,
            "reference_frobenius_norm": reference_norm,
            "difference_frobenius_norm": difference_norm,
            "relative_frobenius_error": difference_norm / denominator,
            "relative_frobenius_zero_reference_floor_used": reference_norm == 0.0,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_absolute_error_label": self.maximum_absolute_error_label,
            "maximum_normalized_error": self.maximum_normalized_error,
            "maximum_normalized_error_label": self.maximum_normalized_error_label,
            "maximum_normalized_error_reference_value": (
                self.maximum_normalized_error_reference_value
            ),
            "maximum_normalized_error_candidate_value": (
                self.maximum_normalized_error_candidate_value
            ),
            "maximum_normalized_error_signed_difference": (
                self.maximum_normalized_error_signed_difference
            ),
            "maximum_normalized_error_tolerance": (
                self.maximum_normalized_error_tolerance
            ),
            "normalized_error_definition": "abs(candidate-reference)/(atol+rtol*abs(reference))",
            "rtol": self.rtol,
            "atol": self.atol,
            "within_declared_tolerance": self.maximum_normalized_error <= 1.0,
        }


def _open_text_table(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _compare_tsv_files(
    reference: Path, candidate: Path, *, key_columns: int,
    rtol: float, atol: float,
) -> dict[str, Any]:
    accumulator = _ErrorAccumulator(rtol=rtol, atol=atol)
    sentinel = object()
    with _open_text_table(reference) as left_handle, _open_text_table(candidate) as right_handle:
        left = csv.reader(left_handle, delimiter="\t")
        right = csv.reader(right_handle, delimiter="\t")
        try:
            left_header = next(left)
            right_header = next(right)
        except StopIteration as exc:
            raise RuntimeError("comparison encountered an empty output table") from exc
        if left_header != right_header:
            raise RuntimeError(
                f"output table schemas differ: {left_header!r} != {right_header!r}"
            )
        if not key_columns < len(left_header):
            raise RuntimeError("output table contains no numerical value columns")
        rows = 0
        for row_number, pair in enumerate(
            itertools.zip_longest(left, right, fillvalue=sentinel), start=2
        ):
            left_row, right_row = pair
            if left_row is sentinel or right_row is sentinel:
                raise RuntimeError("output table row counts differ")
            if len(left_row) != len(left_header) or len(right_row) != len(right_header):
                raise RuntimeError(f"malformed output table row {row_number}")
            if left_row[:key_columns] != right_row[:key_columns]:
                raise RuntimeError(
                    f"output table row identity/order differs at row {row_number}"
                )
            for column, (left_value, right_value) in enumerate(
                zip(left_row[key_columns:], right_row[key_columns:], strict=True),
                start=key_columns,
            ):
                try:
                    accumulator.add(
                        float(left_value), float(right_value),
                        f"{reference.name}:{row_number}:{left_header[column]}",
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"non-numeric output value at row {row_number}, "
                        f"column {left_header[column]!r}"
                    ) from exc
            rows += 1
    return {
        "schema_and_row_order_equal": True,
        "columns": left_header,
        "row_key_columns": left_header[:key_columns],
        "rows": rows,
        **accumulator.record(),
    }


def _compare_json_tree(
    reference: Any, candidate: Any, *, label: str, rtol: float, atol: float
) -> dict[str, Any]:
    accumulator = _ErrorAccumulator(rtol=rtol, atol=atol)

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or list(left) != list(right):
                raise RuntimeError(f"JSON schema/key order differs at {path}")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)):
            if not isinstance(right, (list, tuple)) or len(left) != len(right):
                raise RuntimeError(f"JSON sequence shape differs at {path}")
            for index, (left_value, right_value) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_value, right_value, f"{path}[{index}]")
            return
        if isinstance(left, bool) or left is None or isinstance(left, str):
            if type(right) is not type(left) or right != left:
                raise RuntimeError(f"JSON diagnostic/schema value differs at {path}")
            return
        if isinstance(left, numbers.Real) and not isinstance(left, bool):
            if not isinstance(right, numbers.Real) or isinstance(right, bool):
                raise RuntimeError(f"JSON numerical type differs at {path}")
            accumulator.add(float(left), float(right), path)
            return
        raise RuntimeError(f"unsupported JSON diagnostic type at {path}: {type(left).__name__}")

    compare(reference, candidate, label)
    return {"schema_and_order_equal": True, **accumulator.record()}


def _compare_population_trace(
    reference: Any, candidate: Any, *, rtol: float, atol: float
) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        raise RuntimeError("population_trace must be an object in both references")
    if list(reference) != list(candidate):
        raise RuntimeError("population_trace schema/key order differs")
    matrix_key = "same_individual_kernel_products"
    if matrix_key not in reference:
        raise RuntimeError("population_trace lacks same-person kernel products")
    metadata = {key: value for key, value in reference.items() if key != matrix_key}
    candidate_metadata = {
        key: value for key, value in candidate.items() if key != matrix_key
    }
    if metadata != candidate_metadata:
        raise RuntimeError("population_trace metadata/feature order differs")
    feature_order = reference.get("feature_order")
    left_matrix = reference[matrix_key]
    right_matrix = candidate[matrix_key]
    if not isinstance(feature_order, list) or not feature_order:
        raise RuntimeError("population_trace feature order is missing")
    expected = len(feature_order)
    for matrix in (left_matrix, right_matrix):
        if (
            not isinstance(matrix, list) or len(matrix) != expected
            or any(not isinstance(row, list) or len(row) != expected for row in matrix)
        ):
            raise RuntimeError("same-person kernel product matrix shape is invalid")
    return {
        "schema_metadata_and_feature_order_equal": True,
        "feature_order": list(feature_order),
        "same_individual_kernel_products": _compare_json_tree(
            left_matrix, right_matrix, label=matrix_key, rtol=rtol, atol=atol
        ),
    }


def _declared_artifact(
    manifest_path: Path, payload: Mapping[str, Any], role: str
) -> Path:
    files = payload.get("files")
    hashes = payload.get("artifact_sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError(f"reference manifest {manifest_path} lacks artifact declarations")
    relative = files.get(role)
    expected_hash = hashes.get(role)
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise RuntimeError(f"reference manifest lacks declared {role!r} artifact")
    unresolved = manifest_path.parent / relative
    if unresolved.is_symlink():
        raise RuntimeError(f"declared {role!r} artifact must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise RuntimeError(f"declared {role!r} artifact escapes its output directory") from exc
    if not path.is_file():
        raise RuntimeError(f"declared {role!r} artifact is not a regular file: {path}")
    if _sha256(path) != expected_hash:
        raise RuntimeError(f"declared {role!r} artifact hash is invalid: {path}")
    return path


def _reference_manifests(
    batch_path: Path, payload: Mapping[str, Any], environment_order: Sequence[str]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    records = payload.get("references")
    if not isinstance(records, list):
        raise RuntimeError("completed batch lacks ordered reference declarations")
    observed_order = [record.get("environment") for record in records]
    if observed_order != list(environment_order):
        raise RuntimeError("completed batch reference environment order changed")
    result = {}
    for record in records:
        path = (batch_path.parent / str(record.get("reference", ""))).resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"reference manifest is missing or not regular: {path}")
        if _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"reference manifest hash is invalid: {path}")
        reference = json.loads(path.read_text(encoding="utf-8"))
        environment = str(record["environment"])
        if (
            reference.get("kind") != "summit.gxe.reference"
            or int(reference.get("schema_version", -1)) != 3
            or reference.get("environment") != environment
        ):
            raise RuntimeError(f"unexpected reference schema/environment in {path}")
        result[environment] = (path, reference)
    return result


def _validate_reference_blis_provenance(
    reference: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    native_sha256: str,
    placement: Mapping[str, Any],
) -> dict[str, int]:
    if reference.get("cpu_placement_complete") is not True:
        raise RuntimeError("reference lacks complete OpenMP placement evidence")
    if reference.get("cpu_placement") != dict(placement):
        raise RuntimeError("reference OpenMP placement differs from its group")
    integrity_contracts = []
    for key, expected_stage in (
        ("backend_provenance", "reference"),
        ("feature_backend_provenance", "feature_construction"),
    ):
        provenance = reference.get(key)
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"reference lacks {key}")
        required = {
            "artifact_stage": expected_stage,
            "backend_name": "gxeldcore_direct",
            "backend_version": EXPECTED_NATIVE_BACKEND_VERSION,
            "source_commit": args.expected_source_commit,
            "source_tree_sha256": args.expected_source_tree_sha256,
            "native_binary_sha256": native_sha256,
        }
        mismatches = {
            name: {"expected": value, "observed": provenance.get(name)}
            for name, value in required.items()
            if provenance.get(name) != value
        }
        if mismatches:
            raise RuntimeError(
                f"reference {key} identity mismatch: {mismatches}"
            )
        options = _validate_blis_compile_options(
            provenance.get("compile_options"), args=args, placement=placement,
        )
        integrity_contracts.append(
            {
                "integrity_minimum_vendor_flops": options[
                    "gemm_integrity_minimum_vendor_flops"
                ],
                "snapshot_query_chunk_page_limit": options[
                    "native_integrity_snapshot_numa_query_chunk_page_limit"
                ],
                "output_query_chunk_page_limit": options[
                    "native_gemm_output_numa_query_chunk_page_limit"
                ],
                "output_evidence_capacity": options[
                    "native_gemm_output_numa_evidence_capacity"
                ],
            }
        )
    if integrity_contracts[0] != integrity_contracts[1]:
        raise RuntimeError(
            "reference backend and feature provenance disagree on the native "
            "integrity snapshot contract"
        )
    return integrity_contracts[0]


def _batch_reference_dimensions(
    batch_path: Path, payload: Mapping[str, Any],
    environment_order: Sequence[str],
) -> dict[str, int]:
    references = _reference_manifests(batch_path, payload, environment_order)
    expected_names: list[str] | None = None
    expected_samples: int | None = None
    for environment in environment_order:
        _path, reference = references[environment]
        names = reference.get("annotation_names")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise RuntimeError(
                f"reference {environment!r} lacks unique annotation-bin names"
            )
        if expected_names is None:
            expected_names = list(names)
        elif names != expected_names:
            raise RuntimeError(
                "environment references disagree on annotation-bin identity/order"
            )
        raw_samples = reference.get("n_samples")
        if isinstance(raw_samples, bool) or not isinstance(raw_samples, int):
            raise RuntimeError(
                f"reference {environment!r} lacks a positive integer n_samples"
            )
        samples = int(raw_samples)
        if samples <= 0:
            raise RuntimeError(
                f"reference {environment!r} lacks a positive integer n_samples"
            )
        if expected_samples is None:
            expected_samples = samples
        elif samples != expected_samples:
            raise RuntimeError(
                "environment references disagree on complete-case n_samples"
            )
    if expected_names is None or expected_samples is None:
        raise RuntimeError("completed batch contains no reference dimension metadata")
    return {
        "annotation_bin_count": len(expected_names),
        "complete_case_samples": expected_samples,
    }


def _compare_npz_files(
    reference: Path, candidate: Path, *, rtol: float, atol: float
) -> dict[str, Any]:
    # Imported only after both scientific children have exited.  The controller
    # never shares a NumPy/BLAS image with a running SUMMIT process.
    import numpy as np

    arrays: dict[str, Any] = {}
    with np.load(reference, allow_pickle=False) as left, np.load(
        candidate, allow_pickle=False
    ) as right:
        if left.files != right.files:
            raise RuntimeError("jackknife NPZ key schema/order differs")
        for name in left.files:
            left_array = np.asarray(left[name])
            right_array = np.asarray(right[name])
            if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
                raise RuntimeError(f"jackknife NPZ array schema differs for {name}")
            if left_array.dtype.kind in "US":
                if not np.array_equal(left_array, right_array):
                    raise RuntimeError(f"jackknife label order differs for {name}")
                arrays[name] = {
                    "shape": list(left_array.shape), "dtype": left_array.dtype.str,
                    "labels_equal_in_order": True,
                }
                continue
            if left_array.dtype.kind not in "fiu":
                raise RuntimeError(f"unsupported jackknife NPZ dtype for {name}")
            accumulator = _ErrorAccumulator(rtol=rtol, atol=atol)
            for index, (left_value, right_value) in enumerate(
                zip(left_array.flat, right_array.flat, strict=True)
            ):
                accumulator.add(left_value, right_value, f"{name}[{index}]")
            arrays[name] = {
                "shape": list(left_array.shape), "dtype": left_array.dtype.str,
                **accumulator.record(),
            }
    return {"key_schema_and_order_equal": True, "arrays": arrays}


def _metric_records(value: Any):
    if isinstance(value, Mapping):
        if "maximum_normalized_error" in value:
            yield value
        for child in value.values():
            yield from _metric_records(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _metric_records(child)


def _compare_reference_bundle(
    reference_path: Path, reference: dict[str, Any],
    candidate_path: Path, candidate: dict[str, Any], *,
    rtol: float, atol: float,
) -> dict[str, Any]:
    invariant_keys = (
        "kind", "schema_version", "analysis_fingerprint", "variant_digest",
        "n_samples", "fixed_effect_rank_excluding_intercept", "residual_rank",
        "environment", "environment_transform", "covariates", "kernel_mode",
        "feature_convention", "feature_convention_version", "genotype_scale",
        "ld_scale", "null_corrected", "annotation_names",
    )
    disagreements = [
        key for key in invariant_keys if reference.get(key) != candidate.get(key)
    ]
    if disagreements:
        raise RuntimeError(
            "reference scientific identity/schema differs on: "
            + ", ".join(disagreements)
        )
    reference_genotype = reference.get("genotype_files")
    candidate_genotype = candidate.get("genotype_files")
    if (
        not isinstance(reference_genotype, Mapping)
        or set(reference_genotype) != {".bed", ".bim", ".fam"}
        or not isinstance(candidate_genotype, Mapping)
        or candidate_genotype != reference_genotype
    ):
        raise RuntimeError(
            "reference genotype_files SHA/size identity objects differ"
        )
    reference_resources = reference.get("resource_estimates")
    candidate_resources = candidate.get("resource_estimates")
    if not isinstance(reference_resources, Mapping) or not isinstance(
        candidate_resources, Mapping
    ):
        raise RuntimeError("reference comparison lacks resource_estimates")
    missing_resource_diagnostics = [
        name for name in RESOURCE_DIAGNOSTIC_ALLOWLIST
        if name not in reference_resources or name not in candidate_resources
    ]
    if missing_resource_diagnostics:
        raise RuntimeError(
            "reference resource_estimates lacks required diagnostics: "
            + ", ".join(missing_resource_diagnostics)
        )
    exact_integrity_fields = RESOURCE_DIAGNOSTIC_ALLOWLIST[2:]
    integrity_diagnostics = {}
    for name in exact_integrity_fields:
        left = reference_resources[name]
        right = candidate_resources[name]
        if type(left) is not type(right) or left != right:
            raise RuntimeError(
                f"reference integrity diagnostic {name!r} differs"
            )
        integrity_diagnostics[name] = left
    leakage_diagnostics = _compare_json_tree(
        {
            name: reference_resources[name]
            for name in RESOURCE_DIAGNOSTIC_ALLOWLIST[:2]
        },
        {
            name: candidate_resources[name]
            for name in RESOURCE_DIAGNOSTIC_ALLOWLIST[:2]
        },
        label="resource_estimates",
        rtol=rtol,
        atol=atol,
    )
    reference_files = reference.get("files")
    candidate_files = candidate.get("files")
    reference_hashes = reference.get("artifact_sha256")
    candidate_hashes = candidate.get("artifact_sha256")
    if (
        not isinstance(reference_files, Mapping)
        or not isinstance(candidate_files, Mapping)
        or not isinstance(reference_hashes, Mapping)
        or not isinstance(candidate_hashes, Mapping)
    ):
        raise RuntimeError("reference comparison lacks artifact declarations")
    if list(reference_files) != list(candidate_files):
        raise RuntimeError("declared artifact role schema/order differs")
    if set(reference_files) != set(reference_hashes) or set(candidate_files) != set(
        candidate_hashes
    ):
        raise RuntimeError("reference artifact file/hash role declarations differ")
    score_metrics = {}
    for family in SCORE_FAMILIES:
        score_metrics[family] = _compare_tsv_files(
            _declared_artifact(reference_path, reference, family),
            _declared_artifact(candidate_path, candidate, family),
            key_columns=3, rtol=rtol, atol=atol,
        )
    diagonal = _compare_tsv_files(
        _declared_artifact(reference_path, reference, "diagonal"),
        _declared_artifact(candidate_path, candidate, "diagonal"),
        key_columns=5, rtol=rtol, atol=atol,
    )
    json_diagnostics = {}
    for key in (
        "annotation_masses", "feature_diagnostics", "trace_nxe", "trace_nxe_sq",
        "jackknife",
    ):
        if key not in reference or key not in candidate:
            raise RuntimeError(f"required final reference diagnostic {key!r} is missing")
        json_diagnostics[key] = _compare_json_tree(
            reference[key], candidate[key], label=key, rtol=rtol, atol=atol
        )
    if "population_trace" not in reference or "population_trace" not in candidate:
        raise RuntimeError("required final reference diagnostic 'population_trace' is missing")
    json_diagnostics["population_trace"] = _compare_population_trace(
        reference["population_trace"], candidate["population_trace"],
        rtol=rtol, atol=atol,
    )
    left_jackknife = reference.get("files", {}).get("jackknife")
    right_jackknife = candidate.get("files", {}).get("jackknife")
    if bool(left_jackknife) != bool(right_jackknife):
        raise RuntimeError("jackknife final artifact availability differs")
    jackknife_arrays = None
    if left_jackknife:
        jackknife_arrays = _compare_npz_files(
            _declared_artifact(reference_path, reference, "jackknife"),
            _declared_artifact(candidate_path, candidate, "jackknife"),
            rtol=rtol, atol=atol,
        )
    known_roles = {*SCORE_FAMILIES, "diagonal", "jackknife"}
    additional_artifacts = {}
    for role in reference_files:
        if role in known_roles:
            continue
        left = _declared_artifact(reference_path, reference, role)
        right = _declared_artifact(candidate_path, candidate, role)
        left_sha256 = _sha256(left)
        right_sha256 = _sha256(right)
        additional_artifacts[role] = {
            "reference_bytes": left.stat().st_size,
            "candidate_bytes": right.stat().st_size,
            "reference_sha256": left_sha256,
            "candidate_sha256": right_sha256,
            "byte_identical": (
                left.stat().st_size == right.stat().st_size
                and left_sha256 == right_sha256
            ),
        }
    result = {
        "environment": reference["environment"],
        "reference_schema_version": reference["schema_version"],
        "scientific_identity_and_order_equal": True,
        "score_families": score_metrics,
        "diagonal_diagnostics": diagonal,
        "final_manifest_diagnostics": json_diagnostics,
        "jackknife_arrays": jackknife_arrays,
        "declared_artifact_roles_equal_in_order": True,
        "additional_declared_artifacts": additional_artifacts,
        "genotype_files_sha_size_identity_equal": True,
        "resource_estimate_diagnostics": {
            "explicit_allowlist": list(RESOURCE_DIAGNOSTIC_ALLOWLIST),
            "leakage": leakage_diagnostics,
            "integrity_counters_exactly_equal": True,
            "integrity_values": integrity_diagnostics,
        },
    }
    metrics = list(_metric_records(result))
    result["maximum_normalized_error"] = max(
        (float(item["maximum_normalized_error"]) for item in metrics), default=0.0
    )
    result["accuracy_gate_passed"] = all(
        item.get("within_declared_tolerance") is True for item in metrics
    ) and all(
        item["byte_identical"] for item in additional_artifacts.values()
    )
    return result


def _phase_comparison(
    current: Mapping[str, Any], optimized: Mapping[str, Any]
) -> list[dict[str, Any]]:
    current_phases = current.get("phase_totals", {})
    optimized_phases = optimized.get("phase_totals", {})
    if set(current_phases) != set(optimized_phases):
        raise RuntimeError("layout runs report different final phase sets")
    result = []
    for phase in sorted(current_phases):
        left = float(current_phases[phase].get("wall_seconds", 0.0))
        right = float(optimized_phases[phase].get("wall_seconds", 0.0))
        result.append(
            {
                "phase": phase, "current_wall_seconds": left,
                "optimized_wall_seconds": right,
                "optimized_minus_current_seconds": right - left,
                "current_over_optimized_speedup": left / right if right > 0.0 else None,
            }
        )
    return result


def _compare_completed_runs(
    current_path: Path, current: dict[str, Any],
    optimized_path: Path, optimized: dict[str, Any], *,
    environment_order: Sequence[str], rtol: float, atol: float,
) -> dict[str, Any]:
    invariant_keys = (
        "kind", "schema_version", "execution", "requested_backend",
        "protected_native_gemm", "arithmetic_dtype", "requested_storage_dtype",
        "gemm_backend", "gemm_backend_build_sha256", "native_source_commit",
        "native_source_tree_sha256", "native_gemm_integrity_enabled",
        "native_blas_runtime_isolation", "num_environments",
        "common_complete_case_samples", "num_variants", "randomization",
        "shared_genotype_passes", "environment_tiles",
    )
    disagreements = [
        key for key in invariant_keys if current.get(key) != optimized.get(key)
    ]
    if disagreements:
        raise RuntimeError(
            "current and optimized controls differ on: " + ", ".join(disagreements)
        )
    if _canonical_layout(current.get("full_precision_layout", "")) != "current":
        raise RuntimeError("comparison reference did not use the current layout")
    if _canonical_layout(optimized.get("full_precision_layout", "")) != (
        "source_tt_target_current"
    ):
        raise RuntimeError("comparison candidate did not use the optimized FP64 layout")
    left_references = _reference_manifests(
        current_path, current, environment_order
    )
    right_references = _reference_manifests(
        optimized_path, optimized, environment_order
    )
    environments = []
    for environment in environment_order:
        environments.append(
            _compare_reference_bundle(
                *left_references[environment], *right_references[environment],
                rtol=rtol, atol=atol,
            )
        )
    metrics = list(_metric_records(environments))
    return {
        "reference_layout": "current",
        "candidate_layout": "source_tt_target_current",
        "controlled_batch_identity_equal": True,
        "tolerances": {"rtol": rtol, "atol": atol},
        "environments": environments,
        "maximum_normalized_error": max(
            (float(item["maximum_normalized_error"]) for item in metrics), default=0.0
        ),
        "accuracy_gate_passed": all(
            environment["accuracy_gate_passed"] for environment in environments
        ),
        "phase_runtime_comparison": _phase_comparison(
            current["performance_telemetry"], optimized["performance_telemetry"]
        ),
    }


def _block_width_phase_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
    baseline_width: int, candidate_width: int,
) -> list[dict[str, Any]]:
    baseline_phases = baseline.get("phase_totals", {})
    candidate_phases = candidate.get("phase_totals", {})
    if set(baseline_phases) != set(candidate_phases):
        raise RuntimeError("block-width runs report different final phase sets")
    result = []
    for phase in sorted(baseline_phases):
        left = float(baseline_phases[phase].get("wall_seconds", 0.0))
        right = float(candidate_phases[phase].get("wall_seconds", 0.0))
        result.append(
            {
                "phase": phase,
                "baseline_block_width": baseline_width,
                "candidate_block_width": candidate_width,
                "baseline_wall_seconds": left,
                "candidate_wall_seconds": right,
                "candidate_minus_baseline_seconds": right - left,
                "baseline_over_candidate_speedup": (
                    left / right if right > 0.0 else None
                ),
            }
        )
    return result


def _compare_completed_block_width_runs(
    baseline_path: Path, baseline: dict[str, Any],
    candidate_path: Path, candidate: dict[str, Any], *,
    baseline_width: int, candidate_width: int,
    environment_order: Sequence[str], rtol: float, atol: float,
) -> dict[str, Any]:
    invariant_keys = (
        "kind", "schema_version", "execution", "requested_backend",
        "protected_native_gemm", "arithmetic_dtype", "requested_storage_dtype",
        "gemm_backend", "gemm_backend_build_sha256", "native_source_commit",
        "native_source_tree_sha256", "native_gemm_integrity_enabled",
        "native_blas_runtime_isolation", "num_environments",
        "common_complete_case_samples", "num_variants", "randomization",
        "shared_genotype_passes", "environment_tiles",
    )
    disagreements = [
        key for key in invariant_keys if baseline.get(key) != candidate.get(key)
    ]
    if disagreements:
        raise RuntimeError(
            "block-width controls differ on: " + ", ".join(disagreements)
        )
    for label, payload in (("baseline", baseline), ("candidate", candidate)):
        if _canonical_layout(payload.get("full_precision_layout", "")) != "current":
            raise RuntimeError(
                f"{label} block-width child did not use the current layout"
            )
    left_references = _reference_manifests(
        baseline_path, baseline, environment_order
    )
    right_references = _reference_manifests(
        candidate_path, candidate, environment_order
    )
    environments = []
    for environment in environment_order:
        left_path, left = left_references[environment]
        right_path, right = right_references[environment]
        if list(left) != list(right):
            raise RuntimeError("reference manifest key schema/order differs")
        left_randomization = left.get("randomization")
        right_randomization = right.get("randomization")
        if not isinstance(left_randomization, Mapping) or not isinstance(
            right_randomization, Mapping
        ):
            raise RuntimeError("reference manifest lacks randomization controls")
        if int(left_randomization.get("step_size", -1)) != baseline_width:
            raise RuntimeError("baseline reference reports the wrong block width")
        if int(right_randomization.get("step_size", -1)) != candidate_width:
            raise RuntimeError("candidate reference reports the wrong block width")
        left_control = {
            key: value for key, value in left_randomization.items()
            if key != "step_size"
        }
        right_control = {
            key: value for key, value in right_randomization.items()
            if key != "step_size"
        }
        if left_control != right_control:
            raise RuntimeError(
                "reference randomization controls differ beyond block width"
            )
        observed = _compare_reference_bundle(
            left_path, left, right_path, right, rtol=rtol, atol=atol
        )
        observed["reference_manifest_key_schema_equal_in_order"] = True
        observed["randomization_equal_except_step_size"] = True
        environments.append(observed)
    metrics = list(_metric_records(environments))
    return {
        "layout": "current",
        "baseline_block_width": baseline_width,
        "candidate_block_width": candidate_width,
        "controlled_batch_identity_equal": True,
        "same_literal_subset_probes_environments_and_seed": True,
        "declared_outputs_and_allowlisted_final_diagnostics_compared": True,
        "final_diagnostic_allowlist": [
            "annotation_masses", "feature_diagnostics", "trace_nxe",
            "trace_nxe_sq", "jackknife", "population_trace",
            *RESOURCE_DIAGNOSTIC_ALLOWLIST,
        ],
        "tolerances": {"rtol": rtol, "atol": atol},
        "environments": environments,
        "maximum_normalized_error": max(
            (float(item["maximum_normalized_error"]) for item in metrics),
            default=0.0,
        ),
        "accuracy_gate_passed": all(
            environment["accuracy_gate_passed"] for environment in environments
        ),
        "phase_runtime_comparison": _block_width_phase_comparison(
            baseline["performance_telemetry"],
            candidate["performance_telemetry"],
            baseline_width=baseline_width,
            candidate_width=candidate_width,
        ),
    }


def _compare_completed_blis_outputs(
    reference_path: Path,
    reference: dict[str, Any],
    candidate_path: Path,
    candidate: dict[str, Any],
    *,
    environment_order: Sequence[str],
    rtol: float,
    atol: float,
    reference_run: str,
    candidate_run: str,
    reference_threads: int,
    candidate_threads: int,
    acceptance_reference: bool,
) -> dict[str, Any]:
    if acceptance_reference:
        if reference_threads != 1 or candidate_threads != 32:
            raise RuntimeError(
                "private-BLIS acceptance requires a T1 numerical reference "
                "and T32 selection candidate"
            )
        if rtol != DEFAULT_COMPARISON_RTOL or atol != DEFAULT_COMPARISON_ATOL:
            raise RuntimeError(
                "private-BLIS acceptance requires the predeclared "
                "rtol=atol=5e-12 accuracy gate"
            )
    elif reference_threads != candidate_threads:
        raise RuntimeError(
            "private-BLIS determinism comparison requires equal thread counts"
        )
    for label, payload in (
        (reference_run, reference), (candidate_run, candidate)
    ):
        repairs = payload.get("repaired_gemm_output_columns")
        if payload.get("native_gemm_integrity_enabled") is not True:
            raise RuntimeError(
                f"private-BLIS comparison {label} lacks enabled integrity"
            )
        if type(repairs) is not int or repairs != 0:
            raise RuntimeError(
                f"private-BLIS comparison {label} did not report zero repairs"
            )
    invariant_keys = (
        "kind", "schema_version", "execution", "requested_backend",
        "full_precision_layout", "protected_native_gemm", "arithmetic_dtype",
        "requested_storage_dtype", "gemm_backend",
        "gemm_backend_build_sha256", "native_source_commit",
        "native_source_tree_sha256", "native_gemm_integrity_enabled",
        "native_blas_runtime_isolation", "num_environments",
        "common_complete_case_samples", "num_variants", "randomization",
        "shared_genotype_passes", "environment_tiles",
    )
    disagreements = [
        key for key in invariant_keys
        if reference.get(key) != candidate.get(key)
    ]
    if disagreements:
        raise RuntimeError(
            "private-BLIS comparison controls differ on: "
            + ", ".join(disagreements)
        )
    if _canonical_layout(reference.get("full_precision_layout", "")) != "current":
        raise RuntimeError("private-BLIS numerical reference is not current-layout")
    left_references = _reference_manifests(
        reference_path, reference, environment_order
    )
    right_references = _reference_manifests(
        candidate_path, candidate, environment_order
    )
    environments = []
    for environment in environment_order:
        left_path, left = left_references[environment]
        right_path, right = right_references[environment]
        if list(left) != list(right):
            raise RuntimeError(
                "private-BLIS comparison reference manifest key schema/order differs"
            )
        environments.append(
            _compare_reference_bundle(
                left_path, left, right_path, right, rtol=rtol, atol=atol
            )
        )
    metrics = list(_metric_records(environments))
    reference_phases = reference["performance_telemetry"].get("phase_totals", {})
    candidate_phases = candidate["performance_telemetry"].get("phase_totals", {})
    if set(reference_phases) != set(candidate_phases):
        raise RuntimeError("private-BLIS comparison reports different phase sets")
    phase_runtimes = [
        {
            "phase": phase,
            "reference_wall_seconds": float(
                reference_phases[phase].get("wall_seconds", 0.0)
            ),
            "candidate_wall_seconds": float(
                candidate_phases[phase].get("wall_seconds", 0.0)
            ),
        }
        for phase in sorted(reference_phases)
    ]
    return {
        "reference_run": reference_run,
        "candidate_run": candidate_run,
        "backend": EXPECTED_PRIVATE_BLAS_NAME,
        "layout": "current",
        "fresh_exec_children": 2,
        "children_sequential": True,
        "reference_threads": reference_threads,
        "candidate_threads": candidate_threads,
        "same_literal_subset_probes_environments_seed_layout_and_block_width": True,
        "unsafe_openblas_used_as_reference": False,
        "artifact_level_reference": (
            "fresh_private_blis_t1_numerical_reference"
            if acceptance_reference
            else "fresh_private_blis_same_thread_determinism_reference"
        ),
        "acceptance_reference": bool(acceptance_reference),
        "declared_outputs_and_allowlisted_final_diagnostics_compared": True,
        "tolerances": {"rtol": rtol, "atol": atol},
        "environments": environments,
        "maximum_normalized_error": max(
            (float(item["maximum_normalized_error"]) for item in metrics),
            default=0.0,
        ),
        "accuracy_gate_passed": all(
            environment["accuracy_gate_passed"] for environment in environments
        ),
        "phase_runtime_comparison": phase_runtimes,
    }


def _compact_performance(performance: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version", "backend", "backend_build_sha256",
        "native_source_commit", "native_source_tree_sha256", "arithmetic_dtype",
        "requested_storage_dtype", "full_precision_layout",
        "requested_blas_threads", "vendor_call_telemetry_complete",
        "hot_gemm_telemetry_complete", "optimized_fp64_layout_telemetry_complete",
        "numa_bound_bed_decode_required",
        "numa_bound_bed_decode_complete",
        "repaired_gemm_output_columns", "optimized_fp64_layout_zero_repair",
        "telemetry_complete", "phase_telemetry_complete", "capture_boundary",
        "dropped_gemm_records", "native_telemetry_errors", "gemm_record_count",
        "gemm_wall_seconds", "gemm_process_cpu_seconds", "gemm_average_active_cores",
        "gemm_matrix_minutes", "controller_hot_vendor_numa_validation",
        "controller_current_vendor_acceptance",
        "controller_early_numa_attestation_requirement",
        "controller_numa_bound_bed_decode_validation",
        "controller_native_integrity_snapshot_numa_validation",
        "controller_native_gemm_output_numa_validation",
        "controller_gemm_calling_thread_affinity_validation",
        "phase_totals", "estimator_phase_totals",
    )
    return {key: performance.get(key) for key in keys}


def _projection_speedups(
    current: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    result = {}
    for key in ("full_m_b32", "full_m_b1024_same_tile_schedule"):
        left = float(current[key]["projected_seconds"])
        right = float(optimized[key]["projected_seconds"])
        result[key] = {
            "current_projected_seconds": left,
            "optimized_projected_seconds": right,
            "current_over_optimized_speedup": left / right if right > 0.0 else None,
        }
    return result


def _block_width_projection_speedups(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
    baseline_width: int, candidate_width: int,
) -> dict[str, Any]:
    result = {}
    for key in ("full_m_b32", "full_m_b1024_same_tile_schedule"):
        left = float(baseline[key]["projected_seconds"])
        right = float(candidate[key]["projected_seconds"])
        result[key] = {
            "baseline_block_width": baseline_width,
            "candidate_block_width": candidate_width,
            "baseline_projected_seconds": left,
            "candidate_projected_seconds": right,
            "baseline_over_candidate_speedup": (
                left / right if right > 0.0 else None
            ),
        }
    return result


def _batch_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "kind", "schema_version", "execution", "requested_backend",
        "full_precision_layout", "protected_native_gemm", "arithmetic_dtype",
        "requested_storage_dtype", "gemm_backend", "gemm_backend_build_sha256",
        "native_source_commit", "native_source_tree_sha256",
        "native_gemm_integrity_enabled", "native_blas_runtime_isolation",
        "repaired_gemm_output_columns", "source_panel_memory_order",
        "target_panel_memory_order", "target_genotype_memory_order",
        "source_to_target_layout_transition",
        "source_to_target_layout_copy_count",
        "source_to_target_layout_copy_total_gib",
        "source_to_target_layout_copy_max_tile_gib",
        "modeled_packed_source_panel_gib",
        "modeled_layout_conversion_live_peak_gib",
        "modeled_target_pair_sealing_live_peak_gib",
        "num_environments", "common_complete_case_samples", "num_variants",
        "shared_genotype_passes", "environment_tiles", "fused_gemm_calls",
        "fused_gemm_shapes", "fused_gemm_total_flops",
        "peak_process_rss_gib_at_manifest", "peak_rss_scope",
        "aggregate_genotype_passes", "environment_groups",
        "cpu_placement", "cpu_placements", "cpu_placement_complete",
        "numa_bound_bed_decode_required",
        "numa_bound_bed_decode_complete",
    )
    return {key: payload.get(key) for key in keys}


def _redacted_command(command: Sequence[str], temporary_root: Path) -> list[str]:
    return [
        _redact_temporary(token, temporary_root)
        if token != _CLI_BOOTSTRAP else "<validated_cli_bootstrap>"
        for token in command
    ]


def _comparison_command_signature(command: Sequence[str]) -> str:
    controlled = list(command)
    for option in ("--gxe-fp64-layout", "--out"):
        try:
            index = controlled.index(option)
        except ValueError as exc:
            raise RuntimeError(f"comparison child command lacks {option}") from exc
        if index + 1 >= len(controlled):
            raise RuntimeError(f"comparison child command has no value for {option}")
        controlled[index + 1] = f"<{option[2:]}-controlled-difference>"
    encoded = json.dumps(controlled, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _block_width_command_signature(command: Sequence[str]) -> str:
    controlled = list(command)
    for option in ("--step_size", "--out"):
        try:
            index = controlled.index(option)
        except ValueError as exc:
            raise RuntimeError(
                f"block-width comparison child command lacks {option}"
            ) from exc
        if index + 1 >= len(controlled):
            raise RuntimeError(
                f"block-width comparison command has no value for {option}"
            )
        controlled[index + 1] = f"<{option[2:]}-controlled-difference>"
    try:
        layout_index = controlled.index("--gxe-fp64-layout") + 1
    except ValueError as exc:
        raise RuntimeError("block-width comparison command lacks its layout") from exc
    if controlled[layout_index] != "current":
        raise RuntimeError("block-width comparison command is not current-layout")
    encoded = json.dumps(controlled, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blis_replicate_command_signature(command: Sequence[str]) -> str:
    controlled = list(command)
    try:
        output_index = controlled.index("--out") + 1
        layout_index = controlled.index("--gxe-fp64-layout") + 1
    except ValueError as exc:
        raise RuntimeError(
            "private-BLIS replicate command lacks output/layout controls"
        ) from exc
    if output_index >= len(controlled) or layout_index >= len(controlled):
        raise RuntimeError("private-BLIS replicate command is truncated")
    if controlled[layout_index] != "current":
        raise RuntimeError("private-BLIS replicate command is not current-layout")
    if "--gxe-explicit-openmp-placement" not in controlled:
        raise RuntimeError("private-BLIS replicate lacks explicit placement")
    controlled[output_index] = "<out-controlled-difference>"
    encoded = json.dumps(controlled, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blis_t1_reference_command_signature(command: Sequence[str]) -> str:
    controlled = list(command)
    if len(controlled) < 4 or controlled[1] != "-c":
        raise RuntimeError("BLIS T1/T32 command lacks its taskset CPU control")
    controlled[2] = "<taskset-cpus-controlled-difference>"
    for option in ("--num-threads", "--out"):
        try:
            index = controlled.index(option) + 1
        except ValueError as exc:
            raise RuntimeError(
                f"BLIS T1/T32 comparison command lacks {option}"
            ) from exc
        if index >= len(controlled):
            raise RuntimeError(f"BLIS T1/T32 command has no value for {option}")
        controlled[index] = f"<{option[2:]}-controlled-difference>"
    if "--gxe-explicit-openmp-placement" not in controlled:
        raise RuntimeError("BLIS T1/T32 command lacks explicit placement")
    try:
        memory_scope = controlled.index(
            "--gxe-explicit-openmp-memory-scope"
        ) + 1
        numa_nodes = controlled.index("--numa-nodes") + 1
    except ValueError as exc:
        raise RuntimeError(
            "BLIS T1/T32 command lacks selected-socket memory scope"
        ) from exc
    if (
        memory_scope >= len(controlled)
        or controlled[memory_scope] != EXPLICIT_OPENMP_MEMORY_SCOPE
        or numa_nodes >= len(controlled)
        or controlled[numa_nodes] != "0-3"
    ):
        raise RuntimeError(
            "BLIS T1/T32 command does not share exact socket0 NUMA nodes 0-3"
        )
    encoded = json.dumps(controlled, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execute_layout(
    args: argparse.Namespace, *, layout: str, subset: PlinkMetadata,
    subset_prefix: Path, output_prefix: Path, python: Path,
    cpu_records: Sequence[CpuRecord], child_environment: dict[str, str],
    native_sha256: str, temporary_root: Path, source_variants: int,
    timeout_seconds: float,
    block_width: int | None = None,
    explicit_memory_nodes: Sequence[int] | None = None,
) -> dict[str, Any]:
    selected_block_width = args.block_width if block_width is None else int(block_width)
    command = _command(
        args, subset_prefix=subset_prefix, output_prefix=output_prefix,
        python=python, cpu_records=cpu_records, full_precision_layout=layout,
        block_width=selected_block_width,
        explicit_memory_nodes=explicit_memory_nodes,
    )
    completed, elapsed, maxrss_gib, child_pid = _run_bounded(
        command, child_environment, timeout_seconds
    )
    process = {
        "returncode": completed.returncode,
        "child_pid": child_pid,
        "wall_seconds": elapsed,
        "controller_children_ru_maxrss_gib_high_water": maxrss_gib,
        "controller_children_ru_maxrss_scope": (
            "RUSAGE_CHILDREN process-lifetime high-water; cumulative across "
            "sequential children and not attributable to this layout"
        ),
        "authoritative_per_layout_peak_rss_field": (
            "batch_summary.peak_process_rss_gib_at_manifest"
        ),
        "stdout": _bounded_text_record(completed.stdout, temporary_root),
        "stderr": _bounded_text_record(completed.stderr, temporary_root),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"partial real-genotype {layout} CLI failed with status "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    canonical_manifest_path = Path(f"{output_prefix}.gxe.multi.json")
    if canonical_manifest_path.is_symlink() or not canonical_manifest_path.is_file():
        raise RuntimeError(f"completed {layout} child lacks its batch manifest")
    canonical_payload = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    explicit = _resolve_single_explicit_blis_group(
        canonical_manifest_path,
        canonical_payload,
        subset=subset,
        args=args,
        native_sha256=native_sha256,
        cpu_records=cpu_records,
        outer_child_pid=child_pid,
        expected_numa_nodes=explicit_memory_nodes,
    )
    manifest_path = explicit["group_path"]
    payload = explicit["group_payload"]
    process["outer_child_pid"] = child_pid
    process["placement_worker_pid"] = explicit["worker_pid"]
    process["fresh_exec_worker_distinct_from_outer_child"] = True
    environment_order = [
        item.strip() for item in args.environment_columns.split(",")
    ]
    reference_dimensions = _batch_reference_dimensions(
        manifest_path, payload, environment_order
    )
    performance = _validate_completed_manifest(
        payload, subset=subset, args=args, native_sha256=native_sha256,
        cpu_records=cpu_records, full_precision_layout=layout,
        block_width=selected_block_width,
        annotation_bin_count=reference_dimensions["annotation_bin_count"],
        reference_sample_count=reference_dimensions["complete_case_samples"],
        expected_child_pid=explicit["worker_pid"],
        require_explicit_blis_placement=True,
        expected_placement=explicit["placement"],
        expected_numa_nodes=explicit_memory_nodes,
        integrity_minimum_vendor_flops=explicit[
            "integrity_minimum_vendor_flops"
        ],
        native_integrity_snapshot_numa_query_chunk_page_limit=explicit[
            "snapshot_query_chunk_page_limit"
        ],
        native_gemm_output_numa_query_chunk_page_limit=explicit[
            "output_query_chunk_page_limit"
        ],
        native_gemm_output_numa_evidence_capacity=explicit[
            "output_evidence_capacity"
        ],
    )
    projections = _projections(
        elapsed=elapsed, subset_variants=subset.variants,
        full_variants=source_variants, block_width=selected_block_width,
        measured_probes=args.probes, manifest=payload, performance=performance,
    )
    artifact_identities = _artifact_identities(manifest_path, payload)
    artifact_identities.append(
        {
            "role": "canonical_batch_manifest",
            "name": canonical_manifest_path.name,
            "bytes": canonical_manifest_path.stat().st_size,
            "sha256": _sha256(canonical_manifest_path),
        }
    )
    return {
        "layout": _canonical_layout(layout), "command": command,
        "block_schedule": _block_schedule(subset.variants, selected_block_width),
        "process": process, "manifest_path": manifest_path, "payload": payload,
        "canonical_manifest_path": canonical_manifest_path,
        "canonical_payload": canonical_payload,
        "cpu_placement": explicit["placement"],
        "performance": performance, "projections": projections,
        "gemm_summaries": _gemm_summaries(performance["gemm_records"]),
        "temporary_artifact_identities": artifact_identities,
    }


def _atomic_json_no_replace(payload: dict[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing report: {output}")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o600)
        os.link(staging, output)
        directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.allow_integrity_disabled:
        raise RuntimeError(
            "API-9 private-BLIS real-genotype execution requires integrity checks"
        )
    if args.compare_fp64_layouts:
        raise RuntimeError(
            "--compare-fp64-layouts is disabled at the execution boundary: "
            "rejected API-8 orientations cannot produce real-data evidence"
        )
    block_width_comparison = args.compare_block_widths is not None
    blis_replicate_comparison = bool(args.compare_blis_replicates)
    blis_t1_reference_comparison = bool(args.compare_blis_t1_reference)
    if blis_t1_reference_comparison:
        if (
            args.threads != 32
            or args.blocks != 4
            or args.block_width != 2000
            or args.probes != 32
        ):
            raise RuntimeError(
                "private-BLIS acceptance requires a T32 candidate, four full "
                "K=2000 genotype blocks (the first 8000 variants), and B=32"
            )
        if (
            args.cpu_list == "auto"
            or _parse_integer_ranges(args.cpu_list) != list(range(32))
        ):
            raise RuntimeError(
                "private-BLIS acceptance requires candidate CPUs 0-31 and "
                "T1 reference CPU0"
            )
        if (
            args.comparison_rtol != DEFAULT_COMPARISON_RTOL
            or args.comparison_atol != DEFAULT_COMPARISON_ATOL
        ):
            raise RuntimeError(
                "private-BLIS acceptance requires the predeclared "
                "rtol=atol=5e-12 accuracy gate"
            )
        if (
            not 0.0 < args.reference_timeout_seconds
            <= MAX_BLIS_T1_REFERENCE_CHILD_SECONDS
            or not 0.0 < args.timeout_seconds
            <= MAX_BLIS_T32_CANDIDATE_CHILD_SECONDS
            or not 0.0 < args.controller_timeout_seconds
            <= MAX_BLIS_T1_COMPARISON_CONTROLLER_SECONDS
        ):
            raise RuntimeError(
                "private-BLIS T1/T32 execution exceeds its 420s/240s/600s "
                "hard caps"
            )
    comparison_active = (
        args.compare_fp64_layouts
        or block_width_comparison
        or blis_replicate_comparison
        or blis_t1_reference_comparison
    )
    comparison_controller_started = (
        time.perf_counter()
        if comparison_active and not args.dry_run else None
    )
    source = _validate_plink_prefix(args.geno_prefix)
    subset_variants = (
        args.subset_variants
        if block_width_comparison else args.blocks * args.block_width
    )
    if subset_variants > source.variants:
        raise ValueError(
            f"source has M={source.variants}, fewer than the requested {subset_variants} variants"
        )
    environment_path = _regular_input(args.environment_file, "environment file")
    covariate_path = _regular_input(args.covariate_file, "covariate file")
    install, native, archive, python, dependencies = _native_inputs(args)
    installed_package_identity = _installed_package_identity(install)
    benchmark_script = Path(__file__).resolve()
    cpu_records = _resolve_cpu_records(args.cpu_list, args.threads)
    explicit_memory_nodes = None
    if blis_t1_reference_comparison:
        if (
            [record.cpu for record in cpu_records] != list(range(32))
            or {record.socket for record in cpu_records} != {0}
        ):
            raise RuntimeError(
                "private-BLIS acceptance did not resolve candidate CPUs 0-31 "
                "on socket0"
            )
        explicit_memory_nodes = _full_socket_numa_nodes(cpu_records)
        if explicit_memory_nodes != [0, 1, 2, 3]:
            raise RuntimeError(
                "private-BLIS acceptance requires verified socket0 NUMA nodes 0-3"
            )
    source_paths = [Path(f"{source.prefix}{extension}") for extension in (".bed", ".bim", ".fam")]
    immutable_inputs = [
        *source_paths,
        environment_path,
        covariate_path,
        native,
        archive,
        benchmark_script,
    ]
    initial_identities = {str(path): _file_identity(path) for path in immutable_inputs}
    temporary_parent = (
        None if args.temporary_parent is None else str(args.temporary_parent.expanduser().resolve())
    )
    report: dict[str, Any]
    with tempfile.TemporaryDirectory(
        prefix="summit-gxe-partial-real-", dir=temporary_parent
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        subset_prefix = temporary_root / "plink_subset" / "genotype"
        subset, subset_hashes = _create_prefix_subset(
            source, subset_prefix, subset_variants
        )
        output_root = temporary_root / "reference_outputs"
        output_root.mkdir(mode=0o700)
        child_environment, environment_settings = _child_environment(
            args, install, native, dependencies, cpu_records
        )
        if environment_settings[
            "SUMMIT_GXE_EXPECTED_PACKAGE_MANIFEST_SHA256"
        ] != installed_package_identity["manifest_sha256"]:
            raise RuntimeError("installed summit package changed during setup")
        native_sha256 = environment_settings["SUMMIT_GXE_EXPECTED_NATIVE_SHA256"]
        if block_width_comparison:
            run_specs = tuple(
                (str(width), "current", width)
                for width in args.compare_block_widths
            )
        elif blis_replicate_comparison:
            run_specs = (
                ("replicate_1", "current", args.block_width),
                ("replicate_2", "current", args.block_width),
            )
        elif blis_t1_reference_comparison:
            run_specs = (
                ("reference_t1", "current", args.block_width),
                ("candidate_t32", "current", args.block_width),
            )
        elif args.compare_fp64_layouts:
            run_specs = tuple(
                (_canonical_layout(layout), layout, args.block_width)
                for layout in ("current", "source-tt-target-current")
            )
        else:
            run_specs = (
                (
                    _canonical_layout(args.full_precision_layout),
                    args.full_precision_layout,
                    args.block_width,
                ),
            )
        run_args = {key: args for key, _layout, _width in run_specs}
        run_cpu_records = {
            key: cpu_records for key, _layout, _width in run_specs
        }
        run_environments = {
            key: child_environment for key, _layout, _width in run_specs
        }
        run_environment_settings = {
            key: environment_settings for key, _layout, _width in run_specs
        }
        run_explicit_memory_nodes = {
            key: (
                list(explicit_memory_nodes)
                if explicit_memory_nodes is not None else None
            )
            for key, _layout, _width in run_specs
        }
        if blis_t1_reference_comparison:
            reference_args = argparse.Namespace(**vars(args))
            reference_args.threads = 1
            reference_cpu_records = cpu_records[:1]
            if (
                len(reference_cpu_records) != 1
                or reference_cpu_records[0].cpu != 0
                or reference_cpu_records[0].socket != 0
                or reference_cpu_records[0].node != 0
            ):
                raise RuntimeError(
                    "private-BLIS numerical reference requires CPU0 on socket0/node0"
                )
            reference_environment, reference_settings = _child_environment(
                reference_args,
                install,
                native,
                dependencies,
                reference_cpu_records,
            )
            run_args["reference_t1"] = reference_args
            run_cpu_records["reference_t1"] = reference_cpu_records
            run_environments["reference_t1"] = reference_environment
            run_environment_settings["reference_t1"] = reference_settings
        output_prefixes = {
            key: output_root / (
                "reference"
                if len(run_specs) == 1
                else (
                    f"reference.block_width_{block_width}"
                    if block_width_comparison
                    else f"reference.{key}"
                )
            )
            for key, _layout, block_width in run_specs
        }
        commands = {
            key: _command(
                run_args[key], subset_prefix=subset_prefix,
                output_prefix=output_prefixes[key], python=python,
                cpu_records=run_cpu_records[key], full_precision_layout=layout,
                block_width=block_width,
                explicit_memory_nodes=run_explicit_memory_nodes[key],
            )
            for key, layout, block_width in run_specs
        }
        comparison_command_signature = None
        if comparison_active:
            signatures = {
                (
                    _block_width_command_signature(command)
                    if block_width_comparison
                    else (
                        _blis_replicate_command_signature(command)
                        if blis_replicate_comparison
                        else (
                            _blis_t1_reference_command_signature(command)
                            if blis_t1_reference_comparison
                            else _comparison_command_signature(command)
                        )
                    )
                )
                for command in commands.values()
            }
            if len(signatures) != 1:
                raise RuntimeError(
                    "comparison child commands differ beyond controlled options"
                )
            comparison_command_signature = signatures.pop()
        subset_paths = [
            Path(f"{subset_prefix}{extension}")
            for extension in (".bed", ".bim", ".fam")
        ]
        subset_identities = {
            str(path): _file_identity(path) for path in subset_paths
        }
        base = {
            "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dry_run": bool(args.dry_run),
            "mode": (
                "current_block_width_comparison"
                if block_width_comparison
                else (
                    "private_blis_replicate_comparison"
                    if blis_replicate_comparison
                    else (
                        "private_blis_t1_reference_comparison"
                        if blis_t1_reference_comparison
                        else (
                            "dual_layout_comparison"
                            if args.compare_fp64_layouts else "single_layout"
                        )
                    )
                )
            ),
            "safety": {
                "temporary_subset_and_outputs": True,
                "temporary_workspace_removed_before_report_publication": True,
                "output_no_replace": True,
                "maximum_blocks": MAX_BLOCKS,
                "maximum_probes_executed": MAX_PROBES,
                "maximum_command_seconds": (
                    MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS
                    if block_width_comparison
                    else (
                        MAX_BLIS_REPLICATE_CHILD_SECONDS
                        if blis_replicate_comparison
                        else (
                            MAX_BLIS_T1_REFERENCE_CHILD_SECONDS
                            if blis_t1_reference_comparison
                            else (
                                MAX_COMPARISON_CHILD_SECONDS
                                if args.compare_fp64_layouts else MAX_TIMEOUT_SECONDS
                            )
                        )
                    )
                ),
                "comparison_maximum_child_seconds": MAX_COMPARISON_CHILD_SECONDS,
                "comparison_maximum_controller_seconds": MAX_COMPARISON_SECONDS,
                "comparison_children_are_fresh_exec_and_sequential": bool(
                    comparison_active
                ),
                "block_width_comparison_maximum_child_seconds": (
                    MAX_BLOCK_WIDTH_COMPARISON_CHILD_SECONDS
                ),
                "block_width_comparison_default_controller_seconds": (
                    DEFAULT_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS
                ),
                "block_width_comparison_maximum_controller_seconds": (
                    MAX_BLOCK_WIDTH_COMPARISON_CONTROLLER_SECONDS
                ),
                "block_width_comparison_layout": (
                    "current" if block_width_comparison else None
                ),
                "blis_replicate_maximum_child_seconds": (
                    MAX_BLIS_REPLICATE_CHILD_SECONDS
                ),
                "blis_replicate_default_controller_seconds": (
                    DEFAULT_BLIS_REPLICATE_CONTROLLER_SECONDS
                ),
                "blis_replicate_maximum_controller_seconds": (
                    MAX_BLIS_REPLICATE_CONTROLLER_SECONDS
                ),
                "blis_replicate_is_determinism_only": True,
                "blis_t1_artifact_reference_required_for_acceptance": True,
                "blis_t1_reference_maximum_child_seconds": (
                    MAX_BLIS_T1_REFERENCE_CHILD_SECONDS
                ),
                "blis_t32_candidate_maximum_child_seconds": (
                    MAX_BLIS_T32_CANDIDATE_CHILD_SECONDS
                ),
                "blis_t1_comparison_maximum_controller_seconds": (
                    MAX_BLIS_T1_COMPARISON_CONTROLLER_SECONDS
                ),
                "b1024_is_projection_only": True,
                "private_openblas_acceptance_eligible": False,
                "accepted_private_backend": EXPECTED_PRIVATE_BLAS_NAME,
                "explicit_openmp_placement_required": True,
                "explicit_openmp_memory_scope": (
                    EXPLICIT_OPENMP_MEMORY_SCOPE
                    if blis_t1_reference_comparison else None
                ),
                "verified_memory_numa_nodes": explicit_memory_nodes,
            },
            "arguments": {
                "blocks": args.blocks, "block_width": args.block_width,
                "subset_variants": args.subset_variants,
                "compare_block_widths": (
                    list(args.compare_block_widths)
                    if block_width_comparison else None
                ),
                "probes": args.probes, "seed": args.seed, "threads": args.threads,
                "requested_storage_dtype": args.storage_dtype,
                "full_precision_layout": args.full_precision_layout,
                "compare_fp64_layouts": bool(args.compare_fp64_layouts),
                "compare_blis_replicates": blis_replicate_comparison,
                "compare_blis_t1_reference": blis_t1_reference_comparison,
                "explicit_openmp_memory_scope": (
                    EXPLICIT_OPENMP_MEMORY_SCOPE
                    if blis_t1_reference_comparison else None
                ),
                "environment_columns": args.environment_columns.split(","),
                "workspace_gib": args.workspace_gib,
                "target_memory_gib": args.target_memory_gib,
                "timeout_seconds": args.timeout_seconds,
                "reference_timeout_seconds": args.reference_timeout_seconds,
                "controller_timeout_seconds": args.controller_timeout_seconds,
                "comparison_rtol": args.comparison_rtol,
                "comparison_atol": args.comparison_atol,
                "integrity_required": not args.allow_integrity_disabled,
                "expected_backend": args.expected_backend,
            },
            "host": {
                "hostname": platform.node(), "platform": platform.platform(),
                "python": sys.version, "controller_pid": os.getpid(),
            },
            "cpu_placement": {
                "taskset": _format_integer_ranges([item.cpu for item in cpu_records]),
                "records": [asdict(item) for item in cpu_records],
                "single_socket": True, "physical_cores_only": True,
                "memory_numa_nodes": (
                    list(explicit_memory_nodes)
                    if explicit_memory_nodes is not None
                    else _expected_memory_nodes(cpu_records, None)
                ),
            },
            "inputs": {
                "source_plink": asdict(source), "subset_plink": asdict(subset),
                "source_file_identities": {
                    path.suffix[1:]: initial_identities[str(path)]
                    for path in source_paths
                },
                "source_consumed_prefix_sha256": {
                    name: record["sha256"] for name, record in subset_hashes.items()
                },
                "subset_file_identities": subset_hashes,
                "subset_is_literal_first_variant_prefix": True,
                "environment": {**_file_identity(environment_path), "sha256": _sha256(environment_path)},
                "covariates": {**_file_identity(covariate_path), "sha256": _sha256(covariate_path)},
                "native_module": {**_file_identity(native), "sha256": native_sha256},
                "private_blas_archive": {
                    **_file_identity(archive),
                    "sha256": _sha256(archive),
                },
                "installed_package": installed_package_identity,
                "benchmark_script": {
                    **_file_identity(benchmark_script),
                    "sha256": _sha256(benchmark_script),
                },
                "dependency_paths": list(map(str, dependencies)),
            },
            "expected_private_blis_provenance": {
                "backend": args.expected_backend,
                "native_api_version": EXPECTED_NATIVE_API_VERSION,
                "native_backend_version": EXPECTED_NATIVE_BACKEND_VERSION,
                "native_sha256": args.expected_native_sha256,
                "package_manifest_sha256": (
                    args.expected_package_manifest_sha256
                ),
                "native_source_commit": args.expected_source_commit,
                "native_source_tree_sha256": args.expected_source_tree_sha256,
                "private_blas_backend": EXPECTED_PRIVATE_BLAS_NAME,
                "private_blas_archive_sha256": args.expected_archive_sha256,
                "private_blas_source_commit": (
                    args.expected_private_source_commit
                ),
                "private_blas_source_tree_sha256": (
                    args.expected_private_source_tree_sha256
                ),
                "private_blas_config_family": EXPECTED_BLIS_CONFIG_FAMILY,
                "private_blas_runtime_config": EXPECTED_BLIS_RUNTIME_CONFIG,
                "threading_layer": "openmp",
                "thread_strategy": "automatic",
                "tls_required": True,
                "placement_schema": OPENMP_PLACEMENT_SCHEMA,
            },
            # Preserve the original singular command field for default/single
            # consumers.  Comparison reports add the complete controlled pair.
            "command": _redacted_command(
                commands[
                    "candidate_t32"
                    if blis_t1_reference_comparison else run_specs[0][0]
                ],
                temporary_root,
            ),
            "child_environment": {
                key: value for key, value in environment_settings.items()
                if not key.startswith("SUMMIT_GXE_EXPECTED_")
            },
        }
        if blis_t1_reference_comparison:
            base["child_environments"] = {
                key: {
                    name: value for name, value in settings.items()
                    if not name.startswith("SUMMIT_GXE_EXPECTED_")
                }
                for key, settings in run_environment_settings.items()
            }
        if comparison_active:
            base["commands"] = {
                key: _redacted_command(
                    command, temporary_root
                )
                for key, command in commands.items()
            }
            base["comparison_control_identity"] = {
                "commands_equal_except_controlled_fields": True,
                "controlled_command_template_sha256": comparison_command_signature,
                "child_environment_shared_exactly": not blis_t1_reference_comparison,
                "literal_subset_shared_exactly": True,
                "literal_subset_validation": (
                    "full stat identity plus BED/BIM/FAM SHA-256 after each child"
                ),
                "shared_probe_environment_seed_and_layout": (
                    block_width_comparison
                    or blis_replicate_comparison
                    or blis_t1_reference_comparison
                ),
                "controlled_differences": (
                    ["step_size", "output_prefix"]
                    if block_width_comparison
                    else (
                        ["output_prefix"]
                        if blis_replicate_comparison
                        else (
                            [
                                "taskset_cpu_list", "num_threads",
                                "openmp_places", "output_prefix",
                            ]
                            if blis_t1_reference_comparison
                            else ["full_precision_layout", "output_prefix"]
                        )
                    )
                ),
            }
        if block_width_comparison:
            base["block_width_schedules"] = {
                str(width): _block_schedule(subset.variants, width)
                for width in args.compare_block_widths
            }
        if args.dry_run:
            base["status"] = (
                "validated_current_block_width_comparison_dry_run"
                if block_width_comparison
                else (
                    "validated_private_blis_replicate_dry_run"
                    if blis_replicate_comparison
                    else (
                        "validated_private_blis_t1_reference_dry_run"
                        if blis_t1_reference_comparison
                        else (
                            "validated_dual_layout_dry_run"
                            if args.compare_fp64_layouts else "validated_dry_run"
                        )
                    )
                )
            )
            base["performance_telemetry"] = None
            base["projections"] = None
            base["accepted"] = False
            base["acceptance_eligible"] = False
            base["acceptance_reason"] = "dry_run_performs_no_scientific_execution"
            report = base
        elif not comparison_active:
            key, layout, block_width = run_specs[0]
            observed = _execute_layout(
                args, layout=layout, subset=subset, subset_prefix=subset_prefix,
                output_prefix=output_prefixes[key], python=python,
                cpu_records=cpu_records, child_environment=child_environment,
                native_sha256=native_sha256, temporary_root=temporary_root,
                source_variants=source.variants,
                timeout_seconds=args.timeout_seconds,
                block_width=block_width,
            )
            _assert_staged_subset_integrity(
                subset_paths, subset_identities, subset_hashes
            )
            payload = observed["payload"]
            performance = observed["performance"]
            base["status"] = "completed_single_blis_diagnostic"
            base["process"] = observed["process"]
            base["batch_summary"] = _batch_summary(payload)
            base["canonical_batch_summary"] = _batch_summary(
                observed["canonical_payload"]
            )
            base["openmp_placement_attestation"] = observed["cpu_placement"]
            base["randomization"] = payload.get("randomization")
            base["performance_telemetry"] = performance
            base["gemm_summaries"] = observed["gemm_summaries"]
            base["temporary_artifact_identities"] = observed[
                "temporary_artifact_identities"
            ]
            base["projections"] = observed["projections"]
            base["accepted"] = False
            base["acceptance_eligible"] = False
            base["acceptance_reason"] = (
                "single_blis_run_lacks_independent_artifact_level_reference; "
                "rerun_with_--compare-blis-t1-reference"
            )
            report = base
        elif blis_t1_reference_comparison:
            assert comparison_controller_started is not None
            observed_runs = {}
            for key, layout, block_width in run_specs:
                controller_elapsed = (
                    time.perf_counter() - comparison_controller_started
                )
                remaining = args.controller_timeout_seconds - controller_elapsed
                if remaining <= 0.0:
                    raise TimeoutError(
                        "private-BLIS T1/T32 controller wall cap expired"
                    )
                child_cap = (
                    args.reference_timeout_seconds
                    if key == "reference_t1"
                    else args.timeout_seconds
                )
                observed_runs[key] = _execute_layout(
                    run_args[key],
                    layout=layout,
                    subset=subset,
                    subset_prefix=subset_prefix,
                    output_prefix=output_prefixes[key],
                    python=python,
                    cpu_records=run_cpu_records[key],
                    child_environment=run_environments[key],
                    native_sha256=native_sha256,
                    temporary_root=temporary_root,
                    source_variants=source.variants,
                    timeout_seconds=min(child_cap, remaining),
                    block_width=block_width,
                    explicit_memory_nodes=run_explicit_memory_nodes[key],
                )
                _assert_staged_subset_integrity(
                    subset_paths, subset_identities, subset_hashes
                )
                for path in immutable_inputs:
                    if not _same_file_identity(path, initial_identities[str(path)]):
                        raise RuntimeError(
                            f"controlled input changed after {key}: {path}"
                        )
            reference = observed_runs["reference_t1"]
            candidate = observed_runs["candidate_t32"]
            comparison = _compare_completed_blis_outputs(
                reference["manifest_path"],
                reference["payload"],
                candidate["manifest_path"],
                candidate["payload"],
                environment_order=[
                    item.strip() for item in args.environment_columns.split(",")
                ],
                rtol=args.comparison_rtol,
                atol=args.comparison_atol,
                reference_run="reference_t1",
                candidate_run="candidate_t32",
                reference_threads=1,
                candidate_threads=32,
                acceptance_reference=True,
            )
            controller_elapsed = time.perf_counter() - comparison_controller_started
            if controller_elapsed > args.controller_timeout_seconds:
                raise TimeoutError(
                    "private-BLIS T1/T32 comparison exceeded its hard wall cap"
                )
            accuracy_accepted = comparison.get("accuracy_gate_passed") is True
            base["accuracy_gate_passed"] = accuracy_accepted
            base["accepted"] = accuracy_accepted
            base["acceptance_eligible"] = accuracy_accepted
            base["candidate_selected"] = accuracy_accepted
            base["acceptance_reason"] = (
                "private_blis_t32_passed_full_artifact_gate_against_fresh_t1_reference"
                if accuracy_accepted
                else "private_blis_t32_failed_fresh_t1_artifact_accuracy_gate"
            )
            base["status"] = (
                "completed_private_blis_t1_reference_acceptance"
                if accuracy_accepted
                else "rejected_private_blis_t1_reference_artifact_accuracy"
            )
            base["process"] = {
                "controller_wall_seconds": controller_elapsed,
                "children_sum_wall_seconds": sum(
                    run["process"]["wall_seconds"]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_gib_high_water": max(
                    run["process"][
                        "controller_children_ru_maxrss_gib_high_water"
                    ]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_scope": (
                    "RUSAGE_CHILDREN process-lifetime high-water across the "
                    "sequential T1 reference and T32 candidate"
                ),
                "authoritative_per_run_peak_rss_field": (
                    "blis_t1_reference_runs.<run>.batch_summary."
                    "peak_process_rss_gib_at_manifest"
                ),
                "fresh_exec_children": 2,
                "children_sequential": True,
                "reference_child_cap_seconds": args.reference_timeout_seconds,
                "candidate_child_cap_seconds": args.timeout_seconds,
            }
            base["performance_telemetry"] = None
            base["blis_t1_reference_runs"] = {
                key: {
                    "role": (
                        "numerical_reference"
                        if key == "reference_t1" else "selection_candidate"
                    ),
                    "threads": run_args[key].threads,
                    "cpu_ids": [
                        record.cpu for record in run_cpu_records[key]
                    ],
                    "memory_scope": EXPLICIT_OPENMP_MEMORY_SCOPE,
                    "numa_nodes": list(run_explicit_memory_nodes[key]),
                    "process": run["process"],
                    "batch_summary": _batch_summary(run["payload"]),
                    "canonical_batch_summary": _batch_summary(
                        run["canonical_payload"]
                    ),
                    "openmp_placement_attestation": run["cpu_placement"],
                    "randomization": run["payload"].get("randomization"),
                    "performance_telemetry": _compact_performance(
                        run["performance"]
                    ),
                    "gemm_summaries": run["gemm_summaries"],
                    "temporary_artifact_identities": run[
                        "temporary_artifact_identities"
                    ],
                    "projections": run["projections"],
                }
                for key, run in observed_runs.items()
            }
            base["projections"] = {
                key: run["projections"] for key, run in observed_runs.items()
            }
            base["comparison"] = comparison
            report = base
        elif blis_replicate_comparison:
            assert comparison_controller_started is not None
            observed_runs = {}
            for key, layout, block_width in run_specs:
                controller_elapsed = (
                    time.perf_counter() - comparison_controller_started
                )
                remaining = args.controller_timeout_seconds - controller_elapsed
                if remaining <= 0.0:
                    raise TimeoutError(
                        "private-BLIS replicate controller wall cap expired"
                    )
                observed_runs[key] = _execute_layout(
                    args,
                    layout=layout,
                    subset=subset,
                    subset_prefix=subset_prefix,
                    output_prefix=output_prefixes[key],
                    python=python,
                    cpu_records=cpu_records,
                    child_environment=child_environment,
                    native_sha256=native_sha256,
                    temporary_root=temporary_root,
                    source_variants=source.variants,
                    timeout_seconds=min(args.timeout_seconds, remaining),
                    block_width=block_width,
                )
                _assert_staged_subset_integrity(
                    subset_paths, subset_identities, subset_hashes
                )
                for path in immutable_inputs:
                    if not _same_file_identity(path, initial_identities[str(path)]):
                        raise RuntimeError(
                            f"controlled input changed after {key}: {path}"
                        )
            first = observed_runs["replicate_1"]
            second = observed_runs["replicate_2"]
            comparison = _compare_completed_blis_outputs(
                first["manifest_path"],
                first["payload"],
                second["manifest_path"],
                second["payload"],
                environment_order=[
                    item.strip() for item in args.environment_columns.split(",")
                ],
                rtol=args.comparison_rtol,
                atol=args.comparison_atol,
                reference_run="replicate_1",
                candidate_run="replicate_2",
                reference_threads=args.threads,
                candidate_threads=args.threads,
                acceptance_reference=False,
            )
            controller_elapsed = time.perf_counter() - comparison_controller_started
            if controller_elapsed > args.controller_timeout_seconds:
                raise TimeoutError(
                    "private-BLIS replicate controller exceeded its hard wall cap"
                )
            accuracy_accepted = comparison.get("accuracy_gate_passed") is True
            base["accuracy_gate_passed"] = accuracy_accepted
            base["accepted"] = False
            base["acceptance_eligible"] = False
            base["acceptance_reason"] = (
                "same_thread_blis_replicates_are_determinism_evidence_only; "
                "acceptance_requires_--compare-blis-t1-reference"
            )
            base["status"] = (
                "completed_private_blis_replicate_diagnostic"
                if accuracy_accepted
                else "rejected_private_blis_replicate_artifact_accuracy"
            )
            base["process"] = {
                "controller_wall_seconds": controller_elapsed,
                "children_sum_wall_seconds": sum(
                    run["process"]["wall_seconds"]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_gib_high_water": max(
                    run["process"][
                        "controller_children_ru_maxrss_gib_high_water"
                    ]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_scope": (
                    "RUSAGE_CHILDREN process-lifetime high-water across both "
                    "sequential BLIS replicates; not a sum or per-run measurement"
                ),
                "authoritative_per_run_peak_rss_field": (
                    "blis_replicate_runs.<run>.batch_summary."
                    "peak_process_rss_gib_at_manifest"
                ),
                "fresh_exec_children": 2,
                "children_sequential": True,
            }
            base["performance_telemetry"] = None
            base["blis_replicate_runs"] = {
                key: {
                    "process": run["process"],
                    "batch_summary": _batch_summary(run["payload"]),
                    "canonical_batch_summary": _batch_summary(
                        run["canonical_payload"]
                    ),
                    "openmp_placement_attestation": run["cpu_placement"],
                    "randomization": run["payload"].get("randomization"),
                    "performance_telemetry": _compact_performance(
                        run["performance"]
                    ),
                    "gemm_summaries": run["gemm_summaries"],
                    "temporary_artifact_identities": run[
                        "temporary_artifact_identities"
                    ],
                    "projections": run["projections"],
                }
                for key, run in observed_runs.items()
            }
            base["projections"] = {
                key: run["projections"] for key, run in observed_runs.items()
            }
            base["comparison"] = comparison
            report = base
        elif args.compare_fp64_layouts:
            assert comparison_controller_started is not None
            observed_runs = {}
            for key, layout, block_width in run_specs:
                controller_elapsed = (
                    time.perf_counter() - comparison_controller_started
                )
                remaining = args.controller_timeout_seconds - controller_elapsed
                if remaining <= 0.0:
                    raise TimeoutError("dual-layout controller wall cap expired")
                observed_runs[key] = _execute_layout(
                    args, layout=layout, subset=subset,
                    subset_prefix=subset_prefix,
                    output_prefix=output_prefixes[key], python=python,
                    cpu_records=cpu_records, child_environment=child_environment,
                    native_sha256=native_sha256, temporary_root=temporary_root,
                    source_variants=source.variants,
                    timeout_seconds=min(args.timeout_seconds, remaining),
                    block_width=block_width,
                )
                _assert_staged_subset_integrity(
                    subset_paths, subset_identities, subset_hashes
                )
                for path in immutable_inputs:
                    if not _same_file_identity(path, initial_identities[str(path)]):
                        raise RuntimeError(
                            f"controlled input changed after {layout}: {path}"
                        )
            current = observed_runs["current"]
            optimized = observed_runs["source_tt_target_current"]
            comparison = _compare_completed_runs(
                current["manifest_path"], current["payload"],
                optimized["manifest_path"], optimized["payload"],
                environment_order=[
                    item.strip() for item in args.environment_columns.split(",")
                ],
                rtol=args.comparison_rtol, atol=args.comparison_atol,
            )
            comparison["projection_speedups"] = _projection_speedups(
                current["projections"], optimized["projections"]
            )
            controller_elapsed = time.perf_counter() - comparison_controller_started
            if controller_elapsed > args.controller_timeout_seconds:
                raise TimeoutError("dual-layout controller exceeded its hard wall cap")
            accuracy_accepted = comparison.get("accuracy_gate_passed") is True
            base["accuracy_gate_passed"] = accuracy_accepted
            base["accepted"] = accuracy_accepted
            base["status"] = (
                "completed_dual_layout_comparison"
                if accuracy_accepted else "rejected_dual_layout_accuracy_gate"
            )
            base["process"] = {
                "controller_wall_seconds": controller_elapsed,
                "children_sum_wall_seconds": sum(
                    run["process"]["wall_seconds"] for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_gib_high_water": max(
                    run["process"]["controller_children_ru_maxrss_gib_high_water"]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_scope": (
                    "RUSAGE_CHILDREN process-lifetime high-water across both "
                    "sequential children; not a sum or per-layout measurement"
                ),
                "authoritative_per_layout_peak_rss_field": (
                    "layout_runs.<layout>.batch_summary."
                    "peak_process_rss_gib_at_manifest"
                ),
                "fresh_exec_children": 2,
                "children_sequential": True,
            }
            base["performance_telemetry"] = None
            base["layout_runs"] = {
                layout: {
                    "process": run["process"],
                    "batch_summary": _batch_summary(run["payload"]),
                    "canonical_batch_summary": _batch_summary(
                        run.get("canonical_payload", {})
                    ),
                    "openmp_placement_attestation": run.get("cpu_placement"),
                    "randomization": run["payload"].get("randomization"),
                    "performance_telemetry": _compact_performance(
                        run["performance"]
                    ),
                    "gemm_summaries": run["gemm_summaries"],
                    "temporary_artifact_identities": run[
                        "temporary_artifact_identities"
                    ],
                    "projections": run["projections"],
                }
                for layout, run in observed_runs.items()
            }
            base["projections"] = {
                "current": current["projections"],
                "source_tt_target_current": optimized["projections"],
                "speedups": comparison["projection_speedups"],
            }
            base["comparison"] = comparison
            report = base
        else:
            assert block_width_comparison
            assert comparison_controller_started is not None
            observed_runs = {}
            for key, layout, block_width in run_specs:
                controller_elapsed = (
                    time.perf_counter() - comparison_controller_started
                )
                remaining = args.controller_timeout_seconds - controller_elapsed
                if remaining <= 0.0:
                    raise TimeoutError(
                        "current block-width comparison controller wall cap expired"
                    )
                observed_runs[key] = _execute_layout(
                    args, layout=layout, subset=subset,
                    subset_prefix=subset_prefix,
                    output_prefix=output_prefixes[key], python=python,
                    cpu_records=cpu_records,
                    child_environment=child_environment,
                    native_sha256=native_sha256,
                    temporary_root=temporary_root,
                    source_variants=source.variants,
                    timeout_seconds=min(args.timeout_seconds, remaining),
                    block_width=block_width,
                )
                _assert_staged_subset_integrity(
                    subset_paths, subset_identities, subset_hashes
                )
                for path in immutable_inputs:
                    if not _same_file_identity(path, initial_identities[str(path)]):
                        raise RuntimeError(
                            "controlled input changed after block width "
                            f"{block_width}: {path}"
                        )
            baseline_width, candidate_width = args.compare_block_widths
            baseline = observed_runs[str(baseline_width)]
            candidate = observed_runs[str(candidate_width)]
            comparison = _compare_completed_block_width_runs(
                baseline["manifest_path"], baseline["payload"],
                candidate["manifest_path"], candidate["payload"],
                baseline_width=baseline_width,
                candidate_width=candidate_width,
                environment_order=[
                    item.strip() for item in args.environment_columns.split(",")
                ],
                rtol=args.comparison_rtol,
                atol=args.comparison_atol,
            )
            comparison["projection_speedups"] = (
                _block_width_projection_speedups(
                    baseline["projections"], candidate["projections"],
                    baseline_width=baseline_width,
                    candidate_width=candidate_width,
                )
            )
            controller_elapsed = time.perf_counter() - comparison_controller_started
            if controller_elapsed > args.controller_timeout_seconds:
                raise TimeoutError(
                    "current block-width comparison exceeded its hard wall cap"
                )
            accuracy_passed = comparison.get("accuracy_gate_passed") is True
            observed_speedup = (
                float(baseline["process"]["wall_seconds"])
                / float(candidate["process"]["wall_seconds"])
                if float(candidate["process"]["wall_seconds"]) > 0.0
                else None
            )
            projected_speedups = [
                item.get("baseline_over_candidate_speedup")
                for item in comparison["projection_speedups"].values()
            ]
            material_speedup_passed = (
                observed_speedup is not None
                and math.isfinite(observed_speedup)
                and observed_speedup >= MATERIAL_SPEEDUP_THRESHOLD
                and all(
                    value is not None
                    and math.isfinite(float(value))
                    and float(value) >= MATERIAL_SPEEDUP_THRESHOLD
                    for value in projected_speedups
                )
            )
            rss_bound_gib = args.workspace_gib + args.target_memory_gib
            rss_values = []
            for run in (baseline, candidate):
                raw = run["payload"].get("peak_process_rss_gib_at_manifest")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
                rss_values.append(
                    value if value is not None and math.isfinite(value) else None
                )
            memory_rss_bound_passed = all(
                value is not None and 0.0 < value <= rss_bound_gib
                for value in rss_values
            )
            # This controller does not yet capture bounded PSI or per-child
            # major-fault evidence.  Comparison evidence is publishable, but
            # candidate selection is deliberately ineligible until it does.
            pressure_evidence_available = False
            major_fault_evidence_available = False
            ineligibility_reasons = []
            if not accuracy_passed:
                ineligibility_reasons.append("output_accuracy_gate_failed")
            if not material_speedup_passed:
                ineligibility_reasons.append("material_speedup_gate_failed")
            if not memory_rss_bound_passed:
                ineligibility_reasons.append("memory_rss_bound_gate_failed")
            if not pressure_evidence_available:
                ineligibility_reasons.append("memory_pressure_evidence_unavailable")
            if not major_fault_evidence_available:
                ineligibility_reasons.append("major_fault_evidence_unavailable")
            candidate_eligible = not ineligibility_reasons
            base["comparison_completed"] = True
            base["accuracy_gate_passed"] = accuracy_passed
            base["candidate_eligible"] = candidate_eligible
            base["candidate_selected"] = False
            base["accepted"] = False
            base["status"] = (
                "completed_current_block_width_comparison_candidate_ineligible"
                if accuracy_passed
                else "completed_current_block_width_comparison_accuracy_failed"
            )
            base["candidate_selection"] = {
                "selection_attempted": False,
                "eligible": candidate_eligible,
                "selected": False,
                "accepted": False,
                "material_speedup_threshold": MATERIAL_SPEEDUP_THRESHOLD,
                "observed_wall_speedup": observed_speedup,
                "projected_speedups": projected_speedups,
                "material_speedup_gate_passed": material_speedup_passed,
                "rss_bound_gib": rss_bound_gib,
                "baseline_peak_rss_gib": rss_values[0],
                "candidate_peak_rss_gib": rss_values[1],
                "memory_rss_bound_gate_passed": memory_rss_bound_passed,
                "memory_pressure_evidence_available": (
                    pressure_evidence_available
                ),
                "major_fault_evidence_available": major_fault_evidence_available,
                "ineligibility_reasons": ineligibility_reasons,
                "policy": (
                    "comparison evidence may be published, but candidate "
                    "selection requires output accuracy, >=5% observed and "
                    "projected speedup, bounded RSS, and bounded pressure/"
                    "major-fault evidence"
                ),
            }
            base["process"] = {
                "controller_wall_seconds": controller_elapsed,
                "children_sum_wall_seconds": sum(
                    run["process"]["wall_seconds"]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_gib_high_water": max(
                    run["process"][
                        "controller_children_ru_maxrss_gib_high_water"
                    ]
                    for run in observed_runs.values()
                ),
                "controller_children_ru_maxrss_scope": (
                    "RUSAGE_CHILDREN process-lifetime high-water across both "
                    "sequential children; not a sum or per-width measurement"
                ),
                "authoritative_per_width_peak_rss_field": (
                    "block_width_runs.<width>.batch_summary."
                    "peak_process_rss_gib_at_manifest"
                ),
                "fresh_exec_children": len(observed_runs),
                "children_sequential": True,
            }
            base["performance_telemetry"] = None
            base["block_width_runs"] = {
                width: {
                    "layout": run["layout"],
                    "block_schedule": run["block_schedule"],
                    "process": run["process"],
                    "batch_summary": _batch_summary(run["payload"]),
                    "canonical_batch_summary": _batch_summary(
                        run.get("canonical_payload", {})
                    ),
                    "openmp_placement_attestation": run.get("cpu_placement"),
                    "randomization": run["payload"].get("randomization"),
                    "performance_telemetry": _compact_performance(
                        run["performance"]
                    ),
                    "gemm_summaries": run["gemm_summaries"],
                    "temporary_artifact_identities": run[
                        "temporary_artifact_identities"
                    ],
                    "projections": run["projections"],
                }
                for width, run in observed_runs.items()
            }
            base["projections"] = {
                "by_block_width": {
                    width: run["projections"]
                    for width, run in observed_runs.items()
                },
                "speedups": comparison["projection_speedups"],
            }
            base["comparison"] = comparison
            report = base
        for path in immutable_inputs:
            if not _same_file_identity(path, initial_identities[str(path)]):
                raise RuntimeError(f"input identity changed during benchmark: {path}")
        if _sha256(native) != native_sha256:
            raise RuntimeError("native module bytes changed before workspace cleanup")
        if _sha256(archive) != args.expected_archive_sha256:
            raise RuntimeError("private BLIS archive changed before workspace cleanup")
        if _installed_package_identity(install) != installed_package_identity:
            raise RuntimeError(
                "installed summit package changed before workspace cleanup"
            )
    if comparison_controller_started is not None:
        final_elapsed = time.perf_counter() - comparison_controller_started
        if final_elapsed > args.controller_timeout_seconds:
            raise TimeoutError(
                "comparison controller exceeded its hard wall cap including cleanup"
            )
        report["process"]["controller_wall_seconds_including_cleanup"] = final_elapsed
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    payload = run(args)
    _atomic_json_no_replace(payload, args.output)
    print(args.output.expanduser().resolve())
    if args.dry_run:
        return 0
    return 0 if payload.get("accepted") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
