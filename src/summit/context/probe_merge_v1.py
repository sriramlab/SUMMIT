"""Transient process-merge boundary for contextual variant-probe statistics.

This module deliberately does not define a writer or loader.  A partial carries
an individual/sample axis and is suitable only for authenticated process-local
transport into a merge.  The NumPy finalizer below is an oracle and integration
boundary; production publication still requires a native protected finalizer.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._artifact_io import validate_native_build_provenance_consistency
from .spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
    freeze_context_mapping,
)


CONTEXTUAL_VARIANT_PROBE_PARTIAL_V1_MAGIC = "SUMMIT_CONTEXTUAL_VARIANT_PROBE_PARTIAL_V1"
CONTEXTUAL_VARIANT_PROBE_MERGE_RESULT_V1_MAGIC = (
    "SUMMIT_CONTEXTUAL_VARIANT_PROBE_MERGE_RESULT_V1"
)
CONTEXTUAL_VARIANT_PROBE_PLAN_V1_MAGIC = "SUMMIT_CONTEXTUAL_VARIANT_PROBE_PLAN_V1"
CONTEXTUAL_VARIANT_PROBE_RANGE_V1_MAGIC = "SUMMIT_CONTEXTUAL_VARIANT_PROBE_RANGE_V1"
CONTEXTUAL_VARIANT_PROBE_PARTIAL_COMMITMENT_V1_MAGIC = (
    "SUMMIT_CONTEXTUAL_VARIANT_PROBE_PARTIAL_COMMITMENT_V1"
)
NUMPY_ORACLE_TRANSIENT_FINALIZER_V1 = (
    "numpy_oracle_transient_v1_not_production_protected"
)
CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1 = (
    "annotation_map_sha256",
    "annotation_names_sha256",
    "component_map_sha256",
    "evaluated_phi_sha256",
    "fixed_basis_sha256",
    "group_map_sha256",
    "group_names_sha256",
    "missingness_sha256",
    "pair_map_sha256",
    "retained_sample_map_sha256",
    "retained_variant_order_sha256",
    "scale_plan_sha256",
    "variant_order_allele_sha256",
)
_NATIVE_FULL_RANGE_IDENTITY_KEYS_V1 = frozenset(
    {
        "schema",
        "source_commit",
        "source_tree_sha256",
        "build_id",
        "build_provenance",
        "native_api_version",
        "native_backend_version",
        "native_backend",
        "native_execution_backend",
        "numeric_policy",
        "feature_mode",
        "execution_plan_sha256",
        "sealed_plan_sha256",
        "variant_probe_policy",
        "native_variant_probe_identity_sha256",
        "global_probe_begin",
        "global_probe_end",
        "global_probe_count",
        "ordered_probe_sha256s",
        "global_probe_plan_sha256",
        "sample_count",
        "component_count",
        "probe_sums_layout",
        "probe_cross_layout",
        "full_executor_range_only",
        "partitioned_native_finalizer_claimed",
        "science_identity_inputs",
    }
)
_NATIVE_SCIENCE_IDENTITY_INPUT_KEYS_V1 = frozenset(
    {
        "annotation_map_sha256",
        "evaluated_phi_sha256",
        "fixed_basis_sha256",
        "group_map_sha256",
        "missingness_sha256",
        "retained_sample_map_sha256",
        "retained_variant_order_sha256",
        "scale_plan_sha256",
        "variant_order_allele_sha256",
        "annotation_names",
        "group_names",
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
    }
)
_NATIVE_BUILD_PROVENANCE_KEYS_V1 = frozenset(
    {
        "schema",
        "source_commit",
        "source_tree_sha256",
        "compiler_id",
        "compiler_version",
        "cxx_standard",
        "build_type",
        "sanitizer_mode",
        "asan_enabled",
        "ubsan_enabled",
        "effective_optimization",
        "architecture_tuning",
        "configured_compiler_flags",
        "blas_vendor",
        "gemm_integrity_enabled",
        "gemm_checksum_enabled",
        "private_blas_enabled",
        "private_blas_backend",
        "private_blas_sha256",
        "private_blas_source_commit",
        "private_blas_source_tree_sha256",
        "private_blas_config_family",
        "private_blas_header_sha256",
        "private_blas_cblas_header_sha256",
        "private_openblas_enabled",
        "private_openblas_sha256",
        "native_arch_optimization_enabled",
        "openmp_enabled",
        "contextual_dispatch_backend",
        "contextual_dispatch_vendor_calls",
    }
)


def _sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return value


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _immutable_f64(name: str, value: Any, *, ndim: int) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != ndim or source.dtype.hasobject or source.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a {ndim}-dimensional real numeric array.")
    canonical = np.ascontiguousarray(source, dtype="<f8")
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} must contain only finite values.")
    # Bytes, rather than a caller-owned ndarray, are the immutable backing store.
    result = np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
        canonical.shape
    )
    result.setflags(write=False)
    return result


def _require_exact_symmetry(name: str, value: np.ndarray) -> None:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not np.array_equal(value, value.T):
        raise ValueError(f"{name} must be exactly symmetric.")


def _ordered_unique_integers(
    name: str, values: Sequence[int], *, allow_empty: bool
) -> tuple[int, ...]:
    result = tuple(_integer(name, value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing and unique.")
    return result


def _ordered_probe_digests(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_sha256("ordered_probe_sha256s item", value) for value in values)


def contextual_variant_probe_plan_sha256_v1(
    ordered_probe_sha256s: Sequence[str],
) -> str:
    """Hash the exact ordered global probe plan from per-probe commitments."""
    leaves = _ordered_probe_digests(ordered_probe_sha256s)
    if len(leaves) < 2:
        raise ValueError("A global variant-probe plan requires at least two probes.")
    return canonical_sha256(
        {
            "magic": CONTEXTUAL_VARIANT_PROBE_PLAN_V1_MAGIC,
            "global_probe_count": len(leaves),
            "ordered_probe_sha256s": list(leaves),
        }
    )


@dataclass(frozen=True, slots=True)
class ContextualVariantProbeMergeIdentityV1:
    """Science/build identity shared by every partial in one global merge."""

    global_probe_plan_sha256: str
    science_identity: Mapping[str, str]
    source_tree_sha256: str
    native_build_provenance_sha256: str
    build_id: str
    native_backend: str
    native_execution_backend: str
    variant_probe_policy: str
    native_api_version: int
    native_backend_version: int
    numeric_policy: str = "fp64_v1"
    feature_mode: str = "P_diag_phi_G_v1"
    science_identity_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        for name in (
            "global_probe_plan_sha256",
            "source_tree_sha256",
            "native_build_provenance_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if not isinstance(self.science_identity, Mapping) or set(
            self.science_identity
        ) != set(CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1):
            raise ValueError("Variant-probe per-axis science identity key mismatch.")
        science_identity = {
            name: _sha256(name, self.science_identity[name])
            for name in CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1
        }
        object.__setattr__(
            self, "science_identity", freeze_context_mapping(science_identity)
        )
        object.__setattr__(
            self,
            "science_identity_sha256",
            canonical_sha256(science_identity),
        )
        for name in (
            "build_id",
            "native_backend",
            "native_execution_backend",
            "variant_probe_policy",
        ):
            _nonempty(name, getattr(self, name))
        _integer("native_api_version", self.native_api_version, minimum=1)
        _integer("native_backend_version", self.native_backend_version, minimum=1)
        if self.numeric_policy != "fp64_v1":
            raise ValueError("Variant-probe merge numeric_policy must be fp64_v1.")
        if self.feature_mode != "P_diag_phi_G_v1":
            raise ValueError(
                "Variant-probe merge feature_mode must be P_diag_phi_G_v1."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_probe_plan_sha256": self.global_probe_plan_sha256,
            "science_identity": dict(self.science_identity),
            "science_identity_sha256": self.science_identity_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "native_build_provenance_sha256": (self.native_build_provenance_sha256),
            "build_id": self.build_id,
            "native_backend": self.native_backend,
            "native_execution_backend": self.native_execution_backend,
            "variant_probe_policy": self.variant_probe_policy,
            "native_api_version": self.native_api_version,
            "native_backend_version": self.native_backend_version,
            "numeric_policy": self.numeric_policy,
            "feature_mode": self.feature_mode,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


def _merge_plan_sha256(
    identity: ContextualVariantProbeMergeIdentityV1,
    *,
    global_probe_count: int,
    sample_count: int,
    component_count: int,
) -> str:
    return canonical_sha256(
        {
            "magic": CONTEXTUAL_VARIANT_PROBE_PLAN_V1_MAGIC,
            "identity_sha256": identity.digest,
            "dimensions": {
                "B_D": global_probe_count,
                "N": sample_count,
                "C": component_count,
            },
        }
    )


def _range_commitment_sha256(
    *,
    merge_plan_sha256: str,
    global_probe_begin: int,
    global_probe_end: int,
    global_probe_count: int,
    ordered_probe_sha256s: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "magic": CONTEXTUAL_VARIANT_PROBE_RANGE_V1_MAGIC,
            "merge_plan_sha256": merge_plan_sha256,
            "global_probe_begin": global_probe_begin,
            "global_probe_end": global_probe_end,
            "global_probe_count": global_probe_count,
            "ordered_probe_sha256s": list(ordered_probe_sha256s),
        }
    )


def _partial_commitment_sha256(
    *,
    identity_sha256: str,
    merge_plan_sha256: str,
    range_commitment_sha256: str,
    probe_sums_sha256: str,
    within_probe_cross_sha256: str,
    process_id: int,
    process_slot: int,
    host_name: str,
    process_start_method: str,
    cpu_affinity: Sequence[int],
    socket_ids: Sequence[int],
    lifecycle: str,
) -> str:
    return canonical_sha256(
        {
            "magic": CONTEXTUAL_VARIANT_PROBE_PARTIAL_COMMITMENT_V1_MAGIC,
            "identity_sha256": _sha256("identity_sha256", identity_sha256),
            "merge_plan_sha256": _sha256("merge_plan_sha256", merge_plan_sha256),
            "range_commitment_sha256": _sha256(
                "range_commitment_sha256", range_commitment_sha256
            ),
            "probe_sums_sha256": _sha256("probe_sums_sha256", probe_sums_sha256),
            "within_probe_cross_sha256": _sha256(
                "within_probe_cross_sha256", within_probe_cross_sha256
            ),
            "process_id": _integer("process_id", process_id, minimum=1),
            "process_slot": _integer("process_slot", process_slot),
            "host_name": _nonempty("host_name", host_name),
            "process_start_method": _nonempty(
                "process_start_method", process_start_method
            ),
            "cpu_affinity": list(cpu_affinity),
            "socket_ids": list(socket_ids),
            "lifecycle": _nonempty("lifecycle", lifecycle),
        }
    )


@dataclass(frozen=True, slots=True)
class ContextualVariantProbePartialV1:
    """Immutable raw sufficient statistics for one global probe interval.

    ``probe_sums`` is canonical logical ``[C,N]``.  A partial is intentionally
    sample-bearing and must never be published as a stable contextual artifact.
    ``lifecycle='locally_finalized'`` can represent an invalid transport input;
    the merger accepts only ``partial_complete``.
    """

    probe_sums: np.ndarray
    within_probe_cross: np.ndarray
    global_probe_begin: int
    global_probe_end: int
    global_probe_count: int
    ordered_probe_sha256s: tuple[str, ...]
    identity: ContextualVariantProbeMergeIdentityV1
    process_id: int
    process_slot: int
    cpu_affinity: tuple[int, ...]
    host_name: str
    socket_ids: tuple[int, ...] = ()
    process_start_method: str = "spawn"
    lifecycle: str = "partial_complete"
    magic: str = field(default=CONTEXTUAL_VARIANT_PROBE_PARTIAL_V1_MAGIC, init=False)
    contains_sample_axis: bool = field(default=True, init=False)
    durable_artifact: bool = field(default=False, init=False)
    probe_sums_sha256: str = field(default="", init=False)
    within_probe_cross_sha256: str = field(default="", init=False)
    merge_plan_sha256: str = field(default="", init=False)
    range_commitment_sha256: str = field(default="", init=False)
    partial_commitment_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        probe_sums = _immutable_f64("probe_sums", self.probe_sums, ndim=2)
        within = _immutable_f64("within_probe_cross", self.within_probe_cross, ndim=2)
        object.__setattr__(self, "probe_sums", probe_sums)
        object.__setattr__(self, "within_probe_cross", within)
        object.__setattr__(
            self,
            "ordered_probe_sha256s",
            _ordered_probe_digests(self.ordered_probe_sha256s),
        )
        object.__setattr__(
            self,
            "cpu_affinity",
            _ordered_unique_integers(
                "cpu_affinity", self.cpu_affinity, allow_empty=False
            ),
        )
        object.__setattr__(
            self,
            "socket_ids",
            _ordered_unique_integers("socket_ids", self.socket_ids, allow_empty=True),
        )
        object.__setattr__(self, "probe_sums_sha256", array_sha256(probe_sums))
        object.__setattr__(self, "within_probe_cross_sha256", array_sha256(within))
        if isinstance(self.identity, ContextualVariantProbeMergeIdentityV1):
            plan = _merge_plan_sha256(
                self.identity,
                global_probe_count=self.global_probe_count,
                sample_count=probe_sums.shape[1],
                component_count=probe_sums.shape[0],
            )
            object.__setattr__(self, "merge_plan_sha256", plan)
            object.__setattr__(
                self,
                "range_commitment_sha256",
                _range_commitment_sha256(
                    merge_plan_sha256=plan,
                    global_probe_begin=self.global_probe_begin,
                    global_probe_end=self.global_probe_end,
                    global_probe_count=self.global_probe_count,
                    ordered_probe_sha256s=self.ordered_probe_sha256s,
                ),
            )
            object.__setattr__(
                self,
                "partial_commitment_sha256",
                _partial_commitment_sha256(
                    identity_sha256=self.identity.digest,
                    merge_plan_sha256=self.merge_plan_sha256,
                    range_commitment_sha256=self.range_commitment_sha256,
                    probe_sums_sha256=self.probe_sums_sha256,
                    within_probe_cross_sha256=self.within_probe_cross_sha256,
                    process_id=self.process_id,
                    process_slot=self.process_slot,
                    host_name=self.host_name,
                    process_start_method=self.process_start_method,
                    cpu_affinity=self.cpu_affinity,
                    socket_ids=self.socket_ids,
                    lifecycle=self.lifecycle,
                ),
            )
        self.validate()

    @property
    def component_count(self) -> int:
        return int(self.probe_sums.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.probe_sums.shape[1])

    def validate(self) -> None:
        if self.magic != CONTEXTUAL_VARIANT_PROBE_PARTIAL_V1_MAGIC:
            raise ValueError("Variant-probe partial magic mismatch.")
        if self.contains_sample_axis is not True or self.durable_artifact is not False:
            raise ValueError("Variant-probe partial transient policy mismatch.")
        if self.lifecycle not in {"partial_complete", "locally_finalized"}:
            raise ValueError("Variant-probe partial lifecycle is invalid.")
        if not isinstance(self.identity, ContextualVariantProbeMergeIdentityV1):
            raise ValueError("Variant-probe partial identity has the wrong type.")
        begin = _integer("global_probe_begin", self.global_probe_begin)
        end = _integer("global_probe_end", self.global_probe_end, minimum=1)
        count = _integer("global_probe_count", self.global_probe_count, minimum=2)
        if not 0 <= begin < end <= count:
            raise ValueError("Variant-probe partial global interval is invalid.")
        if len(self.ordered_probe_sha256s) != end - begin:
            raise ValueError("Variant-probe partial commitment width mismatch.")
        if _ordered_probe_digests(self.ordered_probe_sha256s) != (
            self.ordered_probe_sha256s
        ):
            raise ValueError("Variant-probe partial probe commitments are invalid.")
        _integer("process_id", self.process_id, minimum=1)
        _integer("process_slot", self.process_slot)
        _nonempty("host_name", self.host_name)
        _nonempty("process_start_method", self.process_start_method)
        if (
            _ordered_unique_integers(
                "cpu_affinity", self.cpu_affinity, allow_empty=False
            )
            != self.cpu_affinity
        ):
            raise ValueError("Variant-probe partial CPU-affinity evidence is invalid.")
        if (
            _ordered_unique_integers("socket_ids", self.socket_ids, allow_empty=True)
            != self.socket_ids
        ):
            raise ValueError("Variant-probe partial socket evidence is invalid.")
        if (
            self.probe_sums.ndim != 2
            or self.probe_sums.shape[0] == 0
            or self.probe_sums.shape[1] == 0
            or self.probe_sums.dtype.str != "<f8"
            or not self.probe_sums.flags.c_contiguous
            or self.probe_sums.flags.writeable
            or not np.all(np.isfinite(self.probe_sums))
        ):
            raise ValueError("Variant-probe partial probe_sums storage is invalid.")
        expected_cross = (self.component_count, self.component_count)
        if (
            self.within_probe_cross.shape != expected_cross
            or self.within_probe_cross.dtype.str != "<f8"
            or not self.within_probe_cross.flags.c_contiguous
            or self.within_probe_cross.flags.writeable
            or not np.all(np.isfinite(self.within_probe_cross))
        ):
            raise ValueError(
                "Variant-probe partial within_probe_cross storage is invalid."
            )
        _require_exact_symmetry(
            "Variant-probe partial within_probe_cross",
            self.within_probe_cross,
        )
        for name, observed in (
            ("probe_sums_sha256", array_sha256(self.probe_sums)),
            ("within_probe_cross_sha256", array_sha256(self.within_probe_cross)),
        ):
            if _sha256(name, getattr(self, name)) != observed:
                raise ValueError(f"Variant-probe partial {name} mismatch.")
        expected_plan = _merge_plan_sha256(
            self.identity,
            global_probe_count=count,
            sample_count=self.sample_count,
            component_count=self.component_count,
        )
        if _sha256("merge_plan_sha256", self.merge_plan_sha256) != expected_plan:
            raise ValueError("Variant-probe partial merge-plan digest mismatch.")
        expected_range = _range_commitment_sha256(
            merge_plan_sha256=expected_plan,
            global_probe_begin=begin,
            global_probe_end=end,
            global_probe_count=count,
            ordered_probe_sha256s=self.ordered_probe_sha256s,
        )
        if (
            _sha256("range_commitment_sha256", self.range_commitment_sha256)
            != expected_range
        ):
            raise ValueError("Variant-probe partial range commitment mismatch.")
        expected_partial = _partial_commitment_sha256(
            identity_sha256=self.identity.digest,
            merge_plan_sha256=self.merge_plan_sha256,
            range_commitment_sha256=self.range_commitment_sha256,
            probe_sums_sha256=self.probe_sums_sha256,
            within_probe_cross_sha256=self.within_probe_cross_sha256,
            process_id=self.process_id,
            process_slot=self.process_slot,
            host_name=self.host_name,
            process_start_method=self.process_start_method,
            cpu_affinity=self.cpu_affinity,
            socket_ids=self.socket_ids,
            lifecycle=self.lifecycle,
        )
        if (
            _sha256("partial_commitment_sha256", self.partial_commitment_sha256)
            != expected_partial
        ):
            raise ValueError("Variant-probe partial transport commitment mismatch.")


@dataclass(frozen=True, slots=True)
class ContextualVariantProbeMergeResultV1:
    """Compact signed oracle result from a validated transient raw merge."""

    same_person: np.ndarray
    global_probe_count: int
    identity: ContextualVariantProbeMergeIdentityV1
    provenance: Mapping[str, Any]
    magic: str = field(
        default=CONTEXTUAL_VARIANT_PROBE_MERGE_RESULT_V1_MAGIC, init=False
    )
    lifecycle: str = field(default="oracle_merge_complete", init=False)
    finalizer: str = field(default=NUMPY_ORACLE_TRANSIENT_FINALIZER_V1, init=False)
    production_protected_finalizer: bool = field(default=False, init=False)
    contains_sample_axis: bool = field(default=False, init=False)
    durable_artifact: bool = field(default=False, init=False)
    same_person_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        same_person = _immutable_f64("same_person", self.same_person, ndim=2)
        object.__setattr__(self, "same_person", same_person)
        object.__setattr__(self, "same_person_sha256", array_sha256(same_person))
        object.__setattr__(
            self, "provenance", freeze_context_mapping(dict(self.provenance))
        )
        count = _integer("global_probe_count", self.global_probe_count, minimum=2)
        if same_person.shape[0] == 0 or same_person.shape[0] != same_person.shape[1]:
            raise ValueError(
                "Merged same-person statistic must be square and nonempty."
            )
        if not np.allclose(same_person, same_person.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Merged same-person statistic is not symmetric.")
        if count != self.provenance.get("global_probe_count"):
            raise ValueError("Merged provenance probe count mismatch.")
        if self.provenance.get("finalizer") != NUMPY_ORACLE_TRANSIENT_FINALIZER_V1:
            raise ValueError("Merged provenance finalizer policy mismatch.")


def _socket_ids_for_cpus(cpus: Sequence[int]) -> tuple[int, ...]:
    observed: set[int] = set()
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id")
        try:
            observed.add(int(path.read_text(encoding="ascii").strip()))
        except (FileNotFoundError, OSError, ValueError):
            return ()
    return tuple(sorted(observed))


def _native_readonly_i64_vector(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 1
        or result.dtype != np.dtype(np.int64)
        or not result.flags.c_contiguous
        or result.flags.writeable
    ):
        raise ValueError(f"Native variant-probe {name} must be readonly int64.")
    return result


def _native_exact_names(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"Native variant-probe {name} must be a name sequence.")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result) or len(
        set(result)
    ) != len(result):
        raise ValueError(f"Native variant-probe {name} is invalid.")
    return result


def adapt_native_contextual_variant_probe_partial_v1(
    native_snapshot: Mapping[str, Any],
    *,
    identity: ContextualVariantProbeMergeIdentityV1,
    global_probe_begin: int,
    global_probe_end: int,
    global_probe_count: int,
    ordered_probe_sha256s: Sequence[str],
    process_slot: int,
    process_id: int | None = None,
    cpu_affinity: Sequence[int] | None = None,
    socket_ids: Sequence[int] | None = None,
    host_name: str | None = None,
    process_start_method: str = "spawn",
    probe_sums_layout: str | None = None,
) -> ContextualVariantProbePartialV1:
    """Adapt one authenticated native *full-executor* raw probe range.

    Stage 5 does not claim a native partition-local executor/finalizer.  The
    native snapshot therefore authenticates exactly ``[0,B_D)``; caller
    relabeling as a subrange or under a different science/build identity is
    rejected before constructing a transient partial.
    """
    if not isinstance(native_snapshot, Mapping):
        raise ValueError("Native variant-probe snapshot must be a mapping.")
    try:
        raw_sums = np.asarray(native_snapshot["same_probe_sums"])
        raw_cross = np.asarray(native_snapshot["same_probe_cross"])
        native_identity = native_snapshot["same_probe_native_identity"]
    except KeyError as exc:
        raise ValueError(
            "Native variant-probe snapshot lacks raw state or native identity."
        ) from exc
    if not isinstance(native_identity, Mapping) or set(native_identity) != set(
        _NATIVE_FULL_RANGE_IDENTITY_KEYS_V1
    ):
        raise ValueError("Native variant-probe full-range identity schema mismatch.")
    science_inputs = native_identity.get("science_identity_inputs")
    if not isinstance(science_inputs, Mapping) or set(science_inputs) != set(
        _NATIVE_SCIENCE_IDENTITY_INPUT_KEYS_V1
    ):
        raise ValueError("Native variant-probe science identity schema mismatch.")
    if (
        raw_sums.ndim != 2
        or raw_cross.ndim != 2
        or raw_cross.shape[0] == 0
        or raw_cross.shape[0] != raw_cross.shape[1]
        or raw_sums.dtype != np.dtype(np.float64)
        or raw_cross.dtype != np.dtype(np.float64)
        or not raw_sums.flags.c_contiguous
        or not raw_cross.flags.c_contiguous
        or raw_sums.flags.writeable
        or raw_cross.flags.writeable
    ):
        raise ValueError("Native variant-probe raw arrays are noncanonical.")
    components = raw_cross.shape[0]
    if raw_sums.shape[1] != components:
        raise ValueError("Native variant-probe raw array dimensions disagree.")
    if (
        native_snapshot.get("same_probe_sums_layout") != "sample_component_c_v1"
        or native_snapshot.get("same_probe_cross_layout") != "component_component_c_v1"
        or native_identity.get("probe_sums_layout") != "sample_component_c_v1"
        or native_identity.get("probe_cross_layout") != "component_component_c_v1"
        or (
            probe_sums_layout is not None
            and probe_sums_layout != "sample_component_c_v1"
        )
    ):
        raise ValueError("Native variant-probe raw layout identity mismatch.")

    native_count = _integer(
        "native global_probe_count",
        native_identity.get("global_probe_count"),
        minimum=2,
    )
    native_leaves = _ordered_probe_digests(
        native_identity.get("ordered_probe_sha256s", ())
    )
    native_plan = contextual_variant_probe_plan_sha256_v1(native_leaves)
    if (
        native_identity.get("schema")
        != "contextual_native_variant_probe_full_range_identity_v1"
        or native_identity.get("full_executor_range_only") is not True
        or native_identity.get("partitioned_native_finalizer_claimed") is not False
        or native_identity.get("global_probe_begin") != 0
        or native_identity.get("global_probe_end") != native_count
        or len(native_leaves) != native_count
        or _sha256(
            "native global_probe_plan_sha256",
            native_identity.get("global_probe_plan_sha256"),
        )
        != native_plan
        or global_probe_begin != 0
        or global_probe_end != native_count
        or global_probe_count != native_count
        or tuple(ordered_probe_sha256s) != native_leaves
    ):
        raise ValueError("Native variant-probe full-range commitment mismatch.")

    pair_q = _native_readonly_i64_vector("pair_q", science_inputs["pair_q"])
    pair_r = _native_readonly_i64_vector("pair_r", science_inputs["pair_r"])
    pair_eta = _native_readonly_i64_vector("pair_eta", science_inputs["pair_eta"])
    component_annotation = _native_readonly_i64_vector(
        "component_annotation", science_inputs["component_annotation"]
    )
    component_pair = _native_readonly_i64_vector(
        "component_pair", science_inputs["component_pair"]
    )
    if (
        not pair_q.size
        or pair_q.shape != pair_r.shape
        or pair_q.shape != pair_eta.shape
    ):
        raise ValueError("Native variant-probe pair arrays disagree.")
    if np.any(pair_q < 0) or np.any(pair_r < 0):
        raise ValueError("Native variant-probe pair coordinates are invalid.")
    pair_index = ContextPairIndex(int(max(pair_q.max(), pair_r.max())) + 1)
    expected_pair_q = np.asarray([entry.q for entry in pair_index.entries])
    expected_pair_r = np.asarray([entry.r for entry in pair_index.entries])
    expected_pair_eta = np.asarray(
        [entry.kernel_factor for entry in pair_index.entries]
    )
    annotation_names = _native_exact_names(
        "annotation_names", science_inputs["annotation_names"]
    )
    group_names = _native_exact_names("group_names", science_inputs["group_names"])
    component_index = ContextComponentIndex(annotation_names, pair_index)
    expected_component_annotation = np.asarray(
        [entry.annotation_index for entry in component_index.entries]
    )
    expected_component_pair = np.asarray(
        [entry.pair_index for entry in component_index.entries]
    )
    if (
        not np.array_equal(pair_q, expected_pair_q)
        or not np.array_equal(pair_r, expected_pair_r)
        or not np.array_equal(pair_eta, expected_pair_eta)
        or not np.array_equal(component_annotation, expected_component_annotation)
        or not np.array_equal(component_pair, expected_component_pair)
        or len(component_index) != components
    ):
        raise ValueError("Native variant-probe pair/component maps are noncanonical.")

    science_identity = {
        name: _sha256(name, science_inputs[name])
        for name in (
            "annotation_map_sha256",
            "evaluated_phi_sha256",
            "fixed_basis_sha256",
            "group_map_sha256",
            "missingness_sha256",
            "retained_sample_map_sha256",
            "retained_variant_order_sha256",
            "scale_plan_sha256",
            "variant_order_allele_sha256",
        )
    }
    science_identity.update(
        {
            "annotation_names_sha256": canonical_sha256(
                {"axis": "annotation", "names": list(annotation_names)}
            ),
            "group_names_sha256": canonical_sha256(
                {"axis": "deletion_group", "names": list(group_names)}
            ),
            "pair_map_sha256": pair_index.digest,
            "component_map_sha256": component_index.digest,
        }
    )
    if dict(identity.science_identity) != science_identity:
        raise ValueError("Native variant-probe science identity graft rejected.")

    source_commit = native_identity.get("source_commit")
    build_provenance = native_identity.get("build_provenance")
    if not isinstance(build_provenance, Mapping) or set(build_provenance) != set(
        _NATIVE_BUILD_PROVENANCE_KEYS_V1
    ):
        raise ValueError("Native variant-probe build provenance schema mismatch.")
    validate_native_build_provenance_consistency(
        build_provenance,
        family="Native variant-probe snapshot",
    )
    build_provenance_sha256 = canonical_sha256(dict(build_provenance))
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or native_identity.get("build_id") != source_commit
        or identity.build_id != source_commit
        or build_provenance.get("schema") != "contextual_native_build_provenance_v1"
        or build_provenance.get("source_commit") != source_commit
        or build_provenance.get("source_tree_sha256")
        != native_identity.get("source_tree_sha256")
        or build_provenance.get("contextual_dispatch_backend")
        != native_identity.get("native_execution_backend")
        or build_provenance.get("contextual_dispatch_vendor_calls") is not False
        or identity.native_build_provenance_sha256 != build_provenance_sha256
        or identity.source_tree_sha256
        != _sha256(
            "native source_tree_sha256", native_identity.get("source_tree_sha256")
        )
        or identity.native_api_version != native_identity.get("native_api_version")
        or identity.native_backend_version
        != native_identity.get("native_backend_version")
        or identity.native_backend != native_identity.get("native_backend")
        or identity.native_execution_backend
        != native_identity.get("native_execution_backend")
        or identity.numeric_policy != native_identity.get("numeric_policy")
        or identity.feature_mode != native_identity.get("feature_mode")
        or identity.variant_probe_policy != native_identity.get("variant_probe_policy")
        or identity.global_probe_plan_sha256 != native_plan
    ):
        raise ValueError("Native variant-probe build or probe identity graft rejected.")
    for name in (
        "execution_plan_sha256",
        "sealed_plan_sha256",
        "native_variant_probe_identity_sha256",
    ):
        _sha256(f"native {name}", native_identity.get(name))
    if (
        native_identity.get("sample_count") != raw_sums.shape[0]
        or native_identity.get("component_count") != components
        or native_snapshot.get("same_probe_sample_count") != raw_sums.shape[0]
        or native_snapshot.get("same_probe_component_count") != components
        or native_snapshot.get("same_probe_variant_probe_count") != native_count
        or native_snapshot.get("same_probe_variant_probe_identity_sha256")
        != native_identity.get("native_variant_probe_identity_sha256")
        or native_snapshot.get("same_probe_retained_sample_map_sha256")
        != science_identity["retained_sample_map_sha256"]
        or native_snapshot.get("same_probe_component_annotation_sha256")
        != array_sha256(component_annotation)
        or native_snapshot.get("same_probe_component_pair_sha256")
        != array_sha256(component_pair)
        or _sha256(
            "same_probe_sums_sha256",
            native_snapshot.get("same_probe_sums_sha256"),
        )
        != array_sha256(raw_sums)
        or _sha256(
            "same_probe_cross_sha256",
            native_snapshot.get("same_probe_cross_sha256"),
        )
        != array_sha256(raw_cross)
    ):
        raise ValueError("Native variant-probe raw evidence mismatch.")

    probe_sums = raw_sums.T

    if cpu_affinity is None:
        if not hasattr(os, "sched_getaffinity"):
            raise ValueError("CPU-affinity evidence must be supplied on this platform.")
        cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if socket_ids is None:
        socket_ids = _socket_ids_for_cpus(cpu_affinity)
    partial = ContextualVariantProbePartialV1(
        probe_sums=probe_sums,
        within_probe_cross=raw_cross,
        global_probe_begin=global_probe_begin,
        global_probe_end=global_probe_end,
        global_probe_count=global_probe_count,
        ordered_probe_sha256s=tuple(ordered_probe_sha256s),
        identity=identity,
        process_id=os.getpid() if process_id is None else process_id,
        process_slot=process_slot,
        cpu_affinity=tuple(cpu_affinity),
        socket_ids=tuple(socket_ids),
        host_name=socket.gethostname() if host_name is None else host_name,
        process_start_method=process_start_method,
    )
    return partial


def merge_contextual_variant_probe_partials_v1(
    partials: Iterable[ContextualVariantProbePartialV1],
) -> ContextualVariantProbeMergeResultV1:
    """Merge raw partials and finalize a compact signed NumPy oracle statistic.

    This function does not satisfy the production protected-operation contract.
    It exists to qualify process partitioning and to provide the strict boundary
    that a native ``same_person_global_merge_tn`` finalizer can consume.
    """
    values = tuple(partials)
    if not values:
        raise ValueError("At least one variant-probe partial is required.")
    if any(not isinstance(value, ContextualVariantProbePartialV1) for value in values):
        raise TypeError("Variant-probe merge accepts only V1 partials.")
    for value in values:
        value.validate()
        if value.lifecycle != "partial_complete":
            raise ValueError(
                "Already-finalized variant-probe partial is not mergeable."
            )

    ordered = tuple(
        sorted(
            values, key=lambda item: (item.global_probe_begin, item.global_probe_end)
        )
    )
    first = ordered[0]
    cursor = 0
    leaves: list[str] = []
    seen_ranges: set[tuple[int, int]] = set()
    for value in ordered:
        interval = (value.global_probe_begin, value.global_probe_end)
        if interval in seen_ranges:
            raise ValueError("Duplicate variant-probe interval in merge.")
        seen_ranges.add(interval)
        if value.global_probe_begin < cursor:
            raise ValueError("Overlapping variant-probe intervals in merge.")
        if value.global_probe_begin > cursor:
            raise ValueError("Gap in global variant-probe interval coverage.")
        if (
            value.identity != first.identity
            or value.global_probe_count != first.global_probe_count
            or value.sample_count != first.sample_count
            or value.component_count != first.component_count
            or value.merge_plan_sha256 != first.merge_plan_sha256
        ):
            raise ValueError("Mixed identity or dimensions in variant-probe merge.")
        leaves.extend(value.ordered_probe_sha256s)
        cursor = value.global_probe_end
    if cursor != first.global_probe_count:
        raise ValueError("Global variant-probe coverage is incomplete.")
    if (
        contextual_variant_probe_plan_sha256_v1(leaves)
        != first.identity.global_probe_plan_sha256
    ):
        raise ValueError("Global ordered variant-probe commitment mismatch.")

    probe_sum_accumulator = np.zeros(
        (first.component_count, first.sample_count), dtype=np.float64, order="C"
    )
    within_accumulator = np.zeros(
        (first.component_count, first.component_count),
        dtype=np.float64,
        order="C",
    )
    numerator: np.ndarray | None = None
    same_person_work: np.ndarray | None = None
    try:
        for value in ordered:
            np.add(probe_sum_accumulator, value.probe_sums, out=probe_sum_accumulator)
            np.add(
                within_accumulator,
                value.within_probe_cross,
                out=within_accumulator,
            )
        _require_exact_symmetry(
            "Merged within-probe accumulator",
            within_accumulator,
        )
        numerator = probe_sum_accumulator @ probe_sum_accumulator.T
        np.subtract(numerator, within_accumulator, out=numerator)
        _require_exact_symmetry("Merged same-person numerator", numerator)
        denominator = float(first.global_probe_count) * float(
            first.global_probe_count - 1
        )
        same_person_work = numerator / denominator
        if not np.all(np.isfinite(same_person_work)):
            raise ValueError("Merged same-person statistic is nonfinite.")
        _require_exact_symmetry(
            "Merged same-person statistic before publication",
            same_person_work,
        )
        # The scientific statistic is symmetric; match native publication while
        # preserving sign and avoiding any PSD projection or clipping.
        same_person_work = 0.5 * (same_person_work + same_person_work.T)
        provenance = {
            "finalizer": NUMPY_ORACLE_TRANSIENT_FINALIZER_V1,
            "production_protected_finalizer": False,
            "merge_order": "global_probe_interval_ascending_v1",
            "global_probe_count": first.global_probe_count,
            "partial_count": len(ordered),
            "identity_sha256": first.identity.digest,
            "merge_plan_sha256": first.merge_plan_sha256,
            "global_probe_plan_sha256": (first.identity.global_probe_plan_sha256),
            "signed_output_preserved": True,
            "input_partials_mutated": False,
            "local_accumulator_disposition": "zeroed_before_return_v1",
            "ranges": [
                {
                    "global_probe_begin": value.global_probe_begin,
                    "global_probe_end": value.global_probe_end,
                    "range_commitment_sha256": value.range_commitment_sha256,
                    "probe_sums_sha256": value.probe_sums_sha256,
                    "within_probe_cross_sha256": (value.within_probe_cross_sha256),
                    "process_id": value.process_id,
                    "process_slot": value.process_slot,
                    "host_name": value.host_name,
                    "process_start_method": value.process_start_method,
                    "cpu_affinity": list(value.cpu_affinity),
                    "socket_ids": list(value.socket_ids),
                }
                for value in ordered
            ],
        }
        return ContextualVariantProbeMergeResultV1(
            same_person=same_person_work,
            global_probe_count=first.global_probe_count,
            identity=first.identity,
            provenance=provenance,
        )
    finally:
        probe_sum_accumulator.fill(0.0)
        within_accumulator.fill(0.0)
        if numerator is not None:
            numerator.fill(0.0)
        if same_person_work is not None and same_person_work.flags.writeable:
            same_person_work.fill(0.0)


__all__ = [
    "CONTEXTUAL_VARIANT_PROBE_MERGE_RESULT_V1_MAGIC",
    "CONTEXTUAL_VARIANT_PROBE_PARTIAL_V1_MAGIC",
    "CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1",
    "NUMPY_ORACLE_TRANSIENT_FINALIZER_V1",
    "ContextualVariantProbeMergeIdentityV1",
    "ContextualVariantProbeMergeResultV1",
    "ContextualVariantProbePartialV1",
    "adapt_native_contextual_variant_probe_partial_v1",
    "contextual_variant_probe_plan_sha256_v1",
    "merge_contextual_variant_probe_partials_v1",
]
