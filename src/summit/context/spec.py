"""Versioned specifications for the experimental contextual-covariance path.

This module deliberately has no dependency on the production GxE schemas.  It
defines the small, auditable contracts used by the correctness-first Python
implementation; existing SUMMIT defaults and artifacts are unaffected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CONTEXT_SCHEMA_VERSION = 1
CONTEXT_BASIS_KIND = "summit.context.basis"
CONTEXT_REFERENCE_KIND = "summit.context.reference"
CONTEXT_TRAIT_KIND = "summit.context.trait_summary"
CONTEXT_FIT_KIND = "summit.context.fit"
RAW_PROJECTED_FEATURE_MODE = "raw_projected"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_COLUMN_KINDS = frozenset(
    {"constant", "linear", "standardized", "polynomial", "one_hot", "precomputed"}
)
_MANIFEST_KINDS = frozenset(
    {CONTEXT_REFERENCE_KIND, CONTEXT_TRAIT_KIND, CONTEXT_FIT_KIND}
)
_HASH_FIELDS = (
    "basis_hash",
    "fixed_effect_hash",
    "variant_hash",
    "annotation_hash",
    "component_index_hash",
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return the unique compact JSON representation used for hashing."""
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Context specification is not canonical JSON data.") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def array_sha256(value: Any) -> str:
    """Hash one canonical array value with dtype and shape in the identity.

    Numeric payloads are serialized little-endian so the same logical array
    has one digest on little- and big-endian hosts.  Object arrays are rejected
    because their bytes contain process-local pointers rather than a stable
    scientific representation.
    """
    source = np.asarray(value)
    if source.dtype.hasobject:
        raise ValueError("Context array digests do not support object dtype.")
    if source.dtype.kind not in "biufcSU":
        raise ValueError(
            "Context array digests support only scalar numeric, boolean, or string dtype."
        )
    canonical_dtype = source.dtype.newbyteorder("<")
    array = np.ascontiguousarray(source, dtype=canonical_dtype)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def owned_readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return owned C-contiguous storage with no writable caller alias."""
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.hasobject:
        raise ValueError("Context artifacts do not support object arrays.")
    if array.dtype.kind not in "biufc":
        raise ValueError("Context artifacts support only numeric and boolean arrays.")
    array.setflags(write=False)
    return array


class FrozenList(list[Any]):
    """JSON-compatible list whose in-place mutation operations are disabled."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Context artifact metadata is immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(item, memo) for item in self]


class FrozenDict(dict[str, Any]):
    """JSON-compatible mapping whose in-place mutation operations are disabled."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Context artifact metadata is immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self.items()
        }


def freeze_context_value(value: Any) -> Any:
    """Defensively own and recursively freeze JSON/array artifact state."""
    if isinstance(value, np.ndarray):
        return owned_readonly_array(value)
    if isinstance(value, Mapping):
        result = FrozenDict()
        for key, item in value.items():
            dict.__setitem__(result, str(key), freeze_context_value(item))
        return result
    if isinstance(value, list):
        result = FrozenList()
        list.extend(result, (freeze_context_value(item) for item in value))
        return result
    if isinstance(value, tuple):
        return tuple(freeze_context_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_context_value(item) for item in value)
    return value


def freeze_context_mapping(value: Mapping[str, Any]) -> FrozenDict:
    result = freeze_context_value(value)
    if not isinstance(result, FrozenDict):
        raise TypeError("Context artifact metadata must be a mapping.")
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_name(label: str, value: str) -> str:
    value = str(value)
    if not _NAME.fullmatch(value):
        raise ValueError(f"{label} must match {_NAME.pattern!r}; got {value!r}.")
    return value


def _parameter_mapping(parameters: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters:
        if key in result:
            raise ValueError(f"Duplicate basis-column parameter {key!r}.")
        result[str(key)] = value
    canonical_json(result)
    return result


def _typed_one_hot_equal(value: Any, category: Any) -> bool:
    """Compare supported scalar categories without bool/int/string coercion."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(category, np.generic):
        category = category.item()

    def scalar_kind(item: Any) -> type[Any] | None:
        if isinstance(item, bool):
            return bool
        if isinstance(item, int):
            return int
        if isinstance(item, float):
            return float
        if isinstance(item, str):
            return str
        return None

    kind = scalar_kind(value)
    return (
        kind is not None and kind is scalar_kind(category) and bool(value == category)
    )


@dataclass(frozen=True)
class BasisColumnSpec:
    """One deterministic context-basis column."""

    name: str
    kind: str
    source: str | None = None
    include_fixed_effect: bool = True
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        parameters = tuple(
            (str(key), freeze_context_value(value)) for key, value in self.parameters
        )
        object.__setattr__(self, "parameters", parameters)
        _validate_name("basis column name", self.name)
        if self.kind not in _COLUMN_KINDS:
            raise ValueError(
                f"Unsupported basis column kind {self.kind!r}; expected one of "
                f"{sorted(_COLUMN_KINDS)}."
            )
        if self.kind != "constant":
            if self.source is None:
                raise ValueError(f"Basis column {self.name!r} requires a source.")
            _validate_name("basis column source", self.source)
        elif self.source is not None:
            raise ValueError("A constant basis column cannot declare a source.")
        if not isinstance(self.include_fixed_effect, bool):
            raise ValueError("include_fixed_effect must be boolean.")
        params = _parameter_mapping(self.parameters)
        if self.kind == "standardized":
            if set(params) != {"center", "scale"}:
                raise ValueError(
                    "A standardized basis column requires fixed center and scale."
                )
            center = float(params["center"])
            scale = float(params["scale"])
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "Standardization center/scale must be finite and scale > 0."
                )
        elif self.kind == "polynomial":
            if set(params) != {"power"}:
                raise ValueError(
                    "A polynomial basis column requires exactly one power."
                )
            power = params["power"]
            if isinstance(power, bool) or not isinstance(power, int) or power < 1:
                raise ValueError("Polynomial power must be a positive integer.")
        elif self.kind == "one_hot":
            if set(params) != {"category"} or params["category"] is None:
                raise ValueError("A one-hot basis column requires a non-null category.")
        elif params:
            raise ValueError(f"Basis kind {self.kind!r} does not accept parameters.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BasisColumnSpec":
        allowed = {"name", "kind", "source", "include_fixed_effect", "parameters"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown basis-column fields: {sorted(unknown)}.")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("basis-column parameters must be an object.")
        return cls(
            name=str(payload.get("name", "")),
            kind=str(payload.get("kind", "")),
            source=(None if payload.get("source") is None else str(payload["source"])),
            include_fixed_effect=payload.get("include_fixed_effect", True),
            parameters=tuple(sorted(parameters.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "include_fixed_effect": self.include_fixed_effect,
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.parameters:
            payload["parameters"] = _parameter_mapping(self.parameters)
        return payload

    def evaluate(self, data: Mapping[str, Any], n_samples: int) -> np.ndarray:
        if self.kind == "constant":
            return np.ones(n_samples, dtype=np.float64)
        assert self.source is not None
        if self.source not in data:
            raise ValueError(
                f"Basis column {self.name!r} requires missing source {self.source!r}."
            )
        raw = np.asarray(
            data[self.source], dtype=object if self.kind == "one_hot" else None
        )
        if raw.ndim != 1 or raw.shape[0] != n_samples:
            raise ValueError(
                f"Basis source {self.source!r} has shape {raw.shape}; "
                f"expected ({n_samples},)."
            )
        params = _parameter_mapping(self.parameters)
        if self.kind == "one_hot":
            values = np.fromiter(
                (
                    _typed_one_hot_equal(value, params["category"])
                    for value in raw.tolist()
                ),
                dtype=np.float64,
                count=n_samples,
            )
        else:
            try:
                values = raw.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Basis source {self.source!r} must be numeric for {self.kind}."
                ) from exc
            if self.kind == "standardized":
                values = (values - float(params["center"])) / float(params["scale"])
            elif self.kind == "polynomial":
                values = np.power(values, int(params["power"]), dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Basis column {self.name!r} contains non-finite values.")
        return values


@dataclass(frozen=True)
class ContextBasisSpec:
    """Versioned, hash-stable fixed context basis."""

    basis_id: str
    columns: tuple[BasisColumnSpec, ...]
    feature_mode: str = RAW_PROJECTED_FEATURE_MODE
    schema_version: int = CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        _validate_name("basis_id", self.basis_id)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONTEXT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported context basis schema version {self.schema_version}."
            )
        if self.feature_mode != RAW_PROJECTED_FEATURE_MODE:
            raise ValueError(
                "The experimental contextual path requires feature_mode='raw_projected'; "
                f"got {self.feature_mode!r}."
            )
        if not self.columns:
            raise ValueError("Context basis must contain at least one column.")
        names = tuple(column.name for column in self.columns)
        if len(set(names)) != len(names):
            raise ValueError("Context basis column names must be unique.")
        categories: dict[str, list[Any]] = {}
        for column in self.columns:
            if column.kind == "one_hot":
                assert column.source is not None
                categories.setdefault(column.source, []).append(
                    _parameter_mapping(column.parameters)["category"]
                )
        for source, values in categories.items():
            encoded = [canonical_json({"value": value}) for value in values]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"One-hot categories for {source!r} must be unique.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextBasisSpec":
        allowed = {"kind", "schema_version", "basis_id", "feature_mode", "columns"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown context-basis fields: {sorted(unknown)}.")
        if payload.get("kind") != CONTEXT_BASIS_KIND:
            raise ValueError(f"Context basis kind must be {CONTEXT_BASIS_KIND!r}.")
        columns = payload.get("columns")
        if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
            raise ValueError("Context basis columns must be an array.")
        return cls(
            basis_id=str(payload.get("basis_id", "")),
            columns=tuple(BasisColumnSpec.from_dict(column) for column in columns),
            feature_mode=str(payload.get("feature_mode", "")),
            schema_version=payload.get("schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": CONTEXT_BASIS_KIND,
            "schema_version": self.schema_version,
            "basis_id": self.basis_id,
            "feature_mode": self.feature_mode,
            "columns": [column.to_dict() for column in self.columns],
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def fixed_effect_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, column in enumerate(self.columns)
            if column.include_fixed_effect
        )

    def evaluate(
        self, data: Mapping[str, Any], *, n_samples: int | None = None
    ) -> np.ndarray:
        if n_samples is None:
            lengths = {
                np.asarray(value).shape[0]
                for value in data.values()
                if np.asarray(value).ndim >= 1
            }
            if len(lengths) != 1:
                raise ValueError(
                    "Cannot infer one context sample size; provide n_samples explicitly."
                )
            n_samples = lengths.pop()
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")
        result = np.column_stack(
            [column.evaluate(data, n_samples) for column in self.columns]
        )
        if result.shape != (n_samples, len(self.columns)):
            raise AssertionError("Internal context basis shape error.")
        return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True)
class ContextPair:
    index: int
    q: int
    r: int
    kernel_factor: int

    @property
    def is_off_diagonal(self) -> bool:
        return self.q != self.r

    @property
    def name(self) -> str:
        return f"basis:{self.q},{self.r}"

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "q": self.q,
            "r": self.r,
            "kernel_factor": self.kernel_factor,
        }


@dataclass(frozen=True)
class ContextPairIndex:
    """Diagonal-first unordered basis-pair index."""

    num_basis: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_basis, bool)
            or not isinstance(self.num_basis, int)
            or self.num_basis < 1
        ):
            raise ValueError("num_basis must be a positive integer.")

    @property
    def entries(self) -> tuple[ContextPair, ...]:
        pairs = [(q, q) for q in range(self.num_basis)]
        pairs.extend(
            (q, r)
            for q in range(self.num_basis - 1)
            for r in range(q + 1, self.num_basis)
        )
        return tuple(
            ContextPair(index, q, r, 1 if q == r else 2)
            for index, (q, r) in enumerate(pairs)
        )

    def __len__(self) -> int:
        return self.num_basis * (self.num_basis + 1) // 2

    def pair(self, index: int) -> ContextPair:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"Pair index {index!r} is out of range.")
        try:
            return self.entries[index]
        except IndexError as exc:
            raise ValueError(f"Pair index {index} is out of range.") from exc

    def index(self, q: int, r: int) -> int:
        if q > r:
            q, r = r, q
        for entry in self.entries:
            if (entry.q, entry.r) == (q, r):
                return entry.index
        raise ValueError(f"Invalid basis pair ({q}, {r}) for Q={self.num_basis}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordering": "diagonal_then_lexicographic_off_diagonal",
            "num_basis": self.num_basis,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ContextComponent:
    index: int
    annotation_index: int
    annotation_name: str
    pair_index: int
    q: int
    r: int
    kernel_factor: int

    @property
    def name(self) -> str:
        return f"context:{self.annotation_name}:{self.q},{self.r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "annotation_index": self.annotation_index,
            "annotation_name": self.annotation_name,
            "pair_index": self.pair_index,
            "q": self.q,
            "r": self.r,
            "kernel_factor": self.kernel_factor,
            "name": self.name,
        }


@dataclass(frozen=True)
class ContextComponentIndex:
    """Annotation-major Cartesian product of annotations and context pairs."""

    annotation_names: tuple[str, ...]
    pair_index: ContextPairIndex

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "annotation_names", tuple(str(name) for name in self.annotation_names)
        )
        if not self.annotation_names:
            raise ValueError("At least one annotation is required.")
        for name in self.annotation_names:
            _validate_name("annotation name", name)
        if len(set(self.annotation_names)) != len(self.annotation_names):
            raise ValueError("Annotation names must be unique.")

    @property
    def entries(self) -> tuple[ContextComponent, ...]:
        result: list[ContextComponent] = []
        for annotation_index, annotation_name in enumerate(self.annotation_names):
            for pair in self.pair_index.entries:
                result.append(
                    ContextComponent(
                        index=len(result),
                        annotation_index=annotation_index,
                        annotation_name=annotation_name,
                        pair_index=pair.index,
                        q=pair.q,
                        r=pair.r,
                        kernel_factor=pair.kernel_factor,
                    )
                )
        return tuple(result)

    def __len__(self) -> int:
        return len(self.annotation_names) * len(self.pair_index)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordering": "annotation_major_pair_minor",
            "annotation_names": list(self.annotation_names),
            "pair_index": self.pair_index.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


def validate_context_manifest(
    payload: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a private version-1 reference or trait manifest fail-closed."""
    if not isinstance(payload, Mapping):
        raise ValueError("Context manifest must be an object.")
    result = dict(payload)
    kind = result.get("kind")
    if kind not in _MANIFEST_KINDS:
        raise ValueError(f"Unsupported context manifest kind {kind!r}.")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(
            f"Context manifest kind mismatch: expected {expected_kind!r}, got {kind!r}."
        )
    schema_version = result.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported context manifest schema version.")
    if result.get("feature_mode") != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Context manifest does not declare raw_projected features.")
    for field in _HASH_FIELDS:
        if not _is_sha256(result.get(field)):
            raise ValueError(
                f"Context manifest field {field!r} is not a SHA-256 digest."
            )
    loo_hash = result.get("loo_grouping_hash")
    if loo_hash is not None and not _is_sha256(loo_hash):
        raise ValueError("loo_grouping_hash must be a SHA-256 digest or null.")
    dimensions = result.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("Context manifest dimensions must be an object.")
    required_dimensions = {"n_samples", "residual_rank", "n_variants", "q", "k"}
    if not required_dimensions.issubset(dimensions):
        missing = required_dimensions - set(dimensions)
        raise ValueError(f"Context manifest dimensions are missing {sorted(missing)}.")
    for field in required_dimensions:
        value = dimensions[field]
        minimum = 0 if field == "residual_rank" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Context manifest dimension {field!r} is invalid.")
    if dimensions["residual_rank"] > dimensions["n_samples"]:
        raise ValueError("Residual rank cannot exceed sample size.")
    if expected:
        for field, value in expected.items():
            if result.get(field) != value:
                raise ValueError(
                    f"Context manifest mismatch for {field!r}: "
                    f"expected {value!r}, got {result.get(field)!r}."
                )
    canonical_json(result)
    return result
