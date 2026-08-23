"""Private bounded I/O helpers for the stable contextual NPZ families.

The public artifact modules own their schemas.  This module only owns the
container-level rules which must be identical across those families: inspect
ZIP and NPY headers before scientific payload decompression, reject ambiguous
member tables, and make an atomic no-replace publication durable in its parent
directory.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import platform
import re
import stat
import struct
import zipfile
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np


_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 64 * 1024
# Stable artifacts are compact summaries, never row- or variant-axis stores.
# These are format admission caps, not a claim that allocations below the caps
# are safe in every caller's resource environment.
_MAX_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
_MAX_CLASSIC_ZIP_FIELD = zipfile.ZIP64_LIMIT
_MAX_CLASSIC_ZIP_MEMBERS = zipfile.ZIP_FILECOUNT_LIMIT
_MAX_CENTRAL_DIRECTORY_BYTES_PER_MEMBER = 46 + 1024 + _MAX_NPY_HEADER_BYTES
_MEMBER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*[.]npy")
_UNICODE_DESCRIPTOR = re.compile(r"<U([1-9][0-9]*)")
_ALLOWED_COMPRESSION = frozenset((zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED))
_PROVENANCE_SCHEMA = "summit_python_source_runtime_provenance_v1"
_PROVENANCE_NONCLAIM = "provenance_only_no_binary_reproducibility_claim_v1"
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_native_fault_selector_targets_event(
    selector: object,
    *,
    operation: str,
    event: Mapping[str, object],
    grouped: bool,
    identity_required: bool,
    family: str,
) -> None:
    """Bind a native point selector to the protected range that consumed it."""
    if not isinstance(selector, str) or not selector:
        raise ValueError(f"{family} fault selector is invalid.")
    fields = selector.split("|")
    if len(fields) < 7 or fields[0] != operation:
        raise ValueError(f"{family} fault selector operation is invalid.")

    def parse_uint(field: str, prefix: str) -> int:
        if not field.startswith(prefix):
            raise ValueError(f"{family} fault selector field order is invalid.")
        text = field[len(prefix) :]
        if not text.isdecimal() or (len(text) > 1 and text.startswith("0")):
            raise ValueError(f"{family} fault selector integer is noncanonical.")
        return int(text)

    axes = ("resident", "probe", "variant", "annotation", "context", "action")
    points = {
        axis: parse_uint(fields[index], f"{axis}=")
        for index, axis in enumerate(axes, start=1)
    }
    extras: dict[str, str] = {}
    for field in fields[7:]:
        if "=" not in field:
            raise ValueError(f"{family} fault selector extension is invalid.")
        name, value = field.split("=", 1)
        if name in extras or name not in {
            "group",
            "phase",
            "role",
            "placement",
            "canonical",
        }:
            raise ValueError(f"{family} fault selector extension is invalid.")
        extras[name] = value
    if grouped != ("group" in extras):
        raise ValueError(f"{family} grouped fault selector is invalid.")
    identity_fields = {"phase", "role", "placement", "canonical"}
    if identity_required:
        if set(extras) - {"group"} != identity_fields:
            raise ValueError(f"{family} semantic fault identity is incomplete.")
    elif set(extras) - {"group"}:
        raise ValueError(f"{family} semantic fault identity is unexpected.")

    def contains(point: int, begin: object, end: object) -> bool:
        if (
            isinstance(begin, bool)
            or isinstance(end, bool)
            or not isinstance(begin, (int, np.integer))
            or not isinstance(end, (int, np.integer))
        ):
            return False
        lower, upper = int(begin), int(end)
        return point == lower if lower == upper else lower <= point < upper

    for axis, point in points.items():
        if not contains(point, event.get(f"{axis}_begin"), event.get(f"{axis}_end")):
            raise ValueError(f"{family} fault selector misses its protected range.")
    if grouped and not contains(
        parse_uint(f"group={extras['group']}", "group="),
        event.get("group_begin"),
        event.get("group_end"),
    ):
        raise ValueError(f"{family} fault selector misses its protected group.")
    if identity_required:
        if (
            extras["phase"] != event.get("semantic_phase")
            or extras["role"] != event.get("semantic_role")
            or extras["placement"] != event.get("semantic_placement")
            or not contains(
                parse_uint(f"canonical={extras['canonical']}", "canonical="),
                event.get("canonical_begin"),
                event.get("canonical_end"),
            )
        ):
            raise ValueError(f"{family} semantic fault identity is inconsistent.")


def validate_native_build_provenance_consistency(
    value: Mapping[str, object], *, family: str
) -> None:
    """Close mutually dependent compiler, sanitizer, tuning, and BLAS fields."""
    source_commit = value.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or _SOURCE_COMMIT.fullmatch(source_commit) is None
    ):
        raise ValueError(f"{family} native source commit is invalid.")
    required_strings = (
        "compiler_id",
        "compiler_version",
        "build_type",
        "sanitizer_mode",
        "effective_optimization",
        "architecture_tuning",
        "configured_compiler_flags",
        "blas_vendor",
        "private_blas_backend",
        "private_blas_source_commit",
        "private_blas_config_family",
    )
    if any(
        not isinstance(value.get(name), str) or not value[name]
        for name in required_strings
    ):
        raise ValueError(f"{family} native build string is invalid.")
    required_flags = (
        "gemm_integrity_enabled",
        "gemm_checksum_enabled",
        "private_blas_enabled",
        "private_openblas_enabled",
        "native_arch_optimization_enabled",
        "openmp_enabled",
        "asan_enabled",
        "ubsan_enabled",
    )
    if any(type(value.get(name)) is not bool for name in required_flags):
        raise ValueError(f"{family} native build flag is invalid.")

    sanitizer = value["sanitizer_mode"]
    sanitizer_contract = {
        "none": (False, False, "-O3", None),
        "asan_ubsan": (True, True, "-O1", "-fsanitize=address,undefined"),
        "ubsan_only": (False, True, "-O1", "-fsanitize=undefined"),
    }
    expected = sanitizer_contract.get(sanitizer)
    flags = value["configured_compiler_flags"]
    if (
        expected is None
        or (value["asan_enabled"], value["ubsan_enabled"]) != expected[:2]
        or value["effective_optimization"] != expected[2]
        or expected[2] not in flags
        or (expected[3] is None and "-fsanitize=" in flags)
        or (expected[3] is not None and expected[3] not in flags)
    ):
        raise ValueError(f"{family} native sanitizer provenance is contradictory.")

    native_tuning = value["native_arch_optimization_enabled"]
    if native_tuning:
        tuning_valid = (
            value["architecture_tuning"] == "-march=native" and "-march=native" in flags
        )
    else:
        tuning_valid = (
            value["architecture_tuning"] == "portable" and "-march=native" not in flags
        )
    if not tuning_valid:
        raise ValueError(f"{family} native architecture provenance is contradictory.")

    def digest(name: str) -> str:
        observed = value.get(name)
        if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
            raise ValueError(f"{family} native {name} is invalid.")
        return observed

    private_enabled = value["private_blas_enabled"]
    private_openblas = value["private_openblas_enabled"]
    backend = value["private_blas_backend"]
    private_digest_names = (
        "private_blas_sha256",
        "private_blas_source_tree_sha256",
        "private_blas_header_sha256",
        "private_blas_cblas_header_sha256",
        "private_openblas_sha256",
    )
    if not private_enabled:
        if (
            private_openblas
            or backend != "none"
            or value["private_blas_source_commit"] != "none"
            or value["private_blas_config_family"] != "none"
            or any(value.get(name) != "none" for name in private_digest_names)
        ):
            raise ValueError(f"{family} disabled private-BLAS provenance disagrees.")
    elif backend == "openblas":
        private_sha = digest("private_blas_sha256")
        openblas_sha = digest("private_openblas_sha256")
        if (
            not private_openblas
            or private_sha != openblas_sha
            or value["private_blas_source_commit"] != "none"
            or value["private_blas_source_tree_sha256"] != "none"
            or value["private_blas_config_family"] != "none"
            or value["private_blas_header_sha256"] != "none"
            or value["private_blas_cblas_header_sha256"] != "none"
        ):
            raise ValueError(f"{family} private OpenBLAS provenance disagrees.")
    elif backend == "upstream_blis":
        if private_openblas or value["private_openblas_sha256"] != "none":
            raise ValueError(f"{family} private BLIS provenance disagrees.")
        digest("private_blas_sha256")
        for name in (
            "private_blas_source_tree_sha256",
            "private_blas_header_sha256",
            "private_blas_cblas_header_sha256",
        ):
            digest(name)
        if (
            _SOURCE_COMMIT.fullmatch(value["private_blas_source_commit"]) is None
            or value["private_blas_config_family"] == "none"
        ):
            raise ValueError(f"{family} private BLIS provenance disagrees.")
    else:
        raise ValueError(f"{family} private-BLAS backend is unsupported.")


def python_source_runtime_provenance(
    ordered_modules: tuple[str, ...], *, policy: str, include_scipy: bool
) -> dict[str, object]:
    """Identify an ordered source closure and the numerical runtime versions."""
    root = Path(__file__).parent
    module_sha256: dict[str, str] = {}
    for name in ordered_modules:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Python provenance module names must be local .py files.")
        content = (root / name).read_bytes()
        module_sha256[name] = hashlib.sha256(content).hexdigest()
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if include_scipy:
        import scipy

        runtime["scipy_version"] = scipy.__version__
    return {
        "schema": _PROVENANCE_SCHEMA,
        "policy": policy,
        "ordered_modules": list(ordered_modules),
        "module_sha256": module_sha256,
        "ordered_module_set_sha256": _ordered_module_set_sha256(
            ordered_modules, module_sha256
        ),
        "runtime_versions": runtime,
        "claim": _PROVENANCE_NONCLAIM,
    }


def validate_python_source_runtime_provenance(
    value: object,
    *,
    ordered_modules: tuple[str, ...],
    policy: str,
    include_scipy: bool,
    family: str,
) -> None:
    """Validate a persisted provenance record without requiring current-code identity."""
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "policy",
        "ordered_modules",
        "module_sha256",
        "ordered_module_set_sha256",
        "runtime_versions",
        "claim",
    }:
        raise ValueError(f"{family} Python provenance schema mismatch.")
    if (
        value["schema"] != _PROVENANCE_SCHEMA
        or value["policy"] != policy
        or value["ordered_modules"] != list(ordered_modules)
        or value["claim"] != _PROVENANCE_NONCLAIM
    ):
        raise ValueError(f"{family} Python provenance policy mismatch.")
    module_sha256 = value["module_sha256"]
    if not isinstance(module_sha256, Mapping) or set(module_sha256) != set(
        ordered_modules
    ):
        raise ValueError(f"{family} Python provenance module map mismatch.")
    digests = [value["ordered_module_set_sha256"], *module_sha256.values()]
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in digests
    ):
        raise ValueError(f"{family} Python provenance digest is invalid.")
    if value["ordered_module_set_sha256"] != _ordered_module_set_sha256(
        ordered_modules, module_sha256
    ):
        raise ValueError(f"{family} Python provenance aggregate digest mismatch.")
    runtime = value["runtime_versions"]
    runtime_keys = {
        "python_implementation",
        "python_version",
        "numpy_version",
    }
    if include_scipy:
        runtime_keys.add("scipy_version")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_keys
        or any(not isinstance(item, str) or not item for item in runtime.values())
    ):
        raise ValueError(f"{family} Python runtime provenance is invalid.")


def _ordered_module_set_sha256(
    ordered_modules: tuple[str, ...], module_sha256: Mapping[str, object]
) -> str:
    payload = {
        "schema": "summit_ordered_python_module_digest_v1",
        "modules": [
            {"relative_name": name, "content_sha256": module_sha256[name]}
            for name in ordered_modules
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npy_v1_encoded_size(
    header_data: Mapping[str, object],
    *,
    payload_size: int,
    family: str,
    member: str,
) -> int:
    """Return a stable NPY v1 header plus its exact payload size."""
    header = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(header, dict(header_data))
    except ValueError as exc:
        raise ValueError(
            f"{family} member {member!r} cannot use the stable NPY v1 format."
        ) from exc
    return header.tell() + payload_size


def _npy_v1_member_size(value: np.ndarray, *, family: str, member: str) -> int:
    """Return the exact uncompressed size emitted for one stable NPY member."""
    if value.dtype.hasobject:
        raise ValueError(f"{family} member {member!r} has an object dtype.")
    return _npy_v1_encoded_size(
        np.lib.format.header_data_from_array_1_0(value),
        payload_size=value.nbytes,
        family=family,
        member=member,
    )


def _preflight_text_scalar_npy(
    text: str, *, family: str, member: str, maximum_bytes: int
) -> np.ndarray:
    if not isinstance(text, str) or not text:
        raise ValueError(f"{family} member {member!r} must be nonempty text.")
    dtype = np.dtype(f"<U{len(text)}")
    member_size = _npy_v1_encoded_size(
        {
            "descr": dtype.str,
            "fortran_order": False,
            "shape": (),
        },
        payload_size=dtype.itemsize,
        family=family,
        member=member,
    )
    if member_size > maximum_bytes:
        raise ValueError(
            f"{family} {member} size {member_size} exceeds the "
            f"{maximum_bytes}-byte stable bound."
        )
    return np.asarray(text, dtype=dtype)


def _preflight_manifest_json_npy(text: str, *, family: str) -> np.ndarray:
    """Return the canonical scalar only when its complete NPY member is bounded."""
    return _preflight_text_scalar_npy(
        text,
        family=family,
        member="manifest_json.npy",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )


def _deflate_size_bound(source_size: int) -> int:
    """Return zlib's conservative upper bound for one raw-DEFLATE stream."""
    # zlib's general bound ends in +5; retain another 59 bytes of margin so
    # this admission rule is not sensitive to wrapper or flush bookkeeping.
    return source_size + ((source_size + 7) >> 3) + ((source_size + 63) >> 6) + 64


def _preflight_classic_zip(
    member_sizes: tuple[tuple[str, int], ...], *, family: str
) -> None:
    """Guarantee NumPy cannot require ZIP64 fields in the central directory."""
    if len(member_sizes) >= _MAX_CLASSIC_ZIP_MEMBERS:
        raise ValueError(f"{family} member count requires a ZIP64 directory.")
    central_offset_bound = 0
    central_size = 0
    for member, member_size in member_sizes:
        try:
            name_size = len(member.encode("ascii"))
        except UnicodeEncodeError as exc:
            raise ValueError(f"{family} member name is not canonical ASCII.") from exc
        compressed_bound = _deflate_size_bound(member_size)
        if (
            member_size > _MAX_CLASSIC_ZIP_FIELD
            or compressed_bound > _MAX_CLASSIC_ZIP_FIELD
        ):
            raise ValueError(
                f"{family} member {member!r} requires ZIP64 central size fields."
            )
        # NumPy forces a 20-byte ZIP64 size extra in every local header; the
        # stable format permits that local form but requires classic central
        # size and offset fields.
        central_offset_bound += 30 + name_size + 20 + compressed_bound
        central_size += 46 + name_size
    if central_offset_bound > _MAX_CLASSIC_ZIP_FIELD:
        raise ValueError(
            f"{family} conservative central-directory offset bound "
            f"{central_offset_bound} exceeds the {_MAX_CLASSIC_ZIP_FIELD}-byte "
            "classic ZIP bound."
        )
    if central_size > _MAX_CLASSIC_ZIP_FIELD:
        raise ValueError(f"{family} central directory requires ZIP64 fields.")


def _preflight_stable_npz_members(
    *,
    manifest_json: str,
    manifest_sha256: str,
    arrays: Mapping[str, object],
    family: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Preflight every stable writer member before any output path is touched."""
    manifest_value = _preflight_manifest_json_npy(manifest_json, family=family)
    digest_value = _preflight_text_scalar_npy(
        manifest_sha256,
        family=family,
        member="manifest_sha256.npy",
        maximum_bytes=1024,
    )
    array_values: dict[str, np.ndarray] = {}
    scientific_sizes: list[int] = []
    for name, source in arrays.items():
        member = f"{name}.npy"
        value = np.asarray(source)
        member_size = _npy_v1_member_size(value, family=family, member=member)
        array_values[name] = value
        scientific_sizes.append(member_size)

    member_sizes = (
        (
            "manifest_json.npy",
            _npy_v1_member_size(
                manifest_value,
                family=family,
                member="manifest_json.npy",
            ),
        ),
        (
            "manifest_sha256.npy",
            _npy_v1_member_size(
                digest_value,
                family=family,
                member="manifest_sha256.npy",
            ),
        ),
        *(
            (f"{name}.npy", member_size)
            for name, member_size in zip(array_values, scientific_sizes, strict=True)
        ),
    )
    for member, member_size in member_sizes:
        if member_size > _MAX_MEMBER_BYTES:
            raise ValueError(
                f"{family} member {member!r} size {member_size} exceeds the "
                f"{_MAX_MEMBER_BYTES}-byte stable bound."
            )
    total_size = sum(member_size for _, member_size in member_sizes)
    if total_size > _MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"{family} total uncompressed NPY size {total_size} exceeds the "
            f"{_MAX_TOTAL_UNCOMPRESSED_BYTES}-byte stable bound."
        )
    _preflight_classic_zip(member_sizes, family=family)
    return manifest_value, digest_value, array_values


class StableNpzReader:
    """Read one stable NPZ through a single descriptor after bounded preflight."""

    def __init__(self, path: Path, *, family: str, maximum_members: int) -> None:
        self._path = Path(path)
        self._family = family
        self._maximum_members = maximum_members
        self._handle: BinaryIO | None = None
        self._archive: zipfile.ZipFile | None = None
        self._members: dict[str, zipfile.ZipInfo] = {}
        self._headers: dict[str, tuple[str, tuple[int, ...], int]] = {}
        self._central_directory_offset: int | None = None

    def __enter__(self) -> StableNpzReader:
        try:
            self._handle = self._path.open("rb")
            self._preflight_end_of_central_directory()
            self._archive = zipfile.ZipFile(self._handle, mode="r")
            if self._archive.comment:
                self._fail("ZIP comments are not permitted")
            infos = self._archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                self._fail("duplicate ZIP members are not permitted")
            if len(infos) < 2 or len(infos) > self._maximum_members:
                self._fail("ZIP member count is outside the family bound")
            for info in infos:
                if (
                    _MEMBER_NAME.fullmatch(info.filename) is None
                    or info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type not in _ALLOWED_COMPRESSION
                    or info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > _MAX_MEMBER_BYTES
                    or info.compress_size
                    > info.file_size + max(64 * 1024, info.file_size // 100)
                    or info.extra
                    or info.comment
                ):
                    self._fail(f"ZIP member {info.filename!r} has an invalid header")
            self._validate_canonical_zip_layout(infos)
            self._members = {info.filename: info for info in infos}
            return self
        except Exception as exc:
            self.close()
            if isinstance(exc, ValueError) and str(exc).startswith(self._family):
                raise
            raise ValueError(f"{self._family} container preflight failed.") from exc

    def _preflight_end_of_central_directory(self) -> None:
        """Bound ZipFile's central-directory parse before constructing it."""
        if self._handle is None:
            raise ValueError("Stable NPZ reader is closed.")
        descriptor = self._handle.fileno()
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size < 22:
            self._fail("container must be a regular ZIP file")
        # Stable artifacts prohibit comments, so the classic EOCD must be the
        # final 22 bytes.  ZIP64 central directories are rejected below; the
        # canonical local size extra written by NumPy is parsed separately.
        raw = os.pread(descriptor, 22, observed.st_size - 22)
        if len(raw) != 22 or raw[:4] != b"PK\x05\x06":
            self._fail("classic comment-free ZIP end record is missing")
        (
            disk_number,
            directory_disk,
            entries_on_disk,
            total_entries,
            directory_size,
            directory_offset,
            comment_length,
        ) = struct.unpack("<4H2IH", raw[4:])
        if (
            disk_number != 0
            or directory_disk != 0
            or entries_on_disk != total_entries
            or comment_length != 0
        ):
            self._fail("multi-disk or commented ZIP containers are not permitted")
        if (
            entries_on_disk == 0xFFFF
            or total_entries == 0xFFFF
            or directory_size == 0xFFFFFFFF
            or directory_offset == 0xFFFFFFFF
        ):
            self._fail("ZIP64 central directories are not permitted")
        if total_entries < 2 or total_entries > self._maximum_members:
            self._fail("ZIP member count is outside the family bound")
        maximum_directory_size = (
            self._maximum_members * _MAX_CENTRAL_DIRECTORY_BYTES_PER_MEMBER
        )
        if (
            directory_size < total_entries * 46
            or directory_size > maximum_directory_size
            or directory_offset + directory_size != observed.st_size - 22
        ):
            self._fail("ZIP central directory is outside the family bound")
        self._central_directory_offset = directory_offset

    def _validate_canonical_zip_layout(self, infos: list[zipfile.ZipInfo]) -> None:
        """Reject prefixes, gaps, descriptors, and unrecognized ZIP metadata."""
        if self._handle is None or self._central_directory_offset is None:
            raise ValueError("Stable NPZ reader is closed.")
        if infos != sorted(infos, key=lambda info: info.header_offset):
            self._fail("ZIP central and local member orders disagree")
        descriptor = self._handle.fileno()
        cursor = 0
        for info in infos:
            if info.header_offset != cursor:
                self._fail("ZIP local members are not contiguous from byte zero")
            if (
                info.create_system != 3
                or info.create_version != 45
                or info.extract_version != 45
                or info.reserved != 0
                or info.flag_bits != 0
                or info.volume != 0
                or info.internal_attr != 0
                or info.external_attr != 25_165_824
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_DEFLATED
            ):
                self._fail(f"ZIP member {info.filename!r} is not canonical")
            fixed = os.pread(descriptor, 30, cursor)
            if len(fixed) != 30:
                self._fail(f"ZIP member {info.filename!r} has a truncated local header")
            (
                signature,
                extract_version,
                flags,
                compression,
                modified_time,
                modified_date,
                crc,
                local_compressed_size,
                local_file_size,
                name_length,
                extra_length,
            ) = struct.unpack("<4s5H3I2H", fixed)
            name_and_extra = os.pread(
                descriptor,
                name_length + extra_length,
                cursor + 30,
            )
            expected_name = info.filename.encode("ascii")
            expected_extra = struct.pack(
                "<HHQQ",
                0x0001,
                16,
                info.file_size,
                info.compress_size,
            )
            if (
                signature != b"PK\x03\x04"
                or extract_version != 45
                or flags != 0
                or compression != zipfile.ZIP_DEFLATED
                or modified_time != 0
                or modified_date != 33
                or crc != info.CRC
                or local_compressed_size != 0xFFFFFFFF
                or local_file_size != 0xFFFFFFFF
                or name_length != len(expected_name)
                or extra_length != len(expected_extra)
                or name_and_extra != expected_name + expected_extra
            ):
                self._fail(f"ZIP member {info.filename!r} local header is noncanonical")
            cursor += 30 + name_length + extra_length + info.compress_size
        if cursor != self._central_directory_offset:
            self._fail("ZIP central directory does not immediately follow payloads")

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def read_text_scalar(self, key: str, *, maximum_bytes: int) -> str:
        """Read a bounded scalar little-endian Unicode NPY member."""
        if maximum_bytes > _MAX_MANIFEST_BYTES:
            raise ValueError("Internal stable NPZ scalar bound is invalid.")
        member = f"{key}.npy"
        info = self._member(member)
        descriptor, shape, payload_offset = self._inspect(member)
        match = _UNICODE_DESCRIPTOR.fullmatch(descriptor)
        if match is None or shape != () or info.file_size > maximum_bytes:
            self._fail(f"member {member!r} is not a bounded Unicode scalar")
        character_count = int(match.group(1))
        if character_count * 4 != info.file_size - payload_offset:
            self._fail(f"member {member!r} has an inconsistent Unicode payload")
        value = self._load(member)
        if value.shape != () or value.dtype.str != descriptor:
            self._fail(f"member {member!r} changed after preflight")
        return str(value.item())

    def preflight_arrays(
        self, expected: Mapping[str, tuple[np.dtype, tuple[int, ...]]]
    ) -> None:
        """Validate scientific headers and the whole-container NPY size bound."""
        expected_members = {
            "manifest_json.npy",
            "manifest_sha256.npy",
            *(f"{name}.npy" for name in expected),
        }
        if set(self._members) != expected_members:
            self._fail("container key mismatch")
        total = sum(
            self._member(member).file_size
            for member in ("manifest_json.npy", "manifest_sha256.npy")
        )
        if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            self._fail("total uncompressed NPY size exceeds the stable bound")
        for name, (dtype, shape) in expected.items():
            member = f"{name}.npy"
            info = self._member(member)
            descriptor, actual_shape, _ = self._inspect(member)
            expected_descriptor = np.dtype(dtype).newbyteorder("<").str
            if descriptor != expected_descriptor or actual_shape != tuple(shape):
                self._fail(f"array {name!r} has an invalid dtype or shape")
            total += info.file_size
            if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                self._fail("total uncompressed NPY size exceeds the stable bound")

    def load_array(self, name: str) -> np.ndarray:
        """Load one already-preflighted member and force CRC verification."""
        member = f"{name}.npy"
        if member not in self._headers:
            raise ValueError("Internal stable NPZ load occurred before preflight.")
        value = self._load(member)
        descriptor, shape, _ = self._headers[member]
        if value.dtype.str != descriptor or value.shape != shape:
            self._fail(f"array {name!r} changed after preflight")
        return np.array(value, copy=True)

    def _member(self, member: str) -> zipfile.ZipInfo:
        try:
            return self._members[member]
        except KeyError as exc:
            self._fail(f"required member {member!r} is missing")
            raise AssertionError from exc

    def _inspect(self, member: str) -> tuple[str, tuple[int, ...], int]:
        cached = self._headers.get(member)
        if cached is not None:
            return cached
        archive = self._require_archive()
        info = self._member(member)
        try:
            with archive.open(info, mode="r") as stream:
                magic = self._read_exact(stream, 6)
                if magic != b"\x93NUMPY":
                    self._fail(f"member {member!r} lacks the NPY magic")
                version = tuple(self._read_exact(stream, 2))
                if version == (1, 0):
                    header_length = struct.unpack("<H", self._read_exact(stream, 2))[0]
                    encoding = "latin1"
                    prefix_size = 10
                else:
                    self._fail(f"member {member!r} uses an unsupported NPY version")
                if header_length <= 0 or header_length > _MAX_NPY_HEADER_BYTES:
                    self._fail(f"member {member!r} has an invalid NPY header bound")
                raw_header = self._read_exact(stream, header_length)
                if not raw_header.endswith(b"\n"):
                    self._fail(f"member {member!r} has a malformed NPY header")
                header = self._parse_header(raw_header.decode(encoding), member)
                descriptor = header["descr"]
                shape = header["shape"]
                if (
                    not isinstance(descriptor, str)
                    or header["fortran_order"] is not False
                    or not isinstance(shape, tuple)
                    or any(
                        isinstance(dimension, bool)
                        or not isinstance(dimension, int)
                        or dimension < 0
                        for dimension in shape
                    )
                ):
                    self._fail(f"member {member!r} has a noncanonical NPY header")
                try:
                    dtype = np.dtype(descriptor)
                except TypeError as exc:
                    self._fail(f"member {member!r} has an invalid dtype descriptor")
                    raise AssertionError from exc
                if dtype.hasobject:
                    self._fail(f"member {member!r} has an object dtype")
                canonical_header = io.BytesIO()
                np.lib.format.write_array_header_1_0(
                    canonical_header,
                    {
                        "descr": descriptor,
                        "fortran_order": False,
                        "shape": shape,
                    },
                )
                if canonical_header.getvalue()[prefix_size:] != raw_header:
                    self._fail(f"member {member!r} has a noncanonical NPY header")
                element_count = 1
                for dimension in shape:
                    element_count *= dimension
                payload_offset = prefix_size + header_length
                expected_size = payload_offset + element_count * dtype.itemsize
                if expected_size != info.file_size:
                    self._fail(f"member {member!r} has an inconsistent payload size")
                result = (descriptor, shape, payload_offset)
                self._headers[member] = result
                return result
        except Exception as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(self._family):
                raise
            raise ValueError(f"{self._family} member {member!r} is corrupt.") from exc

    def _load(self, member: str) -> np.ndarray:
        archive = self._require_archive()
        info = self._member(member)
        try:
            with archive.open(info, mode="r") as stream:
                value = np.load(
                    stream,
                    allow_pickle=False,
                    max_header_size=_MAX_NPY_HEADER_BYTES,
                )
                if stream.read(1) != b"":
                    self._fail(f"member {member!r} has trailing NPY payload bytes")
            if not isinstance(value, np.ndarray):
                self._fail(f"member {member!r} is not an ndarray")
            return value
        except Exception as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(self._family):
                raise
            raise ValueError(f"{self._family} member {member!r} is corrupt.") from exc

    @staticmethod
    def _read_exact(stream: BinaryIO, count: int) -> bytes:
        value = stream.read(count)
        if len(value) != count:
            raise EOFError("truncated stable NPY member")
        return value

    def _parse_header(self, text: str, member: str) -> dict[str, object]:
        try:
            parsed = ast.parse(text.strip(), mode="eval")
            if not isinstance(parsed.body, ast.Dict):
                self._fail(f"member {member!r} has a non-dictionary NPY header")
            keys = [ast.literal_eval(node) for node in parsed.body.keys]
            if len(keys) != len(set(keys)):
                self._fail(f"member {member!r} has duplicate NPY header keys")
            header = ast.literal_eval(parsed)
        except (SyntaxError, ValueError, TypeError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(self._family):
                raise
            raise ValueError(
                f"{self._family} member {member!r} header is invalid."
            ) from exc
        if not isinstance(header, dict) or set(header) != {
            "descr",
            "fortran_order",
            "shape",
        }:
            self._fail(f"member {member!r} has invalid NPY header keys")
        return header

    def _require_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            raise ValueError("Stable NPZ reader is closed.")
        return self._archive

    def _fail(self, message: str) -> None:
        raise ValueError(f"{self._family} {message}.")


def _validate_stable_npz_writer_temp(
    path: str | Path,
    *,
    family: str,
    manifest_json: np.ndarray,
    manifest_sha256: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Validate the fsynced temporary container before atomic publication."""
    with StableNpzReader(
        Path(path), family=family, maximum_members=len(arrays) + 2
    ) as archive:
        observed_manifest = archive.read_text_scalar(
            "manifest_json", maximum_bytes=_MAX_MANIFEST_BYTES
        )
        observed_digest = archive.read_text_scalar(
            "manifest_sha256", maximum_bytes=1024
        )
        if observed_manifest != str(manifest_json.item()) or observed_digest != str(
            manifest_sha256.item()
        ):
            raise ValueError(f"{family} temporary scalar member mismatch.")
        archive.preflight_arrays(
            {name: (value.dtype, value.shape) for name, value in arrays.items()}
        )


def _publish_stable_npz_no_replace(temporary: str | Path, target: str | Path) -> None:
    """Atomically link a same-directory temporary file to an absent target."""
    os.link(os.fspath(temporary), os.fspath(target))
    os.unlink(temporary)


def fsync_parent_directory(path: Path) -> None:
    """Make a newly published directory entry durable."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(os.fspath(Path(path).parent), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
