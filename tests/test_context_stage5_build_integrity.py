from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src" / "native"

_NATIVE_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
}
_VENDOR_GEMM_CALL = re.compile(
    r"\b("
    r"cblas_[sd](?:gemm|syrk)(?:_batch|_batch_strided)?"
    r"|mkl_[sd](?:gemm|syrk)(?:_batch|_batch_strided)?"
    r"|bli_(?:gemm|gemm_ex|gemm_batch|gemm_batch_ex|syrk|syrk_ex)"
    r"|[sd](?:gemm|syrk)_?"
    r")\s*\("
)
_CONTEXTUAL_GEMM_CALL = re.compile(
    r"\b(?:dgemm_[A-Za-z0-9_]+|observed_vendor_dgemm(?:_general)?)\s*\("
)
_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_CONTEXTUAL_ROOTS = (
    NATIVE / "contextual_dense_v1.inc",
    NATIVE / "contextual_streamed_reference_v1.inc",
)

# These are pre-Stage-5 legacy wrappers plus the one gxeldcore vendor boundary.
# Contextual sources are intentionally absent: they may call a dgemm_* helper
# only from their protected_gemm dispatcher and may never enter a vendor API.
_APPROVED_VENDOR_ENTRIES = Counter(
    {
        ("src/native/gwldcore.cpp", "gemm_col_major_nn", "cblas_dgemm"): 1,
        ("src/native/gwldcore.cpp", "gemm_col_major_nn", "cblas_sgemm"): 1,
        ("src/native/gwldcore.cpp", "gemm_col_major_tn", "cblas_dgemm"): 1,
        ("src/native/gwldcore.cpp", "gemm_col_major_tn", "cblas_sgemm"): 1,
        (
            "src/native/gwldcore_cuda.cu",
            "host_gemm_col_major_nn",
            "cblas_dgemm",
        ): 1,
        (
            "src/native/gwldcore_cuda.cu",
            "host_gemm_col_major_nn",
            "cblas_sgemm",
        ): 1,
        ("src/native/winldcore.cpp", "gemm_col_major_nn", "cblas_dgemm"): 1,
        ("src/native/winldcore.cpp", "gemm_col_major_nn", "cblas_sgemm"): 1,
        ("src/native/winldcore.cpp", "gemm_col_major_tn", "cblas_dgemm"): 1,
        ("src/native/winldcore.cpp", "gemm_col_major_tn", "cblas_sgemm"): 1,
        (
            "src/native/winldcore.cpp",
            "syrk_col_major_upper",
            "cblas_dsyrk",
        ): 1,
        ("src/native/gxeldcore.cpp", "dgemm_nn_raw", "cblas_dgemm"): 1,
        ("src/native/gxeldcore.cpp", "dgemm_tn_raw", "cblas_dgemm"): 1,
        (
            "src/native/gxeldcore.cpp",
            "execute_private_blis_gemm",
            "bli_gemm_ex",
        ): 1,
        (
            "src/native/gxeldcore.cpp",
            "observed_vendor_dgemm_general",
            "cblas_dgemm",
        ): 1,
    }
)


def _mask_non_code(source: str) -> str:
    """Blank comments and literals while preserving source offsets/newlines."""

    masked = list(source)
    index = 0
    while index < len(source):
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = len(source)
            for cursor in range(index, end):
                masked[cursor] = " "
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                end = len(source) - 2
            end = min(len(source), end + 2)
            for cursor in range(index, end):
                if masked[cursor] != "\n":
                    masked[cursor] = " "
            index = end
            continue
        if source[index] in {'"', "'"}:
            quote = source[index]
            cursor = index + 1
            while cursor < len(source):
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                cursor += 1
                if source[cursor - 1] == quote:
                    break
            for position in range(index, min(cursor, len(source))):
                if masked[position] != "\n":
                    masked[position] = " "
            index = cursor
            continue
        index += 1
    return "".join(masked)


def _matching_delimiter(source: str, begin: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(begin, len(source)):
        if source[index] == opening:
            depth += 1
        elif source[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unbalanced {opening}{closing} delimiters")


def _function_body_spans(masked: str, function_name: str) -> tuple[range, ...]:
    spans: list[range] = []
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for match in pattern.finditer(masked):
        opening_parenthesis = masked.find("(", match.start())
        closing_parenthesis = _matching_delimiter(masked, opening_parenthesis, "(", ")")
        cursor = closing_parenthesis + 1
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if cursor >= len(masked) or masked[cursor] != "{":
            continue
        closing_brace = _matching_delimiter(masked, cursor, "{", "}")
        spans.append(range(cursor, closing_brace + 1))
    return tuple(spans)


def _containing_approved_function(path: Path, masked: str, position: int) -> str | None:
    relative = path.relative_to(ROOT).as_posix()
    names = {
        function
        for file_name, function, _symbol in _APPROVED_VENDOR_ENTRIES
        if file_name == relative
    }
    containing = [
        name
        for name in names
        if any(position in span for span in _function_body_spans(masked, name))
    ]
    if len(containing) > 1:
        raise AssertionError(
            f"vendor entry at {relative}:{position} has ambiguous wrappers"
        )
    return containing[0] if containing else None


def _contextual_transitive_sources() -> tuple[Path, ...]:
    pending = list(_CONTEXTUAL_ROOTS)
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        resolved = path.resolve()
        assert resolved.is_relative_to(NATIVE.resolve()), resolved
        assert path.is_file() and path.suffix in _NATIVE_SOURCE_SUFFIXES, path
        if resolved in observed:
            continue
        observed.add(resolved)
        source = path.read_text(encoding="utf-8")
        for match in _LOCAL_INCLUDE.finditer(source):
            candidates = (path.parent / match.group(1), NATIVE / match.group(1))
            included = next((value for value in candidates if value.is_file()), None)
            if included is not None and included.suffix in _NATIVE_SOURCE_SUFFIXES:
                pending.append(included)
    return tuple(sorted(observed))


def test_native_vendor_gemm_entry_sites_match_the_legacy_allowlist() -> None:
    observed: Counter[tuple[str, str | None, str]] = Counter()
    for path in sorted(NATIVE.rglob("*")):
        if not path.is_file() or path.suffix not in _NATIVE_SOURCE_SUFFIXES:
            continue
        masked = _mask_non_code(path.read_text(encoding="utf-8"))
        relative = path.relative_to(ROOT).as_posix()
        for match in _VENDOR_GEMM_CALL.finditer(masked):
            observed[
                (
                    relative,
                    _containing_approved_function(path, masked, match.start()),
                    match.group(1),
                )
            ] += 1

    assert observed == _APPROVED_VENDOR_ENTRIES


def test_contextual_gemm_helpers_are_reached_only_from_protected_dispatchers() -> None:
    contextual_sources = _contextual_transitive_sources()
    assert set(path.resolve() for path in _CONTEXTUAL_ROOTS) <= set(contextual_sources)
    binding = (NATIVE / "gxeldcore.cpp").read_text(encoding="utf-8")
    for root in _CONTEXTUAL_ROOTS:
        assert f'#include "{root.name}"' in binding
    total_calls = 0
    for path in contextual_sources:
        masked = _mask_non_code(path.read_text(encoding="utf-8"))
        assert not list(_VENDOR_GEMM_CALL.finditer(masked)), path.name
        calls = tuple(_CONTEXTUAL_GEMM_CALL.finditer(masked))
        if not calls:
            continue
        total_calls += len(calls)
        protected_spans = _function_body_spans(masked, "protected_gemm")
        assert protected_spans, f"{path.name} has no protected_gemm definition"
        outside = [
            masked.count("\n", 0, call.start()) + 1
            for call in calls
            if not any(call.start() in span for span in protected_spans)
        ]
        assert not outside, (
            f"{path.name} has contextual GEMM calls outside protected_gemm "
            f"at lines {outside}"
        )
    assert total_calls > 0


def test_gxeldcore_build_info_uses_configured_build_policy() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (NATIVE / "gxeldcore.cpp").read_text(encoding="utf-8")

    assert 'result["optimization"] = "-O3"' not in source
    for field, macro in {
        "sanitizer_mode": "GWLDCORE_SANITIZER_MODE",
        "optimization": "GWLDCORE_EFFECTIVE_OPTIMIZATION",
        "architecture_tuning": "GWLDCORE_ARCHITECTURE_TUNING",
        "compiler_flags": "GWLDCORE_CONFIGURED_COMPILER_FLAGS",
    }.items():
        assert f'result["{field}"] = {macro};' in source
        assert macro in cmake

    assert "GWLDCORE_ENABLE_ASAN_UBSAN" in cmake
    assert "GWLDCORE_ENABLE_UBSAN_ONLY" in cmake
    assert "GWLDCORE_NATIVE_OPT=$<BOOL:${GWLDCORE_NATIVE_TUNING_ENABLED}>" in cmake
    assert "gwldcore_apply_sanitizer(gwldcore_core)" in cmake
    assert "gwldcore_apply_sanitizer(${target_name})" in cmake
    assert "summit-tracked-source-tree-sha256-v1" in cmake
    assert 'COMMAND "${GIT_EXECUTABLE}" ls-files --stage' in cmake
    assert 'file(SHA256 "${GWLDCORE_GIT_FILE_ABSOLUTE}"' in cmake
    assert '"git-tree:${GWLDCORE_GIT_TREE}"' not in cmake
