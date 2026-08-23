from __future__ import annotations

import errno
import io
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest

import summit.context._artifact_io as artifact_io
import summit.context.fit_v1 as fit_module
import summit.context.reference_v1 as reference_module
import summit.context.trait_v1 as trait_module
from summit.context.fit_v1 import (
    CONTEXTUAL_FIT_V1_SUFFIX,
    fit_contextual_model_v1,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.reference_v1 import (
    CONTEXTUAL_REFERENCE_V1_SUFFIX,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import canonical_json
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _artifact as _reference_artifact
from test_context_stage4_fit_v1 import _artifacts as _fit_source_artifacts
from test_context_stage4_trait_v1 import _artifact as _trait_artifact


_BOUNDARY_CHARACTER_COUNT = 4_194_272


def _serialized_npy_size(value: object) -> int:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.tell()


@pytest.fixture
def stable_writer_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], ...]:
    reference = _reference_artifact()
    trait = _trait_artifact()
    fit_reference, fit_trait = _fit_source_artifacts(monkeypatch)
    fit = fit_contextual_model_v1(
        fit_reference,
        fit_trait,
        trait_selector="trait-0",
    )
    return (
        {
            "name": "reference",
            "family": "Contextual reference V1",
            "artifact": reference,
            "writer": write_contextual_reference_v1,
            "loader": load_contextual_reference_v1,
            "module": reference_module,
            "suffix": CONTEXTUAL_REFERENCE_V1_SUFFIX,
        },
        {
            "name": "trait",
            "family": "Contextual trait V1",
            "artifact": trait,
            "writer": write_contextual_trait_v1,
            "loader": load_contextual_trait_v1,
            "module": trait_module,
            "suffix": CONTEXTUAL_TRAIT_V1_SUFFIX,
        },
        {
            "name": "fit",
            "family": "Contextual fit V1",
            "artifact": fit,
            "writer": write_contextual_fit_v1,
            "loader": load_contextual_fit_v1,
            "module": fit_module,
            "suffix": CONTEXTUAL_FIT_V1_SUFFIX,
        },
    )


def test_all_stable_writers_publish_unique_target_and_reload(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
) -> None:
    for case in stable_writer_cases:
        requested = tmp_path / f"unique-{case['name']}"
        target = Path(str(requested) + case["suffix"])
        temporary_before = set(target.parent.glob(f".{target.name}.*"))

        written = case["writer"](case["artifact"], requested)

        assert written == target
        assert target.is_file()
        assert set(target.parent.glob(f".{target.name}.*")) == temporary_before
        loaded = case["loader"](target)
        assert loaded.manifest_sha256 == case["artifact"].manifest_sha256


def test_exclusive_generation_directory_precedes_distinct_family_publications(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation"
    os.mkdir(generation)
    with pytest.raises(FileExistsError) as error:
        os.mkdir(generation)
    assert error.value.errno == errno.EEXIST

    published: set[Path] = set()
    for case in stable_writer_cases:
        target = case["writer"](case["artifact"], generation / case["name"])
        assert target not in published
        published.add(target)
        assert case["loader"](target).manifest_sha256 == (
            case["artifact"].manifest_sha256
        )
    assert len(published) == len(stable_writer_cases)


def test_all_stable_writers_refuse_existing_target_without_altering_it(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
) -> None:
    for case in stable_writer_cases:
        target = tmp_path / f"existing-{case['name']}{case['suffix']}"
        preserved = f"preserved-{case['name']}".encode("ascii")
        target.write_bytes(preserved)
        inode_before = target.stat().st_ino
        temporary_before = set(target.parent.glob(f".{target.name}.*"))

        with pytest.raises(FileExistsError) as error:
            case["writer"](case["artifact"], target)

        assert error.value.errno == errno.EEXIST
        assert target.read_bytes() == preserved
        assert target.stat().st_ino == inode_before
        assert set(target.parent.glob(f".{target.name}.*")) == temporary_before

        dangling = tmp_path / f"dangling-{case['name']}{case['suffix']}"
        dangling_destination = Path(f"missing-{case['name']}")
        dangling.symlink_to(dangling_destination)
        temporary_before = set(dangling.parent.glob(f".{dangling.name}.*"))
        with pytest.raises(FileExistsError) as error:
            case["writer"](case["artifact"], dangling)
        assert error.value.errno == errno.EEXIST
        assert dangling.is_symlink()
        assert dangling.readlink() == dangling_destination
        assert set(dangling.parent.glob(f".{dangling.name}.*")) == temporary_before


def test_all_stable_writers_lose_atomic_publication_race_with_eexist(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_link = artifact_io.os.link
    for case in stable_writer_cases:
        winner = case["writer"](
            case["artifact"],
            tmp_path / f"race-winner-{case['name']}",
        )
        winner_bytes = winner.read_bytes()
        target = tmp_path / f"race-target-{case['name']}{case['suffix']}"
        temporary_before = set(target.parent.glob(f".{target.name}.*"))
        link_calls = 0

        def publish_competitor_then_lose(source: object, destination: object) -> None:
            nonlocal link_calls
            link_calls += 1
            real_link(winner, destination)
            real_link(source, destination)

        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io.os,
                "link",
                publish_competitor_then_lose,
            )
            with pytest.raises(FileExistsError) as error:
                case["writer"](case["artifact"], target)

        assert link_calls == 1
        assert error.value.errno == errno.EEXIST
        assert target.read_bytes() == winner_bytes
        assert case["loader"](target).manifest_sha256 == (
            case["artifact"].manifest_sha256
        )
        assert set(target.parent.glob(f".{target.name}.*")) == temporary_before


def test_manifest_npy_preflight_enforces_the_exact_16_mib_boundary() -> None:
    boundary = "x" * _BOUNDARY_CHARACTER_COUNT
    assert _serialized_npy_size(boundary) == artifact_io._MAX_MANIFEST_BYTES
    value = artifact_io._preflight_manifest_json_npy(boundary, family="Stage7 boundary")
    assert value.shape == ()
    assert value.dtype.str == f"<U{_BOUNDARY_CHARACTER_COUNT}"

    over_bound = boundary + "x"
    assert _serialized_npy_size(over_bound) == artifact_io._MAX_MANIFEST_BYTES + 4
    with pytest.raises(
        ValueError,
        match=(
            rf"size {artifact_io._MAX_MANIFEST_BYTES + 4} exceeds the "
            rf"{artifact_io._MAX_MANIFEST_BYTES}-byte stable bound"
        ),
    ):
        artifact_io._preflight_manifest_json_npy(over_bound, family="Stage7 boundary")


def test_all_stable_writers_reject_over_bound_manifest_before_any_output(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    over_bound = "x" * (_BOUNDARY_CHARACTER_COUNT + 1)

    for case in stable_writer_cases:
        fresh_parent = tmp_path / f"fresh-{case['name']}"
        fresh_output = fresh_parent / "artifact"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"],
                "canonical_json",
                lambda _manifest, text=over_bound: text,
            )
            with pytest.raises(ValueError, match=case["family"]):
                case["writer"](case["artifact"], fresh_output)
        assert not fresh_parent.exists()

        existing = tmp_path / f"existing-{case['name']}{case['suffix']}"
        existing.write_bytes(b"preserved stable artifact")
        entries_before = set(tmp_path.iterdir())
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"],
                "canonical_json",
                lambda _manifest, text=over_bound: text,
            )
            with pytest.raises(ValueError, match=case["family"]):
                case["writer"](case["artifact"], existing)
        assert existing.read_bytes() == b"preserved stable artifact"
        assert set(tmp_path.iterdir()) == entries_before


def test_all_stable_writers_preflight_exact_member_and_total_npy_sizes(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for case in stable_writer_cases:
        artifact = case["artifact"]
        scientific_sizes = [
            _serialized_npy_size(getattr(artifact, name))
            for name in artifact.manifest["arrays"]
        ]
        manifest_size = _serialized_npy_size(canonical_json(artifact.manifest))
        digest_size = _serialized_npy_size(artifact.manifest_sha256)
        maximum_member_size = max(
            manifest_size,
            digest_size,
            *scientific_sizes,
        )

        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_MEMBER_BYTES",
                maximum_member_size,
            )
            member_boundary = case["writer"](
                artifact,
                tmp_path / f"member-boundary-{case['name']}",
            )
        assert case["loader"](member_boundary).manifest_sha256 == (
            artifact.manifest_sha256
        )

        member_failure_parent = tmp_path / f"member-over-{case['name']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_MEMBER_BYTES",
                maximum_member_size - 1,
            )
            with pytest.raises(
                ValueError,
                match=rf"member .* exceeds the {maximum_member_size - 1}-byte",
            ):
                case["writer"](artifact, member_failure_parent / "artifact")
        assert not member_failure_parent.exists()

        total_size = sum(scientific_sizes) + manifest_size + digest_size
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_TOTAL_UNCOMPRESSED_BYTES",
                total_size,
            )
            total_boundary = case["writer"](
                artifact,
                tmp_path / f"total-boundary-{case['name']}",
            )
            assert case["loader"](total_boundary).manifest_sha256 == (
                artifact.manifest_sha256
            )

        total_failure_parent = tmp_path / f"total-over-{case['name']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_TOTAL_UNCOMPRESSED_BYTES",
                total_size - 1,
            )
            with pytest.raises(ValueError, match="total uncompressed NPY size"):
                case["loader"](total_boundary)
            with pytest.raises(
                ValueError,
                match=rf"total uncompressed NPY size .* exceeds the {total_size - 1}",
            ):
                case["writer"](artifact, total_failure_parent / "artifact")
        assert not total_failure_parent.exists()


def test_all_stable_writers_preflight_classic_zip_sizes_and_offsets(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for case in stable_writer_cases:
        artifact = case["artifact"]
        member_sizes = (
            (
                "manifest_json.npy",
                _serialized_npy_size(canonical_json(artifact.manifest)),
            ),
            ("manifest_sha256.npy", _serialized_npy_size(artifact.manifest_sha256)),
            *(
                (f"{name}.npy", _serialized_npy_size(getattr(artifact, name)))
                for name in artifact.manifest["arrays"]
            ),
        )
        compressed_bounds = tuple(
            artifact_io._deflate_size_bound(size) for _, size in member_sizes
        )
        count_failure_parent = tmp_path / f"zip64-count-{case['name']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_CLASSIC_ZIP_MEMBERS",
                len(member_sizes),
            )
            with pytest.raises(ValueError, match="member count requires"):
                case["writer"](artifact, count_failure_parent / "artifact")
        assert not count_failure_parent.exists()

        single_member_limit = max(compressed_bounds)
        single_failure_parent = tmp_path / f"zip64-member-{case['name']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_CLASSIC_ZIP_FIELD",
                single_member_limit - 1,
            )
            with pytest.raises(ValueError, match="ZIP64 central size fields"):
                case["writer"](artifact, single_failure_parent / "artifact")
        assert not single_failure_parent.exists()

        central_offset_bound = sum(
            30 + len(name.encode("ascii")) + 20 + compressed_bound
            for (name, _), compressed_bound in zip(
                member_sizes, compressed_bounds, strict=True
            )
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_CLASSIC_ZIP_FIELD",
                central_offset_bound,
            )
            classic_boundary = case["writer"](
                artifact,
                tmp_path / f"classic-boundary-{case['name']}",
            )
        with zipfile.ZipFile(classic_boundary, mode="r") as archive:
            assert archive.start_dir <= central_offset_bound
            assert all(
                info.compress_size <= artifact_io._deflate_size_bound(info.file_size)
                for info in archive.infolist()
            )
        assert case["loader"](classic_boundary).manifest_sha256 == (
            artifact.manifest_sha256
        )

        offset_failure_parent = tmp_path / f"zip64-offset-{case['name']}"
        with monkeypatch.context() as scoped:
            scoped.setattr(
                artifact_io,
                "_MAX_CLASSIC_ZIP_FIELD",
                central_offset_bound - 1,
            )
            with pytest.raises(ValueError, match="central-directory offset bound"):
                case["writer"](artifact, offset_failure_parent / "artifact")
        assert not offset_failure_parent.exists()


def test_all_stable_writers_validate_fsynced_temp_before_publication(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_noncanonical(handle: Any, **_members: object) -> None:
        handle.write(b"not a stable NPZ container")

    for case in stable_writer_cases:
        fresh = tmp_path / f"invalid-temp-fresh-{case['name']}{case['suffix']}"
        temporary_before = set(fresh.parent.glob(f".{fresh.name}.*"))
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"].np,
                "savez_compressed",
                write_noncanonical,
            )
            with pytest.raises(ValueError, match=case["family"]):
                case["writer"](case["artifact"], fresh)
        assert not fresh.exists()
        assert set(fresh.parent.glob(f".{fresh.name}.*")) == temporary_before

        existing = case["writer"](
            case["artifact"],
            tmp_path / f"invalid-temp-existing-{case['name']}",
        )
        original = existing.read_bytes()
        temporary_before = set(existing.parent.glob(f".{existing.name}.*"))
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"].np,
                "savez_compressed",
                write_noncanonical,
            )
            with pytest.raises(ValueError, match=case["family"]):
                case["writer"](case["artifact"], existing)
        assert existing.read_bytes() == original
        assert set(existing.parent.glob(f".{existing.name}.*")) == temporary_before


def test_all_stable_writers_expose_precommit_and_postcommit_fsync_failures(
    stable_writer_cases: tuple[dict[str, Any], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_file_fsync(_descriptor: int) -> None:
        raise OSError("injected file fsync failure")

    def fail_parent_fsync(_path: Path) -> None:
        raise OSError("injected parent fsync failure")

    for case in stable_writer_cases:
        fresh = tmp_path / f"file-fsync-fresh-{case['name']}{case['suffix']}"
        temporary_before = set(fresh.parent.glob(f".{fresh.name}.*"))
        with monkeypatch.context() as scoped:
            scoped.setattr(case["module"].os, "fsync", fail_file_fsync)
            with pytest.raises(OSError, match="injected file fsync failure"):
                case["writer"](case["artifact"], fresh)
        assert not fresh.exists()
        assert set(fresh.parent.glob(f".{fresh.name}.*")) == temporary_before

        existing = case["writer"](
            case["artifact"],
            tmp_path / f"file-fsync-existing-{case['name']}",
        )
        original = existing.read_bytes()
        temporary_before = set(existing.parent.glob(f".{existing.name}.*"))
        with monkeypatch.context() as scoped:
            scoped.setattr(case["module"].os, "fsync", fail_file_fsync)
            with pytest.raises(OSError, match="injected file fsync failure"):
                case["writer"](case["artifact"], existing)
        assert existing.read_bytes() == original
        assert set(existing.parent.glob(f".{existing.name}.*")) == temporary_before

        published = tmp_path / f"parent-fsync-{case['name']}{case['suffix']}"
        temporary_before = set(published.parent.glob(f".{published.name}.*"))
        with monkeypatch.context() as scoped:
            scoped.setattr(
                case["module"],
                "fsync_parent_directory",
                fail_parent_fsync,
            )
            with pytest.raises(OSError, match="injected parent fsync failure"):
                case["writer"](case["artifact"], published)
        assert published.is_file()
        assert set(published.parent.glob(f".{published.name}.*")) == temporary_before
        loaded = case["loader"](published)
        assert loaded.manifest_sha256 == case["artifact"].manifest_sha256
        published_bytes = published.read_bytes()
        published_inode = published.stat().st_ino
        with pytest.raises(FileExistsError) as error:
            case["writer"](case["artifact"], published)
        assert error.value.errno == errno.EEXIST
        assert published.read_bytes() == published_bytes
        assert published.stat().st_ino == published_inode
        assert set(published.parent.glob(f".{published.name}.*")) == temporary_before
