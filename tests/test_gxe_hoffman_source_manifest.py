from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gxe" / "hoffman" / "generate_source_manifest.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "test_generate_source_manifest", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_script()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    source_root = tmp_path / "sources"
    source_root.mkdir(mode=0o700)
    ids = [(str(index), str(index)) for index in range(1, 6)]

    fam = tmp_path / "reference.fam"
    fam.write_text(
        "".join(f"{fid} {iid} 0 0 0 -9\n" for fid, iid in ids), encoding="utf-8"
    )

    paths: dict[str, Path] = {}
    for trait_index, trait in enumerate(("trait_a", "trait_b", "trait_c"), start=1):
        path = source_root / f"{trait}.pheno"
        path.write_text(
            "FID IID pheno\n"
            + "".join(
                f"{fid} {iid} {trait_index + row_index / 10:.1f}\n"
                for row_index, (fid, iid) in enumerate(ids)
            ),
            encoding="utf-8",
        )
        paths[trait] = path

    age_covar = source_root / "age.covar"
    age_covar.write_text(
        "FID IID age pc1\n"
        + "".join(
            f"{fid} {iid} {40 + row_index} {row_index / 10:.1f}\n"
            for row_index, (fid, iid) in enumerate(ids)
        ),
        encoding="utf-8",
    )
    sex_covar = source_root / "sex.covar"
    sex_covar.write_text(
        "FID IID sex pc1\n"
        + "".join(
            f"{fid} {iid} {1 + row_index % 2} {row_index / 10:.1f}\n"
            for row_index, (fid, iid) in enumerate(ids)
        ),
        encoding="utf-8",
    )
    paths["age_covar"] = age_covar
    paths["sex_covar"] = sex_covar

    config = {
        "kind": "summit.gxe.hoffman_panel",
        "schema_version": 1,
        "local_source_layout": {
            "root": str(source_root),
            "phenotype_path_template": "{trait}.pheno",
            "phenotype_header": ["FID", "IID", "pheno"],
        },
        "datasets": {
            "full": {
                "expected_n_samples": len(ids),
                "expected_fam_sha256": _sha256(fam),
            }
        },
        "groups": {
            "age_group": {
                "environment": "age",
                "covariate_source_local": str(age_covar),
                "phenotypes": ["trait_a", "trait_b"],
            },
            "sex_group": {
                "environment": "sex",
                "covariate_source_local": str(sex_covar),
                "phenotypes": ["trait_b", "trait_c"],
            },
        },
    }
    config_path = tmp_path / "panel.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source_root, fam, config_path, paths


def _args(
    source_root: Path, fam: Path, config: Path, output: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        source_root=str(source_root),
        fam=str(fam),
        output=str(output),
    )


def test_source_manifest_is_deterministic_private_and_create_only(tmp_path: Path):
    source_root, fam, config, paths = _write_fixture(tmp_path)
    output_dir = tmp_path / "manifests"
    output_dir.mkdir(mode=0o700)
    first = output_dir / "sources.first.json"
    second = output_dir / "sources.second.json"

    observed = generator.generate_manifest(_args(source_root, fam, config, first))
    generator.generate_manifest(_args(source_root, fam, config, second))
    payload = json.loads(observed.read_text(encoding="utf-8"))

    assert first.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert payload["kind"] == "summit.gxe.local_source_manifest"
    assert payload["config"]["sha256"] == _sha256(config)
    assert payload["counts"] == {
        "groups": 2,
        "unique_phenotype_files": 3,
        "configured_covariate_files": 2,
    }
    assert payload["reference_fam"]["sha256"] == _sha256(fam)
    assert payload["reference_fam"]["n_samples"] == 5
    assert payload["fam_order_digest"] == payload["reference_fam"]["ordered_id_digest"]
    assert list(payload["sources"]["phenotypes"]) == ["trait_a", "trait_b", "trait_c"]
    trait_a = payload["sources"]["phenotypes"]["trait_a"]
    assert trait_a["relative_path"] == "trait_a.pheno"
    assert trait_a["header"] == ["FID", "IID", "pheno"]
    assert trait_a["bytes"] == paths["trait_a"].stat().st_size
    assert trait_a["sha256"] == _sha256(paths["trait_a"])
    assert trait_a["id_unique"] is True
    assert trait_a["order_matches_fam"] is True
    age_record = payload["sources"]["covariates"]["age_group"]
    assert age_record["header"] == [
        "FID",
        "IID",
        "age",
        "pc1",
    ]
    assert age_record["bytes"] == paths["age_covar"].stat().st_size
    assert age_record["sha256"] == _sha256(paths["age_covar"])
    assert age_record["ordered_id_digest"] == payload["fam_order_digest"]

    original = first.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generator.generate_manifest(_args(source_root, fam, config, first))
    assert first.read_bytes() == original


@pytest.mark.parametrize("failure", ["phenotype_header", "covariate_order"])
def test_source_manifest_rejects_header_or_fam_order_mismatch(
    tmp_path: Path, failure: str
):
    source_root, fam, config, paths = _write_fixture(tmp_path)
    if failure == "phenotype_header":
        text = paths["trait_a"].read_text(encoding="utf-8")
        paths["trait_a"].write_text(
            text.replace("FID IID pheno", "FID IID value"), encoding="utf-8"
        )
        message = "header must be"
    else:
        lines = paths["age_covar"].read_text(encoding="utf-8").splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        paths["age_covar"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        message = "order differs"

    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match=message):
        generator.generate_manifest(_args(source_root, fam, config, output))
    assert not output.exists()


def test_source_manifest_rejects_symlink_and_duplicate_fam(tmp_path: Path):
    source_root, fam, config, paths = _write_fixture(tmp_path)
    real_trait = source_root / "trait_a.real"
    paths["trait_a"].rename(real_trait)
    paths["trait_a"].symlink_to(real_trait.name)
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="symlink"):
        generator.generate_manifest(_args(source_root, fam, config, output))
    assert not output.exists()

    paths["trait_a"].unlink()
    real_trait.rename(paths["trait_a"])
    fam_lines = fam.read_text(encoding="utf-8").splitlines()
    fam_lines[-1] = fam_lines[0]
    fam.write_text("\n".join(fam_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate FID/IID"):
        generator.generate_manifest(_args(source_root, fam, config, output))
    assert not output.exists()


def test_source_manifest_rejects_output_below_source_root(tmp_path: Path):
    source_root, fam, config, _ = _write_fixture(tmp_path)
    output = source_root / "must_not_be_written.json"
    with pytest.raises(ValueError, match="outside the read-only source root"):
        generator.generate_manifest(_args(source_root, fam, config, output))
    assert not output.exists()


def test_source_manifest_rejects_symlinked_output_ancestor(tmp_path: Path):
    source_root, fam, config, _ = _write_fixture(tmp_path)
    real_output_parent = source_root / "nested"
    real_output_parent.mkdir()
    source_alias = tmp_path / "source_alias"
    source_alias.symlink_to(source_root, target_is_directory=True)
    output = source_alias / "nested" / "must_not_be_written.json"

    with pytest.raises(ValueError, match="path contains a symlink"):
        generator.generate_manifest(_args(source_root, fam, config, output))
    assert not (real_output_parent / output.name).exists()
