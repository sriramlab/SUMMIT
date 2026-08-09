from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "gxe" / "hoffman"


def _load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_script("build_common_group")
verifier = _load_script("verify_staged_inputs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_fam_order_inputs(root: Path, n: int = 8) -> tuple[Path, Path, Path, Path]:
    ids = [(str(i), str(i)) for i in range(1, n + 1)]
    fam = root / "cohort.fam"
    fam.write_text("".join(f"{fid} {iid} 0 0 0 -9\n" for fid, iid in ids), encoding="utf-8")

    covar = root / "common.covar"
    covar.write_text(
        "FID IID age sex pc1\n"
        + "".join(
            f"{fid} {iid} {30 + 5 * idx} {1 + idx % 2} {((idx * 7) % 11) / 10:.1f}\n"
            for idx, (fid, iid) in enumerate(ids)
        ),
        encoding="utf-8",
    )
    y1 = root / "trait_a.pheno"
    y2 = root / "trait_b.pheno"
    y1.write_text(
        "FID IID pheno\n"
        + "".join(
            f"{fid} {iid} {'NA' if idx == 1 else 1.5 + idx:.3}\n"
            for idx, (fid, iid) in enumerate(ids)
        ),
        encoding="utf-8",
    )
    y2.write_text(
        "FID IID pheno\n"
        + "".join(
            f"{fid} {iid} {'-9' if idx == 5 else 2.5 + 0.5 * idx:.3}\n"
            for idx, (fid, iid) in enumerate(ids)
        ),
        encoding="utf-8",
    )
    for path in (fam, covar, y1, y2):
        os.chmod(path, 0o600)
    return fam, covar, y1, y2


def _builder_args(
    fam: Path,
    covar: Path,
    y1: Path,
    y2: Path,
    out_dir: Path,
    scratch_root: Path | None = None,
):
    return argparse.Namespace(
        scratch_root=str(scratch_root or fam.parent),
        fam=str(fam),
        covar=str(covar),
        environment_column="age",
        phenotype=[f"trait_a={y1}", f"trait_b={y2}"],
        phenotype_column="pheno",
        label="age_group",
        out_dir=str(out_dir),
        missing_values=",".join(builder.DEFAULT_MISSING_VALUES),
        ddof=1,
        allow_source_superset=False,
    )


def _write_plink(prefix: Path, n: int = 8, m: int = 2) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(prefix.parent, 0o700)
    Path(str(prefix) + ".fam").write_text(
        "".join(f"{i} {i} 0 0 0 -9\n" for i in range(1, n + 1)), encoding="utf-8"
    )
    Path(str(prefix) + ".bim").write_text(
        "".join(f"1 rs{i} 0 {100 + i} A G\n" for i in range(1, m + 1)), encoding="utf-8"
    )
    bytes_per_variant = (n + 3) // 4
    Path(str(prefix) + ".bed").write_bytes(b"\x6c\x1b\x01" + bytes([0x55]) * (bytes_per_variant * m))


def test_common_group_builder_fixes_one_private_cohort(tmp_path: Path):
    fam, covar, y1, y2 = _write_fam_order_inputs(tmp_path)
    parent = tmp_path / "groups"
    parent.mkdir(mode=0o700)
    out_dir = parent / "age_group"

    manifest_path = builder.build_group(_builder_args(fam, covar, y1, y2, out_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["kind"] == "summit.gxe.common_cohort_group"
    assert manifest["n_fam_samples"] == 8
    assert manifest["n_selected_samples"] == 6
    assert manifest["phenotype_labels"] == ["trait_a", "trait_b"]
    assert manifest["residual_rank"] == 6 - manifest["fixed_effect_rank_excluding_intercept"] - 1
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    for record in manifest["outputs"].values():
        output = out_dir / record["path"]
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
        assert output.stat().st_size == record["bytes"]
        assert _sha256(output) == record["sha256"]
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    env = pd.read_csv(out_dir / "age_group.env.tsv", sep="\t")
    cov = pd.read_csv(out_dir / "age_group.covar.tsv", sep="\t")
    phen = pd.read_csv(out_dir / "age_group.phenotypes.tsv", sep="\t")
    outside = phen[["trait_a", "trait_b"]].isna().any(axis=1)
    assert int(outside.sum()) == 2
    assert env.loc[outside, "ENV"].isna().all()
    assert cov.loc[outside, ["sex", "pc1"]].isna().all(axis=None)
    assert phen.loc[outside, ["trait_a", "trait_b"]].isna().all(axis=None)

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        builder.build_group(_builder_args(fam, covar, y1, y2, out_dir))


def test_common_group_builder_rejects_nonexact_id_set_before_writing(tmp_path: Path):
    fam, covar, y1, y2 = _write_fam_order_inputs(tmp_path)
    lines = y2.read_text(encoding="utf-8").splitlines()
    y2.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    parent = tmp_path / "groups"
    parent.mkdir(mode=0o700)
    out_dir = parent / "age_group"
    with pytest.raises(ValueError, match="ID set differs"):
        builder.build_group(_builder_args(fam, covar, y1, y2, out_dir))
    assert not out_dir.exists()


def test_common_group_builder_allows_explicit_source_superset(tmp_path: Path):
    full_fam, covar, y1, y2 = _write_fam_order_inputs(tmp_path)
    subset_fam = tmp_path / "subset.fam"
    subset_fam.write_text(
        "".join(full_fam.read_text(encoding="utf-8").splitlines(keepends=True)[:6]),
        encoding="utf-8",
    )
    os.chmod(subset_fam, 0o600)
    parent = tmp_path / "groups"
    parent.mkdir(mode=0o700)
    args = _builder_args(subset_fam, covar, y1, y2, parent / "age_group")
    args.allow_source_superset = True
    manifest_path = builder.build_group(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_fam_samples"] == 6
    assert manifest["source_id_contract"] == "fam_subset_of_each_source"


def test_staged_verifier_hashes_source_and_private_copy(tmp_path: Path):
    source_prefix = tmp_path / "project_source" / "toy"
    _write_plink(source_prefix)

    scratch = tmp_path / "scratch_root"
    scratch.mkdir(mode=0o700)
    os.chmod(scratch, 0o700)
    staged_prefix = scratch / "geno" / "toy"
    _write_plink(staged_prefix)
    for extension in ("bed", "bim", "fam"):
        source = Path(str(source_prefix) + f".{extension}")
        staged = Path(str(staged_prefix) + f".{extension}")
        shutil.copyfile(source, staged)
        os.chmod(staged, 0o600)

    inputs = scratch / "input_sources"
    inputs.mkdir(mode=0o700)
    os.chmod(inputs, 0o700)
    _, covar_src, y1_src, y2_src = _write_fam_order_inputs(inputs)
    for path in (covar_src, y1_src, y2_src):
        os.chmod(path, 0o600)
    groups_parent = scratch / "groups"
    groups_parent.mkdir(mode=0o700)
    os.chmod(groups_parent, 0o700)
    group_dir = groups_parent / "age_group"
    group_manifest = builder.build_group(
        _builder_args(
            Path(str(staged_prefix) + ".fam"),
            covar_src,
            y1_src,
            y2_src,
            group_dir,
            scratch_root=scratch,
        )
    )
    manifest = json.loads(group_manifest.read_text(encoding="utf-8"))

    config = {
        "kind": "summit.gxe.hoffman_panel",
        "schema_version": 1,
        "scratch_root": str(scratch),
        "datasets": {
            "full": {
                "source_prefix": str(source_prefix),
                "expected_n_samples": 8,
                "expected_n_variants": 2,
                "expected_bed_bytes": 7,
                "expected_bed_magic": "6c1b01",
                "expected_fam_sha256": _sha256(Path(str(source_prefix) + ".fam")),
                "expected_bim_sha256": _sha256(Path(str(source_prefix) + ".bim")),
            },
            "subset_50k": {
                "source_prefix": str(source_prefix),
                "expected_n_samples": 8,
                "expected_n_variants": 2,
                "expected_bed_bytes": 7,
                "expected_bed_magic": "6c1b01",
            },
        },
        "groups": {
            "age_group": {
                "environment": "age",
                "phenotypes": ["trait_a", "trait_b"],
                "expected_common_n_full": manifest["n_selected_samples"],
                "expected_rank_full": manifest["fixed_effect_rank_excluding_intercept"],
                "expected_residual_rank_full": manifest["residual_rank"],
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    reports = scratch / "reports"
    reports.mkdir(mode=0o700)
    os.chmod(reports, 0o700)
    report = reports / "verified.json"
    args = argparse.Namespace(
        config=str(config_path),
        dataset="full",
        geno_prefix=str(staged_prefix),
        group_manifest=[str(group_manifest)],
        report=str(report),
    )
    observed_report = verifier.verify(args)
    payload = json.loads(observed_report.read_text(encoding="utf-8"))
    assert payload["plink_shape"]["n_samples"] == 8
    assert payload["plink_shape"]["n_variants"] == 2
    assert payload["groups"][0]["label"] == "age_group"
    assert payload["hashes"]["source"] == payload["hashes"]["staged"]
    assert stat.S_IMODE(observed_report.stat().st_mode) == 0o600

    staged_bed = Path(str(staged_prefix) + ".bed")
    corrupted = bytearray(staged_bed.read_bytes())
    corrupted[-1] ^= 0x01
    staged_bed.write_bytes(corrupted)
    os.chmod(staged_bed, 0o600)
    args.report = str(reports / "must_not_exist.json")
    with pytest.raises(ValueError, match="hash differs"):
        verifier.verify(args)
    assert not Path(args.report).exists()
