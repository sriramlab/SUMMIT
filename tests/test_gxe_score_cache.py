from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.inference.gxe import fit_from_files
from summit.ldscore import gxe_score
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.logger import Logger


@dataclass(frozen=True)
class CachedReferenceFixture:
    root: Path
    genotype_prefix: Path
    environment: Path
    covariates: Path
    phenotype: Path
    annotations: Path
    reference_prefix: Path
    n_samples: int
    n_variants: int

    @property
    def manifest(self) -> Path:
        return Path(str(self.reference_prefix) + ".gxe.ref.json")

    @property
    def original_gwas(self) -> Path:
        return Path(str(self.reference_prefix) + ".gxe.gwas.tsv.gz")

    @property
    def original_gwis(self) -> Path:
        return Path(str(self.reference_prefix) + ".gxe.gwis.tsv.gz")

    @property
    def original_moments(self) -> Path:
        return Path(str(self.reference_prefix) + ".gxe.moments.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def cached_reference(tmp_path_factory) -> CachedReferenceFixture:
    root = tmp_path_factory.mktemp("gxe_cached_reference")
    rng = np.random.default_rng(99017)
    n, m = 43, 15
    genotype = rng.binomial(
        2, rng.uniform(0.12, 0.46, size=m), size=(n, m)
    ).astype(np.float64)
    genotype_prefix = root / "genotype"
    to_bed(str(genotype_prefix) + ".bed", genotype)

    fam = pd.read_csv(str(genotype_prefix) + ".fam", sep=r"\s+", header=None)
    bim = pd.read_csv(str(genotype_prefix) + ".bim", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    environment_values = rng.normal(size=n)
    covariate_values = rng.normal(size=n)
    phenotype_values = rng.normal(size=n)
    environment = root / "environment.tsv"
    covariates = root / "covariates.tsv"
    phenotype = root / "phenotype.tsv"
    ids.assign(E=environment_values).to_csv(environment, sep="\t", index=False)
    ids.assign(C=covariate_values).to_csv(covariates, sep="\t", index=False)
    ids.assign(Y=phenotype_values).to_csv(phenotype, sep="\t", index=False)

    annotations = np.column_stack(
        [
            0.15 + np.arange(m, dtype=np.float64) / 23.0,
            0.2 + (np.arange(m) % 5) / 13.0,
        ]
    )
    annotation_path = root / "annotations.tsv"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "a": annotations[:, 0],
            "b": annotations[:, 1],
        }
    ).to_csv(annotation_path, sep="\t", index=False, float_format="%.17g")

    reference_prefix = root / "reference"
    estimator = GenomewideEnvLDScore(
        bed_path=str(genotype_prefix),
        env_path=str(environment),
        annot_path=str(annotation_path),
        out_path=str(reference_prefix),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(covariates),
        pheno_path=str(phenotype),
        num_vecs=5,
        step_size=m,
        seed=31,
        verbose=False,
        dtype="float32",
        kernel_mode="standardized",
        genotype_scale="sample",
        target_xz_mem=0.01,
    )
    estimator._compute_ldscore()
    return CachedReferenceFixture(
        root=root,
        genotype_prefix=genotype_prefix,
        environment=environment,
        covariates=covariates,
        phenotype=phenotype,
        annotations=annotation_path,
        reference_prefix=reference_prefix,
        n_samples=n,
        n_variants=m,
    )


class _CountingBed:
    def __init__(self, delegate, calls):
        self._delegate = delegate
        self._calls = calls
        self.shape = delegate.shape

    def read(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self._delegate.read(*args, **kwargs)


def test_cached_scoring_matches_reference_generation_and_decodes_once(
    cached_reference, tmp_path, monkeypatch
):
    calls = []
    open_calls = []
    actual_open_bed = gxe_score.open_bed

    def counted_open_bed(path, **kwargs):
        open_calls.append((path, kwargs))
        return _CountingBed(actual_open_bed(path, **kwargs), calls)

    monkeypatch.setattr(gxe_score, "open_bed", counted_open_bed)
    output_prefix = tmp_path / "cached"
    artifacts = gxe_score.score_phenotype_from_reference(
        reference_manifest=cached_reference.manifest,
        bed_path=cached_reference.genotype_prefix,
        env_path=cached_reference.environment,
        covar_path=cached_reference.covariates,
        pheno_path=cached_reference.phenotype,
        output_prefix=output_prefix,
        step_size=4,
        num_threads=2,
    )
    assert len(open_calls) == 1
    assert open_calls[0][1] == {"num_threads": 2}
    assert len(calls) == math.ceil(cached_reference.n_variants / 4)
    assert all(kwargs["num_threads"] == 2 for _, kwargs in calls)

    original_gwas = pd.read_csv(cached_reference.original_gwas, sep=r"\s+")
    original_gwis = pd.read_csv(cached_reference.original_gwis, sep=r"\s+")
    cached_gwas = pd.read_csv(artifacts.gwas, sep=r"\s+")
    cached_gwis = pd.read_csv(artifacts.gwis, sep=r"\s+")
    pd.testing.assert_frame_equal(
        cached_gwas.drop(columns="SCORE"), original_gwas.drop(columns="SCORE")
    )
    pd.testing.assert_frame_equal(
        cached_gwis.drop(columns="SCORE"), original_gwis.drop(columns="SCORE")
    )
    np.testing.assert_array_equal(cached_gwas["SCORE"], original_gwas["SCORE"])
    np.testing.assert_array_equal(cached_gwis["SCORE"], original_gwis["SCORE"])

    original_moments = json.loads(cached_reference.original_moments.read_text())
    cached_moments = json.loads(artifacts.moments.read_text())
    for key in (
        "analysis_fingerprint",
        "variant_digest",
        "phenotype",
        "n_samples",
        "residual_rank",
        "score_definition",
        "q_nxe",
        "q_residual",
        "phenotype_residual_variance_fraction",
    ):
        assert cached_moments[key] == original_moments[key]
    assert cached_moments["reference_manifest_sha256"] == _sha256(
        cached_reference.manifest
    )
    assert cached_moments["score_sha256"]["gwas"] == _sha256(artifacts.gwas)
    assert cached_moments["score_sha256"]["gwis"] == _sha256(artifacts.gwis)
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in (artifacts.gwas, artifacts.gwis, artifacts.moments)
    )
    fitted, equations = fit_from_files(
        cached_reference.manifest,
        artifacts.moments,
        artifacts.gwas,
        artifacts.gwis,
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    assert fitted.rhs.shape == equations.rhs.shape == (6,)


def test_cached_scoring_refuses_overwrite_without_changing_outputs(
    cached_reference, tmp_path
):
    output_prefix = tmp_path / "no-overwrite"
    artifacts = gxe_score.score_phenotype_from_reference(
        reference_manifest=cached_reference.manifest,
        bed_path=cached_reference.genotype_prefix,
        env_path=cached_reference.environment,
        covar_path=cached_reference.covariates,
        pheno_path=cached_reference.phenotype,
        output_prefix=output_prefix,
    )
    before = {
        path: path.read_bytes()
        for path in (artifacts.gwas, artifacts.gwis, artifacts.moments)
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=output_prefix,
        )
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not Path(str(output_prefix) + ".gxe.score.lock").exists()


def test_cached_scoring_preserves_explicit_genie_compatibility(
    cached_reference, tmp_path
):
    genie_prefix = tmp_path / "genie-reference"
    estimator = GenomewideEnvLDScore(
        bed_path=str(cached_reference.genotype_prefix),
        env_path=str(cached_reference.environment),
        annot_path=str(cached_reference.annotations),
        out_path=str(genie_prefix),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(cached_reference.covariates),
        pheno_path=str(cached_reference.phenotype),
        num_vecs=3,
        step_size=cached_reference.n_variants,
        seed=53,
        verbose=False,
        dtype="float32",
        kernel_mode="genie",
        genotype_scale="hwe",
        target_xz_mem=0.01,
    )
    estimator._compute_ldscore()
    cached = gxe_score.score_phenotype_from_reference(
        reference_manifest=Path(str(genie_prefix) + ".gxe.ref.json"),
        bed_path=cached_reference.genotype_prefix,
        env_path=cached_reference.environment,
        covar_path=cached_reference.covariates,
        pheno_path=cached_reference.phenotype,
        output_prefix=tmp_path / "genie-cached",
    )
    original_gwas = pd.read_csv(
        Path(str(genie_prefix) + ".gxe.gwas.tsv.gz"), sep=r"\s+"
    )
    original_gwis = pd.read_csv(
        Path(str(genie_prefix) + ".gxe.gwis.tsv.gz"), sep=r"\s+"
    )
    cached_gwas = pd.read_csv(cached.gwas, sep=r"\s+")
    cached_gwis = pd.read_csv(cached.gwis, sep=r"\s+")
    np.testing.assert_array_equal(cached_gwas["SCORE"], original_gwas["SCORE"])
    np.testing.assert_array_equal(cached_gwis["SCORE"], original_gwis["SCORE"])


def _copy_genotype_triple(source_prefix: Path, target_prefix: Path) -> None:
    for extension in (".bed", ".bim", ".fam"):
        shutil.copy2(str(source_prefix) + extension, str(target_prefix) + extension)


def _copy_reference_bundle(source_manifest: Path, target_directory: Path) -> Path:
    target_directory.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source_manifest.read_text())
    for relative in payload["files"].values():
        source = source_manifest.parent / relative
        shutil.copy2(source, target_directory / Path(relative).name)
    target_manifest = target_directory / source_manifest.name
    target_manifest.write_bytes(source_manifest.read_bytes())
    return target_manifest


def test_cached_scoring_rejects_bed_sha_tamper(cached_reference, tmp_path):
    tampered_prefix = tmp_path / "tampered-genotype"
    _copy_genotype_triple(cached_reference.genotype_prefix, tampered_prefix)
    bed_path = Path(str(tampered_prefix) + ".bed")
    content = bytearray(bed_path.read_bytes())
    content[-1] ^= 0x01
    bed_path.write_bytes(content)

    with pytest.raises(ValueError, match="BED|bed|SHA-256"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=tampered_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=tmp_path / "bed-tamper-output",
        )


def test_cached_scoring_rejects_reference_artifact_tamper(
    cached_reference, tmp_path
):
    copied_manifest = _copy_reference_bundle(
        cached_reference.manifest, tmp_path / "tampered-reference"
    )
    payload = json.loads(copied_manifest.read_text())
    diagonal = copied_manifest.parent / Path(payload["files"]["diagonal"]).name
    diagonal.write_bytes(diagonal.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="artifact.*SHA-256"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=copied_manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=tmp_path / "artifact-tamper-output",
        )


def test_cached_scoring_rejects_analysis_fingerprint_mismatch(
    cached_reference, tmp_path
):
    altered = pd.read_csv(cached_reference.environment, sep=r"\s+")
    altered.loc[0, "E"] += 0.5
    altered_environment = tmp_path / "altered-environment.tsv"
    altered.to_csv(altered_environment, sep="\t", index=False)

    with pytest.raises(ValueError, match="Environment transform|fingerprint"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=altered_environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=tmp_path / "analysis-mismatch-output",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("kernel_mode", "genie", "scale|GENIE"),
        ("genotype_scale", "hwe", "norm|scal"),
        ("residual_rank", 999, "rank"),
    ],
)
def test_cached_scoring_rejects_manifest_feature_definition_mismatch(
    cached_reference, tmp_path, field, replacement, message
):
    copied_manifest = _copy_reference_bundle(
        cached_reference.manifest, tmp_path / f"bad-{field}"
    )
    payload = json.loads(copied_manifest.read_text())
    payload[field] = replacement
    copied_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=copied_manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=tmp_path / f"bad-{field}-output",
        )


def test_cached_scoring_rejects_variant_axis_mismatch_even_with_updated_bim_hash(
    cached_reference, tmp_path
):
    altered_prefix = tmp_path / "altered-axis"
    _copy_genotype_triple(cached_reference.genotype_prefix, altered_prefix)
    bim_path = Path(str(altered_prefix) + ".bim")
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None, dtype=str)
    bim.loc[0, 4] = "T" if bim.loc[0, 4] != "T" else "A"
    bim.to_csv(bim_path, sep="\t", index=False, header=False)

    copied_manifest = _copy_reference_bundle(
        cached_reference.manifest, tmp_path / "altered-axis-reference"
    )
    payload = json.loads(copied_manifest.read_text())
    payload["genotype_files"][".bim"] = {
        "bytes": bim_path.stat().st_size,
        "sha256": _sha256(bim_path),
    }
    copied_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="variant axis|variant digest"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=copied_manifest,
            bed_path=altered_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=cached_reference.phenotype,
            output_prefix=tmp_path / "variant-axis-output",
        )


def test_cached_scoring_rejects_changed_complete_case_sample(cached_reference, tmp_path):
    phenotype = pd.read_csv(cached_reference.phenotype, sep=r"\s+")
    phenotype.loc[0, "Y"] = -9
    missing_phenotype = tmp_path / "missing-phenotype.tsv"
    phenotype.to_csv(missing_phenotype, sep="\t", index=False)

    with pytest.raises(ValueError, match="sample count"):
        gxe_score.score_phenotype_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=missing_phenotype,
            output_prefix=tmp_path / "sample-mismatch-output",
        )


def _write_wide_phenotype(cached_reference, path: Path) -> pd.DataFrame:
    phenotype = pd.read_csv(cached_reference.phenotype, sep=r"\s+")
    sample_index = np.arange(len(phenotype), dtype=np.float64)
    phenotype["Y2"] = (
        np.sin(0.41 * sample_index)
        + 0.31 * np.cos(0.17 * sample_index)
        + 0.007 * sample_index
    )
    phenotype.to_csv(path, sep="\t", index=False, float_format="%.17g")
    return phenotype


def test_wide_cached_scoring_decodes_once_and_matches_single_trait_oracles(
    cached_reference, tmp_path, monkeypatch
):
    wide_path = tmp_path / "wide.tsv"
    wide = _write_wide_phenotype(cached_reference, wide_path)
    second_trait_path = tmp_path / "second-trait.tsv"
    wide.loc[:, ["FID", "IID", "Y2"]].to_csv(
        second_trait_path, sep="\t", index=False, float_format="%.17g"
    )
    second_oracle = gxe_score.score_phenotype_from_reference(
        reference_manifest=cached_reference.manifest,
        bed_path=cached_reference.genotype_prefix,
        env_path=cached_reference.environment,
        covar_path=cached_reference.covariates,
        pheno_path=second_trait_path,
        pheno_col="Y2",
        output_prefix=tmp_path / "second-oracle",
        step_size=4,
    )

    calls = []
    open_calls = []
    actual_open_bed = gxe_score.open_bed

    def counted_open_bed(path, **kwargs):
        open_calls.append((path, kwargs))
        return _CountingBed(actual_open_bed(path, **kwargs), calls)

    monkeypatch.setattr(gxe_score, "open_bed", counted_open_bed)
    results = gxe_score.score_phenotypes_from_reference(
        reference_manifest=cached_reference.manifest,
        bed_path=cached_reference.genotype_prefix,
        env_path=cached_reference.environment,
        covar_path=cached_reference.covariates,
        pheno_path=wide_path,
        pheno_cols=["Y", "Y2"],
        output_prefix=tmp_path / "wide-scores",
        step_size=4,
        num_threads=2,
    )
    assert list(results) == ["Y", "Y2"]
    assert len(open_calls) == 1
    assert open_calls[0][1] == {"num_threads": 2}
    assert len(calls) == math.ceil(cached_reference.n_variants / 4)
    assert all(kwargs["num_threads"] == 2 for _, kwargs in calls)

    oracles = {
        "Y": (
            cached_reference.original_gwas,
            cached_reference.original_gwis,
            cached_reference.original_moments,
        ),
        "Y2": (second_oracle.gwas, second_oracle.gwis, second_oracle.moments),
    }
    for trait, artifacts in results.items():
        oracle_gwas, oracle_gwis, oracle_moments = oracles[trait]
        observed_gwas = pd.read_csv(artifacts.gwas, sep=r"\s+")
        observed_gwis = pd.read_csv(artifacts.gwis, sep=r"\s+")
        expected_gwas = pd.read_csv(oracle_gwas, sep=r"\s+")
        expected_gwis = pd.read_csv(oracle_gwis, sep=r"\s+")
        pd.testing.assert_frame_equal(observed_gwas, expected_gwas, check_exact=True)
        pd.testing.assert_frame_equal(observed_gwis, expected_gwis, check_exact=True)

        observed_moments = json.loads(artifacts.moments.read_text())
        expected_moments = json.loads(oracle_moments.read_text())
        for key in (
            "analysis_fingerprint",
            "variant_digest",
            "phenotype",
            "n_samples",
            "residual_rank",
            "score_definition",
        ):
            assert observed_moments[key] == expected_moments[key]
        for key in (
            "q_nxe",
            "q_residual",
            "phenotype_residual_variance_fraction",
        ):
            # Batched residualization uses matrix reductions whereas the
            # generator's one-phenotype path uses vector reductions.  They can
            # differ by one final floating-point ulp even though the serialized
            # GWAS/GWIS score tables above are exactly equal.
            np.testing.assert_allclose(
                observed_moments[key], expected_moments[key], rtol=2.0e-15, atol=1.0e-14
            )
        assert observed_moments["reference_manifest_sha256"] == _sha256(
            cached_reference.manifest
        )
        assert observed_moments["score_sha256"]["gwas"] == _sha256(artifacts.gwas)
        assert observed_moments["score_sha256"]["gwis"] == _sha256(artifacts.gwis)
        assert all(
            (path.stat().st_mode & 0o777) == 0o600
            for path in (artifacts.gwas, artifacts.gwis, artifacts.moments)
        )


def test_wide_cached_scoring_rejects_any_missing_trait_before_decode(
    cached_reference, tmp_path, monkeypatch
):
    wide_path = tmp_path / "wide-missing.tsv"
    wide = _write_wide_phenotype(cached_reference, wide_path)
    wide.loc[3, "Y2"] = -9
    wide.to_csv(wide_path, sep="\t", index=False, float_format="%.17g")

    def unexpected_open_bed(_path):
        raise AssertionError("BED decoding must not start for an incomplete wide trait")

    monkeypatch.setattr(gxe_score, "open_bed", unexpected_open_bed)
    output_directory = tmp_path / "missing-output"
    with pytest.raises(ValueError, match="complete and finite.*reference cohort"):
        gxe_score.score_phenotypes_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=wide_path,
            pheno_cols=["Y", "Y2"],
            output_prefix=output_directory / "scores",
        )
    assert output_directory.is_dir()
    assert list(output_directory.iterdir()) == []


def test_wide_cached_scoring_rolls_back_every_trait_on_publish_failure(
    cached_reference, tmp_path, monkeypatch
):
    wide_path = tmp_path / "wide-rollback.tsv"
    _write_wide_phenotype(cached_reference, wide_path)
    actual_publish = gxe_score._publish_private_no_replace
    publish_count = 0

    def fail_during_second_trait(temporary, target):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 4:
            raise OSError("injected wide publication failure")
        return actual_publish(temporary, target)

    monkeypatch.setattr(
        gxe_score, "_publish_private_no_replace", fail_during_second_trait
    )
    output_directory = tmp_path / "rollback-output"
    with pytest.raises(OSError, match="injected wide publication failure"):
        gxe_score.score_phenotypes_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=wide_path,
            pheno_cols=["Y", "Y2"],
            output_prefix=output_directory / "scores",
            step_size=4,
        )
    assert publish_count == 4
    assert output_directory.is_dir()
    assert list(output_directory.iterdir()) == []


@pytest.mark.parametrize("unsafe_trait", [".hidden", "_hidden", "-hidden"])
def test_wide_cached_scoring_rejects_hidden_or_non_alphanumeric_trait_labels(
    cached_reference, tmp_path, unsafe_trait
):
    wide_path = tmp_path / f"unsafe-{unsafe_trait[0]}.tsv"
    wide = pd.read_csv(cached_reference.phenotype, sep=r"\s+")
    wide[unsafe_trait] = np.arange(len(wide), dtype=np.float64)
    wide.to_csv(wide_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="start with a letter or digit"):
        gxe_score.score_phenotypes_from_reference(
            reference_manifest=cached_reference.manifest,
            bed_path=cached_reference.genotype_prefix,
            env_path=cached_reference.environment,
            covar_path=cached_reference.covariates,
            pheno_path=wide_path,
            pheno_cols=[unsafe_trait],
            output_prefix=tmp_path / "unsafe-output",
        )
