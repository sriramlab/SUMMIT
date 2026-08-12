from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.inference import gxe as gxe_module
from summit.inference.gxe import (
    GxENormalEquations,
    GxEPhenotypeInput,
    fit_from_files,
    fit_many_from_files,
    solve_normal_equations,
    write_fit,
)
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore, _orthonormalize_columns
from summit.logger import Logger


@dataclass
class TinyGxEInputs:
    root: Path
    prefix: Path
    env_path: Path
    cov_path: Path
    pheno_path: Path
    annot_path: Path
    raw: np.ndarray
    env_raw: np.ndarray
    cov_raw: np.ndarray
    phenotype_raw: np.ndarray
    annotations: np.ndarray


@dataclass
class ExactStandardizedBundle:
    inputs: TinyGxEInputs
    estimator: GenomewideEnvLDScore
    out: Path
    projector: np.ndarray
    x: np.ndarray
    w: np.ndarray
    y: np.ndarray
    residual_rank: int


def _make_inputs(root: Path) -> TinyGxEInputs:
    rng = np.random.default_rng(20260808)
    n, m = 39, 13
    raw = rng.binomial(2, rng.uniform(0.12, 0.44, size=m), size=(n, m)).astype(
        np.float64
    )
    prefix = root / "tiny"
    to_bed(str(prefix) + ".bed", raw)

    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_raw = rng.normal(size=n)
    cov_raw = rng.normal(size=n)
    phenotype_raw = rng.normal(size=n)
    env_path = root / "tiny.env"
    cov_path = root / "tiny.cov"
    pheno_path = root / "tiny.pheno"
    ids.assign(E=env_raw).to_csv(env_path, sep="\t", index=False)
    ids.assign(C=cov_raw).to_csv(cov_path, sep="\t", index=False)
    ids.assign(Y=phenotype_raw).to_csv(pheno_path, sep="\t", index=False)

    # Deliberately non-binary and not exactly representable in float32.  This
    # catches annotation masses computed on a different numeric representation
    # from the weights used in the sketches and written to the diagonal file.
    annotations = np.column_stack(
        [
            0.100000003 + np.arange(m, dtype=np.float64) / 17.0,
            0.200000007 + (np.arange(m) % 4) / 11.0,
        ]
    )
    annot_path = root / "tiny.annot"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "fractional_a": annotations[:, 0],
            "fractional_b": annotations[:, 1],
        }
    ).to_csv(annot_path, sep="\t", index=False, float_format="%.17g")
    return TinyGxEInputs(
        root=root,
        prefix=prefix,
        env_path=env_path,
        cov_path=cov_path,
        pheno_path=pheno_path,
        annot_path=annot_path,
        raw=raw,
        env_raw=env_raw,
        cov_raw=cov_raw,
        phenotype_raw=phenotype_raw,
        annotations=annotations,
    )


def _new_estimator(
    inputs: TinyGxEInputs,
    out: Path,
    *,
    env_path: Path | None = None,
    pheno_path: Path | None = None,
    dtype: str = "float64",
    kernel_mode: str | None = "standardized",
    write_jackknife: bool = False,
) -> GenomewideEnvLDScore:
    kwargs = dict(
        bed_path=str(inputs.prefix),
        env_path=str(inputs.env_path if env_path is None else env_path),
        annot_path=str(inputs.annot_path),
        out_path=str(out),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(inputs.cov_path),
        pheno_path=str(inputs.pheno_path if pheno_path is None else pheno_path),
        num_vecs=inputs.raw.shape[1],
        step_size=inputs.raw.shape[1],
        seed=11,
        verbose=False,
        dtype=dtype,
        genotype_scale="hwe",
        target_xz_mem=0.01,
    )
    if write_jackknife:
        kwargs.update(
            write_jackknife=True,
            jackknife_spec="3",
            allow_low_probe_jackknife=True,
        )
    if kernel_mode is not None:
        kwargs["kernel_mode"] = kernel_mode
    return GenomewideEnvLDScore(**kwargs)


def _install_exact_probes(estimator: GenomewideEnvLDScore) -> None:
    m = estimator.nsnps

    def exact_probes(self, L, v_count, blk_start, v_start):
        assert (v_count, v_start) == (m, 0)
        return np.asfortranarray(
            math.sqrt(m) * np.eye(m)[blk_start:blk_start + L],
            dtype=np.dtype(self.dtype),
        )

    estimator._generate_random_block = types.MethodType(exact_probes, estimator)


def _dense_standardized_design(
    inputs: TinyGxEInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n = inputs.raw.shape[0]
    e = (inputs.env_raw - inputs.env_raw.mean()) / inputs.env_raw.std(ddof=1)
    c = (inputs.cov_raw - inputs.cov_raw.mean()) / inputs.cov_raw.std(ddof=1)
    q = _orthonormalize_columns(np.column_stack([np.ones(n), c, e]))
    projector = np.eye(n) - q @ q.T
    residual_rank = n - q.shape[1]

    dosage_mean = inputs.raw.mean(axis=0)
    hwe_sd = np.sqrt(dosage_mean * (1.0 - 0.5 * dosage_mean))
    g = (inputs.raw - dosage_mean) / hwe_sd
    x = projector @ g
    w = projector @ (e[:, None] * g)
    x *= np.sqrt(residual_rank / np.sum(x * x, axis=0))[None, :]
    w *= np.sqrt(residual_rank / np.sum(w * w, axis=0))[None, :]
    y = projector @ inputs.phenotype_raw
    y *= math.sqrt(residual_rank / float(y @ y))
    return projector, x, w, y, residual_rank


@pytest.fixture(scope="module")
def exact_bundle(tmp_path_factory) -> ExactStandardizedBundle:
    root = tmp_path_factory.mktemp("gxe_summit_contract")
    inputs = _make_inputs(root)
    out = root / "standardized"
    estimator = _new_estimator(inputs, out)
    _install_exact_probes(estimator)
    estimator._compute_ldscore()
    projector, x, w, y, residual_rank = _dense_standardized_design(inputs)
    return ExactStandardizedBundle(
        inputs=inputs,
        estimator=estimator,
        out=out,
        projector=projector,
        x=x,
        w=w,
        y=y,
        residual_rank=residual_rank,
    )


def test_library_default_is_summit_post_projection_standardized():
    default = inspect.signature(GenomewideEnvLDScore).parameters["kernel_mode"].default
    assert default == "standardized"


def test_cli_default_is_summit_post_projection_standardized():
    from summit.cli import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.gxe_kernel_mode == "standardized"
    assert defaults.gxe_native_backend == "python"
    assert defaults.gxe_native_workspace_gib == 16.0
    assert defaults.gxe_native_target_panel_columns == 64
    assert not hasattr(defaults, "gxe_jackknife_scratch_gib")

    explicit = build_parser().parse_args(
        [
            "--gxe-native-backend", "direct",
            "--gxe-native-workspace-gib", "2.5",
            "--gxe-native-target-panel-columns", "32",
        ]
    )
    assert explicit.gxe_native_backend == "direct"
    assert explicit.gxe_native_workspace_gib == 2.5
    assert explicit.gxe_native_target_panel_columns == 32


def test_production_post_projection_norms_and_fixed_effect_leakage(exact_bundle):
    obj = exact_bundle.estimator
    g = obj._read_genotype_block(0, obj.nsnps)
    x = obj._prepare_additive_block(0, obj.nsnps, G=g, out_dtype=np.float64)
    w = obj._prepare_interaction_block(0, obj.nsnps, G=g, out_dtype=np.float64)
    r = exact_bundle.residual_rank

    np.testing.assert_allclose(x, exact_bundle.x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(w, exact_bundle.w, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.sum(x * x, axis=0), r, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(np.sum(w * w, axis=0), r, rtol=2e-14, atol=2e-14)

    fixed = _orthonormalize_columns(
        np.column_stack([np.ones(obj.nsamp), obj.C_int])
    )
    assert np.linalg.norm(fixed.T @ x) / np.linalg.norm(x) < 1e-13
    assert np.linalg.norm(fixed.T @ w) / np.linalg.norm(w) < 1e-13


def test_production_interaction_is_formed_before_projection(exact_bundle):
    obj = exact_bundle.estimator
    g = obj._read_genotype_block(0, obj.nsnps)
    observed = obj._prepare_interaction_block(
        0, obj.nsnps, G=g, out_dtype=np.float64
    )
    p = exact_bundle.projector
    e = obj.env

    expected = p @ (e[:, None] * g)
    expected *= np.sqrt(obj.df_corr / np.sum(expected * expected, axis=0))[None, :]
    projected_first = p @ g
    wrong = p @ (e[:, None] * projected_first)
    wrong *= np.sqrt(obj.df_corr / np.sum(wrong * wrong, axis=0))[None, :]

    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(observed - wrong) / np.linalg.norm(observed) > 1e-3


def _prepared_features(estimator: GenomewideEnvLDScore) -> tuple[np.ndarray, np.ndarray]:
    estimator.inv_sqrt_resvar_x_all, estimator.inv_sqrt_resvar_w_all = (
        estimator._precompute_residual_variances()
    )
    g = estimator._read_genotype_block(0, estimator.nsnps)
    x = estimator._prepare_additive_block(
        0, estimator.nsnps, G=g, out_dtype=np.float64
    )
    w = estimator._prepare_interaction_block(
        0, estimator.nsnps, G=g, out_dtype=np.float64
    )
    return x, w


def test_affine_environment_rescaling_preserves_kernels_and_tracks_sign(exact_bundle):
    inputs = exact_bundle.inputs
    fam = pd.read_csv(str(inputs.prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    positive_path = inputs.root / "env-positive-affine.tsv"
    negative_path = inputs.root / "env-negative-affine.tsv"
    ids.assign(E=3.7 * inputs.env_raw + 11.0).to_csv(
        positive_path, sep="\t", index=False
    )
    ids.assign(E=-2.3 * inputs.env_raw + 5.0).to_csv(
        negative_path, sep="\t", index=False
    )

    base = _new_estimator(inputs, inputs.root / "affine-base")
    positive = _new_estimator(
        inputs, inputs.root / "affine-positive", env_path=positive_path
    )
    negative = _new_estimator(
        inputs, inputs.root / "affine-negative", env_path=negative_path
    )
    xb, wb = _prepared_features(base)
    xp, wp = _prepared_features(positive)
    xn, wn = _prepared_features(negative)

    np.testing.assert_allclose(positive.env, base.env, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(negative.env, -base.env, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(xp, xb, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(xn, xb, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(wp, wb, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(wn, -wb, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(wp @ wp.T, wb @ wb.T, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(wn @ wn.T, wb @ wb.T, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(positive.score_w_all, base.score_w_all, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(negative.score_w_all, -base.score_w_all, rtol=3e-13, atol=3e-13)


def test_common_fixed_effect_signal_does_not_leak_into_scores(exact_bundle):
    inputs = exact_bundle.inputs
    fam = pd.read_csv(str(inputs.prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    shifted_path = inputs.root / "fixed-effect-shifted.pheno"
    shifted = inputs.phenotype_raw + 1.0e5 * (
        2.0 + 3.0 * inputs.env_raw - 4.0 * inputs.cov_raw
    )
    ids.assign(Y=shifted).to_csv(shifted_path, sep="\t", index=False)

    base = _new_estimator(inputs, inputs.root / "leakage-base")
    shifted_estimator = _new_estimator(
        inputs, inputs.root / "leakage-shifted", pheno_path=shifted_path
    )
    _prepared_features(base)
    _prepared_features(shifted_estimator)

    np.testing.assert_allclose(
        shifted_estimator.pheno, base.pheno, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        shifted_estimator.score_x_all, base.score_x_all, rtol=3e-9, atol=3e-9
    )
    np.testing.assert_allclose(
        shifted_estimator.score_w_all, base.score_w_all, rtol=3e-9, atol=3e-9
    )


@pytest.mark.parametrize(
    ("key", "left_name", "source_name"),
    [
        ("xx", "x", "x"),
        ("xw", "x", "w"),
        ("wx", "w", "x"),
        ("ww", "w", "w"),
    ],
)
def test_standardized_files_store_exact_raw_realized_panels(
    exact_bundle, key, left_name, source_name
):
    ref_path = Path(str(exact_bundle.out) + ".gxe.ref.json")
    ref = json.loads(ref_path.read_text())
    panel_path = ref_path.parent / ref["files"][key]
    observed = pd.read_csv(panel_path, sep=r"\s+").loc[
        :, list(ref["annotation_names"])
    ].to_numpy(dtype=np.float64)
    left = getattr(exact_bundle, left_name)
    source = getattr(exact_bundle, source_name)
    expected = (
        ((left.T @ source) / exact_bundle.residual_rank) ** 2
        @ exact_bundle.inputs.annotations
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-9, atol=3e-9)
    assert ref["null_corrected"] is False


def test_standardized_summary_reconstructs_dense_he_system(exact_bundle):
    out = exact_bundle.out
    _, equations = fit_from_files(
        str(out) + ".gxe.ref.json",
        str(out) + ".gxe.moments.json",
        str(out) + ".gxe.gwas.tsv.gz",
        str(out) + ".gxe.gwis.tsv.gz",
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    a = exact_bundle.inputs.annotations
    masses = a.sum(axis=0)
    kernels = []
    for col in range(a.shape[1]):
        kernels.append((exact_bundle.x * a[:, col]) @ exact_bundle.x.T / masses[col])
    for col in range(a.shape[1]):
        kernels.append((exact_bundle.w * a[:, col]) @ exact_bundle.w.T / masses[col])
    kernels.extend(
        [
            exact_bundle.projector
            @ np.diag(exact_bundle.estimator.env**2)
            @ exact_bundle.projector,
            exact_bundle.projector,
        ]
    )
    lhs = np.asarray([[np.trace(x @ y) for y in kernels] for x in kernels])
    rhs = np.asarray([exact_bundle.y @ k @ exact_bundle.y for k in kernels])
    traces = np.asarray([np.trace(k) for k in kernels])
    np.testing.assert_allclose(equations.matrix, lhs, rtol=4e-9, atol=4e-9)
    np.testing.assert_allclose(equations.rhs, rhs, rtol=4e-9, atol=4e-9)
    np.testing.assert_allclose(equations.traces, traces, rtol=4e-9, atol=4e-9)


def test_fractional_annotation_mass_matches_canonical_artifact_weights(exact_bundle):
    inputs = exact_bundle.inputs
    out = inputs.root / "float32-fractional"
    estimator = _new_estimator(inputs, out, dtype="float32")
    canonical_mass = np.asarray(estimator.annot, dtype=np.float64).sum(
        axis=0, dtype=np.float64
    )
    np.testing.assert_array_equal(estimator.nsnps_bin, canonical_mass)

    _install_exact_probes(estimator)
    estimator._compute_ldscore()
    ref_path = Path(str(out) + ".gxe.ref.json")
    ref = json.loads(ref_path.read_text())
    diag = pd.read_csv(ref_path.parent / ref["files"]["diagonal"], sep=r"\s+")
    stored_mass = diag[["ANNOT_0", "ANNOT_1"]].to_numpy(dtype=np.float64).sum(axis=0)
    np.testing.assert_allclose(
        np.asarray(ref["annotation_masses"], dtype=np.float64),
        stored_mass,
        rtol=0.0,
        atol=1e-11,
    )


def _mutated_bound_bundle(
    exact_bundle: ExactStandardizedBundle,
    tmp_path: Path,
    name: str,
    mutate,
) -> tuple[Path, Path]:
    original_ref = Path(str(exact_bundle.out) + ".gxe.ref.json")
    original_moments = Path(str(exact_bundle.out) + ".gxe.moments.json")
    ref = json.loads(original_ref.read_text())
    ref["files"] = {
        key: str((original_ref.parent / value).resolve())
        for key, value in ref["files"].items()
    }
    mutate(ref)
    ref_path = tmp_path / f"{name}.gxe.ref.json"
    ref_path.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")

    moments = json.loads(original_moments.read_text())
    moments["files"] = {
        key: str((original_moments.parent / value).resolve())
        for key, value in moments["files"].items()
    }
    moments["reference_manifest_sha256"] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    moments_path = tmp_path / f"{name}.gxe.moments.json"
    moments_path.write_text(json.dumps(moments, indent=2, sort_keys=True) + "\n")
    return ref_path, moments_path


def _fit_mutated_bundle(
    exact_bundle: ExactStandardizedBundle, ref_path: Path, moments_path: Path
):
    return fit_from_files(
        ref_path,
        moments_path,
        str(exact_bundle.out) + ".gxe.gwas.tsv.gz",
        str(exact_bundle.out) + ".gxe.gwis.tsv.gz",
        allow_ill_conditioned=True,
        max_condition=1e16,
    )


def test_manifest_rejects_non_boolean_null_correction(exact_bundle, tmp_path):
    ref, moments = _mutated_bound_bundle(
        exact_bundle,
        tmp_path,
        "bad-null-type",
        lambda payload: payload.__setitem__("null_corrected", "false"),
    )
    with pytest.raises(ValueError):
        _fit_mutated_bundle(exact_bundle, ref, moments)


def test_manifest_rejects_unknown_kernel_mode(exact_bundle, tmp_path):
    ref, moments = _mutated_bound_bundle(
        exact_bundle,
        tmp_path,
        "bad-kernel-mode",
        lambda payload: payload.__setitem__("kernel_mode", "not-a-kernel"),
    )
    with pytest.raises(ValueError):
        _fit_mutated_bundle(exact_bundle, ref, moments)


def test_manifest_rejects_annotation_mass_mismatch(exact_bundle, tmp_path):
    def alter_mass(payload):
        payload["annotation_masses"][0] += 0.25

    ref, moments = _mutated_bound_bundle(
        exact_bundle, tmp_path, "bad-annotation-mass", alter_mass
    )
    with pytest.raises(ValueError):
        _fit_mutated_bundle(exact_bundle, ref, moments)


def test_manifest_rejects_inconsistent_environment_standardization(exact_bundle, tmp_path):
    def alter_environment(payload):
        payload["environment_transform"]["analysis_sum_squares"] += 1.0

    ref, moments = _mutated_bound_bundle(
        exact_bundle, tmp_path, "bad-environment-transform", alter_environment
    )
    with pytest.raises(ValueError, match="environment sum of squares"):
        _fit_mutated_bundle(exact_bundle, ref, moments)


def test_manifest_rejects_incomplete_feature_diagnostics(exact_bundle, tmp_path):
    def alter_diagnostics(payload):
        payload["feature_diagnostics"]["valid_additive_columns"] -= 1

    ref, moments = _mutated_bound_bundle(
        exact_bundle, tmp_path, "bad-feature-diagnostics", alter_diagnostics
    )
    with pytest.raises(ValueError, match="do not cover every"):
        _fit_mutated_bundle(exact_bundle, ref, moments)


def test_fit_json_records_nxe_residual_alias_diagnostic(exact_bundle, tmp_path):
    fit, equations = fit_from_files(
        str(exact_bundle.out) + ".gxe.ref.json",
        str(exact_bundle.out) + ".gxe.moments.json",
        str(exact_bundle.out) + ".gxe.gwas.tsv.gz",
        str(exact_bundle.out) + ".gxe.gwis.tsv.gz",
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    _, json_path = write_fit(tmp_path / "fit-diagnostics", fit, equations)
    payload = json.loads(json_path.read_text())
    expected = equations.matrix[-2, -1] / math.sqrt(
        equations.matrix[-2, -2] * equations.matrix[-1, -1]
    )
    assert payload["nxe_residual_kernel_correlation"] == pytest.approx(expected)
    correlation = np.asarray(payload["kernel_correlation_matrix"], dtype=np.float64)
    np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0.0, atol=2e-15)


def _clone_fit_bundle_with_bound_cache(exact_bundle, root: Path):
    original_ref_path = Path(str(exact_bundle.out) + ".gxe.ref.json")
    original_moments_path = Path(str(exact_bundle.out) + ".gxe.moments.json")
    reference = json.loads(original_ref_path.read_text(encoding="utf-8"))
    moments = json.loads(original_moments_path.read_text(encoding="utf-8"))

    copied_inputs: list[Path] = []
    for key, value in reference["files"].items():
        source = (original_ref_path.parent / value).resolve()
        destination = root / source.name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        reference["files"][key] = destination.name
        copied_inputs.append(destination)
    score_paths = {}
    for key, value in moments["files"].items():
        source = (original_moments_path.parent / value).resolve()
        destination = root / source.name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        moments["files"][key] = destination.name
        score_paths[key] = destination
        copied_inputs.append(destination)

    cache_path = root / "bound.gxe.cache.npz"
    cache_builder = GenomewideEnvLDScore(
        bed_path=str(exact_bundle.inputs.prefix),
        env_path=str(exact_bundle.inputs.env_path),
        annot_path=str(exact_bundle.inputs.annot_path),
        out_path=str(root / "cache-builder"),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(exact_bundle.inputs.cov_path),
        pheno_path=None,
        num_vecs=exact_bundle.inputs.raw.shape[1],
        step_size=exact_bundle.inputs.raw.shape[1],
        seed=11,
        dtype="float64",
        kernel_mode="standardized",
        genotype_scale="hwe",
        target_xz_mem=0.01,
    )
    cache_builder.write_feature_cache(cache_path)
    cache_builder.close()
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    cache_binding = {"path": cache_path.name, "sha256": cache_hash}
    reference["feature_cache"] = dict(cache_binding)
    moments["feature_cache"] = dict(cache_binding)
    moments["feature_cache_sha256"] = cache_hash
    copied_inputs.append(cache_path)

    ref_path = root / "snapshot.gxe.ref.json"
    ref_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(ref_path, 0o600)
    moments["reference_manifest_sha256"] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    moments_path = root / "snapshot.gxe.moments.json"
    moments_path.write_text(
        json.dumps(moments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(moments_path, 0o600)
    return ref_path, moments_path, score_paths, copied_inputs


def test_fit_hashes_and_parses_every_input_from_private_readonly_snapshots(
    exact_bundle, tmp_path, monkeypatch
):
    expected_fit, expected_equations = fit_from_files(
        str(exact_bundle.out) + ".gxe.ref.json",
        str(exact_bundle.out) + ".gxe.moments.json",
        str(exact_bundle.out) + ".gxe.gwas.tsv.gz",
        str(exact_bundle.out) + ".gxe.gwis.tsv.gz",
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    ref_path, moments_path, scores, mutable_inputs = _clone_fit_bundle_with_bound_cache(
        exact_bundle, tmp_path
    )

    snapshot_paths: list[Path] = []

    def assert_private_snapshot(value) -> Path:
        path = Path(value)
        snapshot_paths.append(path)
        assert path.parent.name.startswith(".summit-gxe-fit-inputs-")
        assert (path.parent.stat().st_mode & 0o777) == 0o700
        assert (path.stat().st_mode & 0o777) == 0o400
        return path

    original_load_json = gxe_module._load_json
    json_mutated = False

    def mutate_json_sources_after_capture(path):
        nonlocal json_mutated
        assert_private_snapshot(path)
        if not json_mutated:
            ref_path.write_text("not json\n", encoding="utf-8")
            moments_path.write_text("not json\n", encoding="utf-8")
            json_mutated = True
        return original_load_json(path)

    original_np_load = gxe_module.np.load
    cache_mutated = False

    def mutate_cache_after_capture(path, *args, **kwargs):
        nonlocal cache_mutated
        snapshot = assert_private_snapshot(path)
        if snapshot.name.endswith(".cache.npz") and not cache_mutated:
            cache_source = next(value for value in mutable_inputs if value.name.endswith(".cache.npz"))
            cache_source.write_bytes(b"not an npz")
            cache_mutated = True
        return original_np_load(path, *args, **kwargs)

    original_read_csv = gxe_module.pd.read_csv
    tabular_inputs_mutated = False

    def mutate_tables_and_jackknife_after_capture(path, *args, **kwargs):
        nonlocal tabular_inputs_mutated
        assert_private_snapshot(path)
        if not tabular_inputs_mutated:
            for source in mutable_inputs:
                if not source.name.endswith(".cache.npz"):
                    source.write_bytes(b"mutated after snapshot")
            tabular_inputs_mutated = True
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(gxe_module, "_load_json", mutate_json_sources_after_capture)
    monkeypatch.setattr(gxe_module.np, "load", mutate_cache_after_capture)
    monkeypatch.setattr(gxe_module.pd, "read_csv", mutate_tables_and_jackknife_after_capture)

    observed_fit, observed_equations = fit_from_files(
        ref_path,
        moments_path,
        scores["gwas"],
        scores["gwis"],
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    assert json_mutated and cache_mutated and tabular_inputs_mutated
    assert snapshot_paths
    assert all(not path.exists() for path in snapshot_paths)
    np.testing.assert_allclose(
        observed_equations.matrix, expected_equations.matrix, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        observed_equations.rhs, expected_equations.rhs, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        observed_fit.proportions, expected_fit.proportions, rtol=0.0, atol=0.0
    )


def test_batch_fit_prepares_reference_once_and_matches_singletons(
    exact_bundle, tmp_path, monkeypatch
):
    reference, moments, scores, _ = _clone_fit_bundle_with_bound_cache(
        exact_bundle, tmp_path
    )
    first = GxEPhenotypeInput(
        phenotype_moments=moments,
        gwas_scores=scores["gwas"],
        gwis_scores=scores["gwis"],
    )
    copied_gwas = tmp_path / "Y2.gxe.gwas.tsv.gz"
    copied_gwis = tmp_path / "Y2.gxe.gwis.tsv.gz"
    shutil.copy2(first.gwas_scores, copied_gwas)
    shutil.copy2(first.gwis_scores, copied_gwis)
    copied_moments = json.loads(Path(first.phenotype_moments).read_text())
    copied_moments["phenotype"] = "Y2"
    copied_moments["files"] = {
        "gwas": copied_gwas.name,
        "gwis": copied_gwis.name,
    }
    copied_moments_path = tmp_path / "Y2.gxe.moments.json"
    copied_moments_path.write_text(
        json.dumps(copied_moments, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second = GxEPhenotypeInput(
        phenotype_moments=copied_moments_path,
        gwas_scores=copied_gwas,
        gwis_scores=copied_gwis,
    )

    expected_first = fit_from_files(
        reference,
        first.phenotype_moments,
        first.gwas_scores,
        first.gwis_scores,
        allow_ill_conditioned=True,
        max_condition=1e16,
    )
    expected_second = fit_from_files(
        reference,
        second.phenotype_moments,
        second.gwas_scores,
        second.gwis_scores,
        allow_ill_conditioned=True,
        max_condition=1e16,
    )

    aligned_calls = 0
    cache_loads = 0
    original_aligned_panel = gxe_module._aligned_panel
    original_cache_loader = gxe_module._load_feature_cache_bundle

    def counted_aligned_panel(*args, **kwargs):
        nonlocal aligned_calls
        aligned_calls += 1
        return original_aligned_panel(*args, **kwargs)

    def counted_cache_loader(*args, **kwargs):
        nonlocal cache_loads
        cache_loads += 1
        return original_cache_loader(*args, **kwargs)

    monkeypatch.setattr(gxe_module, "_aligned_panel", counted_aligned_panel)
    monkeypatch.setattr(gxe_module, "_load_feature_cache_bundle", counted_cache_loader)
    observed = fit_many_from_files(
        reference,
        {"Y1": first, "Y2": second},
        allow_ill_conditioned=True,
        max_condition=1e16,
        scratch_dir=tmp_path,
    )
    assert aligned_calls == 4
    assert cache_loads == 1
    provenance = {
        name: value[0].consumed_input_provenance
        for name, value in observed.items()
    }
    assert provenance["Y1"] is not None
    assert provenance["Y2"] is not None
    assert (
        provenance["Y1"].reference_manifest
        == provenance["Y2"].reference_manifest
    )
    assert provenance["Y1"].feature_cache == provenance["Y2"].feature_cache
    assert provenance["Y1"].phenotype_moments.path == str(Path(moments).resolve())
    assert provenance["Y2"].phenotype_moments.path == str(copied_moments_path.resolve())
    assert provenance["Y1"].gwas.path == str(Path(scores["gwas"]).resolve())
    assert provenance["Y2"].gwas.path == str(copied_gwas.resolve())
    assert provenance["Y1"].gwis.path == str(Path(scores["gwis"]).resolve())
    assert provenance["Y2"].gwis.path == str(copied_gwis.resolve())
    for name, expected in {"Y1": expected_first, "Y2": expected_second}.items():
        fit, equations = observed[name]
        expected_fit, expected_equations = expected
        np.testing.assert_allclose(
            equations.matrix, expected_equations.matrix, rtol=2e-14, atol=2e-12
        )
        np.testing.assert_allclose(
            equations.rhs, expected_equations.rhs, rtol=2e-14, atol=2e-12
        )
        np.testing.assert_allclose(
            fit.proportions, expected_fit.proportions, rtol=2e-12, atol=2e-12
        )

    copied_gwas.write_bytes(b"tampered score bytes")
    with pytest.raises(ValueError, match="GWAS score file SHA-256"):
        fit_many_from_files(
            reference,
            {"Y1": first, "Y2": second},
            allow_ill_conditioned=True,
            max_condition=1e16,
            scratch_dir=tmp_path,
        )


def test_fit_json_provenance_identifies_consumed_snapshot_across_aba_restore(
    exact_bundle, tmp_path, monkeypatch
):
    reference, moments, scores, _ = _clone_fit_bundle_with_bound_cache(
        exact_bundle, tmp_path
    )
    original_moments = Path(moments).read_bytes()
    mutated_payload = json.loads(original_moments)
    mutated_payload["aba_marker"] = "bytes consumed by core fit"
    mutated_moments = (
        json.dumps(mutated_payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    assert mutated_moments != original_moments
    Path(moments).write_bytes(mutated_moments)

    original_load_json = gxe_module._load_json
    restored = False

    def restore_live_moments_after_snapshot_is_parsed(path):
        nonlocal restored
        payload = original_load_json(path)
        if payload.get("aba_marker") == "bytes consumed by core fit":
            Path(moments).write_bytes(original_moments)
            restored = True
        return payload

    monkeypatch.setattr(
        gxe_module,
        "_load_json",
        restore_live_moments_after_snapshot_is_parsed,
    )
    fit, equations = fit_from_files(
        reference,
        moments,
        scores["gwas"],
        scores["gwis"],
        allow_ill_conditioned=True,
        max_condition=1e16,
        scratch_dir=tmp_path,
    )
    assert restored
    assert Path(moments).read_bytes() == original_moments
    _, fit_json = write_fit(tmp_path / "aba-fit", fit, equations)
    observed = json.loads(fit_json.read_text(encoding="utf-8"))[
        "consumed_input_provenance"
    ]
    assert set(observed) == {
        "reference_manifest",
        "feature_cache",
        "phenotype_moments",
        "gwas",
        "gwis",
    }

    reference_payload = json.loads(Path(reference).read_text(encoding="utf-8"))
    cache_path = (
        Path(reference).parent / reference_payload["feature_cache"]["path"]
    ).resolve()

    def live_record(path):
        canonical = Path(path).resolve()
        content = canonical.read_bytes()
        return {
            "path": str(canonical),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    assert observed == {
        "reference_manifest": live_record(reference),
        "feature_cache": live_record(cache_path),
        "phenotype_moments": {
            "path": str(Path(moments).resolve()),
            "bytes": len(mutated_moments),
            "sha256": hashlib.sha256(mutated_moments).hexdigest(),
        },
        "gwas": live_record(scores["gwas"]),
        "gwis": live_record(scores["gwis"]),
    }
    assert observed["phenotype_moments"]["sha256"] != hashlib.sha256(
        original_moments
    ).hexdigest()


def test_schema_v2_null_corrected_jackknife_batch_matches_singletons_and_dense(
    exact_bundle, tmp_path
):
    """Exercise the complete compatibility path, including every delete block."""
    out = tmp_path / "v2-null-jackknife"
    estimator = _new_estimator(
        exact_bundle.inputs,
        out,
        write_jackknife=True,
    )
    _install_exact_probes(estimator)
    estimator._compute_ldscore()
    estimator.close()

    ref_path = Path(str(out) + ".gxe.ref.json")
    moments1_path = Path(str(out) + ".gxe.moments.json")
    gwas1_path = Path(str(out) + ".gxe.gwas.tsv.gz")
    gwis1_path = Path(str(out) + ".gxe.gwis.tsv.gz")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    moments1 = json.loads(moments1_path.read_text(encoding="utf-8"))
    names = tuple(reference["annotation_names"])
    masses = np.asarray(reference["annotation_masses"], dtype=np.float64)
    residual_rank = int(reference["residual_rank"])
    assert len(names) == 2
    assert "jackknife" in reference["files"]

    # Schema v2 stored randomized panels after subtracting the analytic null
    # floor.  Re-encode the exact schema-v3 panels in that supported legacy
    # representation, leaving the raw within-block intersections unchanged.
    for key in ("xx", "xw", "wx", "ww"):
        panel_path = ref_path.parent / reference["files"][key]
        panel = pd.read_csv(panel_path, sep=r"\s+")
        panel.loc[:, list(names)] = (
            panel.loc[:, list(names)].to_numpy(dtype=np.float64)
            - masses[None, :] / residual_rank
        )
        panel.to_csv(
            panel_path,
            sep="\t",
            index=False,
            compression="gzip",
            float_format="%.17g",
        )
        reference["artifact_sha256"][key] = hashlib.sha256(
            panel_path.read_bytes()
        ).hexdigest()
    reference["schema_version"] = 2
    reference["null_corrected"] = True
    ref_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference_hash = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    moments1["schema_version"] = 2
    moments1["reference_manifest_sha256"] = reference_hash
    moments1_path.write_text(
        json.dumps(moments1, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pmat = exact_bundle.projector
    h_nxe = pmat @ np.diag(estimator.env**2) @ pmat
    raw_y2 = np.sin(0.73 * np.arange(estimator.nsamp)) + np.linspace(
        -0.4, 0.6, estimator.nsamp
    )
    y2 = pmat @ raw_y2
    y2 *= np.sqrt(residual_rank / float(y2 @ y2))
    gwas2_path = tmp_path / "Y2.gxe.gwas.tsv.gz"
    gwis2_path = tmp_path / "Y2.gxe.gwis.tsv.gz"
    gwas2 = pd.read_csv(gwas1_path, sep=r"\s+")
    gwis2 = pd.read_csv(gwis1_path, sep=r"\s+")
    gwas2["SCORE"] = exact_bundle.x.T @ y2 / np.sqrt(residual_rank)
    gwis2["SCORE"] = exact_bundle.w.T @ y2 / np.sqrt(residual_rank)
    gwas2.to_csv(
        gwas2_path,
        sep="\t",
        index=False,
        compression="gzip",
        float_format="%.17g",
    )
    gwis2.to_csv(
        gwis2_path,
        sep="\t",
        index=False,
        compression="gzip",
        float_format="%.17g",
    )
    moments2 = dict(moments1)
    moments2["phenotype"] = "Y2"
    moments2["files"] = {"gwas": gwas2_path.name, "gwis": gwis2_path.name}
    moments2["score_sha256"] = {
        "gwas": hashlib.sha256(gwas2_path.read_bytes()).hexdigest(),
        "gwis": hashlib.sha256(gwis2_path.read_bytes()).hexdigest(),
    }
    moments2["q_nxe"] = float(y2 @ h_nxe @ y2)
    moments2["q_residual"] = residual_rank
    moments2.pop("phenotype_residual_variance_fraction", None)
    moments2_path = tmp_path / "Y2.gxe.moments.json"
    moments2_path.write_text(
        json.dumps(moments2, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    phenotype_inputs = {
        "Y1": GxEPhenotypeInput(moments1_path, gwas1_path, gwis1_path),
        "Y2": GxEPhenotypeInput(moments2_path, gwas2_path, gwis2_path),
    }
    singleton = {
        name: fit_from_files(
            ref_path,
            item.phenotype_moments,
            item.gwas_scores,
            item.gwis_scores,
            allow_ill_conditioned=True,
            max_condition=1e16,
            scratch_dir=tmp_path,
        )
        for name, item in phenotype_inputs.items()
    }
    batch = fit_many_from_files(
        ref_path,
        phenotype_inputs,
        allow_ill_conditioned=True,
        max_condition=1e16,
        scratch_dir=tmp_path,
    )
    for name in phenotype_inputs:
        observed_fit, observed_equations = batch[name]
        expected_fit, expected_equations = singleton[name]
        assert observed_fit.jackknife_block_labels == expected_fit.jackknife_block_labels
        assert observed_fit.rank == expected_fit.rank
        for observed, expected in (
            (observed_equations.matrix, expected_equations.matrix),
            (observed_equations.rhs, expected_equations.rhs),
            (observed_equations.traces, expected_equations.traces),
            (observed_fit.coefficients, expected_fit.coefficients),
            (observed_fit.contributions, expected_fit.contributions),
            (observed_fit.proportions, expected_fit.proportions),
            (observed_fit.jackknife_estimates, expected_fit.jackknife_estimates),
            (observed_fit.standard_errors, expected_fit.standard_errors),
        ):
            np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=2e-11)

    diagonal = pd.read_csv(
        ref_path.parent / reference["files"]["diagonal"], sep=r"\s+"
    )
    annotations = diagonal.loc[
        :, [f"ANNOT_{index}" for index in range(len(names))]
    ].to_numpy(dtype=np.float64)
    blocks = diagonal["BLOCK"].to_numpy(dtype=np.int64)
    component_names = tuple(f"G:{name}" for name in names) + tuple(
        f"GxE:{name}" for name in names
    ) + ("NxE", "residual")

    def dense_equations(y, keep):
        kept_annotations = annotations[keep]
        kept_masses = kept_annotations.sum(axis=0)
        kernels = [
            (exact_bundle.x[:, keep] * kept_annotations[:, index])
            @ exact_bundle.x[:, keep].T
            / kept_masses[index]
            for index in range(len(names))
        ] + [
            (exact_bundle.w[:, keep] * kept_annotations[:, index])
            @ exact_bundle.w[:, keep].T
            / kept_masses[index]
            for index in range(len(names))
        ] + [h_nxe, pmat]
        matrix = np.asarray(
            [[np.trace(left @ right) for right in kernels] for left in kernels]
        )
        rhs = np.asarray([y @ kernel @ y for kernel in kernels])
        traces = np.asarray([np.trace(kernel) for kernel in kernels])
        return GxENormalEquations(matrix, rhs, traces, component_names)

    for name, y in {"Y1": exact_bundle.y, "Y2": y2}.items():
        observed_fit, observed_equations = batch[name]
        full = dense_equations(y, np.ones(len(blocks), dtype=bool))
        explicit_fit = solve_normal_equations(
            full, allow_ill_conditioned=True, max_condition=1e16
        )
        np.testing.assert_allclose(
            observed_equations.matrix, full.matrix, rtol=4e-9, atol=4e-9
        )
        np.testing.assert_allclose(
            observed_equations.rhs, full.rhs, rtol=4e-9, atol=4e-9
        )
        for observed, expected in (
            (observed_fit.coefficients, explicit_fit.coefficients),
            (observed_fit.contributions, explicit_fit.contributions),
            (observed_fit.proportions, explicit_fit.proportions),
        ):
            np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-8)

        explicit_replicates = np.asarray(
            [
                solve_normal_equations(
                    dense_equations(y, blocks != block_id),
                    allow_ill_conditioned=True,
                    max_condition=1e16,
                ).proportions
                for block_id in range(len(observed_fit.jackknife_block_labels))
            ]
        )
        center = explicit_replicates.mean(axis=0)
        nblock = len(explicit_replicates)
        explicit_se = np.sqrt(
            (nblock - 1.0)
            / nblock
            * np.sum((explicit_replicates - center) ** 2, axis=0)
        )
        np.testing.assert_allclose(
            observed_fit.jackknife_estimates,
            explicit_replicates,
            rtol=3e-8,
            atol=3e-8,
        )
        np.testing.assert_allclose(
            observed_fit.standard_errors, explicit_se, rtol=3e-8, atol=3e-8
        )
