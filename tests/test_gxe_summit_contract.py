from __future__ import annotations

import inspect
import json
import math
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
    fit_from_files,
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
    kernel_mode: str | None = "standardized_projected",
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
    assert default == "standardized_projected"


def test_cli_default_is_summit_post_projection_standardized():
    from summit.cli import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.gxe_kernel_mode == "standardized_projected"
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
        njack=3,
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
        njack=3,
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
        njack=3,
    )
    _, json_path = write_fit(tmp_path / "fit-diagnostics", fit, equations)
    payload = json.loads(json_path.read_text())
    expected = equations.matrix[-2, -1] / math.sqrt(
        equations.matrix[-2, -2] * equations.matrix[-1, -1]
    )
    assert payload["nxe_residual_kernel_correlation"] == pytest.approx(expected)
    correlation = np.asarray(payload["kernel_correlation_matrix"], dtype=np.float64)
    np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0.0, atol=2e-15)
