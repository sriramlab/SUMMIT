"""Common-cohort multi-environment GxE reference construction.

The statistical model remains one independent G+GxE+NxE+residual system per
environment.  This module shares each standardized genotype block across those
systems; it never writes feature matrices or randomized sketches to disk.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .gwe_ldscore import (
    _BACKEND_PROVENANCE_SCHEMA_VERSION,
    GenomewideEnvLDScore,
    _build_balanced_vtiles,
    _is_canonical_sha256,
    _loaded_native_binary_record,
    _sha256_descriptor,
    _validate_backend_provenance,
    _validate_jackknife_probe_count,
    _validate_native_blas_runtime,
)


_SCORE_NAMES = ("xx", "xw", "wx", "ww")


def safe_environment_suffix(name: str) -> str:
    """Return a stable filename component for an environment column name."""
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    if not suffix:
        raise ValueError(f"Environment name {name!r} has no safe filename characters.")
    return suffix


class _MultiEnvironmentGemm:
    """Matrix executor for the shared stream, with exact native provenance."""

    def __init__(
        self, requested_backend: str, estimator: GenomewideEnvLDScore
    ) -> None:
        backend = str(requested_backend).strip().lower()
        if backend not in {"python", "direct"}:
            raise ValueError("requested_backend must be 'python' or 'direct'.")
        self.protected = backend == "direct"
        self.threads = int(estimator.num_threads)
        self.compute_dtype = np.dtype(
            np.float64 if self.protected else estimator.dtype
        )
        self.repaired_output_columns = 0
        self._module = None
        self._descriptor: int | None = None
        self._binary_record: dict | None = None
        self._template: dict | None = None
        if not self.protected:
            return

        try:
            from .. import gxeldcore
        except Exception as exc:
            raise RuntimeError(
                "The requested protected multi-environment GxE extension is unavailable."
            ) from exc
        for function in ("protected_matmul_nn", "protected_matmul_tn"):
            if not callable(getattr(gxeldcore, function, None)):
                raise RuntimeError(
                    "The loaded GxE extension predates protected shared GEMMs; "
                    "rebuild/install SUMMIT from the current source before running "
                    "--gxe-env-cols with --gxe-native-backend direct."
                )

        descriptor = None
        try:
            descriptor, binary_record = _loaded_native_binary_record(gxeldcore)
            build_info = dict(gxeldcore.build_info())
            runtime_record = _validate_native_blas_runtime(build_info)
            source_commit = build_info.get("source_commit")
            source_tree_sha256 = build_info.get("source_tree_sha256")
            if not (
                isinstance(source_commit, str)
                and len(source_commit) == 40
                and all(character in "0123456789abcdef" for character in source_commit)
                and _is_canonical_sha256(source_tree_sha256)
            ):
                raise RuntimeError(
                    "The protected shared GxE extension lacks exact source provenance."
                )
            compile_options = {
                key: build_info.get(key)
                for key in (
                    "api_version", "compiler_id", "compiler_version", "build_type",
                    "blas_vendor", "cxx_standard", "optimization",
                    "architecture_tuning", "openmp_enabled",
                    "native_optimization_enabled", "platform",
                    "blas_runtime_config",
                )
            }
            compile_options["execution_mode"] = (
                "shared_multi_environment_protected_gemm"
            )
            compile_options["loaded_blas_runtime"] = dict(runtime_record)
            template = {
                "schema_version": _BACKEND_PROVENANCE_SCHEMA_VERSION,
                "artifact_stage": "feature_construction",
                "backend_name": str(
                    build_info.get("backend_name", "gxeldcore_direct")
                ),
                "backend_version": str(build_info.get("backend_version", "unknown")),
                "source_commit": source_commit,
                "source_tree_sha256": source_tree_sha256,
                "native_binary_sha256": binary_record["sha256"],
                "compile_options": compile_options,
                "native_workspace_cap_bytes": int(
                    estimator.native_workspace_gib * 1024**3
                ),
                "configured_target_panel_columns": int(
                    estimator.native_target_panel_columns
                ),
                "actual_global_2b_source_columns": 0,
                "actual_jackknife_2b_source_columns": 0,
                "actual_target_source_columns": 0,
            }
            _validate_backend_provenance(
                template, expected_stage="feature_construction"
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            raise
        self._module = gxeldcore
        self._descriptor = descriptor
        self._binary_record = binary_record
        self._template = template

    def __enter__(self) -> "_MultiEnvironmentGemm":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def provenance_template(self) -> dict | None:
        if not self.protected:
            return None
        if (
            self._descriptor is None
            or self._binary_record is None
            or self._template is None
        ):
            raise RuntimeError("The protected shared GxE executor is closed.")
        observed = os.fstat(self._descriptor)
        identity = (
            observed.st_dev, observed.st_ino, observed.st_size,
            observed.st_mtime_ns, observed.st_ctime_ns,
        )
        if identity != tuple(self._binary_record["identity"]):
            raise RuntimeError(
                "The loaded protected GxE extension inode changed during execution."
            )
        if _sha256_descriptor(self._descriptor) != self._binary_record["sha256"]:
            raise RuntimeError(
                "The loaded protected GxE extension bytes changed during execution."
            )
        return dict(self._template)

    def nn(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if not self.protected:
            return left @ right
        assert self._module is not None
        result, repaired = self._module.protected_matmul_nn(
            np.asfortranarray(left, dtype=np.float64),
            np.asfortranarray(right, dtype=np.float64),
            self.threads,
        )
        self.repaired_output_columns += int(repaired)
        return np.asarray(result)

    def tn(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if not self.protected:
            return left.T @ right
        assert self._module is not None
        result, repaired = self._module.protected_matmul_tn(
            np.asfortranarray(left, dtype=np.float64),
            np.asfortranarray(right, dtype=np.float64),
            self.threads,
        )
        self.repaired_output_columns += int(repaired)
        return np.asarray(result)


def _prepare_feature_block(
    estimator: GenomewideEnvLDScore,
    genotype: np.ndarray,
    start: int,
    stop: int,
    *,
    interaction: bool,
    apply_scale: bool,
    executor: _MultiEnvironmentGemm,
) -> np.ndarray:
    if not executor.protected:
        method = (
            estimator._prepare_interaction_block
            if interaction else estimator._prepare_additive_block
        )
        return method(
            start, stop, G=genotype, apply_scale=apply_scale,
            out_dtype=executor.compute_dtype,
        )

    if interaction:
        feature = np.asarray(
            genotype * estimator.env[:, None], dtype=np.float64, order="F"
        )
    else:
        feature = np.array(genotype, copy=True, dtype=np.float64, order="F")
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, feature)
        feature -= executor.nn(estimator.C_int, coefficients)
    feature -= feature.mean(axis=0, keepdims=True)
    if apply_scale and estimator.kernel_mode == "standardized":
        scales = (
            estimator.inv_sqrt_resvar_w_all
            if interaction else estimator.inv_sqrt_resvar_x_all
        )
        if scales is None:
            label = "Interaction" if interaction else "Additive"
            raise RuntimeError(f"{label} residual variances have not been precomputed.")
        feature *= scales[start:stop].reshape(1, -1)
    return np.asarray(feature, dtype=executor.compute_dtype, order="F")


def _new_feature_state(estimator: GenomewideEnvLDScore) -> dict:
    return {
        name: np.zeros(estimator.nsnps, dtype=np.float64)
        for name in (
            "inv_x", "inv_w", "norm_x", "norm_w", "diag_x", "diag_w",
            "corr_xw",
        )
    } | {"max_leak_x": 0.0, "max_leak_w": 0.0}


def _accumulate_feature_block(
    estimator: GenomewideEnvLDScore,
    state: dict,
    genotype: np.ndarray,
    start: int,
    stop: int,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Consume one shared standardized genotype block for one environment."""
    x = _prepare_feature_block(
        estimator, genotype, start, stop,
        interaction=False, apply_scale=False, executor=executor,
    )
    w = _prepare_feature_block(
        estimator, genotype, start, stop,
        interaction=True, apply_scale=False, executor=executor,
    )
    ssx = np.sum(x * x, axis=0, dtype=np.float64)
    ssw = np.sum(w * w, axis=0, dtype=np.float64)
    varx = ssx / float(estimator.df_corr)
    varw = ssw / float(estimator.df_corr)
    good_x = np.isfinite(varx) & (varx > estimator.eps_var)
    good_w = np.isfinite(varw) & (varw > estimator.eps_var)
    invalid = ~(good_x & good_w)
    if np.any(invalid):
        examples = []
        for offset in np.flatnonzero(invalid)[:5]:
            index = start + int(offset)
            snp = (
                str(estimator.snplist.iloc[index]["SNP"])
                if estimator.snplist is not None else str(index)
            )
            failed = []
            if not good_x[offset]:
                failed.append("additive")
            if not good_w[offset]:
                failed.append("interaction")
            examples.append(f"{snp} ({'+'.join(failed)})")
        raise ValueError(
            f"{int(invalid.sum())} variant feature(s) in block [{start}:{stop}) "
            f"have zero or invalid projected variance for environment "
            f"{estimator.env_name!r}. QC/remove these variants and regenerate the "
            f"complete batch. Examples: {', '.join(examples)}."
        )

    if estimator.kernel_mode == "standardized":
        state["inv_x"][start:stop] = 1.0 / np.sqrt(varx)
        state["inv_w"][start:stop] = 1.0 / np.sqrt(varw)
    else:
        state["inv_x"][start:stop] = 1.0
        state["inv_w"][start:stop] = 1.0
    x *= state["inv_x"][start:stop].reshape(1, -1)
    w *= state["inv_w"][start:stop].reshape(1, -1)

    final_ssx = np.sum(x * x, axis=0, dtype=np.float64)
    final_ssw = np.sum(w * w, axis=0, dtype=np.float64)
    state["norm_x"][start:stop] = final_ssx / float(estimator.df_corr)
    state["norm_w"][start:stop] = final_ssw / float(estimator.df_corr)

    root_n = math.sqrt(float(estimator.nsamp))
    leaked_x_sq = (x.sum(axis=0, dtype=np.float64) / root_n) ** 2
    leaked_w_sq = (w.sum(axis=0, dtype=np.float64) / root_n) ** 2
    if estimator.p_eff > 0:
        projected_x = executor.tn(estimator.C_int, x)
        projected_w = executor.tn(estimator.C_int, w)
        leaked_x_sq += np.sum(projected_x * projected_x, axis=0, dtype=np.float64)
        leaked_w_sq += np.sum(projected_w * projected_w, axis=0, dtype=np.float64)
    state["max_leak_x"] = max(
        float(state["max_leak_x"]),
        float(np.max(np.sqrt(leaked_x_sq / final_ssx))),
    )
    state["max_leak_w"] = max(
        float(state["max_leak_w"]),
        float(np.max(np.sqrt(leaked_w_sq / final_ssw))),
    )

    ex = estimator.env[:, None] * x
    ew = estimator.env[:, None] * w
    state["diag_x"][start:stop] = (
        np.sum(ex * ex, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )
    state["diag_w"][start:stop] = (
        np.sum(ew * ew, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )
    state["corr_xw"][start:stop] = (
        np.sum(x * w, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )


def _finish_feature_state(estimator: GenomewideEnvLDScore, state: dict) -> None:
    estimator.inv_sqrt_resvar_x_all = state["inv_x"]
    estimator.inv_sqrt_resvar_w_all = state["inv_w"]
    estimator.norm_x_all = state["norm_x"]
    estimator.norm_w_all = state["norm_w"]
    estimator.diag_nxe_x_all = state["diag_x"]
    estimator.diag_nxe_w_all = state["diag_w"]
    estimator.corr_xw_all = state["corr_xw"]
    estimator.score_x_all = None
    estimator.score_w_all = None

    annotations = np.asarray(estimator.annot, dtype=np.float64)
    trace_x = (
        float(estimator.df_corr)
        * (annotations.T @ estimator.norm_x_all)
        / estimator.nsnps_bin
    )
    trace_w = (
        float(estimator.df_corr)
        * (annotations.T @ estimator.norm_w_all)
        / estimator.nsnps_bin
    )
    max_norm_x = float(np.max(np.abs(estimator.norm_x_all - 1.0)))
    max_norm_w = float(np.max(np.abs(estimator.norm_w_all - 1.0)))
    max_leak_x = float(state["max_leak_x"])
    max_leak_w = float(state["max_leak_w"])
    if max(max_leak_x, max_leak_w) > 1.0e-9:
        raise RuntimeError(
            f"Projected features for environment {estimator.env_name!r} leak into "
            f"the fixed-effect span: X={max_leak_x:.6g}, W={max_leak_w:.6g}."
        )
    if estimator.kernel_mode == "standardized" and max(max_norm_x, max_norm_w) > 1.0e-9:
        raise RuntimeError(
            f"Post-projection normalization failed for environment "
            f"{estimator.env_name!r}: X={max_norm_x:.6g}, W={max_norm_w:.6g}."
        )
    estimator.feature_diagnostics = {
        "valid_additive_columns": int(estimator.nsnps),
        "valid_interaction_columns": int(estimator.nsnps),
        "max_projection_leakage_additive": max_leak_x,
        "max_projection_leakage_interaction": max_leak_w,
        "min_norm_additive_over_rank": float(np.min(estimator.norm_x_all)),
        "max_norm_additive_over_rank": float(np.max(estimator.norm_x_all)),
        "min_norm_interaction_over_rank": float(np.min(estimator.norm_w_all)),
        "max_norm_interaction_over_rank": float(np.max(estimator.norm_w_all)),
        "max_norm_error_additive": max_norm_x,
        "max_norm_error_interaction": max_norm_w,
        "kernel_traces_additive": trace_x.tolist(),
        "kernel_traces_interaction": trace_w.tolist(),
        "max_trace_error_additive": float(
            np.max(np.abs(trace_x - estimator.df_corr))
        ),
        "max_trace_error_interaction": float(
            np.max(np.abs(trace_w - estimator.df_corr))
        ),
    }
    estimator.log._log(
        f"[gxe:multi:invariants:{estimator.env_name}] max fixed-effect leakage "
        f"X={max_leak_x:.3e}, W={max_leak_w:.3e}; max norm error "
        f"X={max_norm_x:.3e}, W={max_norm_w:.3e}."
    )


def _require_common_contract(estimators: Sequence[GenomewideEnvLDScore]) -> None:
    if len(estimators) < 2:
        raise ValueError("Multi-environment construction requires at least two environments.")
    first = estimators[0]
    if first.genotype_format != "bed":
        raise ValueError("Shared multi-environment construction currently requires BED input.")
    scalar_fields = (
        "nsamp_total", "nsamp", "nsnps", "nbins", "nvecs", "step_size",
        "root_seed", "probe_offset", "rand_dist", "ddof", "kernel_mode",
        "genotype_scale", "write_jackknife", "allow_low_probe_jackknife",
        "eps_var", "dtype",
    )
    for estimator in estimators:
        if estimator.native_backend != "python":
            raise ValueError(
                "Multi-environment construction owns one shared decoded genotype "
                "stream and therefore requires Python-orchestrated BLAS estimators."
            )
        if estimator.pheno is not None or estimator.feature_cache_path is not None:
            raise ValueError(
                "Multi-environment reference construction is phenotype-free and "
                "does not consume feature caches."
            )
        if estimator.shard_mode:
            raise ValueError("Multi-environment reference construction does not use shards.")
        if not np.array_equal(first.row_sel, estimator.row_sel):
            raise ValueError(
                "Selected environments do not retain exactly the same complete-case "
                "cohort. Intersect the input rows explicitly or create separate batches."
            )
        if estimator._construction_genotype_state != first._construction_genotype_state:
            raise ValueError(
                "Multi-environment estimators must refer to the same opened genotype files."
            )
        if not np.array_equal(
            first.sample_ids[["FID", "IID"]].to_numpy(),
            estimator.sample_ids[["FID", "IID"]].to_numpy(),
        ):
            raise ValueError("Multi-environment estimators disagree on genotype sample IDs.")
        for field in scalar_fields:
            if getattr(estimator, field) != getattr(first, field):
                raise ValueError(f"Multi-environment estimators disagree on {field}.")
        for name in ("annot", "nsnps_bin", "jackknife_ids"):
            left = getattr(first, name)
            right = getattr(estimator, name)
            if left is None or right is None:
                if left is not right:
                    raise ValueError(f"Multi-environment estimators disagree on {name}.")
            elif not np.array_equal(left, right):
                raise ValueError(f"Multi-environment estimators disagree on {name}.")
        if estimator.l2cols != first.l2cols:
            raise ValueError("Multi-environment estimators disagree on annotation names.")
        if estimator.jackknife_labels != first.jackknife_labels:
            raise ValueError("Multi-environment estimators disagree on jackknife labels.")
    names = [estimator.env_name for estimator in estimators]
    if len(set(names)) != len(names):
        raise ValueError(f"Multi-environment column names must be unique: {names}.")
    suffixes = [safe_environment_suffix(name) for name in names]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(
            f"Environment names collide after filename normalization: {names}."
        )


def _shared_vtiles(
    estimators: Sequence[GenomewideEnvLDScore],
    compute_dtype: np.dtype,
) -> list[tuple[int, int]]:
    first = estimators[0]
    budget = min(float(estimator.target_xz_mem) for estimator in estimators)
    # Jackknife deletion is performed later from completed per-variant LD
    # scores, exactly as in the additive path.  No environment retains a
    # within-block sketch during reference construction.
    multiplier = 2
    bytes_per_probe = (
        multiplier
        * len(estimators)
        * first.nsamp
        * first.nbins
        * np.dtype(compute_dtype).itemsize
    )
    vmax = int((budget * 1024**3) // max(1, bytes_per_probe))
    if vmax < 1:
        raise RuntimeError(
            "The multi-environment sketch budget cannot hold one probe across all "
            f"{len(estimators)} environments; required={bytes_per_probe} bytes, "
            f"limit={int(budget * 1024**3)} bytes."
        )
    return _build_balanced_vtiles(first.nvecs, vmax)


def _accumulate_sketch_block(
    estimator: GenomewideEnvLDScore,
    target: np.ndarray,
    feature: np.ndarray,
    probes: np.ndarray,
    annotation: np.ndarray,
    executor: _MultiEnvironmentGemm,
) -> None:
    if not executor.protected:
        estimator._accumulate_sketch_block(target, feature, probes, annotation)
        return
    sqrt_annotation = np.sqrt(np.maximum(annotation, 0))
    for index in range(estimator.nbins):
        weights = sqrt_annotation[:, index]
        if not np.any(weights):
            continue
        segment = slice(
            index * probes.shape[1], (index + 1) * probes.shape[1]
        )
        weighted_probes = np.asfortranarray(
            weights.reshape(-1, 1) * probes, dtype=np.float64
        )
        target[:, segment] += executor.nn(feature, weighted_probes)


def _accumulate_unprojected_source_block(
    estimator: GenomewideEnvLDScore,
    target: np.ndarray,
    genotype: np.ndarray,
    probes: np.ndarray,
    annotation: np.ndarray,
    start: int,
    stop: int,
    *,
    interaction: bool,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Accumulate a raw source; one panel projection follows the full pass."""
    scales = (
        estimator.inv_sqrt_resvar_w_all
        if interaction else estimator.inv_sqrt_resvar_x_all
    )
    if scales is None:
        raise RuntimeError("GxE source scales have not been precomputed.")
    sqrt_annotation = np.sqrt(np.maximum(annotation, 0))
    block_scales = scales[start:stop]
    for index in range(estimator.nbins):
        weights = sqrt_annotation[:, index]
        if not np.any(weights):
            continue
        segment = slice(
            index * probes.shape[1], (index + 1) * probes.shape[1]
        )
        weighted_probes = np.asfortranarray(
            (weights * block_scales).reshape(-1, 1) * probes,
            dtype=np.float64,
        )
        contribution = executor.nn(genotype, weighted_probes)
        if interaction:
            contribution *= estimator.env[:, None]
        target[:, segment] += contribution


def _project_source_panel_inplace(
    estimator: GenomewideEnvLDScore,
    panel: np.ndarray,
    executor: _MultiEnvironmentGemm,
) -> float:
    """Project one completed source panel and return relative leakage."""
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, panel)
        panel -= executor.nn(estimator.C_int, coefficients)
    panel -= panel.mean(axis=0, keepdims=True)

    denominator = np.sum(panel * panel, axis=0, dtype=np.float64)
    root_n = math.sqrt(float(estimator.nsamp))
    leaked = (panel.sum(axis=0, dtype=np.float64) / root_n) ** 2
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, panel)
        leaked += np.sum(coefficients * coefficients, axis=0, dtype=np.float64)
    relative = np.sqrt(
        np.divide(
            leaked,
            denominator,
            out=np.full_like(leaked, np.inf),
            where=denominator > 0.0,
        )
    )
    maximum = float(np.max(relative))
    if not np.isfinite(maximum) or maximum > 1.0e-9:
        raise RuntimeError(
            f"Projected source panel for environment {estimator.env_name!r} "
            f"has excessive fixed-effect leakage: {maximum:.6g}."
        )
    return maximum


def _publish_json_no_replace(payload: dict, target: Path) -> tuple[int, int]:
    """Atomically seal a new batch manifest without replacing another writer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        staged = temporary_path.stat(follow_symlinks=False)
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing concurrently created multi-environment manifest: {target}."
            ) from exc
        return staged.st_dev, staged.st_ino
    finally:
        temporary_path.unlink(missing_ok=True)


def _published_file_identity(path: Path) -> tuple[Path, int, int, str]:
    observed = path.stat(follow_symlinks=False)
    return path, observed.st_dev, observed.st_ino, GenomewideEnvLDScore._file_sha256(
        str(path)
    )


def _rollback_published_files(
    published: Sequence[tuple[Path, int, int, str]],
) -> None:
    """Remove only unchanged files published by this failed batch."""
    for path, device, inode, expected_hash in reversed(published):
        try:
            observed = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if observed.st_dev != device or observed.st_ino != inode:
            continue
        if GenomewideEnvLDScore._file_sha256(str(path)) == expected_hash:
            path.unlink(missing_ok=True)


def generate_multi_environment_references(
    estimators: Sequence[GenomewideEnvLDScore],
    *,
    batch_manifest: str | Path,
    requested_backend: str = "python",
) -> Path:
    """Generate independent references while sharing every genotype block read."""
    estimators = tuple(estimators)
    _require_common_contract(estimators)
    with _MultiEnvironmentGemm(requested_backend, estimators[0]) as executor:
        return _generate_multi_environment_references(
            estimators,
            batch_manifest=batch_manifest,
            requested_backend=requested_backend,
            executor=executor,
        )


def _generate_multi_environment_references(
    estimators: Sequence[GenomewideEnvLDScore],
    *,
    batch_manifest: str | Path,
    requested_backend: str,
    executor: _MultiEnvironmentGemm,
) -> Path:
    first = estimators[0]
    shared_provenance = executor.provenance_template()
    for estimator in estimators:
        estimator.shared_backend_provenance = (
            None if shared_provenance is None else dict(shared_provenance)
        )
    target = Path(batch_manifest).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing existing multi-environment manifest: {target}.")
    for estimator in estimators:
        estimator._assert_output_paths_available()
        _validate_jackknife_probe_count(
            estimator.nvecs,
            estimator.write_jackknife,
            estimator.allow_low_probe_jackknife,
        )

    first._assert_construction_genotype_state()
    initial_provenance = first._genotype_provenance()
    blocks = first._make_compute_blocks()
    vtiles = _shared_vtiles(estimators, executor.compute_dtype)
    for estimator in estimators:
        estimator._vtiles_used = list(vtiles)

    first.log._log(
        "[gxe:multi] Sharing each standardized genotype block across "
        f"{len(estimators)} independent environments: "
        f"{[estimator.env_name for estimator in estimators]}."
    )
    if executor.protected:
        first.log._log(
            "[gxe:multi] Shared decoded blocks use double-precision native "
            "partitioned GEMMs with checksum detection and deterministic repair."
        )

    feature_states = [_new_feature_state(estimator) for estimator in estimators]
    for start, stop in blocks:
        genotype = first._read_genotype_block(start, stop)
        for estimator, state in zip(estimators, feature_states, strict=True):
            _accumulate_feature_block(
                estimator, state, genotype, start, stop, executor
            )
        del genotype
    for estimator, state in zip(estimators, feature_states, strict=True):
        _finish_feature_state(estimator, state)
        estimator.feature_backend_provenance = estimator._backend_provenance(
            "feature_construction"
        )
    del feature_states
    gc.collect()

    max_vt = max(size for _, size in vtiles)
    itemsize = executor.compute_dtype.itemsize
    # The protected algebraic target retains E*S beside each projected 2B
    # source panel.  This trades one additional bounded panel per environment
    # for eliminating its repeated construction in every variant block.
    resident_multiplier = 4 if executor.protected else 2
    total_resident = (
        resident_multiplier
        * len(estimators)
        * first.nsamp
        * first.nbins
        * max_vt
        * itemsize
    )
    max_block = min(first.step_size, first.nsnps)
    passes_per_tile = 2
    passes = 1 + len(vtiles) * passes_per_tile
    for estimator in estimators:
        estimator.resource_estimates = {
            "native_direct_backend": int(executor.protected),
            "multi_environment_shared_decode": 1,
            "multi_environment_protected_gemm": int(executor.protected),
            "multi_environment_count": len(estimators),
            "shared_genotype_passes": passes,
            "decoded_genotype_block_gib": float(
                first.nsamp * max_block * 8 / 1024**3
            ),
            "prepared_feature_pair_gib": float(
                2 * first.nsamp * max_block * itemsize / 1024**3
            ),
            "resident_sketch_workspace_gib": float(
                resident_multiplier
                * first.nsamp * first.nbins * max_vt * itemsize / 1024**3
            ),
            "multi_environment_total_resident_sketch_workspace_gib": float(
                total_resident / 1024**3
            ),
            "jackknife_in_memory_block_sketch_gib": 0.0,
            "source_columns": int(first.nbins * max_vt),
            "actual_global_2b_source_columns": int(2 * first.nbins * max_vt),
            "actual_jackknife_2b_source_columns": 0,
            "target_source_columns": int(2 * first.nbins * max_vt),
            "blas_threads": int(first.num_threads),
            "bed_reader_threads": int(first.decode_threads),
        }
    first.log._log(
        "[gxe:multi:resources] modeled total resident paired-global "
        f"sketch workspace={total_resident / 1024**3:.3f} GiB; "
        f"v_tiles={vtiles}."
    )

    accumulators = [
        {
            name: np.zeros((first.nsnps, first.nbins), dtype=np.float64)
            for name in _SCORE_NAMES
        }
        for _ in estimators
    ]
    population_enabled = first.nvecs >= 2
    population_probe_square_sums = (
        [
            np.zeros((estimator.nsamp, 2 * estimator.nbins), dtype=np.float64)
            for estimator in estimators
        ]
        if population_enabled
        else []
    )
    population_same_probe_products = (
        [
            np.zeros((2 * estimator.nbins, 2 * estimator.nbins), dtype=np.float64)
            for estimator in estimators
        ]
        if population_enabled
        else []
    )
    if population_enabled:
        for index, estimator in enumerate(estimators):
            estimator.resource_estimates["population_trace_workspace_gib"] = float(
                population_probe_square_sums[index].nbytes
                + population_same_probe_products[index].nbytes
            ) / 1024**3
    for probe_start, probe_count in vtiles:
        columns = first.nbins * probe_count
        global_sources = [
            np.zeros(
                (first.nsamp, 2 * columns),
                dtype=executor.compute_dtype,
                order="F",
            )
            for _ in estimators
        ]
        for start, stop in blocks:
            genotype = first._read_genotype_block(start, stop)
            probes = first._generate_random_block(
                L=stop - start,
                v_count=probe_count,
                blk_start=start,
                v_start=probe_start,
            )
            for index, estimator in enumerate(estimators):
                annotation = np.asarray(
                    estimator.annot[start:stop], dtype=executor.compute_dtype
                )
                if executor.protected:
                    _accumulate_unprojected_source_block(
                        estimator,
                        global_sources[index][:, :columns],
                        genotype,
                        probes,
                        annotation,
                        start,
                        stop,
                        interaction=False,
                        executor=executor,
                    )
                    _accumulate_unprojected_source_block(
                        estimator,
                        global_sources[index][:, columns:2 * columns],
                        genotype,
                        probes,
                        annotation,
                        start,
                        stop,
                        interaction=True,
                        executor=executor,
                    )
                    del annotation
                    continue
                x = _prepare_feature_block(
                    estimator, genotype, start, stop,
                    interaction=False, apply_scale=True, executor=executor,
                )
                w = _prepare_feature_block(
                    estimator, genotype, start, stop,
                    interaction=True, apply_scale=True, executor=executor,
                )
                _accumulate_sketch_block(
                    estimator, global_sources[index][:, :columns], x,
                    probes, annotation, executor,
                )
                _accumulate_sketch_block(
                    estimator,
                    global_sources[index][:, columns:2 * columns],
                    w, probes, annotation, executor,
                )
                del x, w, annotation
            del genotype, probes

        environment_weighted_sources = []
        if executor.protected:
            for index, estimator in enumerate(estimators):
                leakage = _project_source_panel_inplace(
                    estimator, global_sources[index], executor
                )
                estimator.resource_estimates[
                    "max_native_source_projection_leakage"
                ] = max(
                    float(
                        estimator.resource_estimates.get(
                            "max_native_source_projection_leakage", 0.0
                        )
                    ),
                    leakage,
                )
                environment_weighted_sources.append(
                    np.asfortranarray(
                        estimator.env[:, None] * global_sources[index],
                        dtype=np.float64,
                    )
                )

        if population_enabled:
            for index, estimator in enumerate(estimators):
                estimator._accumulate_population_diagonal_moments(
                    global_sources[index],
                    probe_count,
                    population_probe_square_sums[index],
                    population_same_probe_products[index],
                )

        for start, stop in blocks:
            genotype = first._read_genotype_block(start, stop)
            for index, estimator in enumerate(estimators):
                if executor.protected:
                    work_x = np.asarray(
                        executor.tn(genotype, global_sources[index]),
                        dtype=np.float64,
                    )
                    work_w = np.asarray(
                        executor.tn(
                            genotype, environment_weighted_sources[index]
                        ),
                        dtype=np.float64,
                    )
                    work_x *= estimator.inv_sqrt_resvar_x_all[
                        start:stop
                    ].reshape(-1, 1)
                    work_w *= estimator.inv_sqrt_resvar_w_all[
                        start:stop
                    ].reshape(-1, 1)
                else:
                    x = _prepare_feature_block(
                        estimator, genotype, start, stop,
                        interaction=False, apply_scale=True, executor=executor,
                    )
                    w = _prepare_feature_block(
                        estimator, genotype, start, stop,
                        interaction=True, apply_scale=True, executor=executor,
                    )
                    work_x = np.asarray(
                        executor.tn(x, global_sources[index]), dtype=np.float64
                    )
                    work_w = np.asarray(
                        executor.tn(w, global_sources[index]), dtype=np.float64
                    )
                    del x, w
                estimator._accumulate_left_scores(
                    work_x[:, :columns], accumulators[index]["xx"],
                    start, stop, probe_count,
                )
                estimator._accumulate_left_scores(
                    work_x[:, columns:2 * columns], accumulators[index]["xw"],
                    start, stop, probe_count,
                )
                estimator._accumulate_left_scores(
                    work_w[:, :columns], accumulators[index]["wx"],
                    start, stop, probe_count,
                )
                estimator._accumulate_left_scores(
                    work_w[:, columns:2 * columns], accumulators[index]["ww"],
                    start, stop, probe_count,
                )
                del work_x, work_w
            del genotype
        del environment_weighted_sources
        del global_sources
        gc.collect()

    first._assert_construction_genotype_state()
    if executor.protected:
        # Re-hash the still-loaded extension immediately before sealing any
        # scientific bundle, then expose the aggregate repair count in every
        # independently consumable reference.
        shared_provenance = executor.provenance_template()
        for estimator in estimators:
            estimator.shared_backend_provenance = dict(shared_provenance)
            estimator.resource_estimates[
                "native_repaired_gemm_output_columns"
            ] = int(executor.repaired_output_columns)
            estimator.resource_estimates[
                "native_retried_gemm_input_mutations"
            ] = 0
        first.log._log(
            "[gxe:multi:integrity] ABFT-repaired output columns="
            f"{executor.repaired_output_columns}; input mutation policy=abort."
        )
    if population_enabled:
        for index, estimator in enumerate(estimators):
            estimator.population_same_individual_products = (
                estimator._finalize_population_diagonal_moments(
                    population_probe_square_sums[index],
                    population_same_probe_products[index],
                )
            )
    references = []
    published: list[tuple[Path, int, int, str]] = []
    try:
        for index, estimator in enumerate(estimators):
            scores = {
                name: value / float(estimator.nvecs)
                for name, value in accumulators[index].items()
            }
            estimator._compute_ldscore(
                compute_callback=(
                    lambda estimator=estimator, scores=scores:
                    estimator._finalize_ldscore_outputs(scores, None)
                ),
                expected_provenance=initial_provenance,
                provenance_preverified=True,
            )
            published.extend(
                _published_file_identity(path.resolve())
                for path in estimator._planned_output_paths()
            )
            reference = Path(f"{estimator.outpath}.gxe.ref.json").resolve()
            references.append(
                {
                    "environment": estimator.env_name,
                    "reference": os.path.relpath(reference, start=target.parent),
                    "sha256": estimator._file_sha256(str(reference)),
                }
            )

        payload = {
            "kind": "summit.gxe.multi_environment_reference_batch",
            "schema_version": 1,
            "execution": "shared_in_memory_decoded_blocks",
            "requested_backend": str(requested_backend),
            "protected_native_gemm": bool(executor.protected),
            "repaired_gemm_output_columns": int(
                executor.repaired_output_columns
            ),
            "num_environments": len(estimators),
            "common_complete_case_samples": first.nsamp,
            "num_variants": first.nsnps,
            "randomization": {
                "distribution": first.rand_dist,
                "num_vectors": first.nvecs,
                "seed": first.root_seed,
                "probe_offset": first.probe_offset,
                "probe_tiles": [list(tile) for tile in vtiles],
            },
            "shared_genotype_passes": passes,
            "modeled_total_resident_sketch_workspace_gib": float(
                total_resident / 1024**3
            ),
            "references": references,
        }
        # One final shared hash seals all environment bundles against the same
        # input bytes immediately before the batch manifest becomes visible.
        first._assert_genotype_provenance_unchanged(initial_provenance)
        _publish_json_no_replace(payload, target)
    except Exception:
        _rollback_published_files(published)
        raise
    return target
