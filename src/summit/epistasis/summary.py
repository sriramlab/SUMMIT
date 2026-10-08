"""Genotype-free inference with exact phenotype-dependent analytic-SE moments."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import norm

from summit.context.fit import ContextNormalEquations, solve_context_normal_equations
from summit.context.spec import (
    array_sha256, canonical_json, freeze_context_mapping, owned_readonly_array,
)

KIND = "summit.epistasis.study_matched_moments"


@dataclass(frozen=True)
class EpistasisSummary:
    """C is ordered (trait, left kernel, middle kernel, right kernel).

    Only aggregates are mandatory. No phenotype or sample-level kernel action
    is stored. Metadata binds the cohort-side definitions; transfers to another
    reference population are deliberately unsupported.
    """
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    cubic: np.ndarray
    component_names: tuple[str, ...]
    trait_names: tuple[str, ...]
    n_samples: int
    residual_rank: int
    metadata: dict
    probe_matrices: np.ndarray | None = None

    def __post_init__(self):
        for key in ("component_names", "trait_names"):
            names = tuple(getattr(self, key))
            if not names or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
                raise ValueError(f"{key} must contain unique nonempty names")
            object.__setattr__(self, key, names)
        c, t = len(self.component_names), len(self.trait_names)
        for key, shape in (("matrix", (c, c)), ("rhs", (c, t)),
                           ("traces", (c,)), ("cubic", (t, c, c, c))):
            value = owned_readonly_array(getattr(self, key), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {key}: expected finite {shape}")
            object.__setattr__(self, key, value)
        if (type(self.n_samples) is not int or type(self.residual_rank) is not int
                or not 1 <= self.residual_rank <= self.n_samples):
            raise ValueError("invalid sample count or residual rank")
        if self.component_names[-1] != "residual":
            raise ValueError("last component must be the projected iid residual")
        if not np.allclose(self.matrix, self.matrix.T, rtol=1e-10, atol=1e-10):
            raise ValueError("normal matrix is asymmetric")
        if not np.allclose(self.cubic, self.cubic.transpose(0, 3, 2, 1), rtol=1e-9, atol=1e-9):
            raise ValueError("analytic moments are asymmetric in outer kernels")
        if not np.allclose(self.matrix[-1], self.traces, rtol=1e-9, atol=1e-9):
            raise ValueError("residual cross-traces do not match kernel traces")
        if not np.isclose(self.traces[-1], self.residual_rank):
            raise ValueError("residual trace does not match residual rank")
        if np.any(self.traces <= 0) or np.any(self.rhs[-1] <= 0):
            raise ValueError("empty component or zero residual phenotype variance")
        if not isinstance(self.metadata, dict) or not self.metadata:
            raise ValueError("cohort-side definitions must be recorded")
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))
        if self.probe_matrices is not None:
            probes=owned_readonly_array(self.probe_matrices,dtype=float)
            if (probes.ndim!=3 or probes.shape[1:]!=(c,c) or len(probes)<1
                    or not np.all(np.isfinite(probes)) or not np.allclose(probes,probes.transpose(0,2,1),atol=1e-9)
                    or not np.allclose(probes.mean(axis=0),self.matrix,rtol=1e-9,atol=1e-9)):
                raise ValueError("invalid per-probe normal matrices")
            object.__setattr__(self,"probe_matrices",probes)


def normal_equations(summary, trait=0):
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    if isinstance(trait, bool) or not isinstance(trait, (int, np.integer)) or not 0 <= trait < len(summary.trait_names):
        raise ValueError("invalid trait selection")
    c = int(summary.metadata.get("genetic_count", len(summary.component_names) - 1))
    return ContextNormalEquations(
        matrix=summary.matrix, rhs=summary.rhs[:, trait], traces=summary.traces,
        component_names=summary.component_names, genetic_count=c,
        annotation_masses=np.ones(c), deleted_groups=(),
        reference_genetic_gram=summary.matrix[:c, :c],
        transferred_genetic_gram=summary.matrix[:c, :c],
        reference_n=summary.n_samples, study_n=summary.n_samples,
    )


def fit_epistasis(summary: EpistasisSummary, trait=0):
    """Unconstrained MoM and FAME plug-in covariance, conditional on reference.

    Wald P values are a one-sided normal approximation for positive variance,
    not an exact boundary or small-set test. Invalid covariance yields NaN SEs
    and P values, with the unmodified covariance retained for diagnosis.
    """
    if isinstance(trait, str):
        trait = summary.trait_names.index(trait)
    equations = normal_equations(summary, trait)
    solve = solve_context_normal_equations(equations)
    theta = solve.coefficients
    cov_q = 2 * np.einsum("c,acb->ab", theta, summary.cubic[trait])
    inverse = np.linalg.solve(summary.matrix, np.eye(len(theta)))
    covariance = inverse @ cov_q @ inverse.T
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-10 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    valid = bool(eigenvalues[0] >= -tolerance and np.all(np.diag(covariance) >= 0))
    se = np.sqrt(np.diag(covariance)) if valid else np.full(len(theta), np.nan)
    d = summary.traces / summary.residual_rank
    contribution = theta * d
    contribution_covariance = covariance * np.outer(d, d)
    total = contribution.sum()
    proportion = contribution / total if total > 0 else np.full(len(theta), np.nan)
    gradient = np.diag(d) / total - np.outer(contribution, d) / total**2
    proportion_covariance = gradient @ covariance @ gradient.T
    contribution_se = se * d
    proportion_se = np.full(len(theta), np.nan)
    if valid and total > 0:
        nonnegative = np.diag(proportion_covariance) >= 0
        proportion_se[nonnegative] = np.sqrt(np.diag(proportion_covariance)[nonnegative])
    with np.errstate(divide="ignore", invalid="ignore"):
        p = norm.sf(theta / se)
    p[~np.isfinite(se) | (se <= 0)] = np.nan
    from .probe import probe_diagnostics
    return dict(
        trait=summary.trait_names[trait], component_names=summary.component_names,
        coefficients=theta, covariance=covariance, standard_errors=se,
        variance_contributions=contribution, contribution_covariance=contribution_covariance,
        variance_contribution_se=contribution_se,
        proportions=proportion, proportion_covariance=proportion_covariance, proportion_se=proportion_se,
        wald_p_one_sided=p, covariance_valid=valid,
        covariance_eigenvalues=eigenvalues, negative_coefficients=np.flatnonzero(theta < 0),
        rank=solve.rank, condition_number=solve.condition_number,
        minimum_gram_eigenvalue=solve.minimum_gram_eigenvalue,
        nuisance_fit_status=("nonnegative_components" if np.all(theta >= 0)
                             else "unconstrained_negative_components_covariance_not_certified"),
        relative_solve_residual=solve.relative_residual,
        uncertainty="fame_phenotype_plugin_conditional_on_reference",
        probe_uncertainty=probe_diagnostics(summary,theta),
    )


def write_summary(summary, path):
    """Atomic no-replace publication, with SUMMIT's canonical array digests."""
    arrays = {k: getattr(summary, k) for k in ("matrix", "rhs", "traces", "cubic")}
    if summary.probe_matrices is not None:arrays["probe_matrices"]=summary.probe_matrices
    manifest = dict(kind=KIND, schema_version=2 if summary.probe_matrices is not None else 1, component_names=summary.component_names,
                    trait_names=summary.trait_names, n_samples=summary.n_samples,
                    residual_rank=summary.residual_rank, metadata=summary.metadata,
                    digests={k: array_sha256(v) for k, v in arrays.items()})
    return _publish_bundle(path, manifest, arrays)


def _publish_bundle(path, manifest, arrays):
    """Shared private no-replace writer for epistasis aggregate artifacts."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, manifest=np.asarray(canonical_json(manifest)), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def load_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["kind"] != KIND or manifest["schema_version"] not in (1,2):
            raise ValueError("unsupported epistasis summary")
        fields={"matrix","rhs","traces","cubic"}
        if manifest["schema_version"]==2:fields.add("probe_matrices")
        if set(archive.files)!=fields|{"manifest"}:raise ValueError("unexpected summary fields")
        arrays = {k: archive[k] for k in fields}
        if manifest["digests"] != {k: array_sha256(v) for k, v in arrays.items()}:
            raise ValueError("summary digest mismatch")
        return EpistasisSummary(**arrays, **{k: manifest[k] for k in (
            "component_names", "trait_names", "n_samples", "residual_rank", "metadata")})
