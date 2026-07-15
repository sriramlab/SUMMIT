from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .. import utils
from ..sumstats.moments import build_h2_summary_moment


@dataclass(frozen=True)
class LDSCIRWLSFit:
    h: np.ndarray
    lhs: np.ndarray
    rhs: np.ndarray
    weights: np.ndarray
    h_path: np.ndarray
    n_iter: int
    converged: bool
    final_delta: float
    initialization: str
    rank: int
    condition_number: float


@dataclass(frozen=True)
class LDSCH2Prepared:
    """The per-SNP state needed by constrained LDSC, without HE matrices."""

    trace_view: object
    matched: object
    jackknife: object
    active_mask: np.ndarray
    has_overlap: bool
    unit_sizes: np.ndarray
    n_scale: float
    summary_y_info: dict | None
    y: np.ndarray


@dataclass(frozen=True)
class LDSCReferenceMoments:
    m_annot: np.ndarray
    overlap_matrix: np.ndarray
    source_nsnps: int
    source: str


def prepare_h2_ldsc(trace_view, matched, jackknife) -> LDSCH2Prepared:
    """Prepare score-scale response data without constructing HE normal equations."""
    if int(trace_view.nsnps) != int(matched.nsnps):
        raise ValueError("TraceView and MatchedSumstats must have the same number of SNPs.")
    if not np.array_equal(trace_view.snps, matched.snps):
        raise ValueError("TraceView and MatchedSumstats SNP order mismatch.")
    if int(jackknife.nsnps) != int(trace_view.nsnps):
        raise ValueError("JackknifeDesign was not built on this TraceView SNP axis.")

    y, info = build_h2_summary_moment(matched)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (int(trace_view.nsnps),):
        raise ValueError("Unexpected h2 summary-moment shape.")
    active = np.isfinite(y)
    if not np.any(active):
        raise ValueError("No finite score moments remain for constrained LDSC.")
    n_scale = float(info.get("n_scale", getattr(matched, "n_scale", np.nan)))
    if not (np.isfinite(n_scale) and n_scale > 0.0):
        raise ValueError(f"Invalid score-scale sample size {n_scale}.")

    A = np.asarray(trace_view.annot, dtype=np.float64)
    overlap = A[active, :].T @ A[active, :]
    diag = np.diag(overlap)
    offdiag = overlap - np.diag(diag)
    has_overlap = bool(np.any(np.abs(offdiag) > 1e-12))
    return LDSCH2Prepared(
        trace_view=trace_view,
        matched=matched,
        jackknife=jackknife,
        active_mask=active,
        has_overlap=has_overlap,
        unit_sizes=jackknife.unit_sizes(active_mask=active, dtype=np.float64),
        n_scale=n_scale,
        summary_y_info=info,
        y=y,
    )


def read_ldsc_m(path_spec, *, nbins: int) -> np.ndarray:
    """Read and, for chromosome-split input, sum LDSC ``.l2.M`` vectors."""
    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError("nbins must be positive")

    paths = utils._resolve_chr_split_paths(path_spec, require=True)
    total = np.zeros(nbins, dtype=np.float64)
    for path in paths:
        values = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64).reshape(-1)
        if values.shape != (nbins,):
            raise ValueError(
                f"LDSC M file '{path}' has {values.size} value(s); expected {nbins}."
            )
        if not (np.isfinite(values).all() and np.all(values >= 0.0)):
            raise ValueError(f"LDSC M file '{path}' contains non-finite or negative values.")
        total += values

    if not np.all(total > 0.0):
        bad = np.flatnonzero(total <= 0.0).tolist()
        raise ValueError(f"Summed LDSC M values must be positive; bad bins: {bad}.")
    return total


def read_ldsc_weight_ld_aligned(path_spec, target_snps) -> tuple[np.ndarray, np.ndarray]:
    """Read one scalar weight-LD column without discarding values below one."""
    df = utils._read_csv_maybe_chr_split(
        path_spec,
        compression="infer",
        sep=r"\s+",
        index_col=False,
    )
    cols = df.columns.tolist()
    first4 = cols[:4]
    if not {"CHR", "BP", "SNP"}.issubset(set(first4)):
        raise ValueError(
            "LDSC weight-LD input must have CHR, BP, SNP in its first columns."
        )
    start = 4 if "CM" in first4 else 3
    value_cols = cols[start:]
    if len(value_cols) != 1:
        raise ValueError(
            f"LDSC weight-LD input must have exactly one score column; got {len(value_cols)}."
        )
    snps = df["SNP"].astype(str)
    if snps.duplicated().any():
        first = snps[snps.duplicated(keep=False)].iloc[0]
        raise ValueError(f"LDSC weight-LD input contains duplicate SNP ID {first!r}.")
    values = df[value_cols[0]].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    index = pd.Index(snps[finite].to_numpy())
    target = np.asarray(target_snps, dtype=str).reshape(-1)
    pos = index.get_indexer(target)
    present = pos >= 0
    out = np.full((target.size, 1), np.nan, dtype=np.float64)
    if np.any(present):
        out[present, 0] = values[finite][pos[present]]
    if not np.any(present):
        raise ValueError("No primary LD-score SNPs overlap the LDSC weight-LD input.")
    return out, present


def resolve_ldsc_reference_moments(
    *,
    annot_path,
    trace_annot,
    trace_header,
    m_override=None,
) -> LDSCReferenceMoments:
    """Resolve fixed effect-reference moments, before GWAS/weight-LD filtering.

    A standard LDSC ``.M`` vector and the annotation overlap matrix must describe
    the same effect-SNP universe.  When an annotation file is supplied, read its
    full (pre-regression-merge) rows and enforce that invariant.
    """
    trace_annot = np.asarray(trace_annot, dtype=np.float64)
    headers = [str(x) for x in np.asarray(trace_header).reshape(-1)]
    k = len(headers)
    if trace_annot.ndim != 2 or trace_annot.shape[1] != k:
        raise ValueError("Trace annotation matrix/header mismatch.")

    source = "full_aligned_trace_annotation"
    if annot_path is None:
        A_ref = trace_annot
    else:
        df = utils._read_csv_maybe_chr_split(
            annot_path,
            compression="infer",
            sep=r"\s+",
            index_col=False,
        )
        cols = df.columns.tolist()
        first4 = cols[:4]
        is_full = {"CHR", "BP", "SNP"}.issubset(set(first4))
        if is_full:
            if df["SNP"].astype(str).duplicated().any():
                raise ValueError("Reference annotation file contains duplicate SNP IDs.")
            start = 4 if "CM" in first4 else 3
            annot_cols = [str(x) for x in cols[start:]]
            if annot_cols != headers:
                raise ValueError(
                    "Reference annotation columns do not match the fitted LD-score columns: "
                    f"annotation={annot_cols}, fitted={headers}."
                )
            A_ref = df.iloc[:, start:].to_numpy(dtype=np.float64, copy=False)
            source = "full_reference_annotation_file"
        else:
            thin_header, A_ref = utils._read_with_optional_header(annot_path)
            A_ref = np.asarray(A_ref, dtype=np.float64)
            if A_ref.ndim == 1:
                A_ref = A_ref.reshape(-1, 1)
            if thin_header is not None and [str(x) for x in thin_header] != headers:
                raise ValueError(
                    "Thin reference annotation columns do not match the fitted LD-score columns."
                )
            if A_ref.shape[1] != k:
                raise ValueError(
                    f"Thin reference annotation has {A_ref.shape[1]} columns; expected {k}."
                )
            source = "full_reference_thin_annotation_file"

    A_ref = np.asarray(A_ref, dtype=np.float64, order="C")
    if A_ref.ndim != 2 or A_ref.shape[0] <= 0 or A_ref.shape[1] != k:
        raise ValueError("Reference annotation matrix has an invalid shape.")
    if not np.isfinite(A_ref).all():
        raise ValueError("Reference annotation matrix contains non-finite values.")
    raw_m = A_ref.sum(axis=0, dtype=np.float64)
    if not (np.isfinite(raw_m).all() and np.all(raw_m > 0.0)):
        raise ValueError("Reference annotation masses must be positive and finite.")

    if m_override is None:
        m_annot = raw_m
    else:
        m_annot = np.asarray(m_override, dtype=np.float64).reshape(-1)
        if m_annot.shape != (k,):
            raise ValueError(f"LDSC M vector has shape {m_annot.shape}; expected {(k,)}.")
        if annot_path is not None and not np.allclose(
            m_annot,
            raw_m,
            rtol=1e-8,
            atol=1e-4,
        ):
            raise ValueError(
                "--ldsc-m and --annot describe different effect-SNP universes. "
                "Use the .M file computed from this exact annotation set; .M_5_50 "
                "requires an annotation matrix filtered to the same variants."
            )

    # With no explicit annotation the sole column is all ones.  An external M
    # therefore supplies the exact missing size/overlap moment directly.
    if annot_path is None and m_override is not None:
        if k != 1:
            raise ValueError("Annotation-free LDSC input must have exactly one LD-score column.")
        source_nsnps = int(round(float(m_annot[0])))
        if source_nsnps <= 0 or not np.isclose(float(source_nsnps), float(m_annot[0])):
            raise ValueError("Unpartitioned LDSC M must be a positive integer SNP count.")
        overlap = np.asarray([[float(m_annot[0])]], dtype=np.float64)
        source += "+external_m"
    else:
        source_nsnps = int(A_ref.shape[0])
        overlap = np.asarray(A_ref.T @ A_ref, dtype=np.float64)
        if m_override is not None:
            source += "+validated_external_m"

    return LDSCReferenceMoments(
        m_annot=np.asarray(m_annot, dtype=np.float64),
        overlap_matrix=overlap,
        source_nsnps=source_nsnps,
        source=source,
    )


def _solve_weighted_design(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve WLS on the design itself, without squaring its condition number."""
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or response.shape != (design.shape[0],):
        raise ValueError("LDSC design and response shapes are inconsistent.")

    if weights is None:
        weighted_design = design
        weighted_response = response
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != (design.shape[0],):
            raise ValueError("LDSC weights do not share the design SNP axis.")
        if not (np.isfinite(weights).all() and np.all(weights > 0.0)):
            raise ValueError("LDSC weights must be positive and finite.")
        root_w = np.sqrt(weights)
        weighted_design = design * root_w[:, None]
        weighted_response = response * root_w

    coef, _resid, rank, singular = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )
    k = int(design.shape[1])
    if int(rank) != k:
        raise np.linalg.LinAlgError(
            f"LDSC weighted design is rank deficient ({int(rank)} < {k})."
        )
    lhs = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_response
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0.0
        else np.inf
    )
    return np.asarray(coef, dtype=np.float64), lhs, rhs, int(rank), condition


def _initial_total_h(
    design: np.ndarray,
    q: np.ndarray,
    ref_ld_total: np.ndarray,
    n_scale: float,
    m_total: float,
    keep: np.ndarray,
) -> tuple[np.ndarray, str]:
    dk = design[keep, :]
    qk = q[keep]
    k = int(design.shape[1])
    if k == 1:
        denom = float(np.sum(float(n_scale) * ref_ld_total[keep], dtype=np.float64))
        numer = float(m_total) * float(np.sum(qk, dtype=np.float64))
        if np.isfinite(numer) and np.isfinite(denom) and denom != 0.0:
            return np.asarray([numer / denom], dtype=np.float64), "aggregate_ldsc"

    h, _lhs, _rhs, _rank, _condition = _solve_weighted_design(dk, qk)
    return h, "unweighted_wls"


def fit_constrained_ldsc_irwls(
    design,
    q,
    ref_ld_total,
    weight_ld,
    *,
    n_scale: float,
    m_annot,
    keep=None,
    irwls_iters: int = 3,
    irwls_tol: float = 0.0,
    initial_h=None,
    ld_floor: float = 1.0,
    mean_floor: float = 1e-3,
) -> LDSCIRWLSFit:
    """Fit fixed-intercept LDSC by closed-form WLS updates.

    ``q`` is the fixed-intercept response (for h2, score ``Z^2 - 1``), and
    ``design[:, k] = n_scale * LD[:, k] / M_k``.  The design is never floored;
    flooring applies only to the heteroskedasticity and overcounting weights.
    """
    design = np.asarray(design, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    ref_ld_total = np.asarray(ref_ld_total, dtype=np.float64).reshape(-1)
    weight_ld = np.asarray(weight_ld, dtype=np.float64).reshape(-1)
    m_annot = np.asarray(m_annot, dtype=np.float64).reshape(-1)

    if design.ndim != 2:
        raise ValueError(f"design must be 2D; got {design.shape}.")
    m, k = design.shape
    if q.shape != (m,) or ref_ld_total.shape != (m,) or weight_ld.shape != (m,):
        raise ValueError("LDSC design, response, reference LD, and weight LD must share one SNP axis.")
    if m_annot.shape != (k,):
        raise ValueError(f"m_annot must have shape ({k},); got {m_annot.shape}.")
    if not (
        np.isfinite(design).all()
        and np.isfinite(ref_ld_total).all()
        and np.isfinite(weight_ld).all()
        and np.isfinite(m_annot).all()
        and np.all(m_annot > 0.0)
    ):
        raise ValueError("LDSC inputs contain non-finite values or non-positive annotation masses.")

    n_scale = float(n_scale)
    if not (np.isfinite(n_scale) and n_scale > 0.0):
        raise ValueError(f"n_scale must be positive and finite; got {n_scale}.")
    m_total = float(np.sum(m_annot, dtype=np.float64))
    if not (np.isfinite(m_total) and m_total > 0.0):
        raise ValueError("Total LDSC annotation mass must be positive and finite.")

    if keep is None:
        keep = np.ones(m, dtype=bool)
    else:
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        if keep.shape != (m,):
            raise ValueError(f"keep must have shape ({m},); got {keep.shape}.")
    if int(np.sum(keep)) < k:
        raise ValueError(f"LDSC fit has fewer retained SNPs ({int(np.sum(keep))}) than coefficients ({k}).")
    if not np.isfinite(q[keep]).all():
        raise ValueError("The retained LDSC response contains non-finite values.")

    irwls_iters = int(irwls_iters)
    if irwls_iters < 1:
        raise ValueError("irwls_iters must be at least 1.")
    irwls_tol = float(irwls_tol)
    if not (np.isfinite(irwls_tol) and irwls_tol >= 0.0):
        raise ValueError("irwls_tol must be non-negative and finite.")
    ld_floor = float(ld_floor)
    mean_floor = float(mean_floor)
    if not (np.isfinite(ld_floor) and ld_floor > 0.0):
        raise ValueError("ld_floor must be positive and finite.")
    if not (np.isfinite(mean_floor) and mean_floor > 0.0):
        raise ValueError("mean_floor must be positive and finite.")

    if initial_h is None:
        h_current, initialization = _initial_total_h(
            design, q, ref_ld_total, n_scale, m_total, keep
        )
    else:
        h_current = np.asarray(initial_h, dtype=np.float64).reshape(-1)
        if h_current.shape != (k,) or not np.isfinite(h_current).all():
            raise ValueError(f"initial_h must be a finite vector with shape ({k},).")
        initialization = "provided"

    # Slice once.  Delete-block refits are expensive enough without copying the
    # M-by-K design again on every IRWLS update.
    dk = np.asarray(design[keep, :], dtype=np.float64, order="C")
    qk = np.asarray(q[keep], dtype=np.float64)
    ref_weight = np.maximum(np.asarray(ref_ld_total[keep], dtype=np.float64), ld_floor)
    oc_weight = np.maximum(np.asarray(weight_ld[keep], dtype=np.float64), ld_floor)
    path = [h_current.copy()]
    final_lhs = None
    final_rhs = None
    final_weights = None
    converged = False
    final_delta = np.nan
    n_iter = 0
    final_rank = 0
    final_condition = np.nan

    for iteration in range(irwls_iters):
        h_for_weights = float(np.clip(np.sum(h_current, dtype=np.float64), 0.0, 1.0))
        mean = 1.0 + h_for_weights * (n_scale / m_total) * ref_weight
        mean = np.maximum(mean, mean_floor)
        weights = 1.0 / (2.0 * mean * mean * oc_weight)
        if not (np.isfinite(weights).all() and np.all(weights > 0.0)):
            raise ValueError("LDSC IRWLS produced non-positive or non-finite weights.")

        h_next, lhs, rhs, rank, condition = _solve_weighted_design(dk, qk, weights)

        final_lhs = lhs
        final_rhs = rhs
        final_weights = weights
        final_rank = rank
        final_condition = condition
        n_iter = iteration + 1
        final_delta = float(np.max(np.abs(h_next - h_current)))
        h_current = h_next
        path.append(h_current.copy())
        if irwls_tol > 0.0:
            scale = max(1.0, float(np.max(np.abs(h_current))))
            if final_delta <= irwls_tol * scale:
                converged = True
                break

    return LDSCIRWLSFit(
        h=np.asarray(h_current, dtype=np.float64),
        lhs=np.asarray(final_lhs, dtype=np.float64),
        rhs=np.asarray(final_rhs, dtype=np.float64),
        weights=np.asarray(final_weights, dtype=np.float64),
        h_path=np.asarray(path, dtype=np.float64),
        n_iter=int(n_iter),
        converged=bool(converged),
        final_delta=float(final_delta),
        initialization=str(initialization),
        rank=int(final_rank),
        condition_number=float(final_condition),
    )


def fit_h2_ldsc(
    prepared,
    *,
    m_annot,
    overlap_matrix,
    source_nsnps: int,
    enrich_mode: str,
    report_tau: bool,
    allow_neg_enr: bool,
    clip_nonfinite_vals: bool,
    jack_mode: str,
    nan_policy: str,
    irwls_iters: int,
    irwls_tol: float,
):
    """Fit constrained score-scale LDSC and return the regular ``H2Fit`` API."""
    # Imported lazily to avoid a module-import cycle with h2core.fit_h2().
    from .h2core import H2Fit, _normalize_enrich_mode

    p = prepared
    if np.asarray(p.y).shape != (int(p.trace_view.nsnps),):
        raise ValueError(
            "This prepared h2 object does not expose the per-SNP score response. "
            "The fast batch/cache path is not wired to LDSC IRWLS yet."
        )

    A = np.asarray(p.trace_view.annot, dtype=np.float64)
    L = np.asarray(p.trace_view.ldscores, dtype=np.float64)
    y = np.asarray(p.y, dtype=np.float64)
    active = np.asarray(p.active_mask, dtype=bool)
    m, k = L.shape
    if A.shape != (m, k) or y.shape != (m,) or active.shape != (m,):
        raise ValueError("Inconsistent per-SNP arrays in LDSC h2 preparation.")

    m_annot = np.asarray(m_annot, dtype=np.float64).reshape(-1)
    overlap_matrix = np.asarray(overlap_matrix, dtype=np.float64)
    if m_annot.shape != (k,) or overlap_matrix.shape != (k, k):
        raise ValueError("LDSC reference annotation moments do not match the fitted LD-score columns.")
    source_nsnps = int(source_nsnps)
    if source_nsnps <= 0:
        raise ValueError("source_nsnps must be positive.")

    q = y - 1.0
    valid = active & np.isfinite(q) & np.isfinite(L).all(axis=1)
    if not np.any(valid):
        raise ValueError("No finite SNPs remain for LDSC-weighted h2.")

    design = (float(p.n_scale) * L) / m_annot[None, :]
    ref_ld_total = np.sum(L, axis=1, dtype=np.float64)
    reg_w = getattr(p.trace_view, "ldscores_reg_w", None)
    if reg_w is None:
        weight_ld = ref_ld_total.copy()
        weight_ld_source = "total_primary_ld"
    else:
        reg_w = np.asarray(reg_w, dtype=np.float64)
        if reg_w.shape != (m, 1):
            raise ValueError(f"LDSC weight-LD input must have shape ({m}, 1); got {reg_w.shape}.")
        weight_ld = reg_w[:, 0]
        weight_ld_source = "explicit_ldscores_w"

    R = int(p.jackknife.nrep)
    h_reps = np.full((R + 1, k), np.nan, dtype=np.float64)
    failures = []
    unit_id = np.asarray(p.jackknife.unit_id, dtype=np.int64)
    delete = np.asarray(p.jackknife.D, dtype=np.float64)
    full_fit = None

    # Run the full fit first, then retain only its small diagnostics.  Holding a
    # length-M weight vector for every replicate would otherwise cost O(RM).
    for r in [R, *range(R)]:
        keep = valid if r == R else (valid & (delete[r, unit_id] < 0.5))
        try:
            fit = fit_constrained_ldsc_irwls(
                design,
                q,
                ref_ld_total,
                weight_ld,
                n_scale=float(p.n_scale),
                m_annot=m_annot,
                keep=keep,
                irwls_iters=irwls_iters,
                irwls_tol=irwls_tol,
            )
            h_reps[r, :] = fit.h
            if r == R:
                full_fit = fit
        except Exception as exc:
            failures.append({
                "replicate": "full" if r == R else int(r),
                "error": f"{exc.__class__.__name__}: {exc}",
            })
            if r == R:
                break

    if full_fit is None:
        raise RuntimeError(f"Full-sample LDSC-weighted h2 fit failed: {failures[-1]['error']}")
    if failures:
        first = failures[0]
        raise RuntimeError(
            "One or more exact LDSC jackknife refits failed; refusing to report a partial SE. "
            f"First failure at replicate {first['replicate']}: {first['error']}"
        )

    h_total_reps = np.sum(h_reps, axis=1)
    sigma_reps = np.full((R + 1, k + 2), np.nan, dtype=np.float64)
    sigma_reps[:, :k] = h_reps
    sigma_reps[:, k] = 1.0 - h_total_reps
    sigma_reps[:, k + 1] = h_total_reps

    with np.errstate(divide="ignore", invalid="ignore"):
        tau_reps = h_reps / m_annot[None, :]
        h2_overlap_reps = np.einsum("kl,rl->rk", overlap_matrix, tau_reps, optimize=True)

    requested = _normalize_enrich_mode(enrich_mode)
    ref_diag = np.diag(overlap_matrix)
    ref_offdiag = overlap_matrix - np.diag(ref_diag)
    reference_has_overlap = bool(np.any(np.abs(ref_offdiag) > 1e-12))
    if requested == "auto":
        used = "overlap" if reference_has_overlap else "non-overlap"
        modes = (used,)
    elif requested == "both":
        used = "overlap" if reference_has_overlap else "non-overlap"
        modes = ("non-overlap", "overlap")
    else:
        used = requested
        modes = (requested,)

    prop = m_annot / float(source_nsnps)

    def _enrichment(cat):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (cat / h_total_reps[:, None]) / prop[None, :]
        bad = (~np.isfinite(out)) | (prop[None, :] <= 0.0)
        if not allow_neg_enr:
            bad |= h_total_reps[:, None] <= 0.0
        out[bad] = np.nan
        return out

    enrich_overlap_reps = _enrichment(h2_overlap_reps) if "overlap" in modes else None
    enrich_nonoverlap_reps = _enrichment(h_reps) if "non-overlap" in modes else None
    enrich_reps = enrich_overlap_reps if used == "overlap" else enrich_nonoverlap_reps
    # Match the established H2Fit contract: reported category h2 is always the
    # overlap-adjusted annotation total; enrich_mode only selects enrichment.
    h2_out_reps = np.column_stack([h2_overlap_reps, h_total_reps])

    tau_star_reps = None
    if report_tau:
        mean_a = m_annot / float(source_nsnps)
        mean_a2 = np.diag(overlap_matrix) / float(source_nsnps)
        sd_a = np.sqrt(np.maximum(mean_a2 - mean_a * mean_a, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_star_reps = tau_reps * (sd_a[None, :] / (h_total_reps[:, None] / float(source_nsnps)))

    if clip_nonfinite_vals:
        for arr in (h2_out_reps, enrich_reps, enrich_overlap_reps, enrich_nonoverlap_reps, tau_reps, tau_star_reps):
            if arr is not None:
                arr[~np.isfinite(arr)] = 0.0

    def _summary(arr):
        est, se = p.jackknife.summarize(
            arr,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        return np.column_stack([est, se]) if np.ndim(est) else np.asarray([est, se], dtype=np.float64)

    sigmas = _summary(sigma_reps)
    h2 = _summary(h2_out_reps)
    enrich = _summary(enrich_reps)
    enrich_overlap = None if enrich_overlap_reps is None else _summary(enrich_overlap_reps)
    enrich_nonoverlap = None if enrich_nonoverlap_reps is None else _summary(enrich_nonoverlap_reps)
    tau = _summary(tau_reps) if report_tau else None
    tau_star = _summary(tau_star_reps) if report_tau else None

    full = full_fit
    weight_info = {
        "estimator": "constrained_ldsc_irwls",
        "response": "score_z_squared_minus_1",
        "n_scale": float(p.n_scale),
        "m_annot": m_annot.copy(),
        "m_total_for_weights": float(np.sum(m_annot)),
        "source_nsnps": int(source_nsnps),
        "weight_ld_source": weight_ld_source,
        "irwls_iters": int(full.n_iter),
        "irwls_max_iters": int(irwls_iters),
        "irwls_tol": float(irwls_tol),
        "irwls_converged": bool(full.converged),
        "irwls_final_delta": float(full.final_delta),
        "initialization": str(full.initialization),
        "weighted_design_rank": int(full.rank),
        "weighted_design_condition_number": float(full.condition_number),
        "h_path": full.h_path.copy(),
        "weight_min": float(np.min(full.weights)),
        "weight_max": float(np.max(full.weights)),
        "n_failed_replicates": int(len(failures)),
        "replicate_failures": failures[:20],
        "jackknife_weights": "refit_irwls_per_replicate",
    }

    return H2Fit(
        prepared=prepared,
        sigma_reps=sigma_reps,
        h2_reps=h2_out_reps,
        enrich_reps=enrich_reps,
        enrich_overlap_reps=enrich_overlap_reps,
        enrich_nonoverlap_reps=enrich_nonoverlap_reps,
        tau_reps=tau_reps if report_tau else None,
        tau_star_reps=tau_star_reps,
        sigmas=sigmas,
        h2=h2,
        enrich=enrich,
        enrich_overlap=enrich_overlap,
        enrich_nonoverlap=enrich_nonoverlap,
        tau=tau,
        tau_star=tau_star,
        enrich_mode_requested=requested,
        enrich_mode_used=used,
        weight_mode="ldsc",
        weight_info=weight_info,
    )
