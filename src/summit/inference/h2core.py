from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import json

from .. import utils
from ..sumstats.moments import build_h2_summary_moment


@dataclass(frozen=True)
class H2Prepared:
    trace_view: object
    matched: object
    jackknife: object
    active_mask: np.ndarray
    has_overlap: bool
    unit_sizes: np.ndarray
    n_scale: float
    summary_y_info: dict | None
    y: np.ndarray                  # (M,)
    m_unit: np.ndarray             # (U,)
    Ak_unit: np.ndarray            # (U,K)
    Ay_unit: np.ndarray            # (U,K)
    Ak2_unit: np.ndarray           # (U,K)
    AA_unit: np.ndarray            # (U,K,K)
    AL_unit: np.ndarray            # (U,K,K)
    M_rep: np.ndarray              # (R+1,)
    Ak_rep: np.ndarray             # (R+1,K)
    Ay_rep: np.ndarray             # (R+1,K)
    Ak2_rep: np.ndarray            # (R+1,K)
    AA_rep: np.ndarray             # (R+1,K,K)
    AL_rep: np.ndarray             # (R+1,K,K)
    lhs: np.ndarray                # (R+1,K+1,K+1)
    rhs: np.ndarray                # (R+1,K+1)

@dataclass(frozen=True)
class H2Fit:
    prepared: H2Prepared
    sigma_reps: np.ndarray                 # (R+1,K+2) = sigma_g bins, sigma_e, sum_g
    h2_reps: np.ndarray                    # (R+1,K+1) = h2_cat bins, total h2
    enrich_reps: np.ndarray                # (R+1,K)
    enrich_overlap_reps: np.ndarray | None
    enrich_nonoverlap_reps: np.ndarray | None
    tau_reps: np.ndarray | None
    tau_star_reps: np.ndarray | None
    sigmas: np.ndarray                     # (K+2,2)
    h2: np.ndarray                         # (K+1,2)
    enrich: np.ndarray                     # (K,2)
    enrich_overlap: np.ndarray | None
    enrich_nonoverlap: np.ndarray | None
    tau: np.ndarray | None
    tau_star: np.ndarray | None
    enrich_mode_requested: str
    enrich_mode_used: str


@dataclass(frozen=True)
class H2TraceMetadata:
    nsnps: int
    nbins: int
    annot_header: object
    delta: np.ndarray | None = None
    kmoments: dict | None = None
    kmoments_path: str | None = None
    kmoments_valid: bool = False


@dataclass(frozen=True)
class H2MatchedMetadata:
    nsnps: int
    nsamp: float
    n_scale: float
    cov_rank: int
    cov_rank_source: str
    name: str
    used_summary: dict | None = None
    used_top: list | None = None
    clip_count: int = 0
    clip_threshold: float | None = None


@dataclass(frozen=True)
class H2StructuralUnitStats:
    m: np.ndarray                 # (U,)
    Ak: np.ndarray                # (U,K)
    Ak2: np.ndarray               # (U,K)
    AA: np.ndarray                # (U,K,K)
    AL: np.ndarray                # (U,K,K)


class H2ResultWriter:
    @staticmethod
    def save_jackknife_text(fit: H2Fit, path: str):
        p = fit.prepared
        R = p.jackknife.nrep
        K = p.trace_view.nbins
        with open(path, "w") as fd:
            fd.write("# replicate\t" + "\t".join([f"sigma_g_{k}" for k in range(K)]) + "\t"
                     + "sigma_e\tsum_g\t"
                     + "\t".join([f"h2_cat_{k}" for k in range(K)]) + "\ttotal_h2\n")
            for r in range(R + 1):
                rep_label = "full" if r == R else str(r)
                vals = np.concatenate([fit.sigma_reps[r], fit.h2_reps[r]])
                fd.write(rep_label + "\t" + "\t".join(f"{x:.10g}" for x in vals) + "\n")

    @staticmethod
    def _jsonify_numeric(x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            v = float(arr)
            return v if np.isfinite(v) else None
        return [H2ResultWriter._jsonify_numeric(v) for v in arr]

    @staticmethod
    def build_score_normal_equations_payload(
        fit: H2Fit,
        *,
        system: str | None = None,
    ) -> dict:
        p = fit.prepared
        K = int(p.trace_view.nbins)
        R = int(p.jackknife.nrep)

        T = np.asarray(p.lhs, dtype=np.float64)
        q = np.asarray(p.rhs, dtype=np.float64)
        sigma = np.asarray(fit.sigma_reps[:, : K + 1], dtype=np.float64)

        if T.shape != (R + 1, K + 1, K + 1):
            raise ValueError(f"Unexpected lhs shape: {T.shape}, expected {(R + 1, K + 1, K + 1)}")
        if q.shape != (R + 1, K + 1):
            raise ValueError(f"Unexpected rhs shape: {q.shape}, expected {(R + 1, K + 1)}")
        if sigma.shape != (R + 1, K + 1):
            raise ValueError(f"Unexpected sigma shape: {sigma.shape}, expected {(R + 1, K + 1)}")

        headers = getattr(p.trace_view, "annot_header", None)
        if headers is None or len(headers) != K:
            headers = [f"bin_{k}" for k in range(K)]
        else:
            headers = [str(h) for h in headers]

        matched = p.matched
        info = p.summary_y_info if isinstance(p.summary_y_info, dict) else {}

        system_name = system
        if system_name is None:
            system_name = getattr(matched, "name", None)
        if system_name is None or str(system_name).strip() == "":
            system_name = "trait"
        system_name = str(system_name)

        meta = {
            "equation_type": "score_full",
            "kernel_name": "h2g",
            "multi_component": bool(K > 1),
            "partial_overlap": False,
            "n_summary_raw": float(matched.nsamp),
            "n_scale": float(p.n_scale),
            "nrep": R,
            "n_trace_snps": int(p.trace_view.nsnps),
            "n_active_snps": int(np.sum(p.active_mask)),
            "has_overlapping_annotations": bool(p.has_overlap),
            "annot_headers": headers,
        }

        if "mode" in info:
            meta["summary_y_mode"] = str(info.get("mode"))
        if "n_nonfinite" in info:
            try:
                meta["n_nonfinite_summary_y"] = int(info.get("n_nonfinite"))
            except Exception:
                pass

        cov_rank = getattr(matched, "cov_rank", None)
        if cov_rank is not None:
            try:
                meta["cov_rank"] = int(cov_rank)
            except Exception:
                pass

        cov_rank_source = getattr(matched, "cov_rank_source", None)
        if cov_rank_source is not None:
            meta["cov_rank_source"] = str(cov_rank_source)

        payload = {
            "system": system_name,
            "meta": meta,
            "sigma_names": [f"sigma_g_{k}" for k in range(K)] + ["sigma_e"],
            "moment_names": [f"score_row_{k}" for k in range(K)] + ["variance_row"],
            "full": {
                "replicate": "full",
                "T": H2ResultWriter._jsonify_numeric(T[R]),
                "q": H2ResultWriter._jsonify_numeric(q[R]),
                "sigma": H2ResultWriter._jsonify_numeric(sigma[R]),
            },
            "jackknife": [
                {
                    "replicate": int(r),
                    "T": H2ResultWriter._jsonify_numeric(T[r]),
                    "q": H2ResultWriter._jsonify_numeric(q[r]),
                    "sigma": H2ResultWriter._jsonify_numeric(sigma[r]),
                }
                for r in range(R)
            ],
        }
        return payload

    @staticmethod
    def save_score_normal_equations_json(
        fit: H2Fit,
        path: str,
        *,
        system: str | None = None,
    ):
        payload = H2ResultWriter.build_score_normal_equations_payload(fit, system=system)
        with open(path, "w") as fd:
            json.dump(payload, fd, indent=2, allow_nan=False)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _normalize_enrich_mode(mode: str | None) -> str:
    if mode is None:
        return "auto"
    s = str(mode).strip().lower()
    if s in ("auto", "overlap", "non-overlap", "nonoverlap", "both"):
        return "non-overlap" if s == "nonoverlap" else s
    raise ValueError("enrich_mode must be one of {'auto','overlap','non-overlap','both'}")


def _has_overlapping_annotations(
    A: np.ndarray,
    active_mask: np.ndarray | None = None,
    *,
    chunk_size: int = 250_000,
) -> bool:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] <= 1:
        return False

    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.ndim != 1 or active_mask.size != A.shape[0]:
            raise ValueError(
                f"active_mask must be length {A.shape[0]}; got {active_mask.shape}"
            )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    M = int(A.shape[0])
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        block = A[s:e, :]
        if active_mask is not None:
            mu = active_mask[s:e]
            if not np.any(mu):
                continue
            if not np.all(mu):
                block = block[mu, :]
        if bool(np.any(np.sum(np.abs(block) > 0.0, axis=1) > 1)):
            return True
    return False


def _make_h2_trace_metadata(trace_view) -> H2TraceMetadata:
    return H2TraceMetadata(
        nsnps=int(trace_view.nsnps),
        nbins=int(trace_view.nbins),
        annot_header=getattr(trace_view, "annot_header", None),
        delta=getattr(trace_view, "delta", None),
        kmoments=getattr(trace_view, "kmoments", None),
        kmoments_path=getattr(trace_view, "kmoments_path", None),
        kmoments_valid=bool(getattr(trace_view, "kmoments_valid", False)),
    )


def _select_h2_matrices(trace_view, ld_kind: str):
    A = np.asarray(trace_view.annot, dtype=np.float64, order="C")
    if A.ndim != 2:
        raise ValueError(f"trace_view.annot must be 2D; got {A.shape}")

    if ld_kind == "main":
        L = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")
    elif ld_kind == "reg":
        if getattr(trace_view, "ldscores_reg", None) is None:
            raise ValueError("ld_kind='reg' requested but TraceView has no ldscores_reg")
        L = np.asarray(trace_view.ldscores_reg, dtype=np.float64, order="C")
    else:
        raise ValueError("ld_kind must be 'main' or 'reg'")

    if L.ndim != 2:
        raise ValueError(f"LD-score matrix must be 2D; got {L.shape}")
    if L.shape != A.shape:
        raise ValueError(
            f"For h2, annotation and LD-score matrices must have identical shape; "
            f"got A.shape={A.shape}, L.shape={L.shape}."
        )
    return A, L


def _empty_h2_structural_stats(U: int, K: int) -> H2StructuralUnitStats:
    U = int(U)
    K = int(K)
    return H2StructuralUnitStats(
        m=np.zeros(U, dtype=np.float64),
        Ak=np.zeros((U, K), dtype=np.float64),
        Ak2=np.zeros((U, K), dtype=np.float64),
        AA=np.zeros((U, K, K), dtype=np.float64),
        AL=np.zeros((U, K, K), dtype=np.float64),
    )


def _validate_h2_structural_stats(stats: H2StructuralUnitStats, U: int, K: int, *, label: str):
    shapes = {
        "m": (U,),
        "Ak": (U, K),
        "Ak2": (U, K),
        "AA": (U, K, K),
        "AL": (U, K, K),
    }
    for name, expected in shapes.items():
        arr = np.asarray(getattr(stats, name), dtype=np.float64)
        if arr.shape != expected:
            raise ValueError(
                f"{label}.{name} has shape {arr.shape}; expected {expected}."
            )


def _subtract_h2_structural_stats(
    left: H2StructuralUnitStats,
    right: H2StructuralUnitStats,
) -> H2StructuralUnitStats:
    return H2StructuralUnitStats(
        m=np.asarray(left.m, dtype=np.float64) - np.asarray(right.m, dtype=np.float64),
        Ak=np.asarray(left.Ak, dtype=np.float64) - np.asarray(right.Ak, dtype=np.float64),
        Ak2=np.asarray(left.Ak2, dtype=np.float64) - np.asarray(right.Ak2, dtype=np.float64),
        AA=np.asarray(left.AA, dtype=np.float64) - np.asarray(right.AA, dtype=np.float64),
        AL=np.asarray(left.AL, dtype=np.float64) - np.asarray(right.AL, dtype=np.float64),
    )


def _compute_h2_structural_from_rows(
    A: np.ndarray,
    L: np.ndarray,
    jackknife,
    idx: np.ndarray,
) -> H2StructuralUnitStats:
    A = np.asarray(A, dtype=np.float64, order="C")
    L = np.asarray(L, dtype=np.float64, order="C")
    if A.shape != L.shape:
        raise ValueError(f"A/L shape mismatch: {A.shape} vs {L.shape}")

    U = int(jackknife.nunit)
    K = int(A.shape[1])
    out = _empty_h2_structural_stats(U, K)

    idx = np.asarray(idx, dtype=np.int64).ravel()
    if idx.size == 0:
        return out
    if np.any((idx < 0) | (idx >= A.shape[0])):
        bad = idx[(idx < 0) | (idx >= A.shape[0])][:10].tolist()
        raise ValueError(f"row index out of range; first bad indices: {bad}")

    unit_id = np.asarray(jackknife.unit_id, dtype=np.int64)
    if unit_id.shape != (A.shape[0],):
        raise ValueError(
            f"jackknife.unit_id must have length {A.shape[0]}; got {unit_id.shape}"
        )

    uids = unit_id[idx]
    for u in np.unique(uids):
        rows = idx[uids == u]
        if rows.size == 0:
            continue
        Au = A[rows, :]
        Lu = L[rows, :]
        out.m[u] = float(rows.size)
        out.Ak[u] = Au.sum(axis=0, dtype=np.float64)
        out.Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        out.AA[u] = Au.T @ Au
        out.AL[u] = Au.T @ Lu
    return out


def compute_h2_structural_unit_stats(
    trace_view,
    jackknife,
    *,
    active_mask: np.ndarray | None = None,
    ld_kind: str = "main",
) -> H2StructuralUnitStats:
    if jackknife.nsnps != int(trace_view.nsnps):
        raise ValueError(
            "JackknifeDesign was not built on this trace axis. "
            f"Expected {trace_view.nsnps}, got {jackknife.nsnps}."
        )

    A, L = _select_h2_matrices(trace_view, ld_kind)
    M = int(A.shape[0])
    U = int(jackknife.nunit)
    K = int(A.shape[1])

    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.ndim != 1 or active_mask.size != M:
            raise ValueError(f"active_mask must be length {M}; got {active_mask.shape}")

    out = _empty_h2_structural_stats(U, K)
    for u in range(U):
        s = int(jackknife.starts[u])
        e = int(jackknife.ends[u])
        if e <= s:
            continue

        if active_mask is None:
            Au = A[s:e, :]
            Lu = L[s:e, :]
        else:
            mu = active_mask[s:e]
            if not np.any(mu):
                continue
            if np.all(mu):
                Au = A[s:e, :]
                Lu = L[s:e, :]
            else:
                rows = s + np.flatnonzero(mu)
                Au = A[rows, :]
                Lu = L[rows, :]

        out.m[u] = float(Au.shape[0])
        out.Ak[u] = Au.sum(axis=0, dtype=np.float64)
        out.Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        out.AA[u] = Au.T @ Au
        out.AL[u] = Au.T @ Lu
    return out


def _compute_h2_ay_unit(
    A: np.ndarray,
    jackknife,
    y: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)

    if y.shape != (A.shape[0],):
        raise ValueError(f"y must have shape ({A.shape[0]},); got {y.shape}")
    if active_mask.shape != (A.shape[0],):
        raise ValueError(
            f"active_mask must have shape ({A.shape[0]},); got {active_mask.shape}"
        )

    U = int(jackknife.nunit)
    K = int(A.shape[1])
    Ay = np.zeros((U, K), dtype=np.float64)

    for u in range(U):
        s = int(jackknife.starts[u])
        e = int(jackknife.ends[u])
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        yu = y[s:e]
        if np.all(mu):
            Ay[u] = A[s:e, :].T @ yu
        else:
            y_work = np.zeros(e - s, dtype=np.float64)
            y_work[mu] = yu[mu]
            Ay[u] = A[s:e, :].T @ y_work
    return Ay


def _solve_linear_batch(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.ndim != 3 or rhs.ndim != 2:
        raise ValueError("lhs must be (R+1,p,p) and rhs must be (R+1,p)")
    if lhs.shape[0] != rhs.shape[0] or lhs.shape[1] != lhs.shape[2] or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("lhs/rhs shape mismatch")

    out = np.full(rhs.shape, np.nan, dtype=np.float64)
    for i in range(lhs.shape[0]):
        A = lhs[i]
        b = rhs[i]
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            continue
        try:
            out[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                out[i] = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
    return out


def _symmetrize_with_design(trace_KK: np.ndarray, jackknife, unit_sizes: np.ndarray) -> np.ndarray:
    """
    Symmetrize replicate trace matrices using the current jackknife design.

    Prefer the newer utils API that takes the explicit delete-incidence matrix and
    per-unit active sizes. Fall back to the older shorthand API for backward
    compatibility.
    """
    unit_sizes = np.asarray(unit_sizes, dtype=np.float64)
    try:
        return utils.symmetrize_trace_with_jackknife(
            trace_KK,
            logger=None,
            verbose=False,
            jk_delete_matrix=np.asarray(jackknife.D, dtype=np.float64, order="C"),
            jk_unit_sizes=unit_sizes,
            jk_delete_d=int(jackknife.delete),
        )
    except TypeError:
        if jackknife.mode == "block":
            return utils.symmetrize_trace_with_jackknife(
                trace_KK,
                logger=None,
                verbose=False,
                jk_block_sizes=unit_sizes,
                jk_delete_d=1,
            )
        return utils.symmetrize_trace_with_jackknife(
            trace_KK,
            logger=None,
            verbose=False,
            jk_n_units=int(jackknife.nunit),
            jk_delete_d=int(jackknife.delete),
        )


def _stack_delete_replicates(full: np.ndarray, unit: np.ndarray, D: np.ndarray) -> np.ndarray:
    full = np.asarray(full, dtype=np.float64)
    unit = np.asarray(unit, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if unit.ndim < 1:
        raise ValueError("unit must have at least one dimension")
    if unit.shape[0] != D.shape[1]:
        raise ValueError("unit first axis must match D.shape[1]")

    tail_shape = unit.shape[1:]
    flat_unit = unit.reshape(unit.shape[0], -1)           # (U,P)
    flat_full = full.reshape(-1)                          # (P,)
    deleted = D @ flat_unit                               # (R,P)
    reps = flat_full[None, :] - deleted                   # (R,P)
    reps = reps.reshape((D.shape[0],) + tail_shape)
    return np.concatenate([reps, full[None, ...]], axis=0)


def _pair_correction_from_deleted_mass(A_keep, A_del, delta):
    A_keep = np.asarray(A_keep, dtype=np.float64)
    A_del = np.asarray(A_del, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    K = int(A_keep.shape[1])

    if d.ndim == 0:
        return (A_keep[:, :, None] * A_del[:, None, :]) * float(d)
    if d.ndim == 1:
        if d.size != K:
            raise ValueError(f"delta vector has length {d.size}, expected K={K}")
        return A_keep[:, :, None] * (A_del * d[None, :])[:, None, :]
    if d.ndim == 2:
        if d.shape != (K, K):
            raise ValueError(f"delta matrix has shape {d.shape}, expected ({K},{K})")
        return (A_keep[:, :, None] * A_del[:, None, :]) * d[None, :, :]
    raise ValueError("delta must be scalar, (K,), or (K,K)")


# -----------------------------------------------------------------------------
# preparation
# -----------------------------------------------------------------------------


def _finish_prepare_h2(
    *,
    trace_view,
    matched,
    jackknife,
    active_mask: np.ndarray,
    y: np.ndarray,
    struct: H2StructuralUnitStats,
    Ay_unit: np.ndarray,
    n_scale: float,
    summary_y_info: dict | None,
    has_overlap: bool,
    adjust_delta: bool,
    store_y: bool = True,
) -> H2Prepared:
    K = int(trace_view.nbins)
    R = int(jackknife.nrep)
    U = int(jackknife.nunit)

    _validate_h2_structural_stats(struct, U, K, label="struct")
    Ay_unit = np.asarray(Ay_unit, dtype=np.float64)
    if Ay_unit.shape != (U, K):
        raise ValueError(f"Ay_unit has shape {Ay_unit.shape}; expected {(U, K)}")

    m_unit = np.asarray(struct.m, dtype=np.float64)
    Ak_unit = np.asarray(struct.Ak, dtype=np.float64)
    Ak2_unit = np.asarray(struct.Ak2, dtype=np.float64)
    AA_unit = np.asarray(struct.AA, dtype=np.float64)
    AL_unit = np.asarray(struct.AL, dtype=np.float64)

    M_full = float(m_unit.sum())
    if not (np.isfinite(M_full) and M_full > 0.0):
        raise RuntimeError("No active SNPs remain for H2 preparation.")

    unit_sizes = jackknife.unit_sizes(active_mask=active_mask, dtype=np.float64)

    M_rep = _stack_delete_replicates(np.array(M_full, dtype=np.float64), m_unit, jackknife.D).reshape(R + 1)
    Ak_rep = _stack_delete_replicates(Ak_unit.sum(axis=0), Ak_unit, jackknife.D)
    Ay_rep = _stack_delete_replicates(Ay_unit.sum(axis=0), Ay_unit, jackknife.D)
    Ak2_rep = _stack_delete_replicates(Ak2_unit.sum(axis=0), Ak2_unit, jackknife.D)
    AA_rep = _stack_delete_replicates(AA_unit.sum(axis=0), AA_unit, jackknife.D)
    AL_rep = _stack_delete_replicates(AL_unit.sum(axis=0), AL_unit, jackknife.D)

    if jackknife.mode == "chr" and adjust_delta and getattr(trace_view, "delta", None) is not None:
        delta = np.asarray(trace_view.delta, dtype=np.float64)
        del_Ak = jackknife.D @ Ak_unit
        rep_Ak = Ak_rep[:R]
        AL_rep = AL_rep.copy()
        AL_rep[:R] -= _pair_correction_from_deleted_mass(rep_Ak, del_Ak, delta)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    src_mass = np.broadcast_to(Ak_full[None, :], Ak_rep.shape).copy()
    if jackknife.mode == "chr" and adjust_delta and getattr(trace_view, "delta", None) is not None:
        src_mass[:R] = Ak_rep[:R]

    M_k = Ak_rep[:, :, None]
    M_l = src_mass[:, None, :]
    delta = np.asarray(trace_view.delta, dtype=np.float64) if (adjust_delta and getattr(trace_view, "delta", None) is not None) else None
    trace_KK = utils._calc_trace_from_ld_batch(AL_rep, n_scale, M_k, M_l, delta=delta)

    trace_KK = _symmetrize_with_design(trace_KK, jackknife, unit_sizes)

    lhs = np.full((R + 1, K + 1, K + 1), n_scale, dtype=np.float64)
    lhs[:, :K, :K] = trace_KK
    lhs[:, K, K] = n_scale

    rhs = np.full((R + 1, K + 1), n_scale, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs[:, :K] = (Ay_rep * n_scale) / Ak_rep
    bad_rhs = (~np.isfinite(rhs[:, :K])) | (~np.isfinite(Ak_rep)) | (Ak_rep <= 0.0)
    rhs[:, :K][bad_rhs] = np.nan

    bad_bins = np.flatnonzero(~np.isfinite(Ak_rep[-1]) | (Ak_rep[-1] <= 0.0))
    if bad_bins.size:
        raise RuntimeError(
            "One or more full-sample bins have non-positive total weight after matching. "
            f"Bad bins: {bad_bins.tolist()}"
        )

    y_store = np.asarray(y, dtype=np.float64) if store_y else np.empty(0, dtype=np.float64)

    return H2Prepared(
        trace_view=trace_view,
        matched=matched,
        jackknife=jackknife,
        active_mask=active_mask,
        has_overlap=has_overlap,
        unit_sizes=unit_sizes,
        n_scale=n_scale,
        summary_y_info=summary_y_info,
        y=y_store,
        m_unit=m_unit,
        Ak_unit=Ak_unit,
        Ay_unit=Ay_unit,
        Ak2_unit=Ak2_unit,
        AA_unit=AA_unit,
        AL_unit=AL_unit,
        M_rep=M_rep,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        Ak2_rep=Ak2_rep,
        AA_rep=AA_rep,
        AL_rep=AL_rep,
        lhs=lhs,
        rhs=rhs,
    )


def prepare_h2(
    trace_view,
    matched,
    jackknife,
    *,
    active_mask=None,
    summary_y=None,
    summary_y_info=None,
    ld_kind: str = "main",
    adjust_delta: bool = False,
):
    if trace_view.nsnps != matched.nsnps:
        raise ValueError("TraceView and MatchedSumstats must have the same number of SNPs.")
    if not np.array_equal(trace_view.snps, matched.snps):
        raise ValueError("TraceView and MatchedSumstats SNP order mismatch.")
    if jackknife.nsnps != trace_view.nsnps:
        raise ValueError(
            "JackknifeDesign was not built on this TraceView SNP axis. "
            f"Expected {jackknife.nsnps}, got {trace_view.nsnps}."
        )

    M = trace_view.nsnps
    K = trace_view.nbins
    R = jackknife.nrep
    U = jackknife.nunit

    if summary_y is None:
        summary_y, summary_y_info = build_h2_summary_moment(matched)

    y = np.asarray(summary_y, dtype=np.float64)
    if y.shape != (M,):
        raise ValueError(f"summary_y must have shape ({M},), got {y.shape}")

    if summary_y_info is not None:
        n_scale = float(summary_y_info.get("n_scale", getattr(matched, "n_scale", matched.nsamp)))
    else:
        n_scale = float(getattr(matched, "n_scale", matched.nsamp))

    if not (np.isfinite(n_scale) and n_scale > 0.0):
        raise RuntimeError(f"Invalid univariate n_scale={n_scale}")

    if active_mask is None:
        active_mask = np.isfinite(y)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.ndim != 1 or active_mask.size != M:
            raise ValueError(f"active_mask must be length {M}; got {active_mask.shape}")
        active_mask = active_mask & np.isfinite(y)

    A, _ = _select_h2_matrices(trace_view, ld_kind)
    struct = compute_h2_structural_unit_stats(
        trace_view,
        jackknife,
        active_mask=active_mask,
        ld_kind=ld_kind,
    )
    Ay_unit = _compute_h2_ay_unit(A, jackknife, y, active_mask)
    has_overlap = _has_overlapping_annotations(A, active_mask=active_mask)

    return _finish_prepare_h2(
        trace_view=trace_view,
        matched=matched,
        jackknife=jackknife,
        active_mask=active_mask,
        y=y,
        struct=struct,
        Ay_unit=Ay_unit,
        n_scale=n_scale,
        summary_y_info=summary_y_info,
        has_overlap=has_overlap,
        adjust_delta=adjust_delta,
        store_y=True,
    )


def prepare_h2_reference_axis(
    trace_view,
    matched,
    jackknife,
    keep_mask,
    *,
    summary_y,
    summary_y_info=None,
    full_struct: H2StructuralUnitStats | None = None,
    ld_kind: str = "main",
    adjust_delta: bool = False,
    prefer_drop_correction: bool = True,
):
    """
    Prepare h2 normal equations on the immutable reference SNP axis.

    This is algebraically equivalent to materializing a compact TraceView on
    keep_mask and then aggregating, but it avoids copying the dense annotation
    and LD-score matrices.  Jackknife units are those in the supplied reference
    jackknife design, and unit weights are counted after the trait-specific drop.
    """
    M = int(trace_view.nsnps)
    K = int(trace_view.nbins)
    U = int(jackknife.nunit)

    if jackknife.nsnps != M:
        raise ValueError(
            "JackknifeDesign was not built on this reference trace axis. "
            f"Expected {M}, got {jackknife.nsnps}."
        )

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.ndim != 1 or keep_mask.size != M:
        raise ValueError(f"keep_mask must be length {M}; got {keep_mask.shape}")

    y = np.asarray(summary_y, dtype=np.float64)
    if y.shape != (M,):
        raise ValueError(f"summary_y must have shape ({M},), got {y.shape}")

    if summary_y_info is not None:
        n_scale = float(summary_y_info.get("n_scale", getattr(matched, "n_scale", matched.nsamp)))
    else:
        n_scale = float(getattr(matched, "n_scale", matched.nsamp))

    if not (np.isfinite(n_scale) and n_scale > 0.0):
        raise RuntimeError(f"Invalid univariate n_scale={n_scale}")

    active_mask = keep_mask & np.isfinite(y)
    active_n = int(np.sum(active_mask))
    if active_n <= 0:
        raise RuntimeError("No active SNPs remain for H2 preparation.")

    A, L = _select_h2_matrices(trace_view, ld_kind)
    if A.shape != (M, K):
        raise RuntimeError(f"Unexpected annotation shape {A.shape}; expected {(M, K)}")

    if full_struct is not None:
        _validate_h2_structural_stats(full_struct, U, K, label="full_struct")

    use_drop_correction = False
    if prefer_drop_correction and full_struct is not None:
        n_drop = M - active_n
        use_drop_correction = n_drop <= active_n

    if use_drop_correction:
        n_drop = M - active_n
        if n_drop == 0:
            struct = full_struct
        else:
            drop_idx = np.flatnonzero(~active_mask).astype(np.int64, copy=False)
            corr = _compute_h2_structural_from_rows(A, L, jackknife, drop_idx)
            struct = _subtract_h2_structural_stats(full_struct, corr)
    else:
        struct = compute_h2_structural_unit_stats(
            trace_view,
            jackknife,
            active_mask=active_mask,
            ld_kind=ld_kind,
        )

    Ay_unit = _compute_h2_ay_unit(A, jackknife, y, active_mask)
    has_overlap = _has_overlapping_annotations(A, active_mask=active_mask)

    trace_meta = _make_h2_trace_metadata(trace_view)

    return _finish_prepare_h2(
        trace_view=trace_meta,
        matched=matched,
        jackknife=jackknife,
        active_mask=active_mask,
        y=y,
        struct=struct,
        Ay_unit=Ay_unit,
        n_scale=n_scale,
        summary_y_info=summary_y_info,
        has_overlap=has_overlap,
        adjust_delta=adjust_delta,
        store_y=False,
    )


def prepare_h2_reference_axis_from_sufficient_stats(
    trace_view,
    matched,
    jackknife,
    active_mask,
    *,
    Ay_unit,
    n_scale: float,
    summary_y_info=None,
    full_struct: H2StructuralUnitStats | None = None,
    ld_kind: str = "main",
    adjust_delta: bool = False,
    prefer_drop_correction: bool = True,
    has_overlap: bool | None = None,
):
    """Prepare h2 from exact full-axis sufficient statistics.

    This is the batch-oriented counterpart of :func:`prepare_h2_reference_axis`.
    ``Ay_unit`` must equal the per-jackknife-unit sums of ``A.T @ y`` on
    ``active_mask``.  The structural terms are computed here using the same
    sparse-drop correction as the regular reference-axis path, and the same
    normal-equation builder and fitter are used downstream.

    The function intentionally accepts no approximate or compacted LD input.
    Callers are responsible for constructing ``Ay_unit`` in float64 on the
    immutable Trace SNP axis.
    """
    M = int(trace_view.nsnps)
    K = int(trace_view.nbins)
    U = int(jackknife.nunit)

    if jackknife.nsnps != M:
        raise ValueError(
            "JackknifeDesign was not built on this reference trace axis. "
            f"Expected {M}, got {jackknife.nsnps}."
        )

    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.ndim != 1 or active_mask.size != M:
        raise ValueError(f"active_mask must be length {M}; got {active_mask.shape}")
    active_n = int(np.sum(active_mask))
    if active_n <= 0:
        raise RuntimeError("No active SNPs remain for H2 preparation.")

    Ay_unit = np.asarray(Ay_unit, dtype=np.float64)
    if Ay_unit.shape != (U, K):
        raise ValueError(f"Ay_unit has shape {Ay_unit.shape}; expected {(U, K)}")

    n_scale = float(n_scale)
    if not (np.isfinite(n_scale) and n_scale > 0.0):
        raise RuntimeError(f"Invalid univariate n_scale={n_scale}")

    A, L = _select_h2_matrices(trace_view, ld_kind)
    if A.shape != (M, K):
        raise RuntimeError(f"Unexpected annotation shape {A.shape}; expected {(M, K)}")

    if full_struct is not None:
        _validate_h2_structural_stats(full_struct, U, K, label="full_struct")

    use_drop_correction = False
    if prefer_drop_correction and full_struct is not None:
        n_drop = M - active_n
        use_drop_correction = n_drop <= active_n

    if use_drop_correction:
        n_drop = M - active_n
        if n_drop == 0:
            struct = full_struct
        else:
            drop_idx = np.flatnonzero(~active_mask).astype(np.int64, copy=False)
            corr = _compute_h2_structural_from_rows(A, L, jackknife, drop_idx)
            struct = _subtract_h2_structural_stats(full_struct, corr)
    else:
        struct = compute_h2_structural_unit_stats(
            trace_view,
            jackknife,
            active_mask=active_mask,
            ld_kind=ld_kind,
        )

    if has_overlap is None:
        has_overlap = _has_overlapping_annotations(A, active_mask=active_mask)

    return _finish_prepare_h2(
        trace_view=_make_h2_trace_metadata(trace_view),
        matched=matched,
        jackknife=jackknife,
        active_mask=active_mask,
        y=np.empty(0, dtype=np.float64),
        struct=struct,
        Ay_unit=Ay_unit,
        n_scale=n_scale,
        summary_y_info=summary_y_info,
        has_overlap=bool(has_overlap),
        adjust_delta=adjust_delta,
        store_y=False,
    )


# -----------------------------------------------------------------------------
# fitting
# -----------------------------------------------------------------------------


def fit_h2(
    prepared: H2Prepared,
    *,
    enrich_mode: str = "auto",
    report_tau: bool = True,
    allow_neg_enr: bool = False,
    clip_nonfinite_vals: bool = False,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
) -> H2Fit:
    enrich_mode = _normalize_enrich_mode(enrich_mode)

    p = prepared
    R = p.jackknife.nrep
    K = p.trace_view.nbins

    sigma_core = _solve_linear_batch(p.lhs, p.rhs)                 # (R+1, K+1)
    sigma_reps = np.zeros((R + 1, K + 2), dtype=np.float64)
    sigma_reps[:, :K + 1] = sigma_core
    sigma_reps[:, K + 1] = np.nansum(sigma_core[:, :K], axis=1)

    sigma_g = sigma_core[:, :K]
    h2_tot = np.nansum(sigma_g, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = sigma_g / p.Ak_rep
        h2_cat_overlap = np.einsum("bkl,bl->bk", p.AA_rep, w, optimize=True)

    bad_overlap = ~np.isfinite(h2_cat_overlap)
    if clip_nonfinite_vals:
        h2_cat_overlap[bad_overlap] = 0.0
    else:
        h2_cat_overlap[bad_overlap] = np.nan

    h2_reps = np.zeros((R + 1, K + 1), dtype=np.float64)
    h2_reps[:, :K] = h2_cat_overlap
    h2_reps[:, K] = h2_tot

    with np.errstate(divide="ignore", invalid="ignore"):
        prop = p.Ak_rep / p.M_rep[:, None]

    requested = enrich_mode
    if requested == "auto":
        enrich_mode_used = "overlap" if p.has_overlap else "non-overlap"
        modes_to_compute = (enrich_mode_used,)
    elif requested == "both":
        enrich_mode_used = "overlap" if p.has_overlap else "non-overlap"
        modes_to_compute = ("non-overlap", "overlap")
    else:
        enrich_mode_used = requested
        modes_to_compute = (requested,)

    def _compute_enrichment(h2_cat):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (h2_cat / h2_tot[:, None]) / prop
            invalid = (~np.isfinite(out)) | (~np.isfinite(prop)) | (prop <= 0.0)
            if not allow_neg_enr:
                invalid |= (h2_tot[:, None] <= 0.0)
            out[invalid] = np.nan
        return out

    enrich_overlap_reps = None
    enrich_nonoverlap_reps = None
    if "overlap" in modes_to_compute:
        enrich_overlap_reps = _compute_enrichment(h2_cat_overlap)
    if "non-overlap" in modes_to_compute:
        enrich_nonoverlap_reps = _compute_enrichment(sigma_g)

    if requested == "both":
        enrich_reps = enrich_overlap_reps if enrich_mode_used == "overlap" else enrich_nonoverlap_reps
    else:
        enrich_reps = enrich_overlap_reps if enrich_mode_used == "overlap" else enrich_nonoverlap_reps

    tau_reps = None
    tau_star_reps = None
    if report_tau:
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_reps = sigma_g / p.Ak_rep
            meanA = p.Ak_rep / p.M_rep[:, None]
            meanA2 = p.Ak2_rep / p.M_rep[:, None]
            varA = np.maximum(meanA2 - meanA * meanA, 0.0)
            sdA = np.sqrt(varA, dtype=np.float64)
            denom = h2_tot / p.M_rep
            tau_star_reps = tau_reps * (sdA / denom[:, None])

        bad_tau = ~np.isfinite(tau_reps)
        bad_tau_star = ~np.isfinite(tau_star_reps)
        if clip_nonfinite_vals:
            tau_reps[bad_tau] = 0.0
            tau_star_reps[bad_tau_star] = 0.0
        else:
            tau_reps[bad_tau] = np.nan
            tau_star_reps[bad_tau_star] = np.nan

    est, se = p.jackknife.summarize(
        sigma_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    sigmas = np.column_stack([est, se])

    est, se = p.jackknife.summarize(
        h2_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    h2 = np.column_stack([est, se])

    est, se = p.jackknife.summarize(
        enrich_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    enrich = np.column_stack([est, se])

    enrich_overlap = None
    if enrich_overlap_reps is not None:
        est, se = p.jackknife.summarize(
            enrich_overlap_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        enrich_overlap = np.column_stack([est, se])

    enrich_nonoverlap = None
    if enrich_nonoverlap_reps is not None:
        est, se = p.jackknife.summarize(
            enrich_nonoverlap_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        enrich_nonoverlap = np.column_stack([est, se])

    tau = None
    tau_star = None
    if report_tau:
        est, se = p.jackknife.summarize(
            tau_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        tau = np.column_stack([est, se])

        est, se = p.jackknife.summarize(
            tau_star_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        tau_star = np.column_stack([est, se])

    return H2Fit(
        prepared=prepared,
        sigma_reps=sigma_reps,
        h2_reps=h2_reps,
        enrich_reps=enrich_reps,
        enrich_overlap_reps=enrich_overlap_reps,
        enrich_nonoverlap_reps=enrich_nonoverlap_reps,
        tau_reps=tau_reps,
        tau_star_reps=tau_star_reps,
        sigmas=sigmas,
        h2=h2,
        enrich=enrich,
        enrich_overlap=enrich_overlap,
        enrich_nonoverlap=enrich_nonoverlap,
        tau=tau,
        tau_star=tau_star,
        enrich_mode_requested=requested,
        enrich_mode_used=enrich_mode_used,
    )
