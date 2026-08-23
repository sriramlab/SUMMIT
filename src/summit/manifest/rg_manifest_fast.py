from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .. import utils
from ..inference.h2core import (
    H2Prepared,
    fit_h2,
    _has_overlapping_annotations,
    _stack_delete_replicates as _stack_h2,
    _symmetrize_with_design as _sym_h2,
)
from ..inference.jackknife import JackknifeDesign, JackknifeSpec
from ..sumstats.moments import exact_score_z_from_arrays, effective_n_scale
from ..inference.rgcore import (
    RGPrepared,
    InterceptFit,
    RGResultWriter,
    build_manifest_summary_row,
    fit_rg,
    _stack_delete_replicates as _stack_rg,
    _symmetrize_with_design as _sym_rg,
    _make_intercept_keep_mask,
    _build_simple_intercept_weights,
    _compute_intercept_unit_summaries,
    _compute_weighted_intercept_unit_summaries,
    _solve_constrained_intercept_from_sums,
    _full_intercept_denominator,
    _intercept_gamma_total_from_beta,
    _component_rg,
)
from ..sumstats.sumstats import Sumstats, harmonize_allele_codes
from ..inference.trace import Trace


@dataclass
class _FastTrait:
    phen: str
    spath: str
    nsamp: float
    n_scale: float
    cov_rank: int
    cov_rank_source: str
    keep: np.ndarray
    drop_idx: np.ndarray
    z_h2: np.ndarray
    z_rg: np.ndarray
    a1_code: np.ndarray
    a2_code: np.ndarray
    h2_ay_unit: np.ndarray | None
    matched_stub: object


@dataclass(frozen=True)
class _FastMatchedMetadata:
    nsnps: int
    nsamp: float
    n_scale: float
    cov_rank: int
    cov_rank_source: str
    name: str


@dataclass
class _StructUnitStats:
    m: np.ndarray
    Ak: np.ndarray
    Ak2: np.ndarray
    AA: np.ndarray
    AL: np.ndarray


@dataclass(frozen=True)
class _FastModelSpec:
    name: str
    slug: str
    indices: np.ndarray
    bins: tuple[str, ...]
    aliases: tuple[str, ...]


def _sanitize_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_") or "model"


def _parse_model_columns(value, *, field: str, model: str) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Model '{model}' has an empty {field} field.")
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model '{model}' has invalid JSON in {field}: {exc}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"Model '{model}' {field} JSON must be a list of strings.")
        items = tuple(item.strip() for item in parsed)
    else:
        items = tuple(item.strip() for item in text.split(","))
    if not items or any(not item for item in items):
        raise ValueError(f"Model '{model}' has an empty entry in {field}.")
    if len(set(items)) != len(items):
        raise ValueError(f"Model '{model}' has duplicate entries in {field}: {items}.")
    return items


def _load_fast_model_specs(path, annot_header) -> list[_FastModelSpec]:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"model", "bins"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"RG model manifest '{path}' is missing required column(s): {sorted(missing)}."
        )
    if table.empty:
        raise ValueError(f"RG model manifest '{path}' contains no models.")

    headers = tuple(str(item) for item in np.asarray(annot_header).tolist())
    if len(set(headers)) != len(headers):
        raise ValueError("Union annotation headers must be unique for multi-model RG.")
    header_index = {header: index for index, header in enumerate(headers)}

    specs: list[_FastModelSpec] = []
    names: set[str] = set()
    slugs: set[str] = set()
    common_aliases: tuple[str, ...] | None = None
    for row in table.itertuples(index=False):
        model = str(row.model).strip()
        if not model:
            raise ValueError("RG model manifest contains an empty model name.")
        if model in names:
            raise ValueError(f"RG model manifest contains duplicate model '{model}'.")
        names.add(model)
        slug = _sanitize_model_name(model)
        if slug in slugs:
            raise ValueError(
                f"RG model names are not unique after path sanitization; duplicate slug '{slug}'."
            )
        slugs.add(slug)

        bins = _parse_model_columns(row.bins, field="bins", model=model)
        alias_value = getattr(row, "aliases", "")
        aliases = (
            bins
            if not str(alias_value).strip()
            else _parse_model_columns(alias_value, field="aliases", model=model)
        )
        if len(aliases) != len(bins):
            raise ValueError(
                f"Model '{model}' has {len(bins)} bins but {len(aliases)} aliases."
            )
        unknown = [item for item in bins if item not in header_index]
        if unknown:
            raise ValueError(
                f"Model '{model}' requests annotation bins absent from the union trace: {unknown}."
            )
        if common_aliases is None:
            common_aliases = aliases
        elif aliases != common_aliases:
            raise ValueError(
                "All RG models must use the same ordered aliases so manifest.results.tsv "
                "has a compact, stable schema."
            )
        specs.append(
            _FastModelSpec(
                name=model,
                slug=slug,
                indices=np.asarray([header_index[item] for item in bins], dtype=np.int64),
                bins=bins,
                aliases=aliases,
            )
        )
    return specs


class _FastTraceView:
    def __init__(self, *, nsnps: int, nbins: int, annot_header):
        self._nsnps = int(nsnps)
        self._nbins = int(nbins)
        self.annot_header = annot_header
        self.kmoments = None
        self.kmoments_valid = False

    @property
    def nsnps(self) -> int:
        return self._nsnps

    @property
    def nbins(self) -> int:
        return self._nbins


def _log(log, msg: str):
    if log is not None:
        log._log(msg)


def _write_fast_pair_log(
    pair_prefix: str,
    *,
    phen1: str,
    phen2: str,
    annot_header,
    h2_fit1,
    h2_fit2,
    intercept,
    rg_fit,
    runtime_s: float,
):
    lines: list[str] = []

    def add(msg: str):
        lines.append(msg)

    add(f"[rg:manifest:fast] pair: {phen1} vs {phen2}")

    km_info = getattr(rg_fit, "kmoment_info", None)
    if km_info is not None:
        add(
            "[rg:kmom] "
            f"single-component model-based SE used; "
            f"moment_source={km_info.get('moment_source', 'NA')}, "
            f"alpha_probe={km_info.get('alpha_probe', np.nan):.6g}, "
            f"alpha_probe_err={km_info.get('alpha_probe_err', np.nan):.3e}, "
            f"delta_reff={km_info.get('delta_reff', np.nan):.6g}, "
            f"var_gamma={km_info.get('var_gamma', np.nan):.6g}"
        )

    for name, fit in ((phen1, h2_fit1), (phen2, h2_fit2)):
        if fit.enrich_mode_used:
            add(f"^^^ Phenotype [{name}] enrichment_mode_used: {fit.enrich_mode_used}")

        if len(annot_header) > 1:
            for j, header in enumerate(annot_header):
                line = (
                    f"^^^ Phenotype [{name}] Bin [{header}] "
                    f"sigma_g^2: {fit.sigmas[j, 0]:.6g} (SE: {fit.sigmas[j, 1]:.6g}) "
                    f"h^2_cat: {fit.h2[j, 0]:.6g} (SE: {fit.h2[j, 1]:.6g}) "
                    f"Enrichment: {fit.enrich[j, 0]:.6g} (SE: {fit.enrich[j, 1]:.6g})"
                )
                if fit.enrich_nonoverlap is not None and fit.enrich_overlap is not None:
                    line += (
                        f" Enrichment_nonoverlap: {fit.enrich_nonoverlap[j, 0]:.6g} "
                        f"(SE: {fit.enrich_nonoverlap[j, 1]:.6g})"
                        f" Enrichment_overlap: {fit.enrich_overlap[j, 0]:.6g} "
                        f"(SE: {fit.enrich_overlap[j, 1]:.6g})"
                    )
                if fit.tau is not None and fit.tau_star is not None:
                    line += (
                        f" tau: {fit.tau[j, 0]:.6g} (SE: {fit.tau[j, 1]:.6g})"
                        f" tau_*: {fit.tau_star[j, 0]:.6g} (SE: {fit.tau_star[j, 1]:.6g})"
                    )
                add(line)

        add(
            f"^^^ Phenotype [{name}] Total SNP heritability (h^2): "
            f"{fit.h2[-1, 0]:.6g} SE: {fit.h2[-1, 1]:.6g}"
        )

    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Sample-overlap covariance (c_ov): {intercept.c[0]:.9g} "
        f"(SE: {intercept.c[1]:.6g})"
    )

    for j, header in enumerate(annot_header):
        add(
            f"^^^ Phenotype [{phen1}] & [{phen2}] Bin [{header}] "
            f"gamma_g: {rg_fit.gamma[j, 0]:.6g} (SE: {rg_fit.gamma[j, 1]:.6g}) "
            f"rg: {rg_fit.rg[j, 0]:.6g} (SE: {rg_fit.rg[j, 1]:.6g})"
        )

    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Total genetic covariance (gamma_g): {rg_fit.gamma_total[0]:.6g} "
        f"(SE: {rg_fit.gamma_total[1]:.6g})"
    )
    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Total genetic correlation (rg): {rg_fit.rg_total[0]:.6g} "
        f"(SE: {rg_fit.rg_total[1]:.6g})"
    )

    end_time = utils._get_time()
    add("Analysis ended at: " + utils._get_timestr(end_time))
    add("run time: " + format(float(runtime_s), ".3f") + " s")
    add("Saved log in " + pair_prefix + ".log")

    with open(pair_prefix + ".log", "w") as fd:
        for line in lines:
            fd.write(line + "\n")


def _full_axis_sumstats_arrays(entry, trace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned = entry.aligned
    ss = entry.sumstats
    pos = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    M = int(trace.nsnps)

    beta = np.full(M, np.nan, dtype=np.float64)
    se = np.full(M, np.nan, dtype=np.float64)
    n = np.full(M, np.nan, dtype=np.float64)

    matched = pos >= 0
    if np.any(matched):
        p = pos[matched]
        beta[matched] = ss.beta[p]
        se[matched] = ss.se[p]
        n[matched] = ss.n[p]

    return beta, se, n


def _full_axis_z_array(aligned, trace) -> np.ndarray:
    ss = aligned.sumstats
    pos = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    out = np.full(int(trace.nsnps), np.nan, dtype=np.float64)
    matched = pos >= 0
    if np.any(matched):
        out[matched] = ss.z[pos[matched]]
    return out


def _full_axis_allele_codes(aligned, trace) -> tuple[np.ndarray, np.ndarray]:
    ss = aligned.sumstats
    pos = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    a1 = np.full(int(trace.nsnps), -1, dtype=np.int8)
    a2 = np.full(int(trace.nsnps), -1, dtype=np.int8)
    matched = pos >= 0
    if np.any(matched):
        p = pos[matched]
        a1[matched] = ss.a1_code[p]
        a2[matched] = ss.a2_code[p]
    return a1, a2


def _make_matched_stub(trace, trait: Sumstats) -> _FastMatchedMetadata:
    # The sparse fast path constructs the normal-equation summaries directly on
    # the shared trace axis.  Downstream fitters and writers only need metadata;
    # retaining per-trait SNP/string/Z arrays here is both unnecessary and
    # prohibitive for whole-genome imputed runs.
    return _FastMatchedMetadata(
        nsnps=int(trace.nsnps),
        nsamp=float(trait.nsamp),
        n_scale=float(trait.n_scale),
        cov_rank=int(trait.cov_rank),
        cov_rank_source=str(trait.cov_rank_source),
        name=str(trait.name),
    )


def _compute_struct_unit_stats(A: np.ndarray, L: np.ndarray, jk: JackknifeDesign, *, log=None) -> _StructUnitStats:
    U = int(jk.nunit)
    K = int(A.shape[1])
    m = np.zeros(U, dtype=np.float64)
    Ak = np.zeros((U, K), dtype=np.float64)
    Ak2 = np.zeros((U, K), dtype=np.float64)
    AA = np.zeros((U, K, K), dtype=np.float64)
    AL = np.zeros((U, K, K), dtype=np.float64)

    t0 = time.time()
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        Au = A[s:e, :]
        Lu = L[s:e, :]
        m[u] = float(e - s)
        Ak[u] = Au.sum(axis=0, dtype=np.float64)
        Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        AA[u] = Au.T @ Au
        AL[u] = Au.T @ Lu
    _log(log, f"[rg:manifest:fast] precomputed full unit structural stats in {time.time() - t0:.3f}s.")
    return _StructUnitStats(m=m, Ak=Ak, Ak2=Ak2, AA=AA, AL=AL)


def _select_struct_columns(struct: _StructUnitStats, indices: np.ndarray) -> _StructUnitStats:
    indices = np.asarray(indices, dtype=np.int64)
    return _StructUnitStats(
        m=struct.m,
        Ak=struct.Ak[:, indices],
        Ak2=struct.Ak2[:, indices],
        AA=struct.AA[:, indices, :][:, :, indices],
        AL=struct.AL[:, indices, :][:, :, indices],
    )


def _row_struct_correction(
    A: np.ndarray,
    L: np.ndarray,
    idx: np.ndarray,
    unit_id: np.ndarray,
    U: int,
    K: int,
) -> _StructUnitStats:
    m = np.zeros(U, dtype=np.float64)
    Ak = np.zeros((U, K), dtype=np.float64)
    Ak2 = np.zeros((U, K), dtype=np.float64)
    AA = np.zeros((U, K, K), dtype=np.float64)
    AL = np.zeros((U, K, K), dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return _StructUnitStats(m=m, Ak=Ak, Ak2=Ak2, AA=AA, AL=AL)

    uids = np.asarray(unit_id[idx], dtype=np.int64)
    for u in np.unique(uids):
        rows = idx[uids == u]
        Au = A[rows, :]
        Lu = L[rows, :]
        m[u] = float(rows.size)
        Ak[u] = Au.sum(axis=0, dtype=np.float64)
        Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        AA[u] = Au.T @ Au
        AL[u] = Au.T @ Lu
    return _StructUnitStats(m=m, Ak=Ak, Ak2=Ak2, AA=AA, AL=AL)


def _row_struct_correction_selected(
    A: np.ndarray,
    L: np.ndarray,
    idx: np.ndarray,
    indices: np.ndarray,
    unit_id: np.ndarray,
    U: int,
) -> _StructUnitStats:
    indices = np.asarray(indices, dtype=np.int64)
    K = int(indices.size)
    out = _StructUnitStats(
        m=np.zeros(U, dtype=np.float64),
        Ak=np.zeros((U, K), dtype=np.float64),
        Ak2=np.zeros((U, K), dtype=np.float64),
        AA=np.zeros((U, K, K), dtype=np.float64),
        AL=np.zeros((U, K, K), dtype=np.float64),
    )
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return out

    uids = np.asarray(unit_id[idx], dtype=np.int64)
    for u in np.unique(uids):
        rows = idx[uids == u]
        Au = np.ascontiguousarray(A[np.ix_(rows, indices)], dtype=np.float64)
        Lu = np.ascontiguousarray(L[np.ix_(rows, indices)], dtype=np.float64)
        out.m[u] = float(rows.size)
        out.Ak[u] = Au.sum(axis=0, dtype=np.float64)
        out.Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        out.AA[u] = Au.T @ Au
        out.AL[u] = Au.T @ Lu
    return out


def _row_ay_correction(
    A: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    unit_id: np.ndarray,
    U: int,
    K: int,
) -> np.ndarray:
    out = np.zeros((U, K), dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return out
    uids = np.asarray(unit_id[idx], dtype=np.int64)
    for u in np.unique(uids):
        rows = idx[uids == u]
        out[u] = A[rows, :].T @ y[rows]
    return out


def _compute_full_ay_unit(
    A: np.ndarray,
    y: np.ndarray,
    jk: JackknifeDesign,
) -> np.ndarray:
    U = int(jk.nunit)
    K = int(A.shape[1])
    out = np.zeros((U, K), dtype=np.float64)
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e > s:
            out[u] = A[s:e, :].T @ y[s:e]
    return out


def _build_fast_traits(
    trait_meta,
    shared_trace,
    full_tv,
    jk: JackknifeDesign,
    A: np.ndarray,
    args,
    log,
    verbose_level: int,
    *,
    compute_h2_ay: bool = True,
) -> dict[str, _FastTrait]:
    M = int(shared_trace.nsnps)
    U = int(jk.nunit)
    K = int(full_tv.nbins)

    traits: dict[str, _FastTrait] = {}
    paths = sorted(trait_meta.keys())
    t0_all = time.time()
    for spath in paths:
        meta = trait_meta[spath]
        cov_rank = meta.get("cov_rank", None)
        phen = meta.get("phen", utils._phen_name_from_path(spath))
        t0 = time.time()

        ss = Sumstats.from_file(
            spath,
            name=phen,
            log=log,
            cov_rank=cov_rank,
            cov_rank_source=("manifest" if cov_rank is not None else None),
            compute_diagnostics=(verbose_level >= 1),
            require_alleles=bool(args.align_alleles),
        )
        aligned = ss.align_to_trace(shared_trace)
        a1_code, a2_code = _full_axis_allele_codes(aligned, shared_trace)
        keep = np.asarray(
            aligned.keep_mask(chisq_threshold=args.max_chisq, chisq_action=args.chisq_action),
            dtype=bool,
        )
        entry = type("Entry", (), {"aligned": aligned, "sumstats": ss})
        beta, se, n = _full_axis_sumstats_arrays(entry, shared_trace)
        z_h2 = exact_score_z_from_arrays(beta, se, n, nsamp=float(ss.nsamp), cov_rank=0)
        z_rg = exact_score_z_from_arrays(beta, se, n, nsamp=float(ss.nsamp), cov_rank=int(ss.cov_rank))
        keep &= np.isfinite(z_h2) & np.isfinite(z_rg)

        # Sparse-drop mode keeps the full Trace axis and zeroes excluded SNPs.
        # Pair-specific sufficient statistics are then full sums minus explicit
        # dropped-SNP corrections, which is exact for fixed pre-drop units such
        # as chromosome jackknife.
        z_h2[~keep] = 0.0
        z_rg[~keep] = 0.0
        z_h2[~np.isfinite(z_h2)] = 0.0
        z_rg[~np.isfinite(z_rg)] = 0.0

        h2_ay_unit = (
            _compute_full_ay_unit(A, z_h2 * z_h2, jk)
            if compute_h2_ay
            else None
        )

        trait = _FastTrait(
            phen=str(phen),
            spath=str(spath),
            nsamp=float(ss.nsamp),
            n_scale=float(ss.n_scale),
            cov_rank=int(ss.cov_rank),
            cov_rank_source=str(ss.cov_rank_source),
            keep=keep,
            drop_idx=np.flatnonzero(~keep).astype(np.int64),
            z_h2=z_h2,
            z_rg=z_rg,
            a1_code=a1_code,
            a2_code=a2_code,
            h2_ay_unit=h2_ay_unit,
            matched_stub=_make_matched_stub(shared_trace, ss),
        )
        traits[spath] = trait

        _log(
            log,
            f"[rg:manifest:fast] cached trait '{phen}' in {time.time() - t0:.3f}s; "
            f"kept={int(np.sum(keep))}/{M}, dropped={int(np.sum(~keep))}.",
        )

    _log(log, f"[rg:manifest:fast] cached {len(traits)} trait(s) in {time.time() - t0_all:.3f}s.")
    return traits


def _manifest_trait_use_counts(manifest_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest_df.itertuples(index=False):
        paths = (row.sumstats1,) if row.sumstats1 == row.sumstats2 else (row.sumstats1, row.sumstats2)
        for spath in paths:
            counts[spath] = counts.get(spath, 0) + 1
    return counts


def _compute_struct_for_compact_units(
    A: np.ndarray,
    L: np.ndarray,
    active_idx: np.ndarray,
    jk: JackknifeDesign,
    active_mask: np.ndarray,
) -> _StructUnitStats:
    U = int(jk.nunit)
    K = int(A.shape[1])
    active_idx = np.asarray(active_idx, dtype=np.int64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_idx.ndim != 1 or active_mask.shape != (active_idx.size,):
        raise ValueError("active_idx/active_mask shape mismatch.")

    out = _StructUnitStats(
        m=np.zeros(U, dtype=np.float64),
        Ak=np.zeros((U, K), dtype=np.float64),
        Ak2=np.zeros((U, K), dtype=np.float64),
        AA=np.zeros((U, K, K), dtype=np.float64),
        AL=np.zeros((U, K, K), dtype=np.float64),
    )
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        rows = active_idx[s:e] if np.all(mu) else active_idx[s:e][mu]
        Au = A[rows, :]
        Lu = L[rows, :]
        out.m[u] = float(rows.size)
        out.Ak[u] = Au.sum(axis=0, dtype=np.float64)
        out.Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        out.AA[u] = Au.T @ Au
        out.AL[u] = Au.T @ Lu
    return out


def _compute_ay_for_compact_units(
    A: np.ndarray,
    y: np.ndarray,
    active_idx: np.ndarray,
    jk: JackknifeDesign,
    active_mask: np.ndarray,
) -> np.ndarray:
    U = int(jk.nunit)
    K = int(A.shape[1])
    y = np.asarray(y, dtype=np.float64)
    active_idx = np.asarray(active_idx, dtype=np.int64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if y.shape != (active_idx.size,) or active_mask.shape != (active_idx.size,):
        raise ValueError("y/active_idx/active_mask shape mismatch.")

    out = np.zeros((U, K), dtype=np.float64)
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        rows = active_idx[s:e] if np.all(mu) else active_idx[s:e][mu]
        yu = y[s:e] if np.all(mu) else y[s:e][mu]
        out[u] = A[rows, :].T @ yu
    return out


def _stack_h2_prepared(
    *,
    fast_tv,
    matched,
    jk,
    active_mask,
    has_overlap,
    struct: _StructUnitStats,
    Ay_unit: np.ndarray,
    n_scale: float,
    summary_y_info: dict,
    y: np.ndarray,
    enrich_mode: str,
    report_tau: bool,
    allow_neg_enr: bool,
    clip_nonfinite_vals: bool,
    jack_mode: str,
) :
    R = int(jk.nrep)
    K = int(fast_tv.nbins)

    M_full = float(struct.m.sum())
    M_rep = _stack_h2(np.array(M_full, dtype=np.float64), struct.m, jk.D).reshape(R + 1)
    Ak_rep = _stack_h2(struct.Ak.sum(axis=0), struct.Ak, jk.D)
    Ay_rep = _stack_h2(Ay_unit.sum(axis=0), Ay_unit, jk.D)
    Ak2_rep = _stack_h2(struct.Ak2.sum(axis=0), struct.Ak2, jk.D)
    AA_rep = _stack_h2(struct.AA.sum(axis=0), struct.AA, jk.D)
    AL_rep = _stack_h2(struct.AL.sum(axis=0), struct.AL, jk.D)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    M_k = Ak_rep[:, :, None]
    M_l = np.broadcast_to(Ak_full[None, :], Ak_rep.shape)[:, None, :]
    trace_KK = utils._calc_trace_from_ld_batch(AL_rep, n_scale, M_k, M_l, delta=None)
    unit_sizes = jk.unit_sizes(active_mask=active_mask, dtype=np.float64)
    trace_KK = _sym_h2(trace_KK, jk, unit_sizes, exact_loco_fast=True)

    lhs = np.full((R + 1, K + 1, K + 1), n_scale, dtype=np.float64)
    lhs[:, :K, :K] = trace_KK
    lhs[:, K, K] = n_scale

    rhs = np.full((R + 1, K + 1), n_scale, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs[:, :K] = (Ay_rep * n_scale) / Ak_rep
    bad_rhs = (~np.isfinite(rhs[:, :K])) | (~np.isfinite(Ak_rep)) | (Ak_rep <= 0.0)
    rhs[:, :K][bad_rhs] = np.nan

    prepared = H2Prepared(
        trace_view=fast_tv,
        matched=matched,
        jackknife=jk,
        active_mask=active_mask,
        has_overlap=has_overlap,
        unit_sizes=unit_sizes,
        n_scale=float(n_scale),
        summary_y_info=summary_y_info,
        y=y,
        m_unit=struct.m,
        Ak_unit=struct.Ak,
        Ay_unit=Ay_unit,
        Ak2_unit=struct.Ak2,
        AA_unit=struct.AA,
        AL_unit=struct.AL,
        M_rep=M_rep,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        Ak2_rep=Ak2_rep,
        AA_rep=AA_rep,
        AL_rep=AL_rep,
        lhs=lhs,
        rhs=rhs,
    )
    return fit_h2(
        prepared,
        enrich_mode=enrich_mode,
        report_tau=report_tau,
        allow_neg_enr=allow_neg_enr,
        clip_nonfinite_vals=clip_nonfinite_vals,
        jack_mode=jack_mode,
        nan_policy=("propagate" if clip_nonfinite_vals else "omit"),
    )


def _stack_rg_prepared(
    *,
    fast_tv,
    matched1,
    matched2,
    jk,
    active_mask,
    struct: _StructUnitStats,
    Ay_unit: np.ndarray,
    n1_scale: float,
    n2_scale: float,
    summary_y_info: dict,
    y: np.ndarray,
):
    R = int(jk.nrep)
    Ak_rep = _stack_rg(struct.Ak.sum(axis=0), struct.Ak, jk.D)
    Ay_rep = _stack_rg(Ay_unit.sum(axis=0), Ay_unit, jk.D)
    AL_rep = _stack_rg(struct.AL.sum(axis=0), struct.AL, jk.D)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    M_k = Ak_rep[:, :, None]
    M_l = np.broadcast_to(Ak_full[None, :], Ak_rep.shape)[:, None, :]
    lhs = utils._calc_rg_trace_from_ld_batch(AL_rep, n1_scale, n2_scale, M_k, M_l)
    unit_sizes = jk.unit_sizes(active_mask=active_mask, dtype=np.float64)
    lhs = _sym_rg(lhs, jk, unit_sizes, exact_loco_fast=True)

    return RGPrepared(
        trace_view=fast_tv,
        matched1=matched1,
        matched2=matched2,
        jackknife=jk,
        active_mask=active_mask,
        unit_sizes=unit_sizes,
        y=y,
        n1_scale=float(n1_scale),
        n2_scale=float(n2_scale),
        summary_y_info=summary_y_info,
        Ak_unit=struct.Ak,
        Ay_unit=Ay_unit,
        AL_unit=struct.AL,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        AL_rep=AL_rep,
        lhs=lhs,
    )


def _fit_score_intercept_scalar_fast(
    *,
    fast_tv,
    matched1,
    matched2,
    jk,
    z1: np.ndarray,
    z2: np.ndarray,
    y: np.ndarray,
    reg_ld: np.ndarray,
    n1_scale: float,
    n2_scale: float,
    h2_fit1,
    h2_fit2,
    intercept_chisq_threshold,
    jack_mode: str,
    nan_policy: str,
    reg_ld_source: str,
    log,
) -> InterceptFit:
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    z2 = np.asarray(z2, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    reg_ld = np.asarray(reg_ld, dtype=np.float64).ravel()
    M = int(fast_tv.nsnps)
    if not (z1.size == z2.size == y.size == reg_ld.size == M):
        raise ValueError("Fast overlap-covariance arrays must all live on the pair compact SNP axis.")

    keep, info = _make_intercept_keep_mask(
        z1,
        z2,
        reg_ld,
        y=y,
        weight_ld=reg_ld,
        nsamp_max=max(float(n1_scale), float(n2_scale)),
        chisq_threshold=intercept_chisq_threshold,
        chisq_mode="either",
    )
    if info["n_kept"] <= 1:
        raise RuntimeError("Fast overlap-covariance regression has <=1 SNP after filtering.")
    if log is not None and info.get("threshold") is not None:
        tag = " (auto)" if info.get("threshold_mode") == "auto" else ""
        log._log(
            f"[rg:c] overlap-covariance chi^2 filter: threshold={info['threshold']:.3f}{tag}, "
            f"mode={info['chisq_mode']}, removed={info['n_removed_chisq']} SNPs, "
            f"kept_after_all={info['n_kept']}."
        )

    R = int(jk.nrep)
    a = np.ones((M, 1), dtype=np.float64)
    x = reg_ld.reshape(-1, 1)
    unit_sizes = jk.unit_sizes(active_mask=keep, dtype=np.float64)
    sqrt_n1n2 = float(np.sqrt(float(n1_scale) * float(n2_scale)))
    if not (np.isfinite(sqrt_n1n2) and sqrt_n1n2 > 0.0):
        raise RuntimeError(
            f"Invalid n_scale pair for fast overlap-covariance estimation: "
            f"{n1_scale}, {n2_scale}."
        )

    m_u, t_u, S_u = _compute_intercept_unit_summaries(jk, a, x, y, keep)
    m_full = np.sum(m_u, axis=0, dtype=np.float64)
    t_full = np.sum(t_u, axis=0, dtype=np.float64)
    S_full = np.sum(S_u, axis=0, dtype=np.float64)
    m_rep = _stack_rg(m_full, m_u, jk.D)
    t_rep = _stack_rg(t_full, t_u, jk.D)
    S_rep = _stack_rg(S_full, S_u, jk.D)

    w_score = _build_simple_intercept_weights(reg_ld, keep)
    W_u, XW_u, XXW_u, Sy_u, XWy_u = _compute_weighted_intercept_unit_summaries(jk, x, y, w_score)
    W_rep = _stack_rg(np.sum(W_u, dtype=np.float64), W_u, jk.D)
    XW_rep = _stack_rg(np.sum(XW_u, axis=0, dtype=np.float64), XW_u, jk.D)
    XXW_rep = _stack_rg(np.sum(XXW_u, axis=0, dtype=np.float64), XXW_u, jk.D)
    Sy_rep = _stack_rg(np.sum(Sy_u, dtype=np.float64), Sy_u, jk.D)
    XWy_rep = _stack_rg(np.sum(XWy_u, axis=0, dtype=np.float64), XWy_u, jk.D)

    den0 = _full_intercept_denominator(m_full, S_full, W_rep[-1], XW_rep[-1], XXW_rep[-1])
    denom_floor = 0.0 if not np.isfinite(den0) else max(1e-12 * max(float(den0), 1.0), 0.0)
    c_reps, beta_reps, good = _solve_constrained_intercept_from_sums(
        m_rep,
        S_rep,
        t_rep,
        W_rep,
        XW_rep,
        XXW_rep,
        Sy_rep,
        XWy_rep,
        denom_floor=denom_floor,
    )
    if not bool(good[-1]):
        raise RuntimeError("Fast constrained overlap-covariance solve failed on the full sample.")

    c_est, c_se = jk.summarize(
        c_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    c = np.array([float(c_est), float(c_se)], dtype=np.float64)

    b_reps = np.asarray(beta_reps[:, 0], dtype=np.float64)
    b_est, b_se = jk.summarize(
        b_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    gamma_reg_reps = _intercept_gamma_total_from_beta(beta_reps, m_rep, sqrt_n1n2)
    gamma_reg_est, gamma_reg_se = jk.summarize(
        gamma_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    h2_tot1 = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
    h2_tot2 = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)
    rg_reg_reps = _component_rg(gamma_reg_reps, h2_tot1, h2_tot2)
    rg_reg_est, rg_reg_se = jk.summarize(
        rg_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    info.update({
        "fixed": False,
        "ld_source": str(reg_ld_source),
        "weight_ld_source": "score-legacy-total-ld",
        "regression_design_mode": str(reg_ld_source),
        "regression_ncoef": 1,
        "weight_mode": "score",
        "summary_y_mode": "beta_se_exact",
        "trait1_n_scale": float(n1_scale),
        "trait2_n_scale": float(n2_scale),
        "irwls_iters": 0,
        "regression_slope_full": float(b_reps[-1]),
        "regression_slope": float(b_est),
        "regression_slope_se": float(b_se),
        "regression_gamma_g_total_full": float(gamma_reg_reps[-1]),
        "regression_gamma_g_total": float(gamma_reg_est),
        "regression_gamma_g_total_se": float(gamma_reg_se),
        "regression_rg_total_full": float(rg_reg_reps[-1]) if np.isfinite(rg_reg_reps[-1]) else np.nan,
        "regression_rg_total": float(rg_reg_est),
        "regression_rg_total_se": float(rg_reg_se),
    })

    if log is not None:
        n_bad = int(np.sum(~np.isfinite(c_reps[:R])))
        log._log(
            f"[rg:c] constrained SCORE-weight overlap covariance (scalar): "
            f"final_c={c[0]:.6g}, bad_reps={n_bad}/{R}"
        )

    return InterceptFit(
        trace_view=fast_tv,
        matched1=matched1,
        matched2=matched2,
        jackknife=jk,
        active_mask=keep,
        unit_sizes=unit_sizes,
        ld=reg_ld,
        y=y,
        c_reps=c_reps,
        c=c,
        info=info,
    )


def dispatch_rg_manifest_fast(args, log, manifest_df, trait_meta, verbose_level: int, *, execution_plan=None):
    if str(args.rg_se_method).strip().lower() != "jackknife":
        raise ValueError("--rg-manifest-fast currently supports --rg-se-method jackknife only.")
    if args.adjust_delta:
        raise ValueError("--rg-manifest-fast currently does not support --adjust-delta.")
    parsed_write_jack, parsed_write_normeq = utils._parse_verbose_outputs(args.verbose)
    write_jack = bool(args.write_jack) or bool(parsed_write_jack)
    write_normeq_explicit = bool(args.write_normeq) or str(args.verbose).strip().lower() == "normeq"
    if write_normeq_explicit:
        raise ValueError("--rg-manifest-fast currently does not support normeq dumps.")
    if bool(parsed_write_normeq):
        _log(log, "[rg:manifest:fast] verbose requested normal-equation dumps; fast mode ignores that unsupported dump.")

    write_pair_logs = not bool(getattr(args, "rg_fast_no_pair_logs", False))
    if not write_pair_logs:
        _log(
            log,
            "[rg:manifest:fast] per-pair logs disabled; retaining batch.log and manifest.results.tsv.",
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jk_spec = JackknifeSpec.parse(args.njack)

    intercept_vals = pd.to_numeric(manifest_df["intercept_rg"], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(intercept_vals)):
        raise ValueError(
            "--rg-manifest-fast requires a finite supplied overlap_covariance in every "
            "manifest row. Use regular manifest mode without --rg-manifest-fast for "
            "summary-only overlap-covariance estimation."
        )
    if jk_spec.mode == "block":
        _log(
            log,
            "[rg:manifest:fast] WARNING: using pre-drop contiguous blocks for sparse-drop fast mode. "
            "For exact post-drop block jackknife, rerun without --rg-manifest-fast. "
            "Chromosome jackknife is unaffected.",
        )

    shared_trace = Trace(
        bimpath=args.bim,
        sumpath=None,
        savepath=None,
        log=log,
        ldscores=args.ldscores,
        ldscores_reg=args.ldscores_reg,
        ldscores_reg_w=None,
        annot=args.annot,
        verbose=bool(verbose_level),
        delta=None,
    )
    full_tv = shared_trace.materialize_view()
    model_manifest = getattr(args, "rg_model_manifest", None)
    multi_model = bool(model_manifest)
    if multi_model:
        model_specs = _load_fast_model_specs(model_manifest, full_tv.annot_header)
        _log(
            log,
            f"[rg:manifest:fast] loaded {len(model_specs)} model(s) from {model_manifest}; "
            f"union bins={int(full_tv.nbins)}, per-model bins={len(model_specs[0].indices)}.",
        )
    else:
        headers = tuple(str(item) for item in np.asarray(full_tv.annot_header).tolist())
        model_specs = [
            _FastModelSpec(
                name="default",
                slug="default",
                indices=np.arange(int(full_tv.nbins), dtype=np.int64),
                bins=headers,
                aliases=headers,
            )
        ]
    model_views = {
        model.name: _FastTraceView(
            nsnps=int(full_tv.nsnps),
            nbins=int(model.indices.size),
            annot_header=np.asarray(model.aliases, dtype=object),
        )
        for model in model_specs
    }
    jk = JackknifeDesign.from_trace_view(full_tv, jk_spec, log=log)
    A = np.asarray(full_tv.annot, dtype=np.float64, order="C")
    L = np.asarray(full_tv.ldscores, dtype=np.float64, order="C")
    U = int(jk.nunit)
    K = int(full_tv.nbins)
    unit_id = np.asarray(jk.unit_id, dtype=np.int64)
    if L.shape != (int(full_tv.nsnps), K):
        raise RuntimeError(f"Unexpected LD-score shape {L.shape}; expected {(int(full_tv.nsnps), K)}.")

    traits = _build_fast_traits(
        trait_meta,
        shared_trace,
        full_tv,
        jk,
        A,
        args,
        log,
        verbose_level,
        compute_h2_ay=not multi_model,
    )
    model_matrices: dict[str, np.ndarray] = {}
    model_full_structs: dict[str, _StructUnitStats] = {}
    model_h2_ay: dict[str, dict[str, np.ndarray]] = {}
    if multi_model:
        for model in model_specs:
            model_A = np.ascontiguousarray(A[:, model.indices], dtype=np.float64)
            model_L = np.ascontiguousarray(L[:, model.indices], dtype=np.float64)
            model_matrices[model.name] = model_A
            model_full_structs[model.name] = _compute_struct_unit_stats(
                model_A, model_L, jk, log=log
            )
            model_h2_ay[model.name] = {
                spath: _compute_full_ay_unit(model_A, tr.z_h2 * tr.z_h2, jk)
                for spath, tr in traits.items()
            }
            del model_L
    else:
        model = model_specs[0]
        model_matrices[model.name] = A
        model_full_structs[model.name] = _compute_struct_unit_stats(A, L, jk, log=log)
        model_h2_ay[model.name] = {
            spath: tr.h2_ay_unit for spath, tr in traits.items()
        }
    model_has_overlap = {
        model.name: _has_overlapping_annotations(model_matrices[model.name])
        for model in model_specs
    }

    order = (
        list(range(int(manifest_df.shape[0])))
        if execution_plan is None
        else [int(i) for i in execution_plan]
    )
    rows_by_anchor = {}
    anchor_order: list[str] = []
    for row_pos in order:
        row = manifest_df.iloc[int(row_pos)]
        if row.sumstats1 not in rows_by_anchor:
            rows_by_anchor[row.sumstats1] = []
            anchor_order.append(row.sumstats1)
        rows_by_anchor[row.sumstats1].append((int(row_pos), row))

    n_pairs = int(manifest_df.shape[0])
    n_models = len(model_specs)
    total_fits = n_pairs * n_models
    results = [None] * total_fits
    h2_no_extra_drop_cache = {}
    t0_all = time.time()
    completed = 0

    def h2_fit_for_pair(
        model: _FastModelSpec,
        tr: _FastTrait,
        struct: _StructUnitStats,
        ay: np.ndarray,
        pair_keep: np.ndarray,
        extra_drop: np.ndarray,
    ):
        cache_key = (model.name, tr.spath) if int(extra_drop.size) == 0 else None
        if cache_key is not None and cache_key in h2_no_extra_drop_cache:
            return h2_no_extra_drop_cache[cache_key]

        h2_info = {
            "mode": "beta_se_exact_sparse_drop",
            "cov_rank": 0,
            "cov_rank_source": "forced0_no_covrank_h2",
            "n_scale": float(effective_n_scale(tr.nsamp, 0)),
            "n_nonfinite": 0,
        }
        fit = _stack_h2_prepared(
            fast_tv=model_views[model.name],
            matched=tr.matched_stub,
            jk=jk,
            active_mask=pair_keep,
            has_overlap=model_has_overlap[model.name],
            struct=struct,
            Ay_unit=ay,
            n_scale=h2_info["n_scale"],
            summary_y_info=h2_info,
            y=np.array([], dtype=np.float64),
            enrich_mode=args.enrich_mode,
            report_tau=True,
            allow_neg_enr=args.allow_neg_enr,
            clip_nonfinite_vals=args.clip_nonfinite_vals,
            jack_mode=args.jack_mode,
        )
        if cache_key is not None:
            h2_no_extra_drop_cache[cache_key] = fit
        return fit

    for anchor_path in anchor_order:
        row_items = rows_by_anchor[anchor_path]
        anchor = traits[anchor_path]
        partner_paths = [row.sumstats2 for _, row in row_items]
        partners = [traits[p] for p in partner_paths]
        P = len(partners)
        _log(log, f"[rg:manifest:fast] processing anchor '{anchor.phen}' with {P} partner pair(s).")
        align_alleles = bool(args.align_alleles)
        drop_ambiguous = not bool(getattr(args, "keep_ambiguous", False))

        ay_rg_by_model = {
            model.name: np.zeros(
                (U, int(model.indices.size), P), dtype=np.float64
            )
            for model in model_specs
        }
        for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
            s = int(s)
            e = int(e)
            if e <= s:
                continue
            Zi = anchor.z_rg[s:e]
            if align_alleles:
                zcols = []
                for p in partners:
                    allele_keep_u, allele_flip_u = harmonize_allele_codes(
                        anchor.a1_code[s:e],
                        anchor.a2_code[s:e],
                        p.a1_code[s:e],
                        p.a2_code[s:e],
                        drop_ambiguous=drop_ambiguous,
                    )
                    zcols.append(
                        np.where(
                            allele_keep_u,
                            np.where(allele_flip_u, -p.z_rg[s:e], p.z_rg[s:e]),
                            0.0,
                        )
                    )
                Zp = np.column_stack(zcols)
            else:
                Zp = np.column_stack([p.z_rg[s:e] for p in partners])
            Y = Zi[:, None] * Zp
            for model in model_specs:
                ay_rg_by_model[model.name][u] = (
                    model_matrices[model.name][s:e, :].T @ Y
                )

        for pidx, (row_pos, row) in enumerate(row_items):
            t0_pair = time.time()
            tr1 = anchor
            tr2 = partners[pidx]
            if align_alleles:
                allele_keep, allele_flip = harmonize_allele_codes(
                    tr1.a1_code,
                    tr1.a2_code,
                    tr2.a1_code,
                    tr2.a2_code,
                    drop_ambiguous=drop_ambiguous,
                )
            else:
                allele_keep = None
                allele_flip = None
            pair_keep = np.asarray(
                tr1.keep & tr2.keep
                if allele_keep is None
                else tr1.keep & tr2.keep & allele_keep,
                dtype=bool,
            )
            active_n = int(np.sum(pair_keep))
            if active_n <= 0:
                raise RuntimeError(f"No SNPs remain for pair {row.phen1} vs {row.phen2}.")

            if align_alleles:
                _log(
                    log,
                    f"[rg:manifest:fast] alleles {row.phen1} vs {row.phen2}: "
                    f"dropped={int(np.sum((tr1.keep & tr2.keep) & ~allele_keep))}, "
                    f"flipped={int(np.sum(pair_keep & allele_flip))}, "
                    f"drop_ambiguous={drop_ambiguous}.",
                )

            pair_drop = np.flatnonzero(~pair_keep).astype(np.int64)
            drop2_for_1 = np.flatnonzero(tr1.keep & ~pair_keep).astype(np.int64)
            drop1_for_2 = np.flatnonzero(tr2.keep & ~pair_keep).astype(np.int64)

            union_struct = None
            ay1_union = None
            ay2_union = None
            if not multi_model:
                full_struct = model_full_structs[model_specs[0].name]
                if pair_drop.size == 0:
                    union_struct = full_struct
                else:
                    corr = _row_struct_correction(A, L, pair_drop, unit_id, U, K)
                    union_struct = _StructUnitStats(
                        m=full_struct.m - corr.m,
                        Ak=full_struct.Ak - corr.Ak,
                        Ak2=full_struct.Ak2 - corr.Ak2,
                        AA=full_struct.AA - corr.AA,
                        AL=full_struct.AL - corr.AL,
                    )
                ay1_union = model_h2_ay[model_specs[0].name][tr1.spath]
                if drop2_for_1.size:
                    ay1_union = ay1_union - _row_ay_correction(
                        A,
                        tr1.z_h2 * tr1.z_h2,
                        drop2_for_1,
                        unit_id,
                        U,
                        K,
                    )
                ay2_union = model_h2_ay[model_specs[0].name][tr2.spath]
                if drop1_for_2.size:
                    ay2_union = ay2_union - _row_ay_correction(
                        A,
                        tr2.z_h2 * tr2.z_h2,
                        drop1_for_2,
                        unit_id,
                        U,
                        K,
                    )

            for model_index, model in enumerate(model_specs):
                model_view = model_views[model.name]
                if not multi_model:
                    struct = union_struct
                    ay1 = ay1_union
                    ay2 = ay2_union
                    ay_pair = ay_rg_by_model[model.name][:, :, pidx]
                else:
                    model_A = model_matrices[model.name]
                    full_struct = model_full_structs[model.name]
                    if pair_drop.size == 0:
                        struct = full_struct
                    else:
                        corr = _row_struct_correction_selected(
                            A,
                            L,
                            pair_drop,
                            model.indices,
                            unit_id,
                            U,
                        )
                        struct = _StructUnitStats(
                            m=full_struct.m - corr.m,
                            Ak=full_struct.Ak - corr.Ak,
                            Ak2=full_struct.Ak2 - corr.Ak2,
                            AA=full_struct.AA - corr.AA,
                            AL=full_struct.AL - corr.AL,
                        )
                    ay1 = model_h2_ay[model.name][tr1.spath]
                    if drop2_for_1.size:
                        ay1 = ay1 - _row_ay_correction(
                            model_A,
                            tr1.z_h2 * tr1.z_h2,
                            drop2_for_1,
                            unit_id,
                            U,
                            int(model.indices.size),
                        )
                    ay2 = model_h2_ay[model.name][tr2.spath]
                    if drop1_for_2.size:
                        ay2 = ay2 - _row_ay_correction(
                            model_A,
                            tr2.z_h2 * tr2.z_h2,
                            drop1_for_2,
                            unit_id,
                            U,
                            int(model.indices.size),
                        )
                    ay_pair = ay_rg_by_model[model.name][:, :, pidx]

                h2_fit1 = h2_fit_for_pair(
                    model, tr1, struct, ay1, pair_keep, drop2_for_1
                )
                h2_fit2 = h2_fit_for_pair(
                    model, tr2, struct, ay2, pair_keep, drop1_for_2
                )

                rg_info = {
                    "mode": "beta_se_exact_sparse_drop",
                    "trait1_cov_rank": int(tr1.cov_rank),
                    "trait1_cov_rank_source": str(tr1.cov_rank_source),
                    "trait1_n_scale": float(tr1.n_scale),
                    "trait2_cov_rank": int(tr2.cov_rank),
                    "trait2_cov_rank_source": str(tr2.cov_rank_source),
                    "trait2_n_scale": float(tr2.n_scale),
                    "n_nonfinite": 0,
                    "model": model.name,
                    "model_bins": list(model.bins),
                }
                rg_prepared = _stack_rg_prepared(
                    fast_tv=model_view,
                    matched1=tr1.matched_stub,
                    matched2=tr2.matched_stub,
                    jk=jk,
                    active_mask=pair_keep,
                    struct=struct,
                    Ay_unit=ay_pair,
                    n1_scale=float(tr1.n_scale),
                    n2_scale=float(tr2.n_scale),
                    summary_y_info=rg_info,
                    y=np.array([], dtype=np.float64),
                )

                row_intercept = float(row.intercept_rg)
                intercept = InterceptFit(
                    trace_view=model_view,
                    matched1=tr1.matched_stub,
                    matched2=tr2.matched_stub,
                    jackknife=jk,
                    active_mask=pair_keep,
                    unit_sizes=jk.unit_sizes(active_mask=pair_keep, dtype=np.float64),
                    ld=np.array([], dtype=np.float64),
                    y=np.array([], dtype=np.float64),
                    c_reps=np.full(jk.nrep + 1, row_intercept, dtype=np.float64),
                    c=np.array([row_intercept, 0.0], dtype=np.float64),
                    info={
                        "fixed": True,
                        "source": "manifest",
                        "summary_y_mode": "beta_se_exact_sparse_drop",
                        "trait1_n_scale": float(tr1.n_scale),
                        "trait2_n_scale": float(tr2.n_scale),
                        "model": model.name,
                    },
                )
                rg_fit = fit_rg(
                    rg_prepared,
                    h2_fit1,
                    h2_fit2,
                    intercept,
                    rg_se_method=args.rg_se_method,
                    jack_mode=args.jack_mode,
                    nan_policy=("propagate" if args.clip_nonfinite_vals else "omit"),
                )

                pair_prefix = (
                    str(outdir / model.slug / row.out_stem)
                    if multi_model
                    else str(outdir / row.out_stem)
                )
                if write_pair_logs or write_jack:
                    Path(pair_prefix).parent.mkdir(parents=True, exist_ok=True)
                if write_pair_logs:
                    _write_fast_pair_log(
                        pair_prefix,
                        phen1=row.phen1,
                        phen2=row.phen2,
                        annot_header=model_view.annot_header,
                        h2_fit1=h2_fit1,
                        h2_fit2=h2_fit2,
                        intercept=intercept,
                        rg_fit=rg_fit,
                        runtime_s=(time.time() - t0_pair),
                    )
                if write_jack:
                    jack_path = pair_prefix + ".rg.jack"
                    RGResultWriter.save_jackknife_text(rg_fit, jack_path)
                    _log(log, f"[rg:manifest:fast] saved rg jackknife replicate dump to {jack_path}")

                result = build_manifest_summary_row(
                    phen1=row.phen1,
                    phen2=row.phen2,
                    sumstats1=row.sumstats1,
                    sumstats2=row.sumstats2,
                    cov_rank1=row.cov_rank1,
                    cov_rank2=row.cov_rank2,
                    intercept_rg_input=row.intercept_rg,
                    out_prefix=pair_prefix,
                    n_snps=active_n,
                    annot_header=model_view.annot_header,
                    h2_fit1=h2_fit1,
                    h2_fit2=h2_fit2,
                    intercept=intercept,
                    rg_fit=rg_fit,
                )
                if multi_model:
                    result = {"model": model.name, **result}
                result_index = model_index * n_pairs + int(row_pos)
                results[result_index] = result
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == total_fits:
                    model_msg = f" model={model.name}" if multi_model else ""
                    _log(
                        log,
                        f"[rg:manifest:fast] completed {completed}/{total_fits} fit(s);{model_msg} "
                        f"latest {row.phen1} vs {row.phen2}: rg={float(rg_fit.rg_total[0]):.6g} "
                        f"(SE {float(rg_fit.rg_total[1]):.6g})."
                    )

    out = pd.DataFrame(results)
    summary_path = outdir / "manifest.results.tsv"
    out.to_csv(summary_path, sep="\t", index=False)
    _log(log, f"[rg:manifest:fast] saved batch summary to {summary_path}")
    _log(log, f"[rg:manifest:fast] total fast manifest runtime after Trace load: {time.time() - t0_all:.3f}s.")
