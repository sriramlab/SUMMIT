from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import os
import time

import numpy as np
import pandas as pd

from .. import utils
from ..sumstats.moments import (
    derived_wald_z,
    exact_score_z_from_arrays,
    effective_n_scale,
    resolve_cov_rank,
)
from ..sumstats.sumstats import Sumstats
from .h2core import (
    H2MatchedMetadata,
    H2ResultWriter,
    compute_h2_structural_unit_stats,
    fit_h2,
    prepare_h2_reference_axis_from_sufficient_stats,
)
from .h2_cache import H2TraitCache, trace_axis_digest
from .jackknife import JackknifeDesign, JackknifeSpec
from .trace import Trace


@dataclass(frozen=True)
class _TraitSpec:
    index: int
    path: str
    phen: str
    cov_rank: int | None


@dataclass(frozen=True)
class _LoadedTrait:
    spec: _TraitSpec
    matched: H2MatchedMetadata
    summary_y_info: dict
    n_reported_keep: int
    n_active: int
    load_seconds: float
    source: str


def _log(log, message: str):
    if log is not None:
        log._log(message)


def _parse_cov_rank_values(raw, expected: int) -> list[int | None]:
    expected = int(expected)
    if raw is None:
        return [None] * expected
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"--cov-rank values must be non-negative; got {value}")
        vals.append(value)
    if len(vals) != expected:
        raise ValueError(
            f"--cov-rank must contain exactly {expected} value(s) for --h2; got {len(vals)}"
        )
    return vals


def _trait_payload(item: _LoadedTrait) -> dict:
    matched = item.matched
    return {
        "spec_index": int(item.spec.index),
        "path": str(item.spec.path),
        "phen": str(item.spec.phen),
        "matched": {
            "nsnps": int(matched.nsnps),
            "nsamp": float(matched.nsamp),
            "n_scale": float(matched.n_scale),
            "cov_rank": int(matched.cov_rank),
            "cov_rank_source": str(matched.cov_rank_source),
            "name": str(matched.name),
            "clip_count": int(matched.clip_count),
            "clip_threshold": matched.clip_threshold,
        },
        "summary_y_info": item.summary_y_info,
        "n_reported_keep": int(item.n_reported_keep),
        "n_active": int(item.n_active),
    }


def _loaded_from_payload(spec: _TraitSpec, payload: dict, load_seconds: float) -> _LoadedTrait:
    raw = payload["matched"]
    matched = H2MatchedMetadata(
        nsnps=int(raw["nsnps"]),
        nsamp=float(raw["nsamp"]),
        n_scale=float(raw["n_scale"]),
        cov_rank=int(raw["cov_rank"]),
        cov_rank_source=str(raw["cov_rank_source"]),
        name=str(raw["name"]),
        used_summary=None,
        used_top=None,
        clip_count=int(raw.get("clip_count", 0)),
        clip_threshold=raw.get("clip_threshold"),
    )
    return _LoadedTrait(
        spec=spec,
        matched=matched,
        summary_y_info=dict(payload["summary_y_info"]),
        n_reported_keep=int(payload["n_reported_keep"]),
        n_active=int(payload["n_active"]),
        load_seconds=float(load_seconds),
        source="cache",
    )


def _parse_trait_column(
    spec: _TraitSpec,
    trace: Trace,
    y_column: np.ndarray,
    active_column: np.ndarray,
    *,
    chisq_threshold,
    chisq_action: str,
    compute_diagnostics: bool,
    fast_reader: str,
) -> _LoadedTrait:
    """Load one trait and write its exact h2 moment onto the shared axis."""
    if fast_reader == "stream" and not compute_diagnostics:
        streamed = _stream_trait_column(
            spec,
            trace,
            y_column,
            active_column,
            chisq_threshold=chisq_threshold,
            chisq_action=chisq_action,
        )
        if streamed is not None:
            return streamed

    started = time.monotonic()
    ss = Sumstats.from_file(
        spec.path,
        name=spec.phen,
        log=None,
        cov_rank=spec.cov_rank,
        cov_rank_source=("cli" if spec.cov_rank is not None else None),
        compute_diagnostics=compute_diagnostics,
    )
    aligned = ss.align_to_trace(trace)
    reported_keep = np.asarray(
        aligned.keep_mask(
            chisq_threshold=chisq_threshold,
            chisq_action=chisq_action,
        ),
        dtype=bool,
    )

    positions = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    used_rows = np.flatnonzero(reported_keep)
    source_rows = positions[used_rows]
    if np.any(source_rows < 0):
        raise RuntimeError("Internal alignment error: retained SNP is absent from sumstats.")

    z_star = exact_score_z_from_arrays(
        beta=ss.beta[source_rows],
        se=ss.se[source_rows],
        n_obs=ss.n[source_rows],
        nsamp=float(ss.nsamp),
        cov_rank=0,
    )
    finite = np.isfinite(z_star)
    active_rows = used_rows[finite]

    y_column.fill(0.0)
    active_column.fill(False)
    if active_rows.size:
        y_column[active_rows] = z_star[finite] * z_star[finite]
        active_column[active_rows] = True

    used_summary = used_top = None
    clip_count = 0
    clip_threshold = None
    if compute_diagnostics:
        used_summary, used_top, clip_count, clip_threshold = aligned.diagnostics_for_keep(
            reported_keep,
            chisq_threshold=chisq_threshold,
            chisq_action=chisq_action,
            compute_diagnostics=True,
        )

    matched = H2MatchedMetadata(
        nsnps=int(np.sum(reported_keep)),
        nsamp=float(ss.nsamp),
        n_scale=float(ss.n_scale),
        cov_rank=int(ss.cov_rank),
        cov_rank_source=str(ss.cov_rank_source),
        name=str(ss.name),
        used_summary=used_summary,
        used_top=used_top,
        clip_count=int(clip_count),
        clip_threshold=clip_threshold,
    )
    info = {
        "mode": "beta_se_exact_fast_batch",
        "cov_rank": 0,
        "cov_rank_source": "forced0_no_covrank_h2",
        "n_scale": float(effective_n_scale(ss.nsamp, 0)),
        "n_nonfinite": int(np.sum(~finite)),
    }

    return _LoadedTrait(
        spec=spec,
        matched=matched,
        summary_y_info=info,
        n_reported_keep=int(np.sum(reported_keep)),
        n_active=int(active_rows.size),
        load_seconds=float(time.monotonic() - started),
        source="sumstats",
    )


def _find_column(columns, candidates, *, default_position=None):
    columns = list(columns)
    lower = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        hit = lower.get(str(candidate).lower())
        if hit is not None:
            return hit
    if default_position is not None and 0 <= int(default_position) < len(columns):
        return columns[int(default_position)]
    return None


def _stream_trait_column(
    spec: _TraitSpec,
    trace: Trace,
    y_column: np.ndarray,
    active_column: np.ndarray,
    *,
    chisq_threshold,
    chisq_action: str,
) -> _LoadedTrait | None:
    """Stream normalized text sumstats into compact full-axis numeric buffers.

    ``None`` requests a correctness-preserving fallback to ``Sumstats``. This
    happens for duplicate SNP IDs, whose max-N/first-row resolution is already
    implemented and tested in the legacy reader.
    """
    started = time.monotonic()
    M = int(trace.nsnps)
    beta = np.full(M, np.nan, dtype=np.float64)
    se = np.full(M, np.nan, dtype=np.float64)
    obs_n = np.full(M, np.nan, dtype=np.float64)
    nmax = -np.inf
    cov_rank_values = set()
    expected_columns = None
    duplicate = False
    valid_rows_total = 0

    for file_path in utils._resolve_chr_split_paths(spec.path, require=True):
        header = pd.read_csv(file_path, sep=r"\s+", compression="infer", nrows=0)
        columns = list(header.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"Chromosome-split files for '{spec.path}' have inconsistent columns; "
                f"first columns={expected_columns}, file '{file_path}' columns={columns}."
            )

        id_column = _find_column(columns, ["ID", "id", "snp", "SNP"], default_position=0)
        # Require the same allele fields as Sumstats even though univariate h2
        # does not use their values after source validation.
        a1_column = _find_column(columns, ["A1", "ALT"], default_position=1)
        a2_column = _find_column(columns, ["A2", "REF"], default_position=2)
        n_column = _find_column(columns, ["OBS_CT", "obs_ct", "N", "n"], default_position=3)
        beta_column = _find_column(columns, ["BETA", "beta"])
        se_column = _find_column(columns, ["SE", "se", "STDERR", "stderr"])
        cov_rank_column = _find_column(columns, ["COV_RANK", "cov_rank", "P_EFF", "p_eff"])
        if any(value is None for value in (id_column, a1_column, a2_column, n_column, beta_column, se_column)):
            raise RuntimeError(
                f"Phenotype [{spec.phen}] must contain SNP, A1, A2, N, BETA, and SE columns."
            )

        usecols = [id_column, n_column, beta_column, se_column]
        if cov_rank_column is not None:
            usecols.append(cov_rank_column)
        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            compression="infer",
            usecols=list(dict.fromkeys(usecols)),
            dtype={id_column: str},
        )
        frame[id_column] = frame[id_column].astype(str)
        for column in (n_column, beta_column, se_column):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if cov_rank_column is not None:
            values = pd.to_numeric(frame[cov_rank_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            values = values[np.isfinite(values)]
            if values.size:
                rounded = np.rint(values)
                if np.any(np.abs(values - rounded) > 1e-8):
                    raise RuntimeError(f"Phenotype [{spec.phen}] has non-integer COV_RANK / P_EFF values.")
                cov_rank_values.update(int(value) for value in np.unique(rounded.astype(np.int64)))
                if len(cov_rank_values) > 1:
                    raise RuntimeError(f"Phenotype [{spec.phen}] has non-constant COV_RANK / P_EFF values.")

        n_values = frame[n_column].to_numpy(dtype=np.float64, copy=False)
        beta_values = frame[beta_column].to_numpy(dtype=np.float64, copy=False)
        se_values = frame[se_column].to_numpy(dtype=np.float64, copy=False)
        valid = (
            np.isfinite(n_values)
            & (n_values > 0.0)
            & np.isfinite(beta_values)
            & np.isfinite(se_values)
            & (se_values > 0.0)
        )
        if not np.any(valid):
            continue
        frame = frame.loc[valid, [id_column, n_column, beta_column, se_column]].reset_index(drop=True)
        valid_rows_total += int(frame.shape[0])
        nmax = max(nmax, float(frame[n_column].max()))

        if frame[id_column].duplicated().any():
            duplicate = True
            break
        positions = trace.index.get_indexer(frame[id_column].to_numpy(dtype=str))
        hit = positions >= 0
        positions = positions[hit]
        if positions.size == 0:
            continue
        if np.any(np.isfinite(obs_n[positions])):
            duplicate = True
            break
        beta[positions] = frame.loc[hit, beta_column].to_numpy(dtype=np.float64, copy=False)
        se[positions] = frame.loc[hit, se_column].to_numpy(dtype=np.float64, copy=False)
        obs_n[positions] = frame.loc[hit, n_column].to_numpy(dtype=np.float64, copy=False)

    if duplicate:
        return None
    if valid_rows_total <= 0 or not np.isfinite(nmax):
        raise RuntimeError(f"No valid SNPs remain after basic filtering for phenotype [{spec.phen}].")

    file_cov_rank = None
    if cov_rank_values:
        file_cov_rank = int(next(iter(cov_rank_values)))
        if file_cov_rank < 0:
            raise RuntimeError(f"Phenotype [{spec.phen}] has negative COV_RANK / P_EFF={file_cov_rank}.")

    resolved_cov_rank, resolved_source = resolve_cov_rank(
        explicit=spec.cov_rank,
        explicit_source=("cli" if spec.cov_rank is not None else None),
        sumstats_value=file_cov_rank,
        sumstats_source="file",
    )
    n_scale = float(effective_n_scale(nmax, resolved_cov_rank))
    present = np.isfinite(obs_n)
    rows = np.flatnonzero(present)
    z_qc = derived_wald_z(beta[rows], se[rows], obs_n[rows], n_scale)
    finite_qc = np.isfinite(z_qc)
    if not np.all(finite_qc):
        present[rows[~finite_qc]] = False
        rows = rows[finite_qc]
        z_qc = z_qc[finite_qc]

    action = str(chisq_action).strip().lower()
    if action not in {"drop", "clip", "warn", "none"}:
        raise ValueError(f"Invalid chisq_action={chisq_action!r}")
    threshold, _ = utils._resolve_chisq_threshold(n_scale, chisq_threshold)
    if action == "drop" and threshold is not None and np.isfinite(threshold) and threshold > 0.0:
        keep_qc = (z_qc * z_qc) <= float(threshold)
        present[rows[~keep_qc]] = False
        rows = rows[keep_qc]

    z_star = exact_score_z_from_arrays(
        beta=beta[rows],
        se=se[rows],
        n_obs=obs_n[rows],
        nsamp=nmax,
        cov_rank=0,
    )
    finite_h2 = np.isfinite(z_star)
    active_rows = rows[finite_h2]
    y_column.fill(0.0)
    active_column.fill(False)
    y_column[active_rows] = z_star[finite_h2] * z_star[finite_h2]
    active_column[active_rows] = True

    matched = H2MatchedMetadata(
        nsnps=int(rows.size),
        nsamp=float(nmax),
        n_scale=n_scale,
        cov_rank=int(resolved_cov_rank),
        cov_rank_source=str(resolved_source),
        name=str(spec.phen),
        used_summary=None,
        used_top=None,
        clip_count=0,
        clip_threshold=None,
    )
    info = {
        "mode": "beta_se_exact_fast_batch",
        "cov_rank": 0,
        "cov_rank_source": "forced0_no_covrank_h2",
        "n_scale": float(effective_n_scale(nmax, 0)),
        "n_nonfinite": int(np.sum(~finite_h2)),
    }
    return _LoadedTrait(
        spec=spec,
        matched=matched,
        summary_y_info=info,
        n_reported_keep=int(rows.size),
        n_active=int(active_rows.size),
        load_seconds=float(time.monotonic() - started),
        source="sumstats-stream",
    )


def _load_trait_column(
    spec: _TraitSpec,
    trace: Trace,
    y_column: np.ndarray,
    active_column: np.ndarray,
    *,
    chisq_threshold,
    chisq_action: str,
    compute_diagnostics: bool,
    cache: H2TraitCache | None,
    fast_reader: str,
) -> _LoadedTrait:
    if cache is None:
        return _parse_trait_column(
            spec,
            trace,
            y_column,
            active_column,
            chisq_threshold=chisq_threshold,
            chisq_action=chisq_action,
            compute_diagnostics=compute_diagnostics,
            fast_reader=fast_reader,
        )

    started = time.monotonic()
    entry = cache.entry(
        path=spec.path,
        phen=spec.phen,
        cov_rank=spec.cov_rank,
        chisq_threshold=chisq_threshold,
        chisq_action=chisq_action,
    )
    if cache.mode != "refresh":
        payload = cache.load_into(entry, y_column, active_column)
        if payload is not None:
            return _loaded_from_payload(spec, payload, time.monotonic() - started)
        if cache.mode == "read":
            raise RuntimeError(f"Required h2 cache entry is missing or invalid for {spec.path}")

    with cache.lock(entry):
        if cache.mode != "refresh":
            payload = cache.load_into(entry, y_column, active_column)
            if payload is not None:
                return _loaded_from_payload(spec, payload, time.monotonic() - started)
        item = _parse_trait_column(
            spec,
            trace,
            y_column,
            active_column,
            chisq_threshold=chisq_threshold,
            chisq_action=chisq_action,
            compute_diagnostics=compute_diagnostics,
            fast_reader=fast_reader,
        )
        cache.write(entry, y_column, active_column, _trait_payload(item))
        return _LoadedTrait(
            spec=item.spec,
            matched=item.matched,
            summary_y_info=item.summary_y_info,
            n_reported_keep=item.n_reported_keep,
            n_active=item.n_active,
            load_seconds=float(time.monotonic() - started),
            source="sumstats+cache-write",
        )


def _write_results_table(path: str, rows: list[tuple], nbins: int):
    if not rows:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    header = ["phen_index", "phen", "num_bins", "h2", "h2_se"]
    header.extend(f"h2bin_{j}" for j in range(nbins))
    header.extend(f"h2bin_se_{j}" for j in range(nbins))
    with open(tmp, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for index, phen, fit in rows:
            row = [
                str(index),
                str(phen),
                str(nbins),
                format(float(fit.h2[-1, 0]), ".12g"),
                format(float(fit.h2[-1, 1]), ".12g"),
            ]
            row.extend(format(float(fit.h2[j, 0]), ".12g") for j in range(nbins))
            row.extend(format(float(fit.h2[j, 1]), ".12g") for j in range(nbins))
            handle.write("\t".join(row) + "\n")
    os.replace(tmp, path)


def dispatch_h2_batch_fast(args, log):
    """Run exact chromosome-jackknife h2 in bounded phenotype batches."""
    if args.trace is not None:
        raise ValueError("--h2-batch-fast requires per-SNP --ldscores, not --trace.")
    if args.ldscores is None:
        raise ValueError("--h2-batch-fast requires --ldscores.")
    if str(args.chisq_action).strip().lower() == "clip":
        raise ValueError(
            "--h2-batch-fast does not currently support --chisq-action clip; "
            "use drop, warn, or none."
        )

    jk_spec = JackknifeSpec.parse(args.njack)
    if jk_spec.mode != "chr":
        raise ValueError(
            "--h2-batch-fast currently requires chromosome jackknife (--njack chr[:...]). "
            "Use the regular h2 path for post-drop contiguous block jackknife."
        )

    paths = utils._parse_sumdir(args.h2)
    phen_names = [utils._phen_name_from_path(path) for path in paths]
    cov_ranks = _parse_cov_rank_values(args.cov_rank, len(paths))
    specs = [
        _TraitSpec(index=i, path=path, phen=phen_names[i], cov_rank=cov_ranks[i])
        for i, path in enumerate(paths)
    ]

    workers = int(args.h2_workers)
    batch_size = int(args.h2_batch_size)
    checkpoint_every = int(args.h2_checkpoint_every)
    if workers <= 0:
        raise ValueError("--h2-workers must be positive.")
    if batch_size <= 0:
        raise ValueError("--h2-batch-size must be positive.")
    if checkpoint_every <= 0:
        raise ValueError("--h2-checkpoint-every must be positive.")
    workers = min(workers, batch_size, len(specs))
    fast_reader = str(args.h2_fast_reader).strip().lower()
    if fast_reader not in {"stream", "pandas"}:
        raise ValueError("--h2-fast-reader must be stream or pandas.")

    started = time.monotonic()
    verbose_level = utils._parse_verbose(args.verbose)
    compute_diagnostics = verbose_level >= 1
    parsed_write_jack, _ = utils._parse_verbose_outputs(args.verbose)
    write_jack = bool(args.write_jack) or bool(parsed_write_jack)

    trace_started = time.monotonic()
    trace = Trace(
        bimpath=args.bim,
        sumpath=None,
        savepath=None,
        log=log,
        ldscores=args.ldscores,
        annot=args.annot,
        verbose=bool(verbose_level),
        delta=None,
    )
    trace_view = trace.materialize_view()
    jackknife = JackknifeDesign.from_trace_view(trace_view, jk_spec, log=log)
    # Build pandas' immutable lookup engine before concurrent readers share it.
    trace.index.get_indexer(trace.snps[:1])
    M = int(trace_view.nsnps)
    U = int(jackknife.nunit)
    K = int(trace_view.nbins)
    _log(
        log,
        f"[h2:batch:fast] shared Trace/jackknife ready in "
        f"{time.monotonic() - trace_started:.3f}s.",
    )

    cache = None
    if args.h2_cache_dir is not None:
        digest_started = time.monotonic()
        axis_hash = trace_axis_digest(trace.snps)
        cache = H2TraitCache(
            args.h2_cache_dir,
            axis_digest=axis_hash,
            nsnps=M,
            mode=args.h2_cache_mode,
            verify_checksum=args.h2_cache_verify_checksum,
        )
        _log(
            log,
            f"[h2:batch:fast] cache mode={args.h2_cache_mode} dir={args.h2_cache_dir} "
            f"axis_sha256={axis_hash} digest_time={time.monotonic() - digest_started:.3f}s.",
        )
    elif args.h2_cache_only:
        raise ValueError("--h2-cache-only requires --h2-cache-dir.")

    if args.h2_cache_only:
        full_struct = None
        A = None
        overlap_rows = None
    else:
        structural_started = time.monotonic()
        full_struct = compute_h2_structural_unit_stats(
            trace_view,
            jackknife,
            ld_kind="main",
        )
        A = np.asarray(trace_view.annot, dtype=np.float64, order="C")
        overlap_rows = np.empty(M, dtype=bool)
        for start in range(0, M, 200_000):
            end = min(start + 200_000, M)
            overlap_rows[start:end] = (
                np.sum(np.abs(A[start:end, :]) > 0.0, axis=1) > 1
            )
        _log(
            log,
            f"[h2:batch:fast] shared structural statistics ready in "
            f"{time.monotonic() - structural_started:.3f}s.",
        )

    _log(
        log,
        f"[h2:batch:fast] traits={len(specs)} SNPs={M} bins={K} "
        f"batch_size={batch_size} loader_workers={workers}.",
    )

    result_rows = []
    results_path = f"{args.out}.results.tsv"
    for batch_start in range(0, len(specs), batch_size):
        batch_specs = specs[batch_start : batch_start + batch_size]
        B = len(batch_specs)
        batch_started = time.monotonic()

        # Fortran order keeps each loader-owned column contiguous while still
        # exposing a dense (rows x traits) matrix to BLAS below.
        Y = np.zeros((M, B), dtype=np.float64, order="F")
        active = np.zeros((M, B), dtype=bool, order="F")
        loaded: list[_LoadedTrait | None] = [None] * B

        with ThreadPoolExecutor(max_workers=min(workers, B)) as executor:
            futures = {
                executor.submit(
                    _load_trait_column,
                    spec,
                    trace,
                    Y[:, column],
                    active[:, column],
                    chisq_threshold=args.max_chisq,
                    chisq_action=args.chisq_action,
                    compute_diagnostics=compute_diagnostics,
                    cache=cache,
                    fast_reader=fast_reader,
                ): column
                for column, spec in enumerate(batch_specs)
            }
            for future in as_completed(futures):
                column = futures[future]
                loaded[column] = future.result()

        if args.h2_cache_only:
            for item in loaded:
                if item is None:
                    raise RuntimeError("Internal fast-batch loader did not return a trait.")
                _log(
                    log,
                    f"[h2:batch:fast] cached {item.spec.index + 1}/{len(specs)} "
                    f"{item.spec.phen}: source={item.source} load={item.load_seconds:.3f}s.",
                )
            del active, Y, loaded
            gc.collect()
            continue

        Ay = np.zeros((B, U, K), dtype=np.float64)
        if A is None or full_struct is None or overlap_rows is None:
            raise RuntimeError("Internal fast-h2 fitting state was not initialized.")
        for unit, (start, end) in enumerate(zip(jackknife.starts, jackknife.ends)):
            start = int(start)
            end = int(end)
            if end > start:
                Ay[:, unit, :] = (A[start:end, :].T @ Y[start:end, :]).T

        for column, item in enumerate(loaded):
            if item is None:
                raise RuntimeError("Internal fast-batch loader did not return a trait.")
            trait_started = time.monotonic()
            active_mask = active[:, column]
            prepared = prepare_h2_reference_axis_from_sufficient_stats(
                trace_view,
                item.matched,
                jackknife,
                active_mask,
                Ay_unit=Ay[column],
                n_scale=float(item.summary_y_info["n_scale"]),
                summary_y_info=item.summary_y_info,
                full_struct=full_struct,
                adjust_delta=args.adjust_delta,
                has_overlap=bool(np.any(overlap_rows & active_mask)),
            )
            fit = fit_h2(
                prepared,
                enrich_mode=args.enrich_mode,
                report_tau=True,
                allow_neg_enr=args.allow_neg_enr,
                clip_nonfinite_vals=args.clip_nonfinite_vals,
                jack_mode=args.jack_mode,
                nan_policy=("propagate" if args.clip_nonfinite_vals else "omit"),
            )

            if write_jack:
                H2ResultWriter.save_jackknife_text(
                    fit,
                    f"{args.out}.{item.spec.phen}.jack",
                )

            object.__setattr__(fit, "prepared", None)
            result_rows.append((item.spec.index, item.spec.phen, fit))
            _log(
                log,
                f"[h2:batch:fast] completed {item.spec.index + 1}/{len(specs)} "
                f"{item.spec.phen}: kept={item.n_active}/{M}, "
                f"source={item.source} load={item.load_seconds:.3f}s "
                f"fit={time.monotonic() - trait_started:.3f}s "
                f"h2={fit.h2[-1, 0]:.6g} SE={fit.h2[-1, 1]:.6g}.",
            )

        result_rows.sort(key=lambda value: value[0])
        batch_end = batch_start + B
        crossed_checkpoint = (
            batch_end // checkpoint_every
            > batch_start // checkpoint_every
        )
        if crossed_checkpoint or batch_end == len(specs):
            _write_results_table(results_path, result_rows, K)
        _log(
            log,
            f"[h2:batch:fast] batch {batch_start // batch_size + 1} completed "
            f"({B} traits) in {time.monotonic() - batch_started:.3f}s.",
        )
        del Ay, active, Y, loaded
        gc.collect()

    elapsed = time.monotonic() - started
    _log(log, "Analysis ended at: " + utils._get_timestr(utils._get_time()))
    _log(log, f"run time: {elapsed:.3f} s")
    if args.h2_cache_only:
        _log(log, f"[h2:batch:fast] cache build complete for {len(specs)} trait(s).")
    else:
        _write_results_table(results_path, result_rows, K)
        _log(log, f"Saved h2 results table in {results_path}")
    return result_rows
