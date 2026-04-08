from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

import utils
from logger import Logger
from gw_ldscore import GenomewideLDScore, apply_env
from win_ldscore import WindowedLDScore
from sumrhe import Sumrhe
from sumcore import Sumcore
from trace import Trace
from sumstats import Sumstats
from rg_manifest_builder import build_rg_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SUMMIT: Summary-stats-based Unified Method for Multivariate Inference of Traits"
    )

    # Trace / LD input
    parser.add_argument("--trace", default=None, type=str,
                        help="Path to trace summaries (.tr/.MN). Currently unsupported in the refactored h2/rg path.")
    parser.add_argument("--save-trace", default=None, type=str,
                        help="Output prefix for saving trace summaries. Currently unused in the refactored h2/rg path.")
    parser.add_argument("--bim", default=None, type=str,
                        help="Reference .bim file used for annotation alignment or trace summaries.")
    parser.add_argument("--ldscores", default=None, type=str,
                        help="Path to the primary LD-score file.")
    parser.add_argument("--ldscores-reg", default=None, type=str,
                        help="Optional LD-score file used only for the bivariate intercept regression.")
    parser.add_argument("--ldscores-reg-w", default=None, type=str,
                        help="Optional 1D LD-score weight file used only for the bivariate intercept regression.")
    parser.add_argument("--collapse-reg-ld", action="store_true", default=False,
                        help="Collapse multi-column regression LD to 1D total LD for the bivariate intercept fit.")

    # Sumstats / regression mode
    parser.add_argument("--h2", default=None, type=str,
                        help="Path to one summary-statistics file or a directory of files for univariate h2 estimation.")
    parser.add_argument("--rg", default=None, type=str,
                        help=(
                            "Either a comma-separated pair of summary-statistics files for bivariate rg estimation, "
                            "or a manifest file path for batch rg. Manifest mode currently requires per-row phen1, phen2, "
                            "sumstats1, sumstats2, and intercept_rg columns."
                        ))
    parser.add_argument("--make-rg-manifest", default=None, type=str,
                        help=(
                            "Build an rg manifest TSV from raw phenotype/covariate input and write it to this path. "
                            "Use together with --phen-dir, --sum-dir, and either --pair-list or (--phen-list --all-pairwise)."
                        ))
    
    parser.add_argument("--compact", action="store_true", help="Write a compact rg manifest with only the core columns needed downstream.",)
    parser.add_argument("--phen-dir", default=None, type=str,
                        help=(
                            "Phenotype source for --make-rg-manifest. Either a directory of per-trait phenotype files "
                            "or a single wide phenotype table with FID IID followed by phenotype columns."
                        ))
    parser.add_argument("--cov-dir", default=None, type=str,
                        help=(
                            "Optional covariate source for --make-rg-manifest. Either a directory of per-trait covariate files "
                            "or a single shared covariate table with FID IID followed by covariate columns."
                        ))
    parser.add_argument("--sum-dir", default=None, type=str,
                        help=(
                            "Sumstats source for --make-rg-manifest. Either a directory of per-trait sumstats files "
                            "or a mapping file with columns phen,sumstats."
                        ))
    parser.add_argument("--phen-list", default=None, type=str,
                        help="One-column phenotype list used by --make-rg-manifest.")
    parser.add_argument("--pair-list", default=None, type=str,
                        help="Two-column phenotype pair list used by --make-rg-manifest.")
    parser.add_argument("--all-pairwise", action="store_true", default=False,
                        help="In --make-rg-manifest mode, build all unordered pairs from --phen-list.")
    parser.add_argument("--max-chisq", default=None, type=str,
                        help="Main chi^2 threshold. Use 'auto' for max(80, 0.001*Nmax).")
    parser.add_argument("--intercept-chisq-thr", default=None, type=str,
                        help="Chi^2 threshold used only for the cross-trait intercept regression. Use 'auto' for max(80, 0.001*Nmax).")
    parser.add_argument("--intercept-weight-mode", default="ldsc", type=str,
                        choices=["ldsc", "score"],
                        help="Weighting scheme for the constrained cross-trait intercept fit: 'ldsc' for LDSC-style IRWLS weights, or 'score' for fixed w_j = 1 / w_ld,j.")
    parser.add_argument("--chisq-action", default="drop", type=str,
                        choices=["drop", "clip", "warn", "none"],
                        help="What to do with high-chi^2 SNPs on the main analysis axis.")

    parser.add_argument("--intercept-rg", default=None, type=float, help=(
        "Fix the SUMCORE nuisance offset c for rg estimation. "
        "This must be c = y_overlap^T y_overlap / sqrt(N1*N2) = N_overlap * rho_y,overlap / sqrt(N1*N2)."
    ))
    parser.add_argument("--pheno-rg", default=None, type=str, help=(
        "Comma-separated pair of phenotype files for the traits in --rg. "
        "Each file should contain sample ID column(s) followed by the phenotype in the last column. "
        "SUMCORE standardizes each phenotype on its own study sample, intersects overlapping IDs, "
        "and computes c = y_overlap^T y_overlap / sqrt(N1*N2). Mutually exclusive with --intercept-rg."
    ))
    parser.add_argument(
        "--pheno-rg-cov",
        default=None,
        type=str,
        help=(
            "Comma-separated pair of covariate files aligned with --pheno-rg. "
            "Files must be whitespace-delimited with headers; first two columns must be FID and IID; "
            "all remaining columns are used as covariates. "
            "SUMCORE residualizes each phenotype on its trait-specific covariates before computing c."
        ),
    )
    parser.add_argument(
        "--pheno-rg-missing-values",
        default="-9",
        type=str,
        help=(
            "Comma-separated tokens treated as missing in --pheno-rg phenotype files. "
            "Applies to the phenotype column only. For whitespace-delimited files, blank/omitted fields are not reliably detectable; "
            "use an explicit token such as -9 or NA."
        ),
    )
    parser.add_argument(
        "--pheno-rg-cov-missing-values",
        default="-9,NA,NaN,nan,.,None,NONE,null,NULL",
        type=str,
        help=(
            "Comma-separated tokens treated as missing in --pheno-rg-cov files. "
            "Rows with any missing covariate are dropped before residualization. For whitespace-delimited files, blank/omitted fields are not reliably detectable; "
            "use an explicit token such as -9 or NA."
        ),
    )
    parser.add_argument("--cov-rank", default=None, type=str,
                        help="Specify the rank of the covariate matrix (comma-separated). Must be non-negative. Default is 0.")


    # Additional input
    parser.add_argument("--annot", default=None, type=str,
                        help="Path to the annotation file.")

    # Output / behavior
    parser.add_argument("--out", default=None, type=str,
                        help="Output prefix for single-run modes. In rg manifest mode, this must be an output directory.")
    parser.add_argument("--verbose", nargs="?", const="1", default="0", type=str,
                        help="Verbosity / extra-output mode: 0, 1, 2, 'max', 'jack', or 'normeq'. Passing --verbose with no value implies 1.")
    parser.add_argument("--suppress", action="store_true", default=False,
                        help="Suppress stdout logging; still write to the log file(s).")
    parser.add_argument("--allow-neg-enr", action="store_true", default=False,
                        help="Allow negative enrichment estimates.")
    parser.add_argument("--clip-nonfinite-vals", action="store_true", default=False,
                        help="Clip non-finite h2/tau values to 0.0 instead of propagating NaN.")
    parser.add_argument("--enrich-mode", choices=["auto", "overlap", "non-overlap", "both"], default="auto")

    # SE / jackknife
    parser.add_argument("--njack", default="chr", type=str, help=(
        "Jackknife scheme for LD-score input.\n"
        "  * integer (e.g., 1000): contiguous SNP blocks\n"
        "  * 'chr'               : LOCO delete-1\n"
        "  * 'chr:d'             : delete-d LOCO over chromosomes\n"
        "  * 'chr:d:R'           : random R delete-d replicates\n"
        "  * 'chr:d:R:seed'      : random R delete-d replicates with fixed seed\n"
    ))
    parser.add_argument("--jack-mode", default="mean", type=str,
                        choices=["mean", "median", "full"],
                        help="Center used in jackknife SE calculation.")
    parser.add_argument("--rg-se-method", default="jackknife", type=str,
                        choices=["jackknife", "delta", "robust", "kmoments"],
                        help="SE method for total rg.")
    parser.add_argument("--adjust-delta", action="store_true", default=False,
                        help="Apply delta-based deleted-source correction when Trace.delta is available.")

    # Allele alignment for rg
    parser.add_argument("--align-alleles", action="store_true", default=False,
                        help="Align the second trait to the first trait by allele labels before rg estimation.")
    parser.add_argument("--keep-ambiguous", action="store_true", default=False,
                        help="Keep strand-ambiguous SNPs during allele alignment.")

    # LD-score generation mode
    parser.add_argument("--geno", default=None, type=str,
                        help="Path to the genotype file for LD-score calculation.")
    parser.add_argument("--nvecs", default=1000, type=int,
                        help="Number of random vectors for stochastic genome-wide LD scores.")
    parser.add_argument("--step_size", default=1000, type=int,
                        help="Step size for LD-score computation.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed.")
    parser.add_argument("--covar", default=None, type=str,
                        help="Covariate file for LD-score estimation.")
    parser.add_argument("--rand-dist", default="spherical", type=str,
                        help="Distribution for randomized LD-score estimation.")
    parser.add_argument("--dtype", default="float32", type=str,
                        help="dtype for LD-score computation.")
    parser.add_argument("--rand-samp", default=None, type=str,
                        help="Random subset of samples: ratio in (0,1] or an integer count >100.")
    parser.add_argument("--ddof", default=1, type=int,
                        help="ddof used in LD-score estimation.")
    parser.add_argument("--ld-wind-kb", default=None, type=float,
                        help="If set, compute windowed LD scores with the given kb window.")
    parser.add_argument("--correct-skew", action="store_true",
                        help="Enable optional finite-sample skew diagnostics in genome-wide LD-score estimation.")

    # Resource / performance knobs
    parser.add_argument("--num-threads", default=None, type=int,
                        help="Cap BLAS / compute threads.")
    parser.add_argument("--target-xz-mem", type=float, default=16.0,
                        help="Memory budget (GB) for Phase-1 Xz panel.")
    parser.add_argument("--target-mem", type=float, default=None,
                        help="Overall memory budget (GB) for LD-score estimation.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for GWLD computation.")
    parser.add_argument("--use-tp32", action="store_true", default=False,
                        help="Use TP32 for GPU-backed computation.")
    parser.add_argument("--vchunk", type=int, default=None,
                        help="Fixed V-chunk size.")
    parser.add_argument("--vtiles", type=int, default=None,
                        help="Force number of V-tiles.")
    parser.add_argument("--q-panel", type=int, default=None)
    parser.add_argument("--reduce-blk", type=int, default=None)
    parser.add_argument("--reduce-threads", type=int, default=None)
    parser.add_argument("--malloc-arena-max", type=int, default=2)
    parser.add_argument("--malloc-trim-threshold", type=int, default=131072)
    parser.add_argument("--malloc-mmap-threshold", type=int, default=131072)
    parser.add_argument("--numa-mode", default="interleave", choices=["interleave", "membind", "cpunodebind", "preferred"])
    parser.add_argument("--numa-nodes", default="all")
    parser.add_argument("--force_affinity_all", default=True)
    parser.add_argument("--decode_threads_cap", default=32)

    return parser


def _verbose_to_level(verbose) -> int:
    return utils._parse_verbose(verbose)


def _check_outdir(path_str: str, create: bool = True, log=None):
    if not path_str:
        return
    p = Path(path_str)
    parent = p if path_str.endswith(os.sep) else (p.parent if p.parent != Path("") else Path("."))
    try:
        if create:
            parent.mkdir(parents=True, exist_ok=True)
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Directory '{parent}' is not writable by the current user.")
        with tempfile.NamedTemporaryFile(dir=str(parent), prefix=".summit_perm_check_", delete=True):
            pass
        if log is not None:
            log._log(f"[io] Using output directory: {parent}")
    except Exception as e:
        if log is not None:
            log._log(f"!!! Cannot write to output directory '{parent}': {e} !!!")
        else:
            print(f"!!! Cannot write to output directory '{parent}': {e} !!!", file=sys.stderr)
        raise SystemExit(1)


def _check_output_directory(path_str: str, create: bool = True, log=None):
    if not path_str:
        raise SystemExit(1)
    p = Path(path_str)
    try:
        if p.exists() and not p.is_dir():
            raise NotADirectoryError(f"'{p}' exists but is not a directory.")
        if create:
            p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise PermissionError(f"Directory '{p}' is not writable by the current user.")
        with tempfile.NamedTemporaryFile(dir=str(p), prefix=".summit_perm_check_", delete=True):
            pass
        if log is not None:
            log._log(f"[io] Using output directory: {p}")
    except Exception as e:
        if log is not None:
            log._log(f"!!! Cannot write to output directory '{p}': {e} !!!")
        else:
            print(f"!!! Cannot write to output directory '{p}': {e} !!!", file=sys.stderr)
        raise SystemExit(1)


def _parse_rand_samp(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    if any(ch in s for ch in [".", "e", "E"]):
        return float(s)
    return int(s)


def _log_cli_args(parser, args, log):
    log._log(">>> SUMMIT arguments")
    tokens = sys.argv[1:]
    opts_take_value = set()
    for a in parser._actions:
        if not a.option_strings:
            continue
        if a.nargs == 0:
            continue
        for s in a.option_strings:
            opts_take_value.add(s)

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--") and "=" in t:
            log._log("\t" + t)
            i += 1
            continue
        if t in opts_take_value:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                log._log(f"\t{t} {tokens[i + 1]}")
                i += 2
            else:
                log._log(f"\t{t}")
                i += 1
        elif t.startswith("-"):
            log._log("\t" + t)
            i += 1
        else:
            log._log("\t" + t)
            i += 1

    if _verbose_to_level(args.verbose) > 0:
        log._log(">>> Effective options")
        for k, v in sorted(vars(args).items()):
            log._log(f"  {k} = {v!r}")
    log._log("=========================================================================='".replace("'", ""))


def _make_low_level_env(args):
    return {
        "numa_mode": args.numa_mode,
        "numa_nodes": args.numa_nodes,
        "q_panel": args.q_panel,
        "reduce_blk": args.reduce_blk,
        "reduce_threads": args.reduce_threads,
        "malloc_arena_max": args.malloc_arena_max,
        "malloc_trim_threshold": args.malloc_trim_threshold,
        "malloc_mmap_threshold": args.malloc_mmap_threshold,
        "force_affinity_all": args.force_affinity_all,
        "decode_threads_cap": args.decode_threads_cap,
    }


def _dispatch_ldscore(args, log, verbose_on, low_level):
    if args.ld_wind_kb is not None:
        if args.ld_wind_kb <= 0:
            log._log("!!! --ld-wind-kb must be positive !!!")
            raise SystemExit(1)
        log._log(f">>> LD score mode: windowed, --ld-wind-kb {args.ld_wind_kb}")
        winld = WindowedLDScore(
            bed_path=args.geno,
            annot_path=args.annot,
            out_path=args.out,
            covar_path=args.covar,
            ld_wind_kb=args.ld_wind_kb,
            log=log,
            seed=args.seed,
            step_size=args.step_size,
            verbose=verbose_on,
            dtype=args.dtype,
            rand_samp=args.rand_samp,
            ddof=args.ddof,
            num_threads=args.num_threads,
        )
        winld._compute_ldscore()
        return

    gwld = GenomewideLDScore(
        bed_path=args.geno,
        annot_path=args.annot,
        out_path=args.out,
        covar_path=args.covar,
        rand_dist=args.rand_dist,
        log=log,
        num_vecs=args.nvecs,
        step_size=args.step_size,
        seed=args.seed,
        verbose=verbose_on,
        dtype=args.dtype,
        num_threads=args.num_threads,
        rand_samp=args.rand_samp,
        low_level=low_level,
        target_xz_mem=args.target_xz_mem,
        target_mem=args.target_mem,
        device=args.device,
        use_tp32=args.use_tp32,
        correct_skew=args.correct_skew,
    )
    gwld._compute_ldscore()


def _dispatch_h2(args, log):
    if args.trace is not None:
        log._log("!!! Trace summaries are not supported in the refactored h2 path yet. Use --ldscores. !!!")
        raise SystemExit(1)
    if args.ldscores is None:
        log._log("!!! --ldscores must be provided for refactored h2 estimation. !!!")
        raise SystemExit(1)

    sums = Sumrhe(
        bim_path=args.bim,
        sum_path=None,
        h2_path=args.h2,
        out=args.out,
        chisq_threshold=args.max_chisq,
        log=log,
        verbose=args.verbose,
        ldscores=args.ldscores,
        njack=args.njack,
        annot=args.annot,
        chisq_action=args.chisq_action,
        allow_neg_enr=args.allow_neg_enr,
        clip_nonfinite_vals=args.clip_nonfinite_vals,
        adjust_delta=args.adjust_delta,
        enrich_mode=args.enrich_mode,
        jack_mode=args.jack_mode,
    )
    sums._run()
    sums._logoff()


def _dispatch_rg(args, log):
    if args.trace is not None:
        log._log("!!! Trace summaries are not supported in the refactored rg path yet. Use --ldscores. !!!")
        raise SystemExit(1)
    if args.ldscores is None:
        log._log("!!! --ldscores must be provided for rg estimation. !!!")
        raise SystemExit(1)

    rg = Sumcore(
        bim_path=args.bim,
        rg=args.rg,
        chisq_threshold=args.max_chisq,
        intercept_chisq_thr=args.intercept_chisq_thr,
        intercept_rg=args.intercept_rg,
        pheno_rg=args.pheno_rg,
        pheno_rg_cov=args.pheno_rg_cov,
        pheno_rg_missing_values=args.pheno_rg_missing_values,
        pheno_rg_cov_missing_values=args.pheno_rg_cov_missing_values,
        intercept_weight_mode=args.intercept_weight_mode,
        chisq_action=args.chisq_action,
        log=log,
        verbose=args.verbose,
        out=args.out,
        ldscores=args.ldscores,
        ldscores_reg=args.ldscores_reg,
        ldscores_reg_w=args.ldscores_reg_w,
        njack=args.njack,
        annot=args.annot,
        enrich_mode=args.enrich_mode,
        jack_mode=args.jack_mode,
        collapse_reg_ld=args.collapse_reg_ld,
        clip_nonfinite_vals=args.clip_nonfinite_vals,
        rg_se_method=args.rg_se_method,
        align_alleles=args.align_alleles,
        drop_ambiguous=(not args.keep_ambiguous),
        allow_neg_enr=args.allow_neg_enr,
        report_tau=True,
        adjust_delta=args.adjust_delta,
        cov_rank=args.cov_rank,
    )
    rg._run()
    rg._logoff()


def _manifest_column(df: pd.DataFrame, name: str):
    lower = {str(c).strip().lower(): c for c in df.columns}
    return lower.get(name.strip().lower())


def _read_rg_manifest(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", compression="infer")
    if df.shape[0] == 0:
        raise ValueError(f"RG manifest '{path}' has no rows.")
    return df


def _coerce_optional_cov_rank(val, *, row_label: str, col_name: str):
    if pd.isna(val):
        return None
    fv = float(val)
    if not pd.notna(fv):
        raise ValueError(f"Manifest row {row_label}: non-finite {col_name}={val!r}.")
    iv = int(round(fv))
    if abs(fv - iv) > 1e-8:
        raise ValueError(f"Manifest row {row_label}: {col_name} must be an integer; got {val!r}.")
    if iv < 0:
        raise ValueError(f"Manifest row {row_label}: {col_name} must be non-negative; got {iv}.")
    return iv


def _normalize_rg_manifest(path: str, log=None):
    raw = _read_rg_manifest(path)

    required = ["phen1", "phen2", "sumstats1", "sumstats2", "intercept_rg"]
    cols = {}
    for name in required:
        hit = _manifest_column(raw, name)
        if hit is None:
            raise ValueError(f"RG manifest '{path}' is missing required column '{name}'.")
        cols[name] = hit

    opt_cov1 = _manifest_column(raw, "cov_rank1")
    opt_cov2 = _manifest_column(raw, "cov_rank2")

    rows = []
    for idx, row in raw.iterrows():
        row_id = int(idx) + 1
        phen1 = str(row[cols["phen1"]]).strip() if not pd.isna(row[cols["phen1"]]) else ""
        phen2 = str(row[cols["phen2"]]).strip() if not pd.isna(row[cols["phen2"]]) else ""
        sumstats1_raw = str(row[cols["sumstats1"]]).strip() if not pd.isna(row[cols["sumstats1"]]) else ""
        sumstats2_raw = str(row[cols["sumstats2"]]).strip() if not pd.isna(row[cols["sumstats2"]]) else ""
        if phen1 == "" or phen2 == "":
            raise ValueError(f"Manifest row {row_id}: phen1/phen2 must be non-empty.")
        if sumstats1_raw == "" or sumstats2_raw == "":
            raise ValueError(f"Manifest row {row_id}: sumstats1/sumstats2 must be non-empty.")

        sumstats1 = str(Path(sumstats1_raw).expanduser().resolve())
        sumstats2 = str(Path(sumstats2_raw).expanduser().resolve())
        if not Path(sumstats1).is_file():
            raise ValueError(f"Manifest row {row_id}: could not find sumstats1 file '{sumstats1_raw}'.")
        if not Path(sumstats2).is_file():
            raise ValueError(f"Manifest row {row_id}: could not find sumstats2 file '{sumstats2_raw}'.")

        intercept_rg = pd.to_numeric(pd.Series([row[cols["intercept_rg"]]]), errors="coerce").iloc[0]
        if not pd.notna(intercept_rg):
            raise ValueError(f"Manifest row {row_id}: intercept_rg must be finite.")
        intercept_rg = float(intercept_rg)

        cov_rank1 = _coerce_optional_cov_rank(row[opt_cov1], row_label=str(row_id), col_name="cov_rank1") if opt_cov1 is not None else None
        cov_rank2 = _coerce_optional_cov_rank(row[opt_cov2], row_label=str(row_id), col_name="cov_rank2") if opt_cov2 is not None else None

        rows.append({
            "row_id": row_id,
            "phen1": phen1,
            "phen2": phen2,
            "sumstats1": sumstats1,
            "sumstats2": sumstats2,
            "intercept_rg": intercept_rg,
            "cov_rank1": cov_rank1,
            "cov_rank2": cov_rank2,
            "out_stem": utils._pair_output_stem(phen1, phen2),
        })

    df = pd.DataFrame(rows)
    if df.shape[0] == 0:
        raise ValueError(f"RG manifest '{path}' has no usable rows.")

    dup_cols = ["phen1", "phen2", "sumstats1", "sumstats2", "intercept_rg", "cov_rank1", "cov_rank2"]
    dup_mask = df.duplicated(subset=dup_cols, keep=False)
    if dup_mask.any():
        bad_rows = ", ".join(str(x) for x in df.loc[dup_mask, "row_id"].tolist())
        raise ValueError(f"RG manifest '{path}' contains duplicate rows (row ids: {bad_rows}).")

    stem_dup = df.duplicated(subset=["out_stem"], keep=False)
    if stem_dup.any():
        bad = df.loc[stem_dup, ["row_id", "phen1", "phen2", "out_stem"]]
        raise ValueError(
            "RG manifest output-name collision after sanitization. Conflicting rows:\n" + bad.to_string(index=False)
        )

    path_to_name = {}
    name_to_path = {}
    path_to_cov = {}
    for row in df.itertuples(index=False):
        for phen, spath, cov_rank in ((row.phen1, row.sumstats1, row.cov_rank1), (row.phen2, row.sumstats2, row.cov_rank2)):
            prev_name = path_to_name.get(spath)
            if prev_name is not None and prev_name != phen:
                raise ValueError(
                    f"Manifest path/name conflict: sumstats path '{spath}' is associated with both '{prev_name}' and '{phen}'."
                )
            path_to_name[spath] = phen

            prev_path = name_to_path.get(phen)
            if prev_path is not None and prev_path != spath:
                raise ValueError(
                    f"Manifest name/path conflict: phenotype '{phen}' is associated with both '{prev_path}' and '{spath}'."
                )
            name_to_path[phen] = spath

            if cov_rank is not None:
                prev_cov = path_to_cov.get(spath)
                if prev_cov is not None and prev_cov != cov_rank:
                    raise ValueError(
                        f"Manifest cov_rank conflict: sumstats path '{spath}' appears with conflicting cov_rank values {prev_cov} and {cov_rank}."
                    )
                path_to_cov[spath] = cov_rank
            else:
                path_to_cov.setdefault(spath, None)

    trait_meta = {
        spath: {"phen": path_to_name[spath], "cov_rank": path_to_cov.get(spath, None)}
        for spath in sorted(path_to_name.keys())
    }

    if log is not None:
        log._log(
            f"[rg:manifest] loaded {df.shape[0]} pair(s) from '{path}' "
            f"with {len(trait_meta)} unique trait file(s)."
        )

    return df, trait_meta


def _dispatch_rg_manifest(args, log):
    if args.trace is not None:
        log._log("!!! Trace summaries are not supported in the refactored rg path yet. Use --ldscores. !!!")
        raise SystemExit(1)
    if args.ldscores is None:
        log._log("!!! --ldscores must be provided for rg estimation. !!!")
        raise SystemExit(1)
    if args.intercept_rg is not None or args.pheno_rg is not None or args.pheno_rg_cov is not None:
        log._log("!!! In rg manifest mode, use per-row intercept_rg in the manifest. Global --intercept-rg / --pheno-rg / --pheno-rg-cov are not allowed. !!!")
        raise SystemExit(1)
    if args.cov_rank is not None:
        log._log("!!! In rg manifest mode, provide trait-specific cov_rank via optional manifest columns cov_rank1 / cov_rank2 or via the sumstats files. Global --cov-rank is not allowed. !!!")
        raise SystemExit(1)

    manifest_df, trait_meta = _normalize_rg_manifest(args.rg, log=log)

    shared_trace = Trace(
        bimpath=args.bim,
        sumpath=None,
        savepath=None,
        log=log,
        ldscores=args.ldscores,
        ldscores_reg=args.ldscores_reg,
        ldscores_reg_w=args.ldscores_reg_w,
        annot=args.annot,
        verbose=bool(_verbose_to_level(args.verbose)),
        delta=None,
    )

    sumstats_cache = {}
    for spath, meta in trait_meta.items():
        cov_rank = meta.get("cov_rank", None)
        sumstats_cache[spath] = Sumstats.from_file(
            spath,
            name=meta.get("phen", utils._phen_name_from_path(spath)),
            log=log,
            cov_rank=cov_rank,
            cov_rank_source=("manifest" if cov_rank is not None else None),
        )

    results_rows = []
    outdir = Path(args.out)

    for i, row in enumerate(manifest_df.itertuples(index=False), start=1):
        pair_prefix = str(outdir / row.out_stem)
        pair_log = Logger(suppress=args.suppress)
        pair_log._log(
            f"[rg:manifest] running pair {i}/{manifest_df.shape[0]}: "
            f"{row.phen1} vs {row.phen2}"
        )

        ss1 = sumstats_cache[row.sumstats1]
        ss2 = sumstats_cache[row.sumstats2]
        ss1.log = pair_log
        ss2.log = pair_log

        rg = Sumcore(
            bim_path=args.bim,
            rg=None,
            ldscores=None,
            ldscores_reg=None,
            ldscores_reg_w=None,
            log=pair_log,
            verbose=args.verbose,
            chisq_threshold=args.max_chisq,
            annot=args.annot,
            njack=args.njack,
            out=pair_prefix,
            align_alleles=args.align_alleles,
            drop_ambiguous=(not args.keep_ambiguous),
            collapse_reg_ld=args.collapse_reg_ld,
            enrich_mode=args.enrich_mode,
            jack_mode=args.jack_mode,
            clip_nonfinite_vals=args.clip_nonfinite_vals,
            rg_se_method=args.rg_se_method,
            intercept_chisq_thr=args.intercept_chisq_thr,
            intercept_weight_mode=args.intercept_weight_mode,
            intercept_rg=row.intercept_rg,
            chisq_action=args.chisq_action,
            report_tau=True,
            allow_neg_enr=args.allow_neg_enr,
            adjust_delta=args.adjust_delta,
            cov_rank=None,
            trace_obj=shared_trace,
            sumstats_pair=(ss1, ss2),
            phen_names=(row.phen1, row.phen2),
        )

        try:
            res = rg._run()
            rg._logoff()
        except Exception:
            try:
                pair_log._save_log(pair_prefix + ".log")
            except Exception:
                pass
            raise

        h2_fit1 = res["h2_fit1"]
        h2_fit2 = res["h2_fit2"]
        intercept = res["intercept"]
        rg_fit = res["rg_fit"]

        results_rows.append({
            "phen1": row.phen1,
            "phen2": row.phen2,
            "sumstats1": row.sumstats1,
            "sumstats2": row.sumstats2,
            "cov_rank1": row.cov_rank1,
            "cov_rank2": row.cov_rank2,
            "intercept_rg_input": row.intercept_rg,
            "out_prefix": pair_prefix,
            "h2_trait1": float(h2_fit1.h2[-1, 0]),
            "h2_trait1_se": float(h2_fit1.h2[-1, 1]),
            "h2_trait2": float(h2_fit2.h2[-1, 0]),
            "h2_trait2_se": float(h2_fit2.h2[-1, 1]),
            "intercept_c": float(intercept.c[0]),
            "intercept_c_se": float(intercept.c[1]),
            "gamma_g_total": float(rg_fit.gamma_total[0]),
            "gamma_g_total_se": float(rg_fit.gamma_total[1]),
            "rg_total": float(rg_fit.rg_total[0]),
            "rg_total_se": float(rg_fit.rg_total[1]),
        })

        log._log(
            f"[rg:manifest] completed pair {i}/{manifest_df.shape[0]}: "
            f"{row.phen1} vs {row.phen2}; rg={float(rg_fit.rg_total[0]):.6g} "
            f"(SE: {float(rg_fit.rg_total[1]):.6g})"
        )

    summary_path = outdir / "manifest.results.tsv"
    pd.DataFrame(results_rows).to_csv(summary_path, sep="\t", index=False)
    log._log(f"[rg:manifest] saved batch summary to {summary_path}")


def _dispatch_make_rg_manifest(args, log):
    if args.phen_dir is None:
        log._log("!!! --phen-dir must be provided in --make-rg-manifest mode. !!!")
        raise SystemExit(1)
    if args.sum_dir is None:
        log._log("!!! --sum-dir must be provided in --make-rg-manifest mode. !!!")
        raise SystemExit(1)
    if args.trace is not None or args.ldscores is not None or args.h2 is not None or args.rg is not None or args.geno is not None:
        log._log("!!! --make-rg-manifest is a standalone mode and cannot be combined with --geno / --h2 / --rg / trace inputs. !!!")
        raise SystemExit(1)
    if args.phen_list is None and args.pair_list is None:
        log._log("!!! Provide at least one of --phen-list or --pair-list in --make-rg-manifest mode. !!!")
        raise SystemExit(1)

    out_manifest = str(Path(args.make_rg_manifest).expanduser().resolve())
    _check_outdir(out_manifest, create=True, log=log)

    df = build_rg_manifest(
        output_path=out_manifest,
        phen_source=args.phen_dir,
        sumstats_source=args.sum_dir,
        cov_source=args.cov_dir,
        phen_list_path=args.phen_list,
        pair_list_path=args.pair_list,
        all_pairwise=bool(args.all_pairwise),
        pheno_missing_values=Sumcore._parse_missing_tokens(args.pheno_rg_missing_values),
        cov_missing_values=Sumcore._parse_missing_tokens(args.pheno_rg_cov_missing_values),
        log=log,
    )
    log._log(
        f"[make-rg-manifest] completed successfully: wrote {df.shape[0]} pair(s) to '{out_manifest}'."
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.rand_samp = _parse_rand_samp(args.rand_samp)

    verbose_level = _verbose_to_level(args.verbose)
    verbose_on = verbose_level > 0
    rg_manifest_mode = utils._is_rg_manifest_arg(args.rg) if args.rg is not None else False
    build_manifest_mode = args.make_rg_manifest is not None

    low_level = _make_low_level_env(args)
    apply_env(low_level)

    log = Logger(suppress=args.suppress)
    _log_cli_args(parser, args, log)

    if args.out is None:
        log._log("!!! --out must be provided. !!!")
        raise SystemExit(1)

    if rg_manifest_mode:
        _check_output_directory(args.out, create=True, log=log)
    else:
        _check_outdir(args.out, create=True, log=log)

    log.install_excepthook()
    if rg_manifest_mode:
        log.attach_file(str(Path(args.out) / "batch.log"))
    else:
        log_suffix = ".win.log" if (args.geno and args.ld_wind_kb is not None) else (".gw.log" if args.geno else ".log")
        log.attach_file(args.out + log_suffix)

    modes = int(args.geno is not None) + int(args.h2 is not None) + int(args.rg is not None) + int(build_manifest_mode)
    if modes != 1:
        log._log("!!! Exactly one of --geno / --h2 / --rg / --make-rg-manifest must be specified. !!!")
        raise SystemExit(1)

    if build_manifest_mode:
        _dispatch_make_rg_manifest(args, log)
    elif args.geno is not None:
        _dispatch_ldscore(args, log, verbose_on, low_level)
    elif args.h2 is not None:
        _dispatch_h2(args, log)
    else:
        if rg_manifest_mode:
            _dispatch_rg_manifest(args, log)
        else:
            _dispatch_rg(args, log)


if __name__ == "__main__":
    main()
