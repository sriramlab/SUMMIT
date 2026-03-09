from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from logger import Logger
from gw_ldscore import GenomewideLDScore, apply_env
from win_ldscore import WindowedLDScore
from sumrhe import Sumrhe
from sumcore import Sumcore


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
    parser.add_argument("--collapse-reg-ld", action="store_true", default=False,
                        help="Collapse multi-column regression LD to 1D total LD for the bivariate intercept fit.")

    # Sumstats / regression mode
    parser.add_argument("--h2", default=None, type=str,
                        help="Path to one summary-statistics file or a directory of files for univariate h2 estimation.")
    parser.add_argument("--rg", default=None, type=str,
                        help="Comma-separated pair of summary-statistics files for bivariate rg estimation.")
    parser.add_argument("--max-chisq", default=None, type=str,
                        help="Main chi^2 threshold. Use 'auto' for max(80, 0.001*Nmax).")
    parser.add_argument("--intercept-chisq-thr", default=None, type=str,
                        help="Chi^2 threshold used only for the cross-trait intercept regression. Use 'auto' for max(80, 0.001*Nmax).")
    parser.add_argument("--chisq-action", default="drop", type=str,
                        choices=["drop", "clip", "warn", "none"],
                        help="What to do with high-chi^2 SNPs on the main analysis axis.")

    # Additional input
    parser.add_argument("--annot", default=None, type=str,
                        help="Path to the annotation file.")
    parser.add_argument("--thin-annot", action="store_true", default=False,
                        help="Compatibility flag; the refactored Trace auto-detects thin/full annotation.")

    # Output / behavior
    parser.add_argument("--out", default=None, type=str,
                        help="Output prefix for logs / result dumps.")
    parser.add_argument("--verbose", nargs="?", const="1", default="0", type=str,
                        help="Verbosity level: 0, 1, or 'max'. Passing --verbose with no value implies 1.")
    parser.add_argument("--suppress", action="store_true", default=False,
                        help="Suppress stdout logging; still write to the log file.")
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
                        choices=["jackknife", "delta"],
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
    parser.add_argument("--hybrid", action="store_true",
                        help="Use the hybrid LD-score estimator.")
    parser.add_argument("--hybrid-window-kb", type=float, default=20000.0,
                        help="Local exact window (kb) for the hybrid estimator.")

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
    if isinstance(verbose, str):
        s = verbose.strip().lower()
        if s in ("0", "false", "none", "off", ""):
            return 0
        if s in ("1", "true", "yes", "on"):
            return 1
        if s == "max":
            return 2
        try:
            v = int(s)
            return max(v, 0)
        except Exception:
            return 1
    return 1 if bool(verbose) else 0


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
    log._log("==========================================================================")


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
        hybrid=args.hybrid,
        hybrid_window_kb=args.hybrid_window_kb,
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
        save_path=args.save_trace,
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
        chisq_action=args.chisq_action,
        log=log,
        verbose=args.verbose,
        out=args.out,
        ldscores=args.ldscores,
        ldscores_reg=args.ldscores_reg,
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
    )
    rg._run()
    rg._logoff()


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.rand_samp = _parse_rand_samp(args.rand_samp)

    verbose_level = _verbose_to_level(args.verbose)
    verbose_on = verbose_level > 0

    low_level = _make_low_level_env(args)
    apply_env(low_level)

    log = Logger(suppress=args.suppress)
    _log_cli_args(parser, args, log)

    if args.out is None:
        log._log("!!! --out must be provided. !!!")
        raise SystemExit(1)
    _check_outdir(args.out, create=True, log=log)

    log.install_excepthook()
    log_suffix = ".win.log" if (args.geno and args.ld_wind_kb is not None) else (".gw.log" if args.geno else ".log")
    log.attach_file(args.out + log_suffix)

    modes = int(args.geno is not None) + int(args.h2 is not None) + int(args.rg is not None)
    if modes != 1:
        log._log("!!! Exactly one of --geno / --h2 / --rg must be specified. !!!")
        raise SystemExit(1)

    if args.geno is not None:
        _dispatch_ldscore(args, log, verbose_on, low_level)
    elif args.h2 is not None:
        _dispatch_h2(args, log)
    else:
        _dispatch_rg(args, log)


if __name__ == "__main__":
    main()
