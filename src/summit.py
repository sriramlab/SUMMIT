from logger import Logger
from gw_ldscore import GenomewideLDScore
from sumrhe import Sumrhe
from sumcore import Sumcore
import utils

import argparse
import sys
import numpy as np

from pathlib import Path
import tempfile
import os

parser = argparse.ArgumentParser(description='SUMMIT: integrated tool for heritability & genetic correlation')
parser.add_argument("--trace", default=None, type=str, \
                    help='File path for trace summary statistics (.tr) and corresponding metadata (.MN).'
                     ' If the path is a directory, all trace summaires (ending with .tr) will be used by aggregating them.')
parser.add_argument("--save-trace", default=None, type=str, \
                    help='File path for saving (aggregated) trace summaries (.tr) and corresponding metadata (.MN)')
parser.add_argument("--bim", default=None, type=str, \
                    help='File path for the reference .bim file used for trace calculation (required for trace summaries).')
parser.add_argument("--max-chisq", action='store', default=None, type=float, \
                    help='Filter out SNPs with chi-sq statistic above the threshold.'
                    ' This can be done either only on the yKy or on both sides (use --filter-both-sides);'
                    ' with many non-polygenic SNPs, one-sided filtering might not be accurate')
parser.add_argument("--ldscores", default=None, type=str, \
                    help='File path for LD scores of the reference SNPs. You may use either the traditional (truncated) LD scores (.l2.ldscore.gz) or genome-wide stochastic LD scores (.gw.ldscore.gz)')
parser.add_argument("--out", default=None, type=str, \
                    help='Output file path to save the analysis log and result (.log) or the genome-wide LD scores (.gw.ldscore.gz)')
parser.add_argument("--verbose", action="store_true", default=False,\
                    help='Verbose mode: print out the normal equations')
parser.add_argument("--suppress", action="store_true", default=False,\
                    help='Suppress mode: do not print out the outputs to stdout (log file only)')
parser.add_argument("--njack", default=100, type=int, \
                    help='Number of jackknife blocks (only if using LD scores as input)')
parser.add_argument("--annot", default=None, type=str, \
                    help='Path of the annotation file (only if using partitioned heritability)')
parser.add_argument("--thin-annot", action='store_true', default=False, \
                    help='Use thin annotation (annotation matrix only) instead of full annotation file')
parser.add_argument("--geno", default=None, type=str, \
                    help='Path of the genotype file to calculate the genome-wide LD scores. Calculates partitioned scores if --annot is also specified.')
parser.add_argument("--nvecs", default=1000, type=int, \
                    help='Number of random vectors to use for estimating stochastic genome-wide LD scores. Default is 1000.')
parser.add_argument("--step_size", default=1000, type=int, \
                    help='Number of SNPs to process in each step of estimating stochastic genome-wide LD scores. Default is 1000.')
parser.add_argument("--seed", default=None, type=int, \
                    help='Seed for estimating stochastic genome-wide LD scores. If not specified, the default numpy (pseudo) random number generator will be used.')
parser.add_argument("--h2", default=None, type=str, \
                   help='File path for phenotype-specific summary statistics (.sumstat[.gz]) to estimate heritability.'
                    ' If the path is a directory, all summary statistics (ending with .sumstat[.gz]) will be used.')
parser.add_argument("--rg", default=None, type=str, \
                   help='Comma-separated file path for a pair of phenotype-specific summary statistics (.sumstat[.gz]) to estimate genetic correlation (rg).')
parser.add_argument("--covar", default=None, type=str, \
                    help='Path of the covariate file to adjust for when calculating the genome-wide LD scores. If not specified, no covariates adjustments are made.')
parser.add_argument("--intercept-rg", action='store', default=None, type=float, \
                    help="Constrain the intercept (LDSC-style) for genetic correlation calculation. This is equivalent to N*rho_e / sqrt(N1*N2); note we ask for covariance of environmental factor, not the correlation!")
                    #TODO: in our framework, it might be better to provide N*rho_e instead of N*gamma_e, since our estimates of \sigma^2_e are a lot more accurate?
parser.add_argument("--pheno-rg", default=None, type=str, \
                    help="Comma-separated file path for a pair of (overlapping) individual-level phenotypes used in the pair of summary statistics (--rg). "
                    "This option may yield more accurate estimates (alternative to --intercept-rg).")
parser.add_argument("--rand-dist", default='spherical', type=str, \
                    help="Specify which distribution to use to generate random vectors ('normal', 'rademacher', 'spherical'). Default is spherical distribution.")
parser.add_argument("--dtype", default='float32', type=str, \
                    help="Specify the dtype to use for calculations (either float32 or float64). Default is float32.")
parser.add_argument("--rand-samp", default=None, type=float, \
                    help="Select a random subset of the samples for LD score calculation. Pass a value between (0, 1] for a ratio, and an integer greater than 100 for the number of samples.)")
parser.add_argument("--ddof", default=1, type=int, \
                    help="Specify the delta degrees of freedom (ddof) for estimating genome-wide LD scores. Default is 1 (empirical SD).")
parser.add_argument("--num-threads", default=4, type=int, \
                    help='Cap the number of threads for BLAS to limit CPU usage. Default is 4.')
parser.add_argument("--target-xz-mem", type=float, default=16.0,
                    help="Memory budget (GB) for the Phase-1 Xz panel (N × B × Vt). Used to pick the initial V-tile before balancing. Default: 16.0")
parser.add_argument("--allow-neg-enr", action="store_true", default=False,\
                    help='Allow negative enrichment estimates. Default is False.')

# Low-level performance knobs
parser.add_argument("--ctile", type=int, default=None,
                    help="Manual CTILE override (columns in the RHS tile). Rounded up to a multiple of 64. If set, overrides --ctile-mib and --ctile-l3pct.")
parser.add_argument("--ctile-mb", type=int, default=None,
                    help="Memory-budget-driven CTILE (MiB). If set, overrides --ctile-l3pct. Mutually exclusive with --ctile.")
parser.add_argument("--ctile-l3pct", type=float, default=0.80,
                    help="Fraction of per-socket L3 cache to target per BLAS thread for CTILE auto-sizing. Ignored if --ctile or --ctile-mib is provided. Default: 0.80")
parser.add_argument("--sockets", type=int, default=None,
                    help="Override the number of CPU sockets for CTILE heuristics. By default it is auto-detected from CPU topology.")
parser.add_argument("--malloc-arena-max", type=int, default=2)
parser.add_argument("--malloc-trim-threshold", type=int, default=131072)
parser.add_argument("--malloc-mmap-threshold", type=int, default=131072)
parser.add_argument("--numa-mode", default="interleave", choices=['interleave', 'membind', 'cpunodebind', 'preferred'])
parser.add_argument("--numa-nodes", default="all")

# Optional overrides for V-tiling (balanced split logic still applies)
parser.add_argument("--vchunk", type=int, default=None,
                    help="Fixed V-chunk size. If set, disables auto memory-based guess.")
parser.add_argument("--vtiles", type=int, default=None,
                    help="Force number of V-tiles; the code will split V evenly into this many tiles.")





def _check_outdir(path_str: str, create: bool = True, log=None):
    """
    Check whether the outdir exists & is writable
    Creates the directory (parents=True) if `create` is True.

    Raises SystemExit(1) on failure after logging a clear message.
    """
    if not path_str:
        return  # nothing to do

    p = Path(path_str)

    # Treat --out / --save-trace as FILE paths; guard against someone passing a dir
    if path_str.endswith(os.sep):
        # If the user accidentally gave a trailing slash, treat it as a directory target
        parent = p
    else:
        parent = p.parent if p.parent != Path('') else Path('.')  # current dir if no parent part

    try:
        if create:
            parent.mkdir(parents=True, exist_ok=True)

        # Basic permission check
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Directory '{parent}' is not writable by the current user.")

        # Stronger check: try creating a temp file
        with tempfile.NamedTemporaryFile(dir=str(parent), prefix='.summit_perm_check_', delete=True):
            pass

        if log:
            log._log(f"[io] Using output directory: {parent}")

    except Exception as e:
        if log:
            log._log(f"!!! Cannot write to output directory '{parent}': {e} !!!")
        else:
            print(f"!!! Cannot write to output directory '{parent}': {e} !!!", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    args = parser.parse_args()
    ## low-level config params that most people won't need
    low_level = {
        "numa_mode": args.numa_mode,
        "numa_nodes": args.numa_nodes,
        "ctile":         args.ctile,
        "ctile_mb":      args.ctile_mb,
        "ctile_l3pct":   args.ctile_l3pct,
        "sockets":       args.sockets,
        "malloc_arena_max":        args.malloc_arena_max,
        "malloc_trim_threshold":   args.malloc_trim_threshold,
        "malloc_mmap_threshold":   args.malloc_mmap_threshold,
    }
    from gw_ldscore import apply_env
    apply_env(low_level)
    
    log  = Logger(suppress=args.suppress)

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

    if (args.verbose):
        log._log(">>> Effective options")
        for k, v in sorted(vars(args).items()):
            log._log(f"  {k} = {v!r}")
    log._log("==========================================================================")

    if (args.out is None):
        log._log("!!! An output path to save the results must be provided !!!")
        sys.exit(1)
    else:
        _check_outdir(args.out, create=True, log=log)
    
    log.install_excepthook()
    log.attach_file(args.out + (".gw.log" if args.geno else ".log"))
    
    if (args.geno is not None):
        # set 
        gwld = GenomewideLDScore(bed_path=args.geno, annot_path=args.annot, out_path=args.out, covar_path=args.covar, rand_dist=args.rand_dist,\
            log=log, num_vecs=args.nvecs, step_size=args.step_size, seed=args.seed, verbose=args.verbose, \
                dtype = args.dtype, num_threads=args.num_threads, rand_samp=args.rand_samp, low_level=low_level, target_xz_mem=args.target_xz_mem)
        gwld._compute_ldscore()
    elif (args.h2 is not None):
        if (args.trace is None) and (args.ldscores is None):
            log._log("!!! Either trace summary or LD score (truncated or genome-wide) must be provided !!!")
            if (args.trace is not None) and (args.bim is None):
                log._log("!!! .bim file used for trace summary calculation must also be provided !!!")
                sys.exit(1)
            sys.exit(1)
        if (args.max_chisq is not None):
            if (args.max_chisq <= .0):
                log._log("!!! max-chisq must be a positive value !!!")
                sys.exit(1)
        sums = Sumrhe(bim_path=args.bim, sum_path=args.trace, save_path = args.save_trace, h2_path=args.h2,\
            chisq_threshold=args.max_chisq, log=log, out=args.out, verbose=args.verbose, ldscores=args.ldscores,\
            njack=args.njack, annot=args.annot, allow_neg_enr=args.allow_neg_enr)
        sums._run()
        sums._logoff()
    elif (args.rg is not None):
        if (args.ldscores is None):
            log._log("!!! LD score (truncated or genome-wide) must be provided for estimation of genetic correlation !!!")
            sys.exit(1)
        if (args.intercept_rg is not None and args.pheno_rg is not None):
            log._log("!!! --intercept-rg and --pheno-rg cannot be used together; please use one of the two options !!!")
            sys.exit(1)
        rg = Sumcore(bim_path=args.bim, save_path=args.save_trace, rg=args.rg,\
            chisq_threshold=args.max_chisq, log=log, verbose=args.verbose, out=args.out, \
            ldscores=args.ldscores, njack=args.njack, annot=args.annot, \
            intercept=args.intercept_rg, phenos=args.pheno_rg)
        rg._run()
        rg._logoff()
    else:
        log._log("!!! At least one of the options (--geno / --h2 / --rg) must be specified. !!!")
        sys.exit(1)