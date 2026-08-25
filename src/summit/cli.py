from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _canonicalize_cpu_affinity_mask(values):
    """Return a sorted immutable CPU mask, or ``None`` when malformed."""
    try:
        observed = tuple(values)
    except (TypeError, ValueError):
        return None
    if not observed or any(type(cpu) is not int or cpu < 0 for cpu in observed):
        return None
    return tuple(sorted(set(observed)))


def _capture_pre_numerical_cpu_affinity():
    """Snapshot the launch CPU mask before a numerical runtime can bind it."""
    try:
        observed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    return _canonicalize_cpu_affinity_mask(observed)


_PRE_NUMERICAL_CPU_AFFINITY = _capture_pre_numerical_cpu_affinity()


from ._early_numa import preconfigure_numa_from_argv


_EARLY_NUMA_ATTESTATION = preconfigure_numa_from_argv(sys.argv[1:])

try:
    from ._native_build_config import RECOMMENDED_OMP_WAIT_POLICY
except ImportError:
    # A pure source checkout has no CMake-generated runtime description. Keep
    # the conservative policy used by shared/pthread BLAS builds.
    RECOMMENDED_OMP_WAIT_POLICY = "PASSIVE"


def _preparse_num_threads_from_argv(argv):
    tokens = list(argv)
    for i, tok in enumerate(tokens):
        if tok == "--num-threads":
            if i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except Exception:
                    return None
            return None
        if tok.startswith("--num-threads="):
            try:
                return int(tok.split("=", 1)[1].strip())
            except Exception:
                return None
    return None


def _set_thread_env_vars(num_threads):
    if num_threads is None:
        return
    try:
        n = int(num_threads)
    except Exception:
        return
    if n <= 0:
        return

    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)

    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    # Install the build-specific policy during argv preparse, before either
    # numerical runtime starts. Numerical teams must sleep during streamed
    # Python/native reductions; otherwise an idle OpenMP BLAS team can consume
    # the full allocation at its barrier. Explicit user settings remain
    # authoritative.
    os.environ.setdefault("OMP_WAIT_POLICY", RECOMMENDED_OMP_WAIT_POLICY)
    if os.environ["OMP_WAIT_POLICY"].strip().upper() == "PASSIVE":
        os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "1")


def _python_isolation_prefix():
    """Recreate import/bytecode isolation for fresh internal workers."""
    prefix = [sys.executable]
    if bool(getattr(sys.flags, "isolated", 0)):
        prefix.append("-I")
    else:
        if bool(getattr(sys.flags, "ignore_environment", 0)):
            prefix.append("-E")
        if bool(getattr(sys.flags, "no_user_site", 0)):
            prefix.append("-s")
        if bool(getattr(sys.flags, "safe_path", 0)):
            prefix.append("-P")
    if bool(getattr(sys.flags, "no_site", 0)):
        prefix.append("-S")
    if bool(getattr(sys.flags, "dont_write_bytecode", 0)):
        prefix.append("-B")
    optimize = int(getattr(sys.flags, "optimize", 0))
    if optimize > 0:
        prefix.append("-" + "O" * min(optimize, 2))
    return prefix


def _parse_mailman_mode(value):
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return str2bool(text)


_PREPARSED_NUM_THREADS = _preparse_num_threads_from_argv(sys.argv[1:])
if _PREPARSED_NUM_THREADS is not None:
    _set_thread_env_vars(_PREPARSED_NUM_THREADS)


_GXE_BATCH_REFERENCE_OPTIONS = frozenset(
    {
        "--geno",
        "--env",
        "--gxe-env-cols",
        "--gxe-parallel-environment-groups",
        "--gxe-explicit-openmp-placement",
        "--gxe-explicit-openmp-memory-scope",
        "--covar",
        "--annot",
        "--gxe-pheno",
        "--gxe-pheno-col",
        "--gxe-pheno-cols",
        "--gxe-missing-values",
        "--gxe-kernel-mode",
        "--gxe-genotype-scale",
        "--gxe-native-backend",
        "--gxe-native-workspace-gib",
        "--gxe-native-target-panel-columns",
        "--nvecs",
        "--step_size",
        "--seed",
        "--rand-dist",
        "--dtype",
        "--rand-samp",
        "--ddof",
        "--impute-method",
        "--target-xz-mem",
        "--target-mem",
        "--gxe-total-memory-gib",
        "--device",
    }
)


def _provided_long_options(argv, *, parser=None):
    """Return canonical explicit long options, including accepted abbreviations."""
    observed = {
        token.split("=", 1)[0]
        for token in argv
        if isinstance(token, str) and token.startswith("--")
    }
    if parser is None:
        return observed
    available = tuple(
        option
        for option in parser._option_string_actions
        if option.startswith("--")
    )
    canonical = set()
    for option in observed:
        if option in available:
            canonical.add(option)
            continue
        matches = [candidate for candidate in available if candidate.startswith(option)]
        if len(matches) == 1:
            canonical.add(matches[0])
        else:
            canonical.add(option)
    return canonical

import numpy as np
import pandas as pd

from . import utils
from .logger import Logger
from .ldscore.gw_ldscore import (
    GenomewideLDScore,
    _validate_cpu_placement_attestation,
    _validate_openmp_placement_build_contract,
    apply_env,
)
from .ldscore.gwe_ldscore import GenomewideEnvLDScore
from .ldscore.gxe_multi import (
    combine_multi_environment_reference_batches,
    generate_multi_environment_references,
    safe_environment_suffix,
)
from .ldscore.gxe_score import (
    score_phenotype_from_reference,
    score_phenotypes_from_reference,
)
from .ldscore.win_ldscore import WindowedLDScore
from .inference.sumrhe import Sumrhe
from .inference.h2_batch_fast import dispatch_h2_batch_fast
from .inference.sumcore import Sumcore
from .inference.trace import Trace
from .sumstats.sumstats import Sumstats
from .inference.rgcore import build_manifest_summary_row
from .inference.gxe import (
    fit_from_files as fit_gxe_from_files,
    fit_many_from_files as fit_many_gxe_from_files,
    load_fit_batch_manifest as load_gxe_fit_batch_manifest,
    write_fit as write_gxe_fit,
    write_fits as write_gxe_fits,
)
from .manifest.rg_manifest_builder import build_rg_manifest
from .manifest.rg_manifest_fast import dispatch_rg_manifest_fast


_THREADPOOL_LIMITER = None


def _apply_runtime_thread_cap(num_threads, log=None):
    if num_threads is None:
        return False

    try:
        n = int(num_threads)
    except Exception:
        raise SystemExit("!!! --num-threads must be an integer. !!!")

    if n <= 0:
        raise SystemExit("!!! --num-threads must be positive. !!!")

    _set_thread_env_vars(n)

    global _THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=n)
        if log is not None:
            log._log(f"[threads] capped BLAS/OpenMP thread pools to {n} thread(s).")
        return True
    except Exception as e:
        if log is not None:
            log._log(
                f"[threads] requested --num-threads {n}; set common thread-count environment variables. "
                f"Runtime threadpool cap unavailable ({e.__class__.__name__}: {e})."
            )
        return False



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
                        help="Path to the primary LD-score file. Use '@' as a chromosome placeholder for split files.")
    parser.add_argument("--ldscores-reg", default=None, type=str,
                        help=(
                            "Optional LD-score file used only for summary-only overlap-covariance estimation. "
                            "A 1D total-LD file is preferred; multi-column files are collapsed to "
                            "total LD by default and must be non-overlapping."
                        ))
    parser.add_argument("--ldscores-w", default=None, type=str,
                        help=(
                            "Optional one-column LD-score file for the LDSC overcounting weight. "
                            "Used only with --weight-mode ldsc; when omitted, total primary LD is used."
                        ))
    parser.add_argument("--collapse-reg-ld", action="store_true", default=True,
                        help=(
                            "Deprecated no-op: multi-column overlap-covariance LD is collapsed "
                            "to total LD by default."
                        ))

    # Sumstats / regression mode
    parser.add_argument("--h2", default=None, type=str,
                        help=(
                            "Path to one summary-statistics file, a chromosome-split '@' spec, "
                            "a directory of files, or a chromosome-split directory spec such as "
                            "'.../chr@' for batched univariate h2 estimation."
                        ))
    parser.add_argument("--h2-batch-fast", action="store_true", default=False,
                        help=(
                            "Use the exact chromosome-jackknife HE fast path for batched h2. "
                            "Summary statistics are loaded concurrently in bounded batches, "
                            "while one shared Trace and vectorized sufficient statistics are reused. "
                            "Constrained LDSC and --chisq-action clip require regular h2."
                        ))
    parser.add_argument("--h2-batch-size", default=4, type=int,
                        help="Number of traits held in each bounded fast-h2 batch (default: 4).")
    parser.add_argument("--h2-workers", default=4, type=int,
                        help="Concurrent sumstat loaders used by --h2-batch-fast (default: 4).")
    parser.add_argument("--h2-fast-reader", default="stream", type=str,
                        choices=["stream", "pandas"],
                        help=(
                            "Sumstat reader for fast h2: chromosome-streamed compact buffers or "
                            "the legacy all-file pandas reader (default: stream)."
                        ))
    parser.add_argument("--h2-checkpoint-every", default=64, type=int,
                        help="Rewrite the atomic fast-h2 results checkpoint every N traits (default: 64).")
    parser.add_argument("--h2-cache-dir", default=None, type=str,
                        help=(
                            "Optional reusable fast-h2 cache directory. Entries contain exact float64 "
                            "h2 moments and packed active masks keyed to the source files and Trace SNP axis."
                        ))
    parser.add_argument("--h2-cache-mode", default="readwrite", type=str,
                        choices=["read", "readwrite", "refresh"],
                        help="Fast-h2 cache policy when --h2-cache-dir is provided (default: readwrite).")
    parser.add_argument("--h2-cache-only", action="store_true", default=False,
                        help="Build/validate fast-h2 cache entries without fitting h2.")
    parser.add_argument("--h2-cache-verify-checksum", action="store_true", default=False,
                        help="Verify cached array SHA-256 checksums on every read (slower).")
    parser.add_argument("--rg", default=None, type=str,
                        help=(
                            "Either a comma-separated pair of summary-statistics files for bivariate rg estimation, "
                            "where each file may be a chromosome-split '@' spec, "
                            "or a manifest file path for batch rg. Manifest mode requires per-row phen1, phen2, "
                            "and sumstats1, sumstats2 columns. A finite per-row overlap_covariance is required in fast "
                            "manifest mode and optional in regular mode; omitted regular values use SUMMIT's "
                            "summary-only overlap-covariance estimation with delete refits."
                        ))
    parser.add_argument("--make-rg-manifest", default=None, type=str,
                        help=(
                            "Build an rg manifest TSV from raw phenotype/covariate input and write it to this path. "
                            "Use together with --phen-dir, --sum-dir, and either --pair-list or (--phen-list --all-pairwise)."
                        ))
    parser.add_argument(
        "--gxe-fit",
        default=None,
        type=str,
        help="Fit the full G + GxE + NxE model from a SUMMIT GxE reference-manifest JSON.",
    )
    parser.add_argument(
        "--gxe-fit-batch",
        default=None,
        type=str,
        help=(
            "Fit multiple phenotype summary triplets against one GxE reference "
            "using a summit.gxe.fit_batch JSON manifest."
        ),
    )
    parser.add_argument("--gxe-gwas", default=None, type=str,
                        help="Marginal additive score file for --gxe-fit (direct SCORE contract).")
    parser.add_argument("--gwis", default=None, type=str,
                        help="Marginal interaction score file for --gxe-fit; conditional PLINK ADDxE statistics are rejected.")
    parser.add_argument("--gxe-moments", default=None, type=str,
                        help="Phenotype/NxE moments JSON generated with the GxE reference bundle.")
    parser.add_argument("--gxe-max-condition", default=1e12, type=float,
                        help="Maximum allowed GxE normal-equation condition number.")
    parser.add_argument("--allow-ill-conditioned-gxe", action="store_true", default=False,
                        help="Solve a poorly identified GxE system by least squares after reporting diagnostics.")
    parser.add_argument(
        "--_gxe-probe-offset",
        dest="_gxe_probe_offset",
        default=0,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gxe-score-reference",
        default=None,
        type=str,
        help=(
            "Score quantitative traits in --gxe-pheno against this GxE "
            "reference. The default matched-cohort mode supports "
            "wide one-pass scoring; --gxe-population-reference scores one "
            "trait-specific cohort."
        ),
    )
    parser.add_argument(
        "--gxe-pheno-cols",
        default=None,
        type=str,
        help="Comma-separated wide-phenotype columns for --gxe-score-reference; default is all value columns.",
    )
    parser.add_argument(
        "--gxe-population-reference",
        action="store_true",
        default=False,
        help=(
            "Use GxE LD/kernel moments estimated in the full reference cohort with a "
            "different trait-specific GWAS/GWIS cohort. Requires exactly one "
            "--gxe-pheno-col per scoring command."
        ),
    )

    parser.add_argument("--compact", action="store_true", help="Write a compact rg manifest with only the core columns needed downstream.",)
    parser.add_argument("--rg-manifest-fast", action="store_true", default=False,
                        help=(
                            "Use the HE/jackknife sparse-drop fast path for supplied-overlap rg manifest mode. "
                            "This reuses cached sumstats and shared unit-level moment summaries, "
                            "writes manifest.results.tsv with total and per-bin rg/gamma columns, "
                            "and also emits per-pair .log files. Every row requires a finite overlap_covariance; "
                            "constrained cov-LDSC and summary-only overlap-covariance estimation require regular mode."
                        ))
    parser.add_argument("--rg-fast-no-pair-logs", action="store_true", default=False,
                        help=(
                            "With --rg-manifest-fast, omit per-pair .log files and retain the "
                            "batch log plus manifest.results.tsv. Intended for very large batches."
                        ))
    parser.add_argument("--rg-model-manifest", default=None, type=str,
                        help=(
                            "Optional multi-model specification for --rg-manifest-fast. The TSV must "
                            "contain model and bins columns, with optional aliases. Each row selects an "
                            "ordered subset of bins from the union --annot/--ldscores inputs. Shared "
                            "phenotypes and union sufficient statistics are computed once across models."
                        ))
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
    parser.add_argument("--allow-zero-rg-overlap", action="store_true", default=False,
                        help=(
                            "In --make-rg-manifest mode, retain phenotype pairs with no overlapping "
                            "individuals and set their exact sample-overlap covariance to zero."
                        ))
    parser.add_argument("--max-chisq", default=None, type=str,
                        help="Main chi^2 threshold. Use 'auto' for max(80, 0.001*Nmax).")
    parser.add_argument(
        "--overlap-covariance-chisq-thr",
        dest="intercept_chisq_thr",
        default=None,
        type=str,
        help=(
            "Chi^2 threshold used only for summary-only overlap-covariance "
            "estimation. Use 'auto' for max(80, 0.001*Nmax)."
        ),
    )
    parser.add_argument(
        "--intercept-chisq-thr",
        dest="intercept_chisq_thr",
        default=None,
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overlap-covariance-weight-mode",
        dest="intercept_weight_mode",
        default="score",
        type=str,
        choices=["ldsc", "score"],
        help=(
            "Weighting scheme for summary-only overlap-covariance estimation: "
            "'ldsc' for LDSC-style IRWLS weights, or 'score' for fixed "
            "w_j = 1 / w_ld,j."
        ),
    )
    parser.add_argument(
        "--intercept-weight-mode",
        dest="intercept_weight_mode",
        default="score",
        type=str,
        choices=["ldsc", "score"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--weight-mode", default="he", type=str,
                        choices=["he", "ldsc"],
                        help=(
                            "Main h2/genetic-covariance estimating equation: 'he' keeps the "
                            "SUMMIT/HE score moments (default); 'ldsc' fits constrained score-scale "
                            "LDSC for h2 and constrained score-scale cov-LDSC for genetic covariance "
                            "by closed-form IRWLS. rg is formed from covariance and h2 refits. It retains "
                            "SUMMIT's exact-score "
                            "response, scalar effective-sample-size convention, and overlap-covariance "
                            "refit semantics, so it is not literal ldsc.py when per-SNP sample sizes vary."
                        ))
    parser.add_argument("--ldsc-m", default=None, type=str,
                        help=(
                            "Optional LDSC .l2.M file for --weight-mode ldsc. Use '@' for "
                            "chromosome-split files, which are summed. It must describe the same "
                            "effect-SNP universe as --annot. The default is the fixed full-reference "
                            "annotation mass."
                        ))
    parser.add_argument("--ldsc-irwls-iters", default=3, type=int,
                        help="Number of closed-form LDSC IRWLS updates (default: 3).")
    parser.add_argument("--ldsc-irwls-tol", default=0.0, type=float,
                        help="Optional relative LDSC IRWLS stopping tolerance; zero disables early stopping.")
    parser.add_argument("--chisq-action", default="drop", type=str,
                        choices=["drop", "clip", "warn", "none"],
                        help="What to do with high-chi^2 SNPs on the main analysis axis.")

    parser.add_argument(
        "--overlap-covariance-rg",
        dest="intercept_rg",
        default=None,
        type=float,
        help=(
            "Supply the SUM-CORE overlapping phenotype covariance c_ov on the "
            "HE scale: c_ov = y_overlap^T y_overlap / sqrt(N1*N2) = "
            "N_overlap * rho_y,overlap / sqrt(N1*N2)."
        ),
    )
    parser.add_argument(
        "--intercept-rg",
        dest="intercept_rg",
        default=None,
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pheno-rg", default=None, type=str, help=(
        "Comma-separated pair of phenotype files for the traits in --rg. "
        "Each file should contain sample ID column(s) followed by the phenotype in the last column. "
        "SUMCORE standardizes each phenotype on its own study sample, intersects overlapping IDs, "
        "and computes c_ov = y_overlap^T y_overlap / sqrt(N1*N2). Mutually exclusive "
        "with --overlap-covariance-rg."
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
                        help="Path to the annotation file. Use '@' as a chromosome placeholder for split files.")

    # Output / behavior
    parser.add_argument("--out", default=None, type=str,
                        help="Output prefix for single-run modes. In rg manifest mode, this must be an output directory.")
    parser.add_argument("--verbose", nargs="?", const="1", default="0", type=str,
                        help=("Verbosity level: 0, 1, 2, or 'max'. "
                              "Legacy values 'jack' and 'normeq' request extra output files without enabling verbose diagnostics. "
                              "Passing --verbose with no value implies 1."))
    parser.add_argument("--write-jack", action="store_true", default=False,
                        help="Write jackknife replicate dumps without enabling verbose diagnostics. For h2 this writes <out>.<phen>.jack; for rg this writes <out>.rg.jack.")
    parser.add_argument("--write-normeq", action="store_true", default=False,
                        help="Write the SCORE normal-equation JSON dump to <out>.rg.scoreeq.json without enabling verbose diagnostics.")
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
    allele_group = parser.add_mutually_exclusive_group()
    allele_group.add_argument("--align-alleles", dest="align_alleles", action="store_true",
                              help="Align the second trait to the first trait by allele labels (the default).")
    allele_group.add_argument("--no-align-alleles", dest="align_alleles", action="store_false",
                              help="Assume both rg inputs are already identically oriented; skip allele validation/alignment.")
    parser.set_defaults(align_alleles=True)
    parser.add_argument("--keep-ambiguous", action="store_true", default=False,
                        help=("Keep strand-ambiguous A/T and C/G SNPs during allele alignment. "
                              "Their orientation then follows literal A1/A2 labels because strand is unresolved."))

    # LD-score generation mode
    parser.add_argument("--geno", default=None, type=str,
                        help=(
                            "BED/BIM/FAM or PGEN/PVAR/PSAM path/prefix for LD-score calculation. "
                            "Pass an explicit .bed or .pgen path when both trios share a prefix."
                        ))
    parser.add_argument("--nvecs", default=1000, type=int,
                        help="Number of random vectors for stochastic genome-wide LD scores.")
    parser.add_argument("--step_size", default=1000, type=_step_size_argument,
                        help="Step size for LD-score computation. GxE reference "
                             "generation also accepts 'auto', which picks a "
                             "deterministic canonical block width from the "
                             "variant count; the resolved value is recorded in "
                             "the manifest and defines the finite-probe "
                             "realization exactly like an explicit width.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed.")
    parser.add_argument("--covar", default=None, type=str,
                        help="Covariate file for LD-score estimation.")
    parser.add_argument("--env", default=None, type=str,
                        help=("Environment file for genome-wide GxE LD-score estimation. "
                              "Must be used with --geno. File must contain FID, IID, and one environment column "
                              "unless --gxe-env-cols selects a common-cohort multi-environment batch. "
                              "This mode writes the XX/XW/WX/WW GENIE trace bundle."))
    parser.add_argument(
        "--gxe-env-cols", default=None, type=str,
        help=("Comma-separated columns in a wide --env file to process as independent "
              "G+GxE+NxE+residual references while sharing streamed genotype reads. "
              "All columns must retain exactly the same complete-case cohort."),
    )
    parser.add_argument(
        "--gxe-parallel-environment-groups",
        default="auto",
        choices=["auto", "1", "2"],
        help=(
            "Run a large direct multi-environment reference as one group or as "
            "two process-isolated, socket-local groups (default: auto)."
        ),
    )
    parser.add_argument(
        "--gxe-explicit-openmp-placement",
        action="store_true",
        default=False,
        help=(
            "Run a direct single-group multi-environment reference in a fresh "
            "process with an explicit, verified socket-local OpenMP CPU "
            "placement contract; requires --num-threads."
        ),
    )
    parser.add_argument(
        "--gxe-explicit-openmp-memory-scope",
        default="selected-cpus",
        choices=["selected-cpus", "selected-socket"],
        help=(
            "NUMA memory scope for --gxe-explicit-openmp-placement. The "
            "default binds only nodes covered by the selected CPUs; "
            "selected-socket binds the full verified NUMA-node set of that "
            "same socket without changing CPU or OpenMP placement."
        ),
    )
    parser.add_argument(
        "--_gxe-multi-batch-manifest",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_gxe-environment-group-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_gxe-worker-cpus",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_gxe-worker-auth-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_gxe-log-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gxe-pheno", default=None, type=str,
                        help="Optional quantitative phenotype file; writes aligned marginal GWAS/GWIS scores and NxE moments.")
    parser.add_argument("--gxe-pheno-col", default=None, type=str,
                        help="Phenotype column name when --gxe-pheno contains more than one value column.")
    parser.add_argument("--gxe-missing-values", default="-9,NA,NaN,nan,.,None,null", type=str,
                        help="Comma-separated missing tokens for GxE environment, covariate, and phenotype inputs.")
    parser.add_argument(
        "--gxe-kernel-mode",
        default="standardized_projected",
        choices=["standardized_projected", "raw_projected"],
        help=("Feature convention: post-projection per-variant standardization "
              "(standardized_projected, default) or naturally scaled projected "
              "columns (raw_projected)."),
    )
    parser.add_argument("--gxe-genotype-scale", default=None, choices=["hwe", "sample"],
                        help=("Pre-projection genotype scaling. Defaults to sample scaling for standardized "
                              "SUMMIT kernels and HWE scaling for GENIE compatibility."))
    parser.add_argument(
        "--gxe-native-backend", default="python", choices=["python", "direct"],
        help=("Opt-in descriptor-owned C++ BED reference pipeline for "
              "phenotype-free standardized/sample references with Rademacher "
              "probes. The same native path handles one or multiple "
              "environments; Python "
              "remains the default oracle."),
    )
    parser.add_argument(
        "--gxe-native-workspace-gib", default=16.0, type=float,
        help="Hard allocation ceiling in GiB for each direct native GxE call.",
    )
    parser.add_argument(
        "--gxe-native-target-panel-columns", default=64, type=int,
        help=("Temporary panel width for the generic native raw-source target method. "
              "The production opaque in-memory GxE target uses full-width GEMMs."),
    )
    parser.add_argument("--gxe-overwrite", action="store_true", default=False,
                        help="Explicitly permit replacement of existing fixed-prefix GxE generation or fit outputs.")
    parser.add_argument("--rand-dist", default="spherical", type=str, choices=["spherical", "gaussian", "normal", "rademacher"],
                        help="Distribution for randomized LD-score estimation.")
    parser.add_argument("--dtype", default="float32", type=str,
                        help="Retained storage dtype for randomized probe/sketch panels. "
                             "Annotation values, masses, and native arithmetic always stay binary64.")
    parser.add_argument("--rand-samp", default=None, type=str,
                        help="Random subset of samples: ratio in (0,1] or an integer count >100.")
    parser.add_argument("--ddof", default=1, type=int,
                        help="ddof used in LD-score estimation.")
    parser.add_argument("--ld-wind-kb", default=None, type=float,
                        help="If set, compute windowed LD scores with the given kb window.")
    parser.add_argument("--win-panel-cols", default=None, type=int,
                        help=("PGEN windowed-LD dosage columns decoded per panel. "
                              "By default this is chosen from sample count, chunk size, and cache size."))
    parser.add_argument("--win-cache-mb", default=-1, type=int,
                        help=("PGEN/BED windowed-LD prepared-panel cache in MiB; -1 selects automatically "
                              "(bounded by available/target memory and honoring SUMMIT_WIN_CACHE_MB), "
                              "while 0 disables caching."))
    parser.add_argument("--correct-skew", action="store_true",
                        help="Enable optional finite-sample skew diagnostics in genome-wide LD-score estimation.")
    parser.add_argument("--write-kmoments", action="store_true",
                        help="Write .gw.kmoments for unpartitioned genome-wide LD-score estimation.")
    mc_group = parser.add_mutually_exclusive_group()
    mc_group.add_argument(
        "--write-ld-mc-var", "--write-ld-mc-ci", dest="write_ld_mc_var",
        action="store_true",
        help=("Write optional per-SNP Monte Carlo variances, SEs, and pointwise "
              "95%% conditional MC intervals for genome-wide LD scores."),
    )
    mc_group.add_argument("--skip-ld-mc", action="store_true",
                          help="Disable the default annotation-level genome-wide LD-score MC noise diagnostic.")
    parser.add_argument("--skip-kmoments", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-mailman", default="auto", type=_parse_mailman_mode,
        help=(
            "Mailman mode: auto uses it for <=10 probes when HWE imputation "
            "makes the existing implementation exact; true/false force the choice."
        ),
    )
    parser.add_argument("--impute-method", default='mean', type=str, choices=['mean', 'hwe'],
                        help="Method for imputing missing genotype.")

    # Resource / performance knobs
    parser.add_argument("--num-threads", default=None, type=int,
                        help="Cap BLAS / compute threads.")
    parser.add_argument(
        "--target-xz-mem", type=utils.parse_memory_budget, default="auto",
        help="Memory budget in GiB for sketch panels, or 'auto' (default).",
    )
    parser.add_argument(
        "--target-mem", type=utils.parse_memory_budget, default=None,
        help="Alias overriding --target-xz-mem with a GiB value or 'auto'.",
    )
    parser.add_argument(
        "--gxe-total-memory-gib",
        type=utils.parse_memory_budget,
        default="auto",
        help=(
            "Maximum modeled GxE process peak in GiB, or 'auto' (default). "
            "This is independent of the sketch-panel budget."
        ),
    )
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
    parser.add_argument("--force_affinity_all", default=False, type=str2bool,
                        help=(
                            "Expand CPU affinity to all online CPUs (true/false; "
                            "default false preserves taskset/scheduler placement)."
                        ))
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
    sensitive_options = {"--_gxe-worker-auth-sha256"}
    while i < len(tokens):
        t = tokens[i]
        token_option = t.split("=", 1)[0]
        if any(option.startswith(token_option) for option in sensitive_options):
            i += 1 if "=" in t else 2
            continue
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
            if k == "_gxe_worker_auth_sha256":
                continue
            log._log(f"  {k} = {v!r}")
    log._log("=========================================================================='".replace("'", ""))


def _make_low_level_env(args):
    low_level = {
        "num_threads": args.num_threads,
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
    if getattr(args, "_gxe_group_worker_authenticated", False):
        low_level.update(
            {
                "_gxe_group_worker_authenticated": True,
                "_gxe_worker_cpu_ids": tuple(args._gxe_worker_cpu_ids),
                "_gxe_cpu_placement": dict(args._gxe_cpu_placement),
                "_gxe_cpu_placement_complete": True,
            }
        )
    return low_level


def _make_gxe_generator(args, log, verbose_on, low_level, *, env_col=None, out_path=None,
                        native_backend=None):
    return GenomewideEnvLDScore(
        bed_path=args.geno,
        env_path=args.env,
        annot_path=args.annot,
        out_path=args.out if out_path is None else out_path,
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
        ddof=args.ddof,
        target_xz_mem=args.target_xz_mem,
        target_mem=args.target_mem,
        gxe_total_memory_gib=getattr(args, "gxe_total_memory_gib", "auto"),
        device=args.device,
        impute_method=args.impute_method,
        kernel_mode=args.gxe_kernel_mode,
        genotype_scale=args.gxe_genotype_scale,
        pheno_path=args.gxe_pheno,
        pheno_col=args.gxe_pheno_col,
        missing_values=tuple(x.strip() for x in args.gxe_missing_values.split(",") if x.strip()),
        overwrite=args.gxe_overwrite,
        probe_offset=args._gxe_probe_offset,
        native_backend=args.gxe_native_backend if native_backend is None else native_backend,
        native_workspace_gib=args.gxe_native_workspace_gib,
        native_target_panel_columns=args.gxe_native_target_panel_columns,
        env_col=env_col,
    )


def _dispatch_gxe_score(args, log):
    _require_integer_step_size(args, "GxE phenotype scoring")
    columns = None
    if args.gxe_pheno_col is not None and args.gxe_pheno_cols is not None:
        raise SystemExit(
            "!!! Use only one of --gxe-pheno-col or --gxe-pheno-cols for reference scoring. !!!"
        )
    if args.gxe_pheno_cols is not None:
        columns = tuple(value.strip() for value in args.gxe_pheno_cols.split(",") if value.strip())
        if not columns:
            raise SystemExit("!!! --gxe-pheno-cols must contain at least one column name. !!!")
    elif args.gxe_pheno_col is not None:
        columns = (args.gxe_pheno_col,)
    common = dict(
        reference_manifest=args.gxe_score_reference,
        bed_path=args.geno,
        env_path=args.env,
        covar_path=args.covar,
        pheno_path=args.gxe_pheno,
        output_prefix=args.out,
        missing_values=tuple(x.strip() for x in args.gxe_missing_values.split(",") if x.strip()),
        step_size=args.step_size,
        num_threads=args.num_threads,
    )
    if args.gxe_population_reference:
        if args.gxe_pheno_cols is not None or args.gxe_pheno_col is None:
            raise SystemExit(
                "!!! --gxe-population-reference requires exactly one --gxe-pheno-col. !!!"
            )
        bundle = score_phenotype_from_reference(
            **common,
            pheno_col=columns[0],
            population_transfer=True,
        )
        artifacts = {columns[0]: bundle}
    else:
        artifacts = score_phenotypes_from_reference(
            **common,
            pheno_cols=columns,
        )
    log._log(
        f"[gxe:score] wrote {len(artifacts)} phenotype score/moment triplet(s) "
        "from one genotype pass."
    )
    for trait, bundle in artifacts.items():
        log._log(
            f"[gxe:score] {trait}: {bundle.gwas}, {bundle.gwis}, {bundle.moments}."
        )


def _replace_long_option(tokens, option, value):
    """Return argv with one long option set exactly once."""
    result = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    result.extend((option, str(value)))
    return result


_GXE_GROUP_WORKER_TOKEN_ENV = "SUMMIT_GXE_GROUP_WORKER_TOKEN"
_GXE_OMP_AFFINITY_CONFLICTS = (
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "KMP_HW_SUBSET",
    "KMP_PLACE_THREADS",
    "OMP_NESTED",
)
_GXE_BLIS_AUTOMATIC_CONFLICTS = (
    "BLIS_NT",
    "BLIS_TI",
    "BLIS_THREAD_IMPL",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
)


@dataclass(frozen=True)
class _GxeGroupWorkerCpuContract:
    cpu_ids: tuple[int, ...]
    threads: int


def _canonical_omp_places(cpu_ids):
    return ",".join(f"{{{int(cpu)}}}" for cpu in cpu_ids)


def _parse_canonical_cpu_ranges(value):
    text = str(value)
    parsed = set()
    try:
        for component in text.split(","):
            bounds = component.split("-", 1)
            if not component or len(bounds) > 2:
                raise ValueError
            start = int(bounds[0])
            stop = int(bounds[-1])
            if start < 0 or stop < start:
                raise ValueError
            parsed.update(range(start, stop + 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Internal GxE worker CPU list is malformed.") from exc
    cpus = tuple(sorted(parsed))
    if not cpus or _format_integer_ranges(cpus) != text:
        raise RuntimeError("Internal GxE worker CPU list is not canonical.")
    return cpus


def _parse_canonical_kernel_integer_ranges(value):
    """Parse a canonical Linux cpulist/nodelist, failing closed."""
    text = str(value).strip()
    parsed = set()
    try:
        for component in text.split(","):
            bounds = component.split("-", 1)
            if not component or len(bounds) > 2:
                raise ValueError
            start = int(bounds[0])
            stop = int(bounds[-1])
            if start < 0 or stop < start:
                raise ValueError
            parsed.update(range(start, stop + 1))
    except (TypeError, ValueError):
        return None
    values = tuple(sorted(parsed))
    if not values or _format_integer_ranges(values) != text:
        return None
    return values


def _verified_full_socket_numa_nodes(
    socket_id,
    *,
    node_root=Path("/sys/devices/system/node"),
    cpu_root=Path("/sys/devices/system/cpu"),
    process_status=Path("/proc/self/status"),
):
    """Resolve all online, process-allowed NUMA nodes for one CPU package."""
    if type(socket_id) is not int or socket_id < 0:
        return None
    try:
        online_nodes = _parse_canonical_kernel_integer_ranges(
            (node_root / "online").read_text(encoding="utf-8")
        )
        status_lines = process_status.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    allowed_lines = [
        line.split(":", 1)[1].strip()
        for line in status_lines
        if line.split(":", 1)[0] == "Mems_allowed_list"
    ]
    if online_nodes is None or len(allowed_lines) != 1:
        return None
    allowed_nodes = _parse_canonical_kernel_integer_ranges(allowed_lines[0])
    if allowed_nodes is None:
        return None

    package_by_node = {}
    for node in online_nodes:
        try:
            node_cpus = _parse_canonical_kernel_integer_ranges(
                (node_root / f"node{node}" / "cpulist").read_text(
                    encoding="utf-8"
                )
            )
        except OSError:
            return None
        if node_cpus is None:
            return None
        packages = set()
        for cpu in node_cpus:
            try:
                package_text = (
                    cpu_root
                    / f"cpu{cpu}"
                    / "topology"
                    / "physical_package_id"
                ).read_text(encoding="utf-8").strip()
                if not re.fullmatch(r"[0-9]+", package_text):
                    return None
                package = int(package_text)
            except (OSError, ValueError):
                return None
            packages.add(package)
        if len(packages) != 1:
            return None
        package_by_node[node] = packages.pop()

    selected_nodes = tuple(
        node for node in online_nodes if package_by_node[node] == socket_id
    )
    if not selected_nodes or not set(selected_nodes).issubset(allowed_nodes):
        return None
    return selected_nodes


def _authenticate_gxe_group_worker(args):
    raw_token = os.environ.pop(_GXE_GROUP_WORKER_TOKEN_ENV, None)
    cpu_text = getattr(args, "_gxe_worker_cpus", None)
    expected_digest = getattr(args, "_gxe_worker_auth_sha256", None)
    if not args._gxe_environment_group_worker:
        if cpu_text is not None or expected_digest is not None or raw_token is not None:
            raise RuntimeError(
                "Internal GxE group-worker credentials require the worker flag."
            )
        return None
    if cpu_text is None or expected_digest is None or raw_token is None:
        raise RuntimeError("Internal GxE group-worker authentication is incomplete.")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise RuntimeError("Internal GxE group-worker authentication hash is malformed.")
    observed_digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise RuntimeError("Internal GxE group-worker authentication failed.")
    cpus = _parse_canonical_cpu_ranges(cpu_text)
    if args.num_threads is None or int(args.num_threads) != len(cpus):
        raise RuntimeError(
            "Internal GxE worker threads disagree with its authenticated CPU list."
        )
    if (
        args.gxe_native_backend != "direct"
        or str(args.gxe_parallel_environment_groups) != "1"
        or args.gxe_env_cols is None
        or args._gxe_multi_batch_manifest is None
    ):
        raise RuntimeError(
            "Authenticated GxE group workers require direct, single-group "
            "multi-environment execution with a private batch manifest."
        )
    args._gxe_worker_auth_sha256 = None
    return _GxeGroupWorkerCpuContract(cpu_ids=cpus, threads=len(cpus))


def _validate_gxe_group_worker_openmp_environment(contract):
    expected = {
        "OMP_NUM_THREADS": str(contract.threads),
        "OMP_THREAD_LIMIT": str(contract.threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": _canonical_omp_places(contract.cpu_ids),
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": str(contract.threads),
    }
    disagreements = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    conflicts = [
        name
        for name in (
            *_GXE_OMP_AFFINITY_CONFLICTS,
            *_GXE_BLIS_AUTOMATIC_CONFLICTS,
        )
        if name in os.environ
    ]
    if disagreements or conflicts:
        raise RuntimeError(
            "Authenticated GxE group worker has a noncanonical threading environment: "
            f"disagreements={disagreements}, conflicts={conflicts}."
        )


def _configure_gxe_group_worker_placement(contract, native_module=None):
    _validate_gxe_group_worker_openmp_environment(contract)
    if native_module is None:
        from . import gxeldcore as native_module
    configure = getattr(native_module, "configure_openmp_placement", None)
    if not callable(configure):
        raise RuntimeError(
            "The direct GxE extension lacks the OpenMP placement contract API."
        )
    placement = _validate_cpu_placement_attestation(
        dict(configure(list(contract.cpu_ids), contract.threads)),
        expected_cpu_ids=contract.cpu_ids,
        expected_threads=contract.threads,
    )
    _validate_openmp_placement_build_contract(native_module.build_info(), placement)
    return placement


def _validated_explicit_outer_cpu_affinity():
    """Recover a launch mask narrowed only by canonical OpenMP binding."""
    captured = _PRE_NUMERICAL_CPU_AFFINITY
    if (
        not isinstance(captured, tuple)
        or _canonicalize_cpu_affinity_mask(captured) != captured
    ):
        raise RuntimeError(
            "Explicit OpenMP placement requires a valid immutable "
            "pre-numerical CPU-affinity capture."
        )
    try:
        live = _canonicalize_cpu_affinity_mask(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        live = None
    if live is None:
        raise RuntimeError(
            "Explicit OpenMP placement requires a nonempty current CPU affinity."
        )
    if not set(live).issubset(captured):
        raise RuntimeError(
            "Current CPU affinity escapes the pre-numerical launch mask."
        )
    if live != captured:
        threads = str(len(captured))
        expected = {
            "OMP_NUM_THREADS": threads,
            "OMP_THREAD_LIMIT": threads,
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "SPREAD",
            "OMP_PLACES": _canonical_omp_places(captured),
            "OMP_MAX_ACTIVE_LEVELS": "1",
        }
        disagreements = {
            name: {"expected": value, "observed": os.environ.get(name)}
            for name, value in expected.items()
            if os.environ.get(name) != value
        }
        if disagreements:
            raise RuntimeError(
                "A strict subset of the pre-numerical CPU-affinity mask is "
                "trusted only with the canonical explicit OpenMP launcher "
                f"contract; disagreements={disagreements}."
            )
    return captured


def _socket_local_core_groups(*, allowed_cpu_ids=None):
    """Return one allowed logical CPU per physical core, grouped by socket."""
    if allowed_cpu_ids is None:
        try:
            allowed = _canonicalize_cpu_affinity_mask(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return []
    else:
        allowed = _canonicalize_cpu_affinity_mask(allowed_cpu_ids)
    if allowed is None:
        return []
    physical = {}
    node_by_cpu = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text().strip())
            core = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError):
            return []
        if package < 0 or core < 0:
            return []
        physical.setdefault((package, core), cpu)
        node_paths = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
        if len(node_paths) != 1:
            return []
        try:
            node_by_cpu[cpu] = int(node_paths[0].name[4:])
        except ValueError:
            return []
        if node_by_cpu[cpu] < 0:
            return []
    by_socket = {}
    for (package, _core), cpu in sorted(physical.items()):
        record = by_socket.setdefault(package, {"cpus": [], "nodes": set()})
        record["cpus"].append(cpu)
        record["nodes"].add(node_by_cpu[cpu])
    return [
        {
            "socket": package,
            "cpus": tuple(sorted(record["cpus"])),
            "nodes": tuple(sorted(record["nodes"])),
            "node_by_cpu": {
                cpu: node_by_cpu[cpu] for cpu in sorted(record["cpus"])
            },
        }
        for package, record in sorted(by_socket.items())
        if record["cpus"] and record["nodes"]
    ]


def _format_integer_ranges(values):
    ordered = sorted(set(int(value) for value in values))
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _validate_explicit_openmp_placement_request(args):
    explicit_placement = bool(
        getattr(args, "gxe_explicit_openmp_placement", False)
    )
    memory_scope = getattr(
        args, "gxe_explicit_openmp_memory_scope", "selected-cpus"
    )
    if memory_scope not in {"selected-cpus", "selected-socket"}:
        raise ValueError(
            "--gxe-explicit-openmp-memory-scope must be selected-cpus or "
            "selected-socket."
        )
    if memory_scope != "selected-cpus" and not explicit_placement:
        raise ValueError(
            "--gxe-explicit-openmp-memory-scope=selected-socket requires "
            "--gxe-explicit-openmp-placement."
        )
    if not explicit_placement:
        return
    valid = (
        getattr(args, "gxe_native_backend", None) == "direct"
        and str(getattr(args, "gxe_parallel_environment_groups", "")) == "1"
        and getattr(args, "geno", None) is not None
        and getattr(args, "env", None) is not None
        and getattr(args, "gxe_env_cols", None) is not None
        and type(getattr(args, "num_threads", None)) is int
        and args.num_threads > 0
        and getattr(args, "gxe_score_reference", None) is None
        and getattr(args, "gxe_fit", None) is None
        and getattr(args, "gxe_fit_batch", None) is None
    )
    if not valid:
        raise ValueError(
            "--gxe-explicit-openmp-placement is valid only for direct "
            "multi-environment reference construction with "
            "--gxe-parallel-environment-groups=1 and an explicit positive "
            "--num-threads."
        )


def _verified_physical_cpu_inventory(sockets):
    cpu_to_node = {}
    for record in sockets:
        cpus = record.get("cpus") if isinstance(record, dict) else None
        nodes = record.get("nodes") if isinstance(record, dict) else None
        node_by_cpu = (
            record.get("node_by_cpu") if isinstance(record, dict) else None
        )
        if (
            not isinstance(cpus, tuple)
            or not cpus
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or tuple(sorted(set(cpus))) != cpus
            or not isinstance(nodes, tuple)
            or not nodes
            or any(type(node) is not int or node < 0 for node in nodes)
            or tuple(sorted(set(nodes))) != nodes
            or not isinstance(node_by_cpu, dict)
            or any(type(cpu) is not int or cpu < 0 for cpu in node_by_cpu)
            or set(node_by_cpu) != set(cpus)
            or set(node_by_cpu.values()) != set(nodes)
            or any(type(node) is not int or node < 0 for node in node_by_cpu.values())
            or set(cpus) & set(cpu_to_node)
        ):
            return None
        cpu_to_node.update(node_by_cpu)
    if not cpu_to_node:
        return None
    return tuple(sorted(cpu_to_node)), cpu_to_node


def _parallel_environment_layout(args, columns):
    requested = str(args.gxe_parallel_environment_groups)
    explicit_single_group = bool(
        getattr(args, "gxe_explicit_openmp_placement", False)
    )
    memory_scope = getattr(
        args, "gxe_explicit_openmp_memory_scope", "selected-cpus"
    )
    if memory_scope not in {"selected-cpus", "selected-socket"}:
        raise RuntimeError(
            "Explicit OpenMP memory scope must be selected-cpus or "
            "selected-socket."
        )
    if memory_scope != "selected-cpus" and not explicit_single_group:
        raise RuntimeError(
            "A nondefault explicit OpenMP memory scope requires explicit "
            "single-group placement."
        )
    if args._gxe_environment_group_worker:
        return None
    if requested == "1" and not explicit_single_group:
        return None
    if explicit_single_group:
        allowed_cpu_ids = _validated_explicit_outer_cpu_affinity()
        sockets = _socket_local_core_groups(allowed_cpu_ids=allowed_cpu_ids)
        if (
            requested != "1"
            or args.gxe_native_backend != "direct"
            or len(columns) < 2
        ):
            raise RuntimeError(
                "Explicit OpenMP placement requires a direct single-group "
                "multi-environment reference."
            )
        if _verified_physical_cpu_inventory(sockets) is None:
            raise RuntimeError(
                "Explicit OpenMP placement requires complete allowed physical "
                "CPU and NUMA topology."
            )
        if memory_scope == "selected-socket":
            socket_ids = tuple(record.get("socket") for record in sockets)
            if (
                any(
                    type(socket_id) is not int or socket_id < 0
                    for socket_id in socket_ids
                )
                or len(set(socket_ids)) != len(socket_ids)
            ):
                raise RuntimeError(
                    "Full-socket NUMA scope requires unambiguous physical "
                    "CPU package identities."
                )
        if type(args.num_threads) is not int or args.num_threads <= 0:
            raise RuntimeError(
                "Explicit OpenMP placement requires an explicit positive thread count."
            )
        requested_threads = args.num_threads
        selected_socket = next(
            (
                record
                for record in sockets
                if len(record["cpus"]) >= requested_threads
            ),
            None,
        )
        if selected_socket is None:
            raise RuntimeError(
                "Explicit OpenMP placement cannot satisfy the requested thread "
                "count from one verified physical CPU socket."
            )
        selected_cpus = tuple(selected_socket["cpus"][:requested_threads])
        if memory_scope == "selected-socket":
            selected_nodes = _verified_full_socket_numa_nodes(
                selected_socket["socket"]
            )
            if (
                selected_nodes is None
                or not set(selected_socket["nodes"]).issubset(selected_nodes)
            ):
                raise RuntimeError(
                    "Full-socket NUMA scope requires a complete, unambiguous "
                    "online socket topology within the process memory-node "
                    "allowlist."
                )
        else:
            selected_nodes = tuple(
                sorted(
                    {
                        selected_socket["node_by_cpu"][cpu]
                        for cpu in selected_cpus
                    }
                )
            )
        return (
            {
                "columns": tuple(columns),
                "cpus": selected_cpus,
                "nodes": selected_nodes,
                "threads": requested_threads,
            },
        )
    sockets = _socket_local_core_groups()
    total_available = sum(len(record["cpus"]) for record in sockets[:2])
    requested_threads = (
        total_available if args.num_threads is None else int(args.num_threads)
    )
    eligible = (
        args.gxe_native_backend == "direct"
        and len(columns) >= 4
        and len(sockets) >= 2
        and bool(sockets[0]["nodes"])
        and bool(sockets[1]["nodes"])
        and set(sockets[0]["cpus"]).isdisjoint(sockets[1]["cpus"])
        and set(sockets[0]["nodes"]).isdisjoint(sockets[1]["nodes"])
        and all(
            isinstance(record.get("node_by_cpu"), dict)
            and set(record["node_by_cpu"]) == set(record["cpus"])
            and set(record["node_by_cpu"].values()) == set(record["nodes"])
            for record in sockets[:2]
        )
        and min(len(record["cpus"]) for record in sockets[:2]) >= 8
        and requested_threads >= 32
    )
    if requested == "auto" and not eligible:
        return None
    if requested == "2" and not eligible:
        raise RuntimeError(
            "Two GxE environment groups require the direct backend, at least four "
            "environments, at least two allowed CPU sockets with eight physical "
            "cores each, and at least 32 total requested threads."
        )
    if requested == "2" and requested_threads > total_available:
        raise RuntimeError(
            "Two GxE environment groups cannot satisfy the requested thread "
            "count from the verified allowed physical cores."
        )
    total_threads = min(requested_threads, total_available)
    first_threads = min(len(sockets[0]["cpus"]), (total_threads + 1) // 2)
    second_threads = min(len(sockets[1]["cpus"]), total_threads - first_threads)
    unassigned = total_threads - first_threads - second_threads
    if unassigned:
        additional_first = min(
            len(sockets[0]["cpus"]) - first_threads, unassigned
        )
        first_threads += additional_first
        unassigned -= additional_first
    if unassigned:
        additional_second = min(
            len(sockets[1]["cpus"]) - second_threads, unassigned
        )
        second_threads += additional_second
        unassigned -= additional_second
    if min(first_threads, second_threads) < 1 or unassigned:
        if requested == "2":
            raise RuntimeError("Could not allocate threads to both GxE socket groups.")
        return None
    split = (len(columns) + 1) // 2
    column_groups = (columns[:split], columns[split:])
    layout = []
    for group, socket, threads in zip(
        column_groups, sockets[:2], (first_threads, second_threads), strict=True
    ):
        cpus = tuple(socket["cpus"][:threads])
        nodes = tuple(sorted({socket["node_by_cpu"][cpu] for cpu in cpus}))
        if not nodes:
            if requested == "2":
                raise RuntimeError(
                    "Could not resolve NUMA nodes for a GxE socket group."
                )
            return None
        layout.append(
            {
                "columns": group,
                "cpus": cpus,
                "nodes": nodes,
                "threads": threads,
            }
        )
    return tuple(layout)


def _dispatch_parallel_gxe_environment_groups(args, columns, layout, log):
    canonical_manifest = Path(
        args._gxe_multi_batch_manifest or f"{args.out}.gxe.multi.json"
    ).expanduser().resolve()
    if canonical_manifest.exists():
        raise FileExistsError(
            f"Refusing existing multi-environment manifest: {canonical_manifest}."
        )
    base_tokens = list(sys.argv[1:])
    processes = []
    group_manifests = []
    try:
        for index, record in enumerate(layout):
            group_manifest = Path(
                f"{args.out}.group{index}.gxe.multi.json"
            ).expanduser().resolve()
            group_log = Path(f"{args.out}.group{index}.gxe.log").expanduser().resolve()
            for target in (group_manifest, group_log):
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"Refusing existing group output: {target}.")
            tokens = _replace_long_option(
                base_tokens, "--gxe-env-cols", ",".join(record["columns"])
            )
            tokens = _replace_long_option(
                tokens, "--gxe-parallel-environment-groups", "1"
            )
            tokens = _replace_long_option(tokens, "--num-threads", record["threads"])
            tokens = _replace_long_option(tokens, "--force_affinity_all", "false")
            tokens = _replace_long_option(
                tokens, "--_gxe-multi-batch-manifest", group_manifest
            )
            tokens = _replace_long_option(tokens, "--_gxe-log-path", group_log)
            raw_token = secrets.token_urlsafe(32)
            authentication_digest = hashlib.sha256(
                raw_token.encode("utf-8")
            ).hexdigest()
            tokens = _replace_long_option(
                tokens,
                "--_gxe-worker-cpus",
                _format_integer_ranges(record["cpus"]),
            )
            tokens = _replace_long_option(
                tokens,
                "--_gxe-worker-auth-sha256",
                authentication_digest,
            )
            if "--_gxe-environment-group-worker" not in tokens:
                tokens.append("--_gxe-environment-group-worker")
            if record["nodes"]:
                tokens = _replace_long_option(tokens, "--numa-mode", "membind")
                tokens = _replace_long_option(
                    tokens, "--numa-nodes", _format_integer_ranges(record["nodes"])
                )
            command = [
                shutil.which("taskset") or "taskset",
                "-c",
                _format_integer_ranges(record["cpus"]),
                *_python_isolation_prefix(),
                "-m",
                "summit.cli",
                *tokens,
            ]
            environment = os.environ.copy()
            for name in (
                "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = str(record["threads"])
            environment["OMP_DYNAMIC"] = "FALSE"
            environment["OMP_WAIT_POLICY"] = "PASSIVE"
            environment["GOMP_SPINCOUNT"] = "0"
            environment["OMP_PROC_BIND"] = "SPREAD"
            environment["OMP_PLACES"] = _canonical_omp_places(record["cpus"])
            environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
            for name in _GXE_OMP_AFFINITY_CONFLICTS:
                environment.pop(name, None)
            for name in _GXE_BLIS_AUTOMATIC_CONFLICTS:
                environment.pop(name, None)
            environment[_GXE_GROUP_WORKER_TOKEN_ENV] = raw_token
            # A parent sentinel is meaningful only for the parent's outer
            # invocation. Each socket worker must establish its own policy.
            environment.pop("SUMMIT_NUMACTL_WRAPPED", None)
            log._log(
                f"[gxe:multi:parallel] group {index}: environments="
                f"{list(record['columns'])}; CPUs={_format_integer_ranges(record['cpus'])}; "
                f"NUMA nodes={_format_integer_ranges(record['nodes']) if record['nodes'] else 'local'}; "
                f"threads={record['threads']}."
            )
            processes.append(subprocess.Popen(command, env=environment))
            group_manifests.append(group_manifest)

        failed = None
        while processes:
            remaining = []
            for process in processes:
                status = process.poll()
                if status is None:
                    remaining.append(process)
                elif status != 0 and failed is None:
                    failed = status
            if failed is not None:
                for process in remaining:
                    process.terminate()
                for process in remaining:
                    process.wait()
                raise RuntimeError(
                    f"A socket-isolated GxE environment group exited with status {failed}."
                )
            processes = remaining
            if processes:
                import time
                time.sleep(1.0)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise

    manifest = combine_multi_environment_reference_batches(
        group_manifests,
        batch_manifest=canonical_manifest,
        environment_order=columns,
        require_cpu_placement=True,
        expected_cpu_groups=[record["cpus"] for record in layout],
        expected_numa_groups=[record["nodes"] for record in layout],
    )
    log._log(
        f"[gxe:multi:parallel] published {len(columns)} references through "
        f"canonical batch manifest {manifest}."
    )
    return manifest


def _dispatch_gxe_multi_reference(args, log, verbose_on, low_level):
    raw_columns = str(args.gxe_env_cols).split(",")
    if any(not column.strip() for column in raw_columns):
        raise ValueError("--gxe-env-cols contains an empty column name.")
    columns = tuple(column.strip() for column in raw_columns)
    if len(columns) < 2 or len(set(columns)) != len(columns):
        raise ValueError(
            "--gxe-env-cols must contain at least two unique column names."
        )
    suffixes = tuple(safe_environment_suffix(column) for column in columns)
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(
            "--gxe-env-cols names collide after filename normalization."
        )

    layout = _parallel_environment_layout(args, columns)
    if (
        bool(getattr(args, "gxe_explicit_openmp_placement", False))
        and getattr(args, "_gxe_group_worker_authenticated", False) is not True
        and not layout
    ):
        raise RuntimeError(
            "Explicit single-group OpenMP placement did not produce its "
            "mandatory fresh-worker layout; a nonempty layout is required "
            "before dispatch, and in-process estimator construction is refused."
        )
    if layout is not None:
        _dispatch_parallel_gxe_environment_groups(args, columns, layout, log)
        return

    estimators = []
    try:
        for column, suffix in zip(columns, suffixes, strict=True):
            estimators.append(
                _make_gxe_generator(
                    args,
                    log,
                    verbose_on,
                    low_level,
                    env_col=column,
                    out_path=f"{args.out}.{suffix}",
                    # This executor shares a single decoded block and invokes
                    # native BLAS on each environment sequentially.
                    native_backend="python",
                )
            )
        manifest = generate_multi_environment_references(
            estimators,
            batch_manifest=(
                args._gxe_multi_batch_manifest or f"{args.out}.gxe.multi.json"
            ),
            requested_backend=args.gxe_native_backend,
            full_precision_layout="current",
        )
        log._log(
            f"[gxe:multi] wrote {len(estimators)} independent references and "
            f"batch manifest {manifest}."
        )
    finally:
        for estimator in estimators:
            estimator.close()


def _dispatch_ldscore(args, log, verbose_on, low_level):
    if args.env is not None:
        if args.write_ld_mc_var or args.skip_ld_mc:
            raise ValueError(
                "Genome-wide GxE LD scores do not yet support the --write-ld-mc-var/"
                "--write-ld-mc-ci diagnostic; --skip-ld-mc is specific to standard GWLD."
            )
        if args.ld_wind_kb is not None:
            log._log("!!! --env is currently supported only for genome-wide LD scores (not --ld-wind-kb). !!!")
            raise SystemExit(1)
        if args.gxe_env_cols is not None:
            log._log(
                ">>> LD score mode: common-cohort multi-environment GxE "
                f"references, --gxe-env-cols {args.gxe_env_cols}"
            )
            _dispatch_gxe_multi_reference(args, log, verbose_on, low_level)
            return
        log._log(f">>> LD score mode: genome-wide GxE cross/interaction LD scores, --env {args.env}")
        unified_native_reference = bool(
            args.gxe_native_backend == "direct"
            and args.gxe_pheno is None
        )
        gwe = _make_gxe_generator(
            args,
            log,
            verbose_on,
            low_level,
            native_backend=("python" if unified_native_reference else None),
        )
        try:
            if unified_native_reference:
                manifest = generate_multi_environment_references(
                    [gwe],
                    batch_manifest=(
                        args._gxe_multi_batch_manifest
                        or f"{args.out}.gxe.multi.json"
                    ),
                    requested_backend="direct",
                    full_precision_layout="current",
                )
                log._log(
                    "[gxe:native] wrote the single-environment reference "
                    f"through the unified descriptor-owned pipeline: {manifest}."
                )
            else:
                gwe._compute_ldscore()
        finally:
            gwe.close()
        return

    if args.ld_wind_kb is not None:
        if args.write_ld_mc_var or args.skip_ld_mc:
            raise ValueError(
                "Windowed LD scores are deterministic and do not use the genome-wide "
                "random-probe MC diagnostic flags."
            )
        if not np.isfinite(args.ld_wind_kb) or args.ld_wind_kb <= 0:
            log._log("!!! --ld-wind-kb must be finite and positive !!!")
            raise SystemExit(1)
        log._log(f">>> LD score mode: windowed, --ld-wind-kb {args.ld_wind_kb}")
        _require_integer_step_size(args, "windowed LD scores")
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
            impute_method=args.impute_method,
            panel_cols=args.win_panel_cols,
            cache_mb=args.win_cache_mb,
            target_mem=args.target_mem,
        )
        try:
            winld._compute_ldscore()
        finally:
            winld.close()
        return

    _require_integer_step_size(args, "genome-wide additive LD scores")
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
        write_kmoments=(args.write_kmoments and not args.skip_kmoments),
        estimate_mc_noise=(not args.skip_ld_mc),
        write_ld_mc_var=args.write_ld_mc_var,
        use_mailman=args.use_mailman,
        impute_method=args.impute_method,
        ddof=args.ddof,
    )
    try:
        gwld._compute_ldscore()
    finally:
        gwld.close()


def _dispatch_h2(args, log):
    if args.trace is not None:
        log._log("!!! Trace summaries are not supported in the refactored h2 path yet. Use --ldscores. !!!")
        raise SystemExit(1)
    if args.ldscores is None:
        log._log("!!! --ldscores must be provided for refactored h2 estimation. !!!")
        raise SystemExit(1)

    if (args.h2_cache_dir is not None or args.h2_cache_only) and not args.h2_batch_fast:
        raise ValueError("--h2-cache-dir/--h2-cache-only require --h2-batch-fast.")

    if args.h2_batch_fast:
        if args.weight_mode != "he":
            raise ValueError(
                "--weight-mode ldsc is not implemented in --h2-batch-fast yet; use the regular "
                "--h2 path for exact per-block IRWLS refits."
            )
        dispatch_h2_batch_fast(args, log)
        return

    sums = Sumrhe(
        bim_path=args.bim,
        sum_path=None,
        h2_path=args.h2,
        out=args.out,
        chisq_threshold=args.max_chisq,
        log=log,
        verbose=args.verbose,
        ldscores=args.ldscores,
        ldscores_w=args.ldscores_w,
        njack=args.njack,
        annot=args.annot,
        chisq_action=args.chisq_action,
        allow_neg_enr=args.allow_neg_enr,
        clip_nonfinite_vals=args.clip_nonfinite_vals,
        adjust_delta=args.adjust_delta,
        enrich_mode=args.enrich_mode,
        jack_mode=args.jack_mode,
        write_jack=args.write_jack,
        weight_mode=args.weight_mode,
        ldsc_m=args.ldsc_m,
        ldsc_irwls_iters=args.ldsc_irwls_iters,
        ldsc_irwls_tol=args.ldsc_irwls_tol,
    )
    sums._run()
    sums._logoff()


def _dispatch_gxe_fit_batch(args, log):
    reference, entries = load_gxe_fit_batch_manifest(
        args.gxe_fit_batch,
    )
    fitted = fit_many_gxe_from_files(
        reference,
        {entry.name: entry.phenotype_input for entry in entries},
        njack=args.njack,
        allow_ill_conditioned=args.allow_ill_conditioned_gxe,
        max_condition=args.gxe_max_condition,
    )
    outputs = write_gxe_fits(
        {
            entry.name: (
                entry.output_prefix,
                fitted[entry.name][0],
                fitted[entry.name][1],
            )
            for entry in entries
        }
    )
    log._log(
        f"[gxe] transactionally fitted and published {len(outputs)} phenotypes "
        f"against one validated reference."
    )
    for entry in entries:
        fit, _ = fitted[entry.name]
        table_path, json_path = outputs[entry.name]
        log._log(
            f"[gxe] {entry.name}: rank={fit.rank}, condition={fit.condition_number:.6g}; "
            f"results={table_path}; diagnostics={json_path}."
        )


def _dispatch_gxe_fit(args, log):
    missing = [
        name
        for name, value in (
            ("--gxe-gwas", args.gxe_gwas),
            ("--gwis", args.gwis),
            ("--gxe-moments", args.gxe_moments),
        )
        if value is None
    ]
    if missing:
        log._log(f"!!! --gxe-fit requires {', '.join(missing)}. !!!")
        raise SystemExit(1)
    fit, equations = fit_gxe_from_files(
        args.gxe_fit,
        args.gxe_moments,
        args.gxe_gwas,
        args.gwis,
        njack=args.njack,
        allow_ill_conditioned=args.allow_ill_conditioned_gxe,
        max_condition=args.gxe_max_condition,
    )
    table_path, json_path = write_gxe_fit(args.out, fit, equations, overwrite=args.gxe_overwrite)
    log._log(
        f"[gxe] fitted {len(fit.component_names)} variance components; "
        f"rank={fit.rank}, condition={fit.condition_number:.6g}, "
        f"relative residual={fit.relative_residual:.3g}."
    )
    for name, estimate in zip(fit.component_names, fit.proportions):
        log._log(f"[gxe] {name}: proportion={estimate:.8g}")
    nxe_index = len(fit.component_names) - 2
    residual_index = len(fit.component_names) - 1
    denom = np.sqrt(
        equations.matrix[nxe_index, nxe_index]
        * equations.matrix[residual_index, residual_index]
    )
    if denom > 0.0:
        nxe_residual_correlation = equations.matrix[nxe_index, residual_index] / denom
        log._log(
            "[gxe] NxE/residual kernel Frobenius correlation="
            f"{nxe_residual_correlation:.8g}."
        )
        if abs(nxe_residual_correlation) >= 0.995:
            log._log(
                "[gxe:warning] NxE and residual kernels are nearly collinear; "
                "their separate estimates can be unstable even when their sum is well determined."
            )
    log._log(f"[gxe] wrote {table_path} and {json_path}.")


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
        ldscores_reg_w=None,
        ldscores_w=args.ldscores_w,
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
        write_jack=args.write_jack,
        write_normeq=args.write_normeq,
        weight_mode=args.weight_mode,
        ldsc_m=args.ldsc_m,
        ldsc_irwls_iters=args.ldsc_irwls_iters,
        ldsc_irwls_tol=args.ldsc_irwls_tol,
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


def _normalize_rg_manifest(path: str, log=None, *, require_intercept: bool = True):
    raw = _read_rg_manifest(path)

    required = ["phen1", "phen2", "sumstats1", "sumstats2"]
    cols = {}
    for name in required:
        hit = _manifest_column(raw, name)
        if hit is None:
            raise ValueError(f"RG manifest '{path}' is missing required column '{name}'.")
        cols[name] = hit

    overlap_covariance_col = _manifest_column(raw, "overlap_covariance")
    legacy_overlap_covariance_col = _manifest_column(raw, "intercept_rg")
    if (
        overlap_covariance_col is not None
        and legacy_overlap_covariance_col is not None
    ):
        raise ValueError(
            f"RG manifest '{path}' contains both overlap_covariance and its "
            "legacy alias; provide only overlap_covariance."
        )
    if overlap_covariance_col is None:
        overlap_covariance_col = legacy_overlap_covariance_col
    if require_intercept and overlap_covariance_col is None:
        raise ValueError(
            f"RG manifest '{path}' is missing required column "
            "'overlap_covariance'."
        )
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

        sumstats1 = utils._normalize_path_spec(sumstats1_raw)
        sumstats2 = utils._normalize_path_spec(sumstats2_raw)
        if not utils._path_spec_exists(sumstats1):
            raise ValueError(f"Manifest row {row_id}: could not find sumstats1 file/spec '{sumstats1_raw}'.")
        if not utils._path_spec_exists(sumstats2):
            raise ValueError(f"Manifest row {row_id}: could not find sumstats2 file/spec '{sumstats2_raw}'.")

        raw_intercept = (
            None
            if overlap_covariance_col is None
            else row[overlap_covariance_col]
        )
        intercept_missing = pd.isna(raw_intercept) or str(raw_intercept).strip() == ""
        if intercept_missing:
            if require_intercept:
                raise ValueError(
                    f"Manifest row {row_id}: overlap_covariance must be finite."
                )
            intercept_rg = np.nan
        else:
            try:
                intercept_rg = float(raw_intercept)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest row {row_id}: overlap_covariance must be a finite numeric value "
                    "or omitted in regular manifest mode."
                ) from exc
            if not np.isfinite(intercept_rg):
                raise ValueError(
                    f"Manifest row {row_id}: overlap_covariance must be finite."
                )

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



@dataclass
class _ManifestTraitCacheEntry:
    path: str
    phen: str
    cov_rank: int | None
    sumstats: Sumstats
    aligned: object
    keep_mask: np.ndarray


def _manifest_row_traits(row) -> tuple[str, ...]:
    if row.sumstats1 == row.sumstats2:
        return (row.sumstats1,)
    return (row.sumstats1, row.sumstats2)


def _build_manifest_trait_use_counts(manifest_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest_df.itertuples(index=False):
        for spath in _manifest_row_traits(row):
            counts[spath] = counts.get(spath, 0) + 1
    return counts


def _plan_rg_manifest_order(manifest_df: pd.DataFrame, log=None) -> list[int]:
    nrows = int(manifest_df.shape[0])
    if nrows <= 1:
        return list(range(nrows))

    rows = list(manifest_df.itertuples(index=False))
    row_traits = [_manifest_row_traits(row) for row in rows]
    remaining = _build_manifest_trait_use_counts(manifest_df)

    resident: set[str] = set()
    unprocessed = set(range(nrows))
    plan: list[int] = []
    peak_resident = 0

    while unprocessed:
        best_idx = None
        best_key = None

        for idx in unprocessed:
            traits = row_traits[idx]
            overlap = sum(1 for t in traits if t in resident)
            remaining_score = sum(int(remaining.get(t, 0)) for t in traits)
            post_score = sum(max(int(remaining.get(t, 0)) - 1, 0) for t in traits)
            new_traits = sum(1 for t in traits if t not in resident)
            row_id = int(rows[idx].row_id)

            key = (
                overlap,
                remaining_score,
                post_score,
                -new_traits,
                -row_id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        assert best_idx is not None
        plan.append(best_idx)
        unprocessed.remove(best_idx)

        for trait_path in row_traits[best_idx]:
            resident.add(trait_path)
        peak_resident = max(peak_resident, len(resident))

        for trait_path in row_traits[best_idx]:
            remaining[trait_path] = int(remaining.get(trait_path, 0)) - 1
            if remaining[trait_path] <= 0:
                resident.discard(trait_path)

    if log is not None:
        if plan != list(range(nrows)):
            preview = ", ".join(str(int(rows[i].row_id)) for i in plan[: min(10, len(plan))])
            log._log(
                f"[rg:manifest] optimized pair order for trait reuse; "
                f"peak resident traits in plan={peak_resident}; "
                f"first manifest row ids in execution plan: {preview}"
            )
        else:
            log._log("[rg:manifest] manifest order already near-optimal; keeping input order.")

    return plan


def _load_manifest_trait_entry(
    *,
    spath: str,
    meta: dict,
    shared_trace,
    args,
    cache: dict[str, _ManifestTraitCacheEntry],
    log,
    verbose_level: int,
) -> _ManifestTraitCacheEntry:
    cached = cache.get(spath)
    if cached is not None:
        return cached

    cov_rank = meta.get("cov_rank", None)
    phen = meta.get("phen", utils._phen_name_from_path(spath))
    t0 = utils._get_time()

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
    keep_mask = aligned.keep_mask(
        chisq_threshold=args.max_chisq,
        chisq_action=args.chisq_action,
    )

    entry = _ManifestTraitCacheEntry(
        path=spath,
        phen=phen,
        cov_rank=cov_rank,
        sumstats=ss,
        aligned=aligned,
        keep_mask=np.asarray(keep_mask, dtype=bool),
    )
    cache[spath] = entry

    if log is not None and verbose_level >= 1:
        matched_n = int(np.sum(aligned.matched_mask()))
        keep_n = int(np.sum(entry.keep_mask))
        dt = utils._get_time() - t0
        log._log(
            f"[rg:manifest] cached trait '{phen}' from '{spath}' in {dt:.3f}s; "
            f"matched={matched_n}/{shared_trace.nsnps}, kept={keep_n}."
        )

    return entry



def _dispatch_rg_manifest(args, log):
    if args.trace is not None:
        log._log("!!! Trace summaries are not supported in the refactored rg path yet. Use --ldscores. !!!")
        raise SystemExit(1)
    if args.ldscores is None:
        log._log("!!! --ldscores must be provided for rg estimation. !!!")
        raise SystemExit(1)
    if args.intercept_rg is not None or args.pheno_rg is not None or args.pheno_rg_cov is not None:
        log._log(
            "!!! In rg manifest mode, use optional per-row overlap_covariance "
            "values; omit them in regular mode for summary-only estimation. "
            "Global --overlap-covariance-rg / --pheno-rg / --pheno-rg-cov "
            "are not allowed. !!!"
        )
        raise SystemExit(1)
    if args.cov_rank is not None:
        log._log("!!! In rg manifest mode, provide trait-specific cov_rank via optional manifest columns cov_rank1 / cov_rank2 or via the sumstats files. Global --cov-rank is not allowed. !!!")
        raise SystemExit(1)
    if bool(getattr(args, "rg_fast_no_pair_logs", False)) and not bool(
        getattr(args, "rg_manifest_fast", False)
    ):
        log._log("!!! --rg-fast-no-pair-logs requires --rg-manifest-fast. !!!")
        raise SystemExit(1)
    if getattr(args, "rg_model_manifest", None) and not bool(
        getattr(args, "rg_manifest_fast", False)
    ):
        log._log("!!! --rg-model-manifest requires --rg-manifest-fast. !!!")
        raise SystemExit(1)

    verbose_level = _verbose_to_level(args.verbose)
    manifest_df, trait_meta = _normalize_rg_manifest(
        args.rg,
        log=log,
        require_intercept=False,
    )

    if bool(getattr(args, "rg_manifest_fast", False)):
        if args.weight_mode != "he":
            raise ValueError(
                "--rg-manifest-fast does not retain the per-SNP covariance response required "
                "by --weight-mode ldsc; use regular rg manifest mode."
            )
        execution_plan = _plan_rg_manifest_order(manifest_df, log=log)
        dispatch_rg_manifest_fast(args, log, manifest_df, trait_meta, verbose_level, execution_plan=execution_plan)
        return

    execution_plan = _plan_rg_manifest_order(manifest_df, log=log)
    remaining_uses = _build_manifest_trait_use_counts(manifest_df)

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

    trait_cache: dict[str, _ManifestTraitCacheEntry] = {}
    results_rows = []
    outdir = Path(args.out)

    for exec_pos, row_idx in enumerate(execution_plan, start=1):
        row = manifest_df.iloc[int(row_idx)]
        pair_prefix = str(outdir / row.out_stem)

        pair_log = Logger(suppress=args.suppress)
        pair_log._log(
            f"[rg:manifest] running pair {exec_pos}/{manifest_df.shape[0]} "
            f"(manifest_row={int(row.row_id)}): {row.phen1} vs {row.phen2}"
        )

        entry1 = _load_manifest_trait_entry(
            spath=row.sumstats1,
            meta=trait_meta[row.sumstats1],
            shared_trace=shared_trace,
            args=args,
            cache=trait_cache,
            log=log,
            verbose_level=verbose_level,
        )
        entry2 = _load_manifest_trait_entry(
            spath=row.sumstats2,
            meta=trait_meta[row.sumstats2],
            shared_trace=shared_trace,
            args=args,
            cache=trait_cache,
            log=log,
            verbose_level=verbose_level,
        )

        entry1.sumstats.log = pair_log
        entry2.sumstats.log = pair_log

        row_intercept = (
            None if pd.isna(row.intercept_rg) else float(row.intercept_rg)
        )
        rg = Sumcore(
            bim_path=args.bim,
            rg=None,
            ldscores=None,
            ldscores_reg=None,
            ldscores_reg_w=None,
            ldscores_w=args.ldscores_w,
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
            intercept_rg=row_intercept,
            intercept_rg_source="manifest",
            chisq_action=args.chisq_action,
            report_tau=True,
            allow_neg_enr=args.allow_neg_enr,
            adjust_delta=args.adjust_delta,
            cov_rank=None,
            trace_obj=shared_trace,
            sumstats_pair=(entry1.sumstats, entry2.sumstats),
            aligned_pair=(entry1.aligned, entry2.aligned),
            keep_masks=(entry1.keep_mask, entry2.keep_mask),
            phen_names=(row.phen1, row.phen2),
            write_jack=args.write_jack,
            write_normeq=args.write_normeq,
            weight_mode=args.weight_mode,
            ldsc_m=args.ldsc_m,
            ldsc_irwls_iters=args.ldsc_irwls_iters,
            ldsc_irwls_tol=args.ldsc_irwls_tol,
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

        row_summary = build_manifest_summary_row(
            phen1=row.phen1,
            phen2=row.phen2,
            sumstats1=row.sumstats1,
            sumstats2=row.sumstats2,
            cov_rank1=row.cov_rank1,
            cov_rank2=row.cov_rank2,
            intercept_rg_input=row.intercept_rg,
            out_prefix=pair_prefix,
            n_snps=int(rg_fit.prepared.trace_view.nsnps),
            annot_header=rg.trace_view.annot_header,
            h2_fit1=h2_fit1,
            h2_fit2=h2_fit2,
            intercept=intercept,
            rg_fit=rg_fit,
        )
        row_summary["_row_id"] = int(row.row_id)
        results_rows.append(row_summary)

        log._log(
            f"[rg:manifest] completed pair {exec_pos}/{manifest_df.shape[0]} "
            f"(manifest_row={int(row.row_id)}): {row.phen1} vs {row.phen2}; "
            f"rg={float(rg_fit.rg_total[0]):.6g} (SE: {float(rg_fit.rg_total[1]):.6g})"
        )

        for trait_path in _manifest_row_traits(row):
            remaining_uses[trait_path] = int(remaining_uses.get(trait_path, 0)) - 1
            if remaining_uses[trait_path] <= 0:
                evicted = trait_cache.pop(trait_path, None)
                if evicted is not None and verbose_level >= 1:
                    log._log(
                        f"[rg:manifest] evicted trait '{evicted.phen}' from cache after its final pair."
                    )

    summary_path = outdir / "manifest.results.tsv"
    results_df = pd.DataFrame(results_rows).sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"])
    results_df.to_csv(summary_path, sep="\t", index=False)
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
        allow_zero_overlap=bool(args.allow_zero_rg_overlap),
        log=log,
    )
    log._log(
        f"[make-rg-manifest] completed successfully: wrote {df.shape[0]} pair(s) to '{out_manifest}'."
    )


def _step_size_argument(text):
    """Parse --step_size as a positive integer or the literal 'auto'."""
    value = str(text).strip().lower()
    if value == "auto":
        return "auto"
    return int(text)


def _require_integer_step_size(args, command: str) -> None:
    if getattr(args, "step_size", None) == "auto":
        raise ValueError(
            f"--step_size auto is only supported for GxE reference "
            f"generation; {command} requires an explicit integer step size."
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_explicit_openmp_placement_request(args)
    except ValueError as exc:
        parser.error(str(exc))
    worker_contract = _authenticate_gxe_group_worker(args)
    args._gxe_group_worker_authenticated = worker_contract is not None
    args._gxe_worker_cpu_ids = None
    args._gxe_cpu_placement = None
    args._gxe_cpu_placement_complete = False
    if worker_contract is not None:
        args._gxe_worker_cpu_ids = worker_contract.cpu_ids
        args._gxe_cpu_placement = _configure_gxe_group_worker_placement(
            worker_contract
        )
        args._gxe_cpu_placement_complete = True
    args.rand_samp = _parse_rand_samp(args.rand_samp)

    verbose_level = _verbose_to_level(args.verbose)
    verbose_on = verbose_level > 0
    rg_manifest_mode = utils._is_rg_manifest_arg(args.rg) if args.rg is not None else False
    build_manifest_mode = args.make_rg_manifest is not None

    log = Logger(suppress=args.suppress)

    if args.out is None:
        log._log("!!! --out must be provided. !!!")
        raise SystemExit(1)

    if rg_manifest_mode:
        _check_output_directory(args.out, create=True, log=log)
    else:
        _check_outdir(args.out, create=True, log=log)

    gxe_score_mode = args.gxe_score_reference is not None
    gxe_fit_mode = args.gxe_fit is not None
    gxe_fit_batch_mode = args.gxe_fit_batch is not None
    gxe_trace_mode = bool(
        args.geno is not None
        and args.env is not None
        and not gxe_score_mode
    )
    gxe_workflow_mode = bool(
        gxe_score_mode
        or gxe_fit_mode
        or gxe_fit_batch_mode
        or gxe_trace_mode
    )

    special_gxe_modes = sum(
        int(value)
        for value in (
            gxe_score_mode,
            gxe_fit_mode,
            gxe_fit_batch_mode,
        )
    )
    if special_gxe_modes > 1:
        log._log(
            "!!! Choose only one public GxE workflow mode. !!!"
        )
        raise SystemExit(1)
    if gxe_score_mode:
        if args.geno is None or args.env is None or args.gxe_pheno is None:
            log._log(
                "!!! --gxe-score-reference requires --geno, --env, and --gxe-pheno. !!!"
            )
            raise SystemExit(1)
        if args.gxe_overwrite:
            log._log(
                "!!! Wide GxE scoring is transactionally no-overwrite; choose a fresh --out prefix. !!!"
            )
            raise SystemExit(1)
    if gxe_fit_batch_mode:
        if any(
            value is not None
            for value in (args.gxe_gwas, args.gwis, args.gxe_moments)
        ):
            log._log(
                "!!! --gxe-fit-batch takes every phenotype triplet from its manifest; "
                "do not also pass --gxe-gwas, --gwis, or --gxe-moments. !!!"
            )
            raise SystemExit(1)
        conflicting_reference_options = sorted(
            _provided_long_options(sys.argv[1:], parser=parser)
            & _GXE_BATCH_REFERENCE_OPTIONS
        )
        if conflicting_reference_options:
            log._log(
                "!!! --gxe-fit-batch takes its reference and phenotype definitions "
                "exclusively from the batch manifest and sealed artifacts; remove: "
                f"{', '.join(conflicting_reference_options)}. !!!"
            )
            raise SystemExit(1)
        if args.gxe_overwrite:
            log._log(
                "!!! Batch GxE fitting is transactionally no-overwrite; choose fresh output prefixes. !!!"
            )
            raise SystemExit(1)
    if args.gxe_pheno_cols is not None and not gxe_score_mode:
        log._log("!!! --gxe-pheno-cols is valid only with --gxe-score-reference. !!!")
        raise SystemExit(1)
    if args.gxe_population_reference and not gxe_score_mode:
        log._log(
            "!!! --gxe-population-reference is valid only with --gxe-score-reference. !!!"
        )
        raise SystemExit(1)
    if args.gxe_env_cols is not None:
        if not gxe_trace_mode:
            log._log(
                "!!! --gxe-env-cols is valid only for --geno/--env reference generation. !!!"
            )
            raise SystemExit(1)
        if args.gxe_pheno is not None:
            log._log(
                "!!! Multi-environment construction is phenotype-free; score traits "
                "against each generated reference afterward. !!!"
            )
            raise SystemExit(1)
        if args.gxe_overwrite:
            log._log(
                "!!! Multi-environment batches are transactionally no-overwrite; "
                "choose a fresh --out prefix. !!!"
            )
            raise SystemExit(1)

    if gxe_workflow_mode and not args.gxe_overwrite:
        if gxe_score_mode:
            # Trait-specific triplets are reserved transactionally by the scorer.
            suffixes = []
        elif gxe_fit_batch_mode:
            # Trait-specific pairs are reserved transactionally by write_fits().
            suffixes = []
        elif gxe_trace_mode:
            suffixes = [
                ".gxx.ldscore.gz", ".gxe.ldscore.gz", ".exg.ldscore.gz", ".gee.ldscore.gz",
            ]
            suffixes.extend([".gxe.diag.tsv.gz", ".gxe.ref.json"])
            if args.gxe_pheno is not None:
                suffixes.extend([".gxe.gwas.tsv.gz", ".gxe.gwis.tsv.gz", ".gxe.moments.json"])
        else:
            suffixes = [".gxe.results.tsv", ".gxe.fit.json"]
        existing = [args.out + suffix for suffix in suffixes if Path(args.out + suffix).exists()]
        if existing:
            log._log(
                "!!! Refusing to overwrite existing GxE output(s); choose a new --out prefix or pass "
                f"--gxe-overwrite explicitly: {', '.join(existing[:5])} !!!"
            )
            raise SystemExit(1)

    log.install_excepthook()
    if rg_manifest_mode:
        log.attach_file(str(Path(args.out) / "batch.log"))
    else:
        log_suffix = (".gxe.log" if gxe_workflow_mode else (".win.log" if (args.geno and args.ld_wind_kb is not None) else (".gw.log" if args.geno else ".log")))
        log_path = args._gxe_log_path or (args.out + log_suffix)
        log.attach_file(log_path, mode=("w" if (gxe_workflow_mode and args.gxe_overwrite) else "a"))

    explicit_outer_dispatch = bool(args.gxe_explicit_openmp_placement) and (
        worker_contract is None
    )
    if explicit_outer_dispatch:
        # This outer controller performs no numerical work. In particular, it
        # must retain the pre-import launch mask until it has derived and
        # authenticated the fresh worker's CPU placement.
        low_level = None
    else:
        low_level = _make_low_level_env(args)
        low_level["_runtime_threadpool_capped"] = _apply_runtime_thread_cap(
            args.num_threads, log=log
        )
        actual_runtime_threads = apply_env(low_level)
        # Constructors receive the same settings for provenance and derived
        # decoder controls, but must not resize process-global numerical pools.
        low_level["_runtime_preconfigured"] = True
        low_level["_actual_runtime_threads"] = actual_runtime_threads

    _log_cli_args(parser, args, log)

    if args.env is not None and args.geno is None:
        log._log("!!! --env requires --geno. !!!")
        raise SystemExit(1)
    if args.gxe_pheno is not None and (args.geno is None or args.env is None):
        log._log("!!! --gxe-pheno requires --geno and --env. !!!")
        raise SystemExit(1)

    if args.geno is not None and args.weight_mode != "he":
        raise ValueError(
            "--weight-mode applies to --h2/--rg inference, not LD-score "
            "estimation with --geno."
        )

    base_genotype_mode = bool(
        args.geno is not None and not gxe_score_mode
    )
    modes = (
        int(base_genotype_mode)
        + int(gxe_score_mode)
        + int(args.h2 is not None)
        + int(args.rg is not None)
        + int(build_manifest_mode)
        + int(args.gxe_fit is not None)
        + int(args.gxe_fit_batch is not None)
    )
    if modes != 1:
        log._log(
            "!!! Select exactly one primary SUMMIT mode (genotype LD, GxE score/fit, "
            "h2, rg, or manifest building). !!!"
        )
        raise SystemExit(1)

    if gxe_score_mode:
        _dispatch_gxe_score(args, log)
    elif gxe_fit_batch_mode:
        _dispatch_gxe_fit_batch(args, log)
    elif build_manifest_mode:
        _dispatch_make_rg_manifest(args, log)
    elif args.geno is not None:
        _dispatch_ldscore(args, log, verbose_on, low_level)
    elif args.gxe_fit is not None:
        _dispatch_gxe_fit(args, log)
    elif args.h2 is not None:
        _dispatch_h2(args, log)
    else:
        if rg_manifest_mode:
            _dispatch_rg_manifest(args, log)
        else:
            _dispatch_rg(args, log)


if __name__ == "__main__":
    main()
