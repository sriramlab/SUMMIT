from logger import Logger
from gw_ldscore import GenomewideLDScore
from sumrhe import Sumrhe
from sumcore import Sumcore
import utils

import argparse
import sys
import numpy as np

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
# parser.add_argument("--filter-both-sides", action='store_true', default=False, \
#                     help='When filtering SNPs, remove their effects on both trace and yKy.'
#                     ' This requires the (truncated) LD scores of all the SNPs used in trace calculation')
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
parser.add_argument("--nworkers", default=4, type=int, \
                    help='Number of workers for multiprocessing to calculate stochastic genome-wide LD scores. Default is 4.')
parser.add_argument("--nvecs", default=10, type=int, \
                    help='Number of random vectors to use for estimating stochastic genome-wide LD scores. Default is 10.')
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


if __name__ == '__main__':
    args = parser.parse_args()
    log = Logger(suppress = args.suppress)
    log._log(">>> SUMMIT arguments")
    log._log("python3 summit.py", end=" ")
    arg = sys.argv[1:]
    i = 0
    while i < len(arg):
        if arg[i].startswith('-'):
            if (i == 0):
                log._log("\t"+arg[i]+" "+arg[i + 1] if i + 1 < len(arg) and not arg[i+1].startswith('-') else "")
                i += 1
            elif (i + 1 < len(arg)):
                if arg[i+1].startswith('-'):
                    log._log("\t"+arg[i])
                else:
                    log._log("\t"+arg[i]+" "+arg[i + 1])
                    i += 1
        else:
            log._log(arg[i] if i==0 else '\t\t'+arg[i])
        i += 1

    if (args.geno is not None):
        if (args.out is None):
            log._log("!!! An output path to save the genome-wide LD scores must be provided !!!")
            sys.exit(1)
        gwld = GenomewideLDScore(bed_path=args.geno, annot_path=args.annot, out_path=args.out, covar_path=args.covar, \
            log=log, num_vecs=args.nvecs, num_workers=args.nworkers, step_size=args.step_size, seed=args.seed, verbose=args.verbose)
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
            njack=args.njack, annot=args.annot)
        sums._run()
        sums._logoff()
    elif (args.rg is not None):
        if (args.ldscores is None):
            log._log("!!! LD score (truncated or genome-wide) must be provided for estimation of genetic correlation !!!")
            sys.exit(1)
        if (args.intercept_rg is not None and args.pheno_rg is not None):
            log._log("!!! --intercept-rg and --pheno-rg cannot be used together; please use one of the two options !!!")
            sys.exit(1)
        rg = Sumcore(bim_path=args.bim, save_path = args.save_trace, rg=args.rg,\
            chisq_threshold=args.max_chisq, log=log, verbose=args.verbose, out=args.out, \
            ldscores=args.ldscores, njack=args.njack, annot=args.annot, \
            intercept=args.intercept_rg, phenos=args.pheno_rg)
        rg._run()
        rg._logoff()
    else:
        log._log("!!! At least one of the options (--geno / --h2 / --rg) must be specified. !!!")
        sys.exit(1)