"""LD scores of an admixed GWAS sample composed from ancestry reference panels (`summit admixed-ld ...`).

A cohort that cannot compute or release in-sample LD scores can still be analysed with SUMMIT. Its SNP covariance is
composed from the covariances of two source populations, the allele-frequency differences between them, and the
covariance of local ancestry along the genome:

    C_jk = pi_A(d) C^A_jk + pi_E(d) C^E_jk + delta_j delta_k kappa(d),
    pi_A(d) = m r(d),  pi_E(d) = (1 - m) r(d),  r(d) = exp(-lambda d)   (d in cM; recombination since admixture),

with m the mean ancestry from the first source, delta = p^A - p^E, and kappa(d) the covariance of local ancestry after
the GWAS covariates. The LD scores written are SUMMIT's genome-wide, V-weighted LD scores of that covariance, with the
term from each person's pairing with themselves, so that `summit --h2` and `--rg` read them unchanged.

Commands (see the wiki page "Admixed cohorts without in-sample LD"):
  reference  build the resource from source panels (banded covariances, their sampling variances, frequencies, map)
  curve      run by the cohort: the weighted-LD curve of its genotypes (aggregate), global-ancestry moments and,
             optionally, the squared relatedness of relatives after the covariates (one number)
  compose    LD scores from the resource, the GWAS summary statistics (mean ancestry and kappa(0) from the standard
             errors) and the decay of kappa (the curve, a published admixture date, or a decay rate)
"""
