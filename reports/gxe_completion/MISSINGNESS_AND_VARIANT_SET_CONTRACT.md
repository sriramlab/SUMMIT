# Missingness and variant-set contract

## Supported reference construction

### Missing genotype calls

For a retained variant and retained common cohort, SUMMIT imputes the
cohort/variant dosage mean before centering, fixed-effect projection, and any
post-projection standardization. Missingness is captured before imputation.
Every reference manifest reports:

- variants and calls with missing genotypes;
- minimum call rate;
- maximum absolute point-biserial correlation of the missing-call indicator
  with the modeled environment and, when present, phenotype;
- thresholds and a validity status.

The current warning thresholds are missing fraction `>0.05` for any variant or
absolute missingness correlation `>0.10`. Crossing either sets
`missingness_warning=true` and `mean_imputation_validity` to
`requires_sensitivity_analysis`. This is a warning, not a correction for MNAR
or differential missingness, and the resulting reference must not be described
as unbiased without sensitivity analysis. A fully missing or zero-variance
projected feature fails the existing feature-variance gate.

The direct native feature path requires missing-free blocks. If missing calls
are detected, construction falls back to the Python streaming decoder so the
same imputation and diagnostic contract is applied rather than silently using a
different native estimand.

### Missing environments and phenotypes

Multi-environment fusion requires exactly the same retained FID/IID rows for all
environments. It does not mean-impute environments. Inputs with differing masks
are rejected and must be explicitly intersected or constructed as separate
batches. Grouping identical masks is not implemented.

## Strict v1 variant axis

The active variant set is the entire ordered reference axis. It is defined by
the exact ordered `CHR,SNP,BP,A1,A2` rows and their digest. The fitter requires:

- both marginal score tables to have exactly the reference row count and order;
- all four `XX,XW,WX,WW` panels to have the same ordered axis;
- annotation masses recomputed from the reference diagonal table to match the
  manifest; and
- file, analysis, feature-convention, and variant digests to agree.

Therefore a missing, filtered, duplicated, reordered, or allele-inconsistent
score row fails closed. SUMMIT does not filter only the target side of a kernel
product. Stored v1 references are not subset-composable; rebuilding the complete
reference on the desired variant set is required. This also makes every full
and delete-block annotation mass refer to the same active kernel.

Historical cache-bound reference readers remain so sealed artifacts can be
validated. New cache/shard construction is retired: the CLI controls were
removed and Hoffman cache/shard/merge workers are hard-gated before execution.

## Variant-specific samples

Schema-v3 additive and interaction score rows require a single exact `N` and
`DF`, equal to the study phenotype sample count and residual rank (or to the
matched reference quantities). Heterogeneous `N_j` or `r_j` is rejected.
`N_j` alone is insufficient because pairwise overlap and mask-specific
projectors are needed. SUMMIT does not currently implement that mask-aware
estimator.

Accepted summary input is `SCORE_MODE=marginal_cross_product`. A conventional
conditional/joint interaction Wald statistic is rejected because it is not
automatically the HE-scale marginal cross-product.

## Feature convention

The canonical versioned field is `feature_convention_version=1` with either:

- `standardized_projected`: each retained projected additive and interaction
  feature has squared norm equal to residual rank; or
- `raw_projected`: naturally scaled projected features and their realized
  traces are retained.

Legacy labels `kernel_mode=standardized` and `kernel_mode=genie` map to these
two conventions, respectively. Explicit and legacy fields must agree.
Reference, moment, and score bundles with different conventions are rejected.
Population transfer remains restricted to the standardized convention because
raw projected traces are cohort-specific.
