# Binary traits and PCGC

PCGC estimates genetic variance on a liability scale while accounting for
case–control sampling and individual disease risks. Select it with
`--binary-method`. The ordinary HE/LDSC workflows keep their existing defaults.

Preparation needs individual genotypes and binary phenotypes. It produces
compatible score and reference moments together; ordinary logistic GWAS
beta/SE files and ordinary LD scores are not interchangeable with these inputs.

## Inputs

- BED or biallelic diploid PGEN genotypes, with their companion files.
- A table with `FID IID Y`, where Y is 0 for controls and 1 for cases. Include
  numeric risk covariates or a column of supplied population risks if needed.
- Population prevalence K. The sample case fraction P is computed from Y.
- Population genotype means and inverse standard deviations, on the same
  variant, allele, and genome-build axes as the genotypes.
- Optional nonnegative SNP annotation weights, including overlapping columns.

The population-scale TSV has exactly `SNP A1 A2 MEAN INV_SD`. `MEAN` refers
to the reader's counted allele: BED A1 or PGEN REF. A saved SUMMIT genotype-scale
directory is also accepted. Do not estimate these scales from an ascertained
study and label them population scales.

The sample table selects the analyzed people. IDs and SNPs must be unique;
missing phenotypes, risks, or selected covariates are rejected. Encode categorical
covariates numerically and omit an intercept column; the risk fit includes one.

## Prepare and fit

For age and another exogenous risk factor:

```bash
summit --binary-method pcgc \
  --make-binary-sumstats people.tsv --geno study.bed \
  --binary-prevalence 0.1 --binary-genome-build GRCh38 \
  --binary-scale population_scale.tsv \
  --binary-covariates AGE,RISK_FACTOR \
  --binary-probes 256 --binary-memory-gib 4 \
  --num-threads 4 --out results/trait_pcgc

summit --binary-method pcgc \
  --h2 results/trait_pcgc.binary.npz --out results/trait_pcgc_fit
```

The first command fits an ascertainment-aware population-probit risk model.
For supplied individual population risks, replace `--binary-covariates` with
`--binary-risk-column RISK`. `--binary-covariate-variance` optionally supplies
the population variance of the covariate liability predictor.

Add `--annot annotations.tsv` at preparation for partitioned estimates.
This table contains `SNP` followed by weight columns, with every genotype SNP
present exactly once. Without it, SUMMIT fits one component over all SNPs.

Preparation writes `.binary.npz`; inference writes `.binary.json`. The output
contains conditional and marginal component estimates, totals, risk diagnostics,
and reference diagnostics. Existing output files are not overwritten.
Use the same method at both stages. Changing risks, prevalence, or the sample
selection requires preparing new moments.

## Choose a method

| Flag value | Use | Availability |
|---|---|---|
| `pcgc` | Individual risk weights in scores and the study reference | Point estimates |
| `liability` | Constant population risk; scalar liability conversion | Point estimates |
| `pcgc-basis` | Supplied basis and coefficients that exactly reproduce the risk sensitivity | Point estimates |
| `pcgc-inverse` | Inverse risk weighting with an unweighted genotype reference | Requires `--binary-research` |
| `pcgc-ld` | Transfer from an independent population LD reference | Requires `--binary-research` |

`pcgc-basis` takes `--binary-basis-columns` and
`--binary-basis-coefficients`. An exact span gives the same moments as `pcgc`;
arbitrary risk binning is not an exact span. `pcgc-ld` additionally needs
`--binary-reference-geno`. Its factorization can fail when risk weights and
genotypes are dependent after ascertainment. Inverse weighting can be unstable
with strong continuous risk factors. Neither research method is automatically
selected from the data.

## Standard errors and interpretation

Point estimates are available without a jackknife. Experimental SNP-block
uncertainty requires an explicit research flag and integer block count:

```bash
summit --binary-method pcgc --binary-research \
  --h2 results/trait_pcgc.binary.npz --njack 50 \
  --out results/trait_pcgc_jackknife
```

The jackknife removes target SNP blocks and retains the full source reference.
An integer block count divides the saved variant order into contiguous groups.
Use genotypes ordered by chromosome and position, with blocks large enough
to contain local LD.
It holds the risk fit, prevalence, population scales, and reference probes fixed.
It therefore excludes uncertainty in those quantities. Old schema-1 artifacts
support point estimates only; regenerate them to obtain corrected uncertainty.

Conditional genetic variance is measured relative to liability variance after
the covariate predictor, fixed at one. Marginal estimates divide by
`1 + V_cov`. Negative moment estimates are retained. With overlapping
annotations, components are conditional contributions, not heritability of
each annotation's SNP set in isolation.

The model assumes case-status sampling, exogenous risk covariates, and a
compatible population genotype scale. It does not implement ancestry adjustment
by ordinary covariate projection. Calibration is setting-dependent; research
status is retained for binary uncertainty. The
[qualification reports](../pcgc_release_validation.md) give the tested settings
and distinguish numerical agreement from statistical calibration.

## Computation and cross-trait work

Study-specific preparation uses two genotype traversals and SUMMIT's existing
readers, variant probes, and protected matrix products. Exact basis contraction
is performed before reference calculation. The external-LD method uses one
study scoring traversal and two reference traversals.

`--binary-memory-gib` budgets reference workspace; other process allocations
need additional memory. Reducing `--binary-block-size` changes tiling, while
reducing `--binary-probes` also changes numerical precision. Annotation count
increases both sketch memory and computation. Full-cohort memory plans and a
bounded full-sample benchmark are available in the
[scaling audit](../pcgc_second_audit.md); full-genome throughput is not yet measured.

Binary–binary and binary–quantitative covariance are available through
`summit.pcgc.cross.prepare_pair` and `fit_pair` under an explicit marginal
case-status sampling contract. Shared controls require aligned identities and
a compatible selection design. This remains a research Python API; binary
`--rg` and general cross-study binary artifacts are not exposed.

See the [PCGC scientific contract](../pcgc.md) for the equations and API limits.
