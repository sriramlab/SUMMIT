# Binary traits and PCGC

PCGC estimates genetic variance on a liability scale while accounting for
case–control sampling and individual disease risks. Select it with
`--binary-method`.

Preparation needs individual genotypes and binary phenotypes. It produces
compatible score and reference moments together; ordinary logistic GWAS
beta/SE files and ordinary LD scores are not interchangeable with these inputs.

## Binary G×E

Joint additive and interaction covariance estimation is available with
`--binary-context-columns` and an explicit conditional liability scale.
All four PCGC modes are supported; external LD remains a factorization
approximation. See [Generalized G×E PCGC](../pcgc_gxe.md) for the model,
inputs, examples, and interpretation.

Contextual preparation also supports genotype-PC adjustment through
`--binary-genotype-covariates`. Experimental sampling SEs use
`--binary-sampling-partners`; `--binary-architecture-probes` adds a Gaussian
SNP-effect variance model. These intervals remain experimental, with
undercoverage in some rare-disease and overlapping-LD simulations. The
[G×E guide](../pcgc_gxe.md#estimates-and-uncertainty) explains their scope.

## Inputs

- BED or biallelic diploid PGEN genotypes, with their companion files.
- A table with `FID IID Y`, where Y is 0 for controls and 1 for cases. Include
  numeric risk covariates or a column of supplied population risks if needed.
- Population prevalence K. The sample case fraction P is computed from Y.
- Population genotype means and inverse standard deviations, on the same
  SNP order and counted alleles as the genotypes.
- Optional nonnegative SNP annotation weights, including overlapping columns.

The population-scale TSV has exactly `SNP A1 A2 MEAN INV_SD`. `MEAN` refers
to the reader's counted allele: BED A1 or PGEN REF. A saved SUMMIT genotype-scale
directory is also accepted; if it was prepared with a build label, supply
the same `--genome-build`. Build labels are otherwise optional. Do not estimate these scales from an ascertained
study and label them population scales.

The sample table selects the analyzed people. IDs and SNPs must be unique;
missing phenotypes, risks, or selected covariates are rejected. Encode categorical
covariates numerically and omit an intercept column; the risk fit includes one.

## Prepare reference and trait moments

`--make-binary-sumstats` computes both the PCGC reference and trait moments.
For age and another exogenous risk factor:

```bash
summit --binary-method pcgc \
  --make-binary-sumstats people.tsv --geno study.bed \
  --binary-prevalence 0.1 \
  --binary-scale population_scale.tsv \
  --binary-covariates AGE,RISK_FACTOR \
  --nvecs 256 --memory-gib 4 \
  --num-threads 4 --out results/trait_pcgc
```

This command fits an ascertainment-aware population-probit risk model.
For supplied individual population risks, replace `--binary-covariates` with
`--binary-risk-column RISK`. `--binary-covariate-variance` optionally supplies
the population variance of the covariate liability predictor.

Add `--annot annotations.tsv` at preparation for partitioned estimates.
This table contains `SNP` followed by weight columns, with every genotype SNP
present exactly once. Without it, SUMMIT fits one component over all SNPs.

Preparation writes `results/trait_pcgc.binary.npz`. This NumPy archive contains
PCGC LD-score rows, trait score products, same-person corrections, annotations,
and their shared variant, scale, and risk metadata. Standard PCGC uses a
study-specific, risk-weighted reference; saving it with the trait moments keeps
the two consistent. The [input guide](Input-files.md#files-used-for-inference)
compares this format with quantitative-trait summaries.

Changes to risks, prevalence, population genotype scaling, or sample selection
require new preparation. Existing output files are not overwritten.

## Fit the saved moments

```bash
summit --binary-method pcgc \
  --h2 results/trait_pcgc.binary.npz --out results/trait_pcgc_fit
```

Use the same method at both stages. The archive supplies both reference and
trait moments, so fitting needs no separate `--ldscores` file or genotype input.
It writes `results/trait_pcgc_fit.binary.json`, containing conditional and
marginal component estimates and SEs, totals, risk diagnostics, and reference
diagnostics.

## Choose a method

| Flag value | Use | Main condition |
|---|---|---|
| `pcgc` | Individual risk weights in scores and the study reference | Exogenous risk covariates and population genotype scaling |
| `liability` | Constant population risk; scalar liability conversion | Constant population risk |
| `pcgc-basis` | Supplied basis and coefficients that exactly reproduce the risk sensitivity | Exact sensitivity span |
| `pcgc-inverse` | Inverse risk weighting with an unweighted genotype reference | Inverse weights must have finite variance |
| `pcgc-ld` | Transfer from an independent population LD reference | Matched reference and valid risk/genotype factorization |

`pcgc-basis` takes `--binary-basis-columns` and
`--binary-basis-coefficients`. An exact span gives the same moments as `pcgc`;
arbitrary risk binning is not an exact span. `pcgc-ld` additionally needs
`--binary-reference-geno`. Its factorization can fail when risk weights and
genotypes are dependent after ascertainment. Inverse weighting can be unstable
with strong continuous risk factors. Neither method is automatically
selected from the data.

## Standard errors and interpretation

All five methods report SNP-block jackknife SEs for annotation components
and totals by default. Binary inference uses 200 contiguous SNP blocks. Set
`--njack` to another integer of at least two when needed for the SNP count
and LD structure; it cannot exceed the number of SNPs. The HE/LDSC default
is chromosome deletion (`chr`).

For example, to use 100 blocks:

```bash
summit --binary-method pcgc \
  --h2 results/trait_pcgc.binary.npz --njack 100 \
  --out results/trait_pcgc_100_blocks
```

The jackknife removes target SNP blocks and retains the full source reference.
An integer block count divides the saved variant order into contiguous groups.
Use genotypes ordered by chromosome and position, with blocks large enough
to contain local LD.
It holds the risk fit, prevalence, population scales, and reference probes fixed.
It therefore excludes uncertainty in those quantities.

Conditional genetic variance is measured relative to liability variance after
the covariate predictor, fixed at one. Marginal estimates divide by
`1 + V_cov`. Negative moment estimates are retained. With overlapping
annotations, components are conditional contributions, not heritability of
each annotation's SNP set in isolation.

The additive model assumes case-status sampling, exogenous risk covariates,
and a compatible population genotype scale, with unprojected genotype features.
For genotype-PC adjustment in the generalized G×E path, see
[Covariate adjustment](../pcgc_gxe.md#covariate-adjustment).
The [PCGC methods](../pcgc.md) give the additive estimating equations.

## Computation and cross-trait work

Study-specific preparation uses two genotype traversals and SUMMIT's existing
genotype readers and matrix operations. Exact basis contraction
is performed before reference calculation. The external-LD method uses one
study scoring traversal and two reference traversals.

`--memory-gib` budgets reference workspace; other process allocations
need additional memory. Reducing `--block-size` changes tiling, while
reducing `--nvecs` also changes numerical precision. Annotation count
increases both sketch memory and computation.

Binary–binary and binary–quantitative covariance are available through
`summit.pcgc.cross.prepare_pair` and `fit_pair` under an explicit marginal
case-status sampling assumption. Shared controls require aligned sample IDs and
a compatible selection design. This remains a research Python API; binary
`--rg` and general cross-study binary summary files are not exposed.

See the [PCGC methods](../pcgc.md) for the equations and API limits.
