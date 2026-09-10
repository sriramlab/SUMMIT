# G×E models

The one-environment model estimates additive genetic variance, interaction
variance, residual variance, and environment-dependent residual variance.
It can reuse one reference calculation across traits.

## Inputs

Supply genotypes, a numeric environment table, covariates, and optional SNP
annotations. Environment and covariate tables start with `FID IID`. The
reference uses one fixed set of samples and variants.

## 1. Build the reference

```bash
summit --geno reference.bed --env environment.tsv --covar covariates.tsv \
  --nvecs 1024 --seed 1 --rand-dist rademacher --target-mem 16 \
  --num-threads 8 --out results/reference
```

The default standardizes the environment and adjusts for the intercept,
environment, and covariates. Genetic and interaction feature columns are
normalized after that adjustment. The reference can be reused only with
compatible SNPs, annotations, and feature definitions.

## 2. Compute trait summaries

```bash
summit --gxe-score-reference results/reference.gxe.ref.json \
  --geno reference.bed --env environment.tsv --covar covariates.tsv \
  --gxe-pheno phenotypes.tsv --gxe-pheno-cols trait1,trait2 \
  --num-threads 8 --out results/scores
```

All selected traits must be finite on the reference sample set. SUMMIT adjusts
and normalizes the phenotypes and computes marginal additive and interaction
scores in one genotype pass.

A conventional conditional interaction statistic, such as PLINK's `ADDxE`
coefficient, is not the marginal interaction score required here. Use SUMMIT's
trait summaries unless you have verified the generating formula independently.

## 3. Fit a trait

```bash
summit --gxe-fit results/reference.gxe.ref.json \
  --gxe-gwas results/scores.trait1.gxe.gwas.tsv.gz \
  --gwis results/scores.trait1.gxe.gwis.tsv.gz \
  --gxe-moments results/scores.trait1.gxe.moments.json \
  --njack 200 --out results/trait1
```

Results include `.gxe.results.tsv`, `.gxe.fit.json`, and `.gxe.log`.
Standard errors use SNP-block deletion of the completed reference and trait
summaries. Reference LD scores are held fixed during those deletions.

Use a new output prefix. `--gxe-overwrite` explicitly permits replacing an
existing result.

## Many traits or environments

`--gxe-fit-batch fit_batch.json` fits multiple traits against one reference.
The manifest identifies each trait's moments, GWAS, GWIS, and output prefix:

```json
{
  "kind": "summit.gxe.fit_batch",
  "schema_version": 1,
  "reference": "reference.gxe.ref.json",
  "traits": [{
    "name": "trait1",
    "moments": "scores.trait1.gxe.moments.json",
    "gwas": "scores.trait1.gxe.gwas.tsv.gz",
    "gwis": "scores.trait1.gxe.gwis.tsv.gz",
    "out": "fits/trait1"
  }]
}
```

`--gxe-env-cols exposure1,exposure2` builds independent one-environment models
while sharing genotype reads. The environments must have the same complete
sample set. For a joint model that includes covariance between environmental
responses, see [Multiple environments](Multiple-environments.md).

## A different study cohort

`--gxe-population-reference` allows scoring one selected trait in a different
cohort. Supply that cohort's genotypes, environment, covariates, and phenotype,
with `--gxe-pheno-col`. The model requires matching variant and feature
definitions and assumes compatible genotype–environment distributions between
reference and study. Cohort-specific residual moments are computed in the study.

## Alternative scaling

`--gxe-kernel-mode raw_projected --gxe-genotype-scale hwe` uses the natural
projected column norms of HWE-scaled genotypes for comparison with GENIE's
kernel definition. It changes the variance-component interpretation. Keep
reference and trait scaling consistent and report the chosen mode.

[Methods](Methods.md) describes the feature definitions and reference transfer.
