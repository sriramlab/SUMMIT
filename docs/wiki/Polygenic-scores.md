# G×E polygenic scores

`summit-pgs` fits SNP weights whose contribution can depend on a person's
context. It supports several candidate priors and traits in one run, sharing
genotype reads across them.

Unlike summary-statistic h²/rg analysis, fitting PGS requires discovery
genotypes and phenotypes. Keep discovery, calibration, and final evaluation
samples separate.

## Try the synthetic example

```bash
python scripts/prediction/checkout.py demo --out example/out/pgs-demo
```

This fits six models across two traits, reloads the saved weights, and scores
24 held-out synthetic samples. It writes model files, predictions, and a short
summary. Use a new output directory for each run.

## Required inputs

| Input | Purpose |
|---|---|
| BED or PGEN genotypes | Discovery variants and sample IDs |
| Phenotype table and discovery keep file | Trait values, units, and training rows |
| Context table and saved coding recipe | Continuous scaling, categorical levels, basis order |
| Covariate table and fixed-effect recipe | Intercept, covariates, context terms, and specified products |
| Genotype scale | Variant means and inverse scales used by the architecture model |
| Genetic prior | Positive-semidefinite effect covariance in that basis and scale |
| Residual variances | Positive per-sample values in the modeled phenotype's variance units |

Use the original architecture model's genotype scale. Estimating a fresh scale
from a different cohort changes the model. [PGS API](PGS-API.md) describes how to
prepare these inputs and define candidate priors.

## Fit and inspect

```bash
summit-pgs plan --spec fit.json --genotype-storage compact \
  --memory-gib 16 --num-threads 8
summit-pgs fit --spec fit.json --out models \
  --genotype-storage compact --memory-gib 16 --num-threads 8
summit-pgs inspect models
```

`plan` checks metadata, feature tables, and model inputs without scanning
genotypes. A fit specification names each trait's inputs and candidate models:

```json
{
  "kind": "summit.prediction.fit_spec",
  "schema_version": 1,
  "genotypes": {"geno": "discovery.bed", "genome_build": "GRCh37"},
  "traits": [{
    "id": "trait1",
    "phenotype": {"file": "traits.tsv", "column": "trait1", "units": "trait units"},
    "samples": "discovery.keep",
    "contexts": "contexts.tsv",
    "context_spec": "contexts.json",
    "covariates": "covariates.tsv",
    "fixed_spec": "fixed.json",
    "genotype_scale": "discovery.scale",
    "architecture_prior": "prior.json",
    "residual_spec": "residual.json",
    "candidates": [
      {"id": "full", "operation": "common_scale", "kappa": 1.0},
      {"id": "shrunk", "operation": "common_scale", "kappa": 0.5}
    ]
  }],
  "solver": {"rtol": 0.0005, "max_iterations": 150}
}
```

Paths are relative to the specification file. Keep files and data tables use
string `FID IID` columns. A trait's optional `variants` file lists SNP IDs
without a header. Replace `geno` with `shards: ["chr1.bed", "chr2.bed"]` for
ordered chromosome files with identical sample order.

All candidates for a trait use the same supplied residual variances. Optional
phenotype `center` and `scale` transform its values before fitting; residual
variances must already be in those transformed units.

## Score new samples

```bash
summit-pgs score --models models --spec score.json --out predictions \
  --memory-gib 16 --num-threads 8
```

A score specification uses the saved model's context and fixed-effect recipes:

```json
{
  "kind": "summit.prediction.score_spec",
  "schema_version": 1,
  "genotypes": {"geno": "evaluation.pgen", "genome_build": "GRCh37"},
  "traits": [{"id": "trait1", "samples": "evaluation.keep",
              "contexts": "contexts.tsv", "covariates": "covariates.tsv"}]
}
```

Scoring matches SNP IDs, positions, build, and allele pairs. It handles explicit
allele swaps but does not infer strand. Missing calls use the training mean;
unknown categorical levels are rejected. A missing model SNP is an error by
default. Explicit `missing_variants: "mean_impute"` retains the original model
size and reports variant coverage.

Outputs include response components, genetic scores, and predictions including
fixed effects. They remain in the fitted phenotype units. If a linear phenotype
transformation was used, convert back with `scale*prediction + center`.

## Memory and speed

| Storage mode | Use when |
|---|---|
| `stream` | The genotype panel does not fit in RAM |
| `compact` | BED calls fit in RAM; avoids repeated decoding |
| `standardized` | More RAM is available; avoids repeated standardization |

All computation is FP64. Compact PGEN retains fractional dosages and uses more
memory than compact BED. `--block-size` and `--rhs-columns` control temporary
arrays. Planning estimates memory; measure peak RSS for the intended workload.

Each streamed solver round reads the shared SNP blocks once, even with many
traits or candidates. Cached modes read the source once, then reuse memory.
The scorer uses one traversal for all models that fit its budget.

See [Benchmarks](Benchmarks.md) for measured small-run timings and
[PGS API](PGS-API.md) for calibration and model files.
