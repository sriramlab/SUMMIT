# Batch analyses

Batch modes reduce repeated input loading when many traits share an LD-score
and annotation model.

## Many heritability estimates

```bash
summit --h2 sumstats/ --ldscores reference.gw.ldscore.gz \
  --annot annotations.tsv --h2-batch-fast --njack chr \
  --h2-batch-size 4 --h2-workers 2 --num-threads 4 --out results/h2
```

Fast h² supports HE weighting and chromosome jackknife. For integer SNP blocks,
LDSC weighting, or chi-square clipping, use regular `--h2`.

For repeated annotation models, add `--h2-cache-dir cache/h2`. The default
read/write cache can be reused when the SNP order is identical. After building
it, `--h2-cache-mode read` prevents new entries. `--h2-cache-only` prepares or
checks entries without fitting.

Each cached trait needs about 8.125 bytes per SNP for its values and active
mask, before metadata. Loader workers, batch size, and BLAS threads control
different resources; increasing all three can increase memory substantially.

## Many genetic correlations

A tab-delimited manifest specifies one pair per row:

```text
phen1 phen2 sumstats1 sumstats2 overlap_covariance cov_rank1 cov_rank2
trait_a trait_b trait_a.sumstats trait_b.sumstats 0 0 0
```

```bash
summit --rg pairs.tsv --ldscores reference.gw.ldscore.gz \
  --rg-manifest-fast --njack chr --out results/rg
```

Fast rg requires a supplied finite overlap covariance in every row, HE
weighting, and jackknife standard errors. Ordinary manifests can mix supplied
and summary-estimated overlap covariance. Use regular mode for other weighting
or uncertainty methods.

Outputs include `batch.log`, `manifest.results.tsv`, and per-pair results.
`--rg-fast-no-pair-logs` keeps only the batch log and combined table.

Chromosome jackknife reproduces regular supplied-overlap fits. Integer block
mode uses blocks defined before pair-specific filtering and may differ from
regular mode's block assignment.

## Many annotation models

Store all unique annotation columns and corresponding LD scores once. A model
manifest selects each model's columns:

```text
model bins aliases
model_a ["base","annotation_a"] ["base","focal"]
model_b ["base","annotation_b"] ["base","focal"]
```

Use tabs between fields and the same ordered aliases for each model.

```bash
summit --rg pairs.tsv --rg-manifest-fast --rg-model-manifest models.tsv \
  --ldscores union.gw.ldscore.gz --annot union.annot.tsv \
  --njack chr --out results/models
```

SUMMIT loads shared traits and SNP statistics once, then fits each selected
model. Verify that shared columns are identical before combining existing
annotation datasets.

PGS batching is described in [Polygenic scores](Polygenic-scores.md).
