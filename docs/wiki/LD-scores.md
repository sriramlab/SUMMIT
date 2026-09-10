# LD scores

SUMMIT estimates LD scores from a reference genotype panel. Use a panel suited
to the ancestry and variant set of the GWAS being analyzed.

## Genome-wide LD scores

```bash
summit --geno reference.bed --out results/reference \
  --nvecs 1000 --step_size 1000 --num-threads 8
```

`--nvecs` controls the number of random vectors. More vectors reduce Monte Carlo
error and increase computation. Set `--seed` to reproduce the randomization.

Add annotations and covariates when needed:

```bash
summit --geno reference.bed --annot annotations.tsv --covar covariates.tsv \
  --out results/partitioned --nvecs 1000 --step_size 1000 --num-threads 8
```

Outputs are `<out>.gw.ldscore.gz`, `<out>.gw.M`, and `<out>.gw.log`.
`<out>.gw.mc.tsv` summarizes random-vector error by annotation.

## Windowed LD scores

```bash
summit --geno reference.bed --ld-wind-kb 1000 \
  --out results/reference.1mb --num-threads 8
```

`--ld-wind-kb` sets the genomic distance limit in kilobases. Windowed estimation
is deterministic and does not use `--nvecs`. Annotation and covariate options
work as above. Outputs use `.win.ldscore.gz`, `.win.M`, `.win.M_5_50`, and `.win.log`.

## Dosage input

Use an explicit `.pgen` path in either command. SUMMIT reads stored REF dosages,
mean-imputes missing values, and estimates LD in that dosage matrix. Hard calls
and imputed dosages can therefore yield different LD scores.

Genome-wide PGEN requires CPU dense computation, mean imputation, and `--ddof 1`.
It does not support Mailman, HWE imputation, skew correction, or kernel-moment
output. Windowed PGEN uses `--win-panel-cols` and `--win-cache-mb` to limit its
decoded panels and cache.

## Choosing resources

`--num-threads` sets the CPU thread count. `--step_size` controls variants per
block; smaller blocks reduce temporary memory. `--target-mem` and
`--target-xz-mem` set memory budgets in GiB. Use an explicit budget on a cluster
when the process cannot detect its scheduler allocation.

CUDA is available for supported BED genome-wide runs. The default Monte Carlo
diagnostic makes phase 2 run on the CPU; `--skip-ld-mc` keeps the CUDA path.

## Monte Carlo error

The default `.gw.mc.tsv` reports the RMS per-SNP Monte Carlo standard error and
its size relative to the RMS LD score. This measures random-vector error
conditional on the reference panel. It excludes uncertainty from sampling that panel.

For per-SNP variances and approximate 95% intervals, add `--write-ld-mc-var`
(or `--write-ld-mc-ci`). This needs at least two vectors and additional storage.
The intervals are pointwise. They may extend below zero.

The [Methods](Methods.md) page gives the variance formula. For a runnable
synthetic example, use `bash example/estimate_partitioned_gwldscore.sh`.
