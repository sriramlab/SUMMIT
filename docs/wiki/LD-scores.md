# LD scores

SUMMIT estimates LD scores from a reference genotype panel. Use a panel suited
to the ancestry and variant set of the GWAS being analyzed.

## Genome-wide LD scores

```bash
summit --geno reference.bed --out results/reference \
  --nvecs 1000 --block-size 1000 --num-threads 8
```

`--nvecs` controls the number of random vectors. More vectors reduce Monte Carlo
error and increase computation. Set `--seed` to reproduce the randomization.

Add annotations and covariates when needed:

```bash
summit --geno reference.bed --annot annotations.tsv --covar covariates.tsv \
  --out results/partitioned --nvecs 1000 --block-size 1000 --num-threads 8
```

Outputs are `<out>.gw.ldscore.gz`, `<out>.gw.M`, and `<out>.gw.log`.
`<out>.gw.mc.tsv` summarizes random-vector error by annotation. `<out>.gw.M`
holds the annotation masses (SNP counts per annotation).

## Covariates and SNP-variance weights

With `--covar`, genome-wide LD scores are built from covariate-adjusted
genotypes, and the output gains a `V` column after `BP`. V is each SNP's
share of genotype variance left after covariate adjustment, `1 - adjusted R²`
of the SNP on the covariates. It is close to 1 for a SNP that the covariates
explain no better than chance and 0 for a monomorphic SNP. The LD scores
weight each partner SNP by its V:

```text
CHR SNP BP V L2_0
1 sim_variant_000001 10583 0.991204 41.802113
```

With these weights, `--h2` and `--rg` fit the covariate-adjusted GRM of the
genotypes before adjustment, the GRM that GCTA and RHE-mc use with the same
covariates, and report h² on that scale. Without them, each SNP is rescaled
to unit variance after adjustment, which down-weights SNPs that the
covariates partly explain. h² is then too low in admixed cohorts: in
simulations on UK Biobank AFR genotypes with 10 PCs, h² of 0.5 was estimated
at about 0.48.

The weights describe the LD-score sample and its covariates. Use the GWAS
covariates; in admixed cohorts, compute the LD scores in the GWAS sample
itself. `--snp-var-weights-off` writes ordinary LD scores without V.
Windowed LD scores are written without V. `--correct-skew` and
`--write-kmoments` require `--snp-var-weights-off`.

## Related individuals in the LD sample

When in-sample LD scores include relatives, `--relative-pairs pairs.tsv`
adds an `LD_REL_<column>` column after the LD scores for each LD-score column.
Each holds the related pairs' share of that LD score, computed with the same
samples, covariates, annotation and V weights. `--kinship-col`, `--kinship-min` and
`--no-closure` select pairs as in `summit relatives moments`; use the same
pairs and options for both. `--h2` and `--rg` subtract LD_REL only for summary
statistics with `REL_M`, and use the LD scores unchanged otherwise. See
[Related individuals](Related-individuals.md#ld-scores).

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
With covariates, a PGEN run reads the dosages once more before phase 1 to compute V.
It does not support Mailman, HWE imputation, skew correction, or kernel-moment
output. Windowed PGEN uses `--win-panel-cols` and `--win-cache-mb` to limit its
decoded panels and cache.

## Choosing resources

`--num-threads` sets the CPU thread count. `--block-size` controls variants per
block; smaller blocks reduce temporary memory. `--memory-gib` sets the
sketch-panel or windowed-LD memory budget in GiB. Use an explicit budget on a cluster
when the process cannot detect its scheduler allocation.

CUDA is available for supported BED genome-wide runs. The default Monte Carlo
diagnostic makes phase 2 run on the CPU; `--skip-ld-mc` keeps the CUDA path.

## Monte Carlo error

The default `.gw.mc.tsv` reports the RMS per-SNP Monte Carlo standard error and
its size relative to the RMS LD score. This measures random-vector error
conditional on the reference panel. It excludes uncertainty from sampling that panel.

For per-SNP variances and approximate 95% intervals, add `--write-ld-mc-var`. This needs at least two vectors and additional storage.
The intervals are pointwise. They may extend below zero.

The [Methods](Methods.md) page gives the variance formula. For a runnable
synthetic example, use `bash example/estimate_partitioned_gwldscore.sh`.
