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

With `--covar`, genome-wide LD scores are computed from genotypes adjusted
for the covariates, and the output gains a `V` column after `BP`. V is one
minus the adjusted R² of the SNP on the covariates, the share of its variance
that the covariates leave. It is close to 1 for a SNP that the covariates
explain no better than chance and 0 for a monomorphic SNP. Each LD score
weights its partner SNPs by their V. The file then begins

```text
CHR SNP BP V L2_0
1 sim_variant_000001 10583 0.991204 41.802113
```

With these weights, `--h2` and `--rg` fit the GRM of the standardized
genotypes with the covariates as fixed effects, as GCTA and RHE-mc do, and
report h² for that model. Ordinary LD scores rescale each SNP to unit variance
after adjustment, which gives more weight to SNPs that the covariates partly
explain. In admixed cohorts h² is then too low. In simulations on UK Biobank
AFR genotypes with 10 PCs, an h² of 0.5 was estimated at about 0.48 without
the weights.

Use the GWAS covariates for the LD scores, and in admixed cohorts compute the
LD scores in the GWAS sample itself. Because V uses the adjusted R², it does
not depend on the size of the LD-score sample, so LD scores from a random
subset of the GWAS sample (`--rand-samp`) can be used as with ordinary LD
scores. `--snp-var-weights-off` writes ordinary
LD scores without V. Windowed LD scores have no V column. `--correct-skew` and
`--write-kmoments` require `--snp-var-weights-off`.

## Related individuals in the LD sample

When in-sample LD scores include relatives, `--relative-pairs pairs.tsv`
adds one `LD_REL_<column>` column per LD-score column, after the LD scores.
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
