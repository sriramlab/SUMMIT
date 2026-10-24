# Heritability and genetic correlation

These analyses require GWAS summary statistics, reference LD scores, and
optional SNP annotations. See [Input files](Input-files.md) for column names.

For binary traits, [PCGC](Binary-traits-and-PCGC.md) prepares ascertainment-aware
moments from individual data. Ordinary logistic beta/SE files cannot substitute
for those moments. To estimate covariance between two traits' environmental
genetic responses, use the [cross-trait research API](Cross-trait-analysis.md).

## Heritability

```bash
summit --h2 trait.sumstats.gz --ldscores reference.gw.ldscore.gz \
  --out results/trait --num-threads 4
```

For partitioned heritability, add `--annot annotations.tsv` and use the
corresponding multi-column LD scores. `--h2` also accepts a directory of traits
or chromosome-split paths.

Results are written to `<out>.results.tsv` and `<out>.log`. The table includes
component estimates, standard errors, and enrichment where applicable.

For a GWAS that includes related individuals, add the related-pair moments
described in [Related individuals](Related-individuals.md) instead of pruning
to an unrelated set.

## Genetic correlation

For studies with no overlapping participants:

```bash
summit --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores reference.gw.ldscore.gz --overlap-covariance-rg 0 \
  --out results/trait1.trait2
```

For overlapping studies, either supply `--overlap-covariance-rg` or let SUMMIT
estimate the overlap covariance from summary statistics by omitting that option.
The supplied value is the standardized phenotype cross-product on the shared
samples divided by `sqrt(N1*N2)`. It is not the number or proportion of overlapping participants.

When N varies across SNPs, the overlap correction assumes a constant overlap
covariance on the local score scale. This is appropriate for shared genotype
missingness with similar missing fractions in both traits and their overlap.
Different cohort contributions or trait-specific missingness require additional
overlap information; per-SNP N alone is insufficient.

If you have participant-level inputs locally, `--pheno-rg a.tsv,b.tsv` and
optional `--pheno-rg-cov a.cov,b.cov` compute this value. Phenotype tables have
`FID IID` and the phenotype as their last column.

Partitioned analyses with summary-only overlap estimation need a scalar
regression LD score:

```bash
summit --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores partitioned.gw.ldscore.gz --annot annotations.tsv \
  --ldscores-reg total.gw.ldscore.gz --out results/pair
```

For disjoint annotations, partition LD scores can be summed to obtain total LD.
For overlapping annotations, supply a separately computed total LD score.

## Allele alignment

SUMMIT aligns alleles by default. It handles allele swaps and strand
complements, changing the second trait's effect sign when required. Invalid or
incompatible alleles are excluded. A/T and C/G variants are excluded by default
because their strand is ambiguous.

`--keep-ambiguous` uses the supplied literal allele orientation. Use
`--no-align-alleles` only after independently harmonizing the inputs.

## Standard errors and filtering

The default jackknife deletes one of 200 contiguous SNP blocks at a time; an
integer sets another block count and `--njack chr` deletes chromosomes. With 200
blocks the SE is far less variable than with 22 chromosomes. In recently
admixed cohorts, long-range admixture LD links neighbouring blocks on a
chromosome and block-jackknife SEs can be too small; chromosome deletion is
less affected.
The default jackknife supports varying per-SNP N. The optional
`--rg-se-method kmoments` and full SCORE normal-equation export require
constant per-SNP sample sizes because they assume one shared sample matrix.

`--max-chisq auto` uses `max(80, 0.001*Nmax)`; `--chisq-action` chooses `drop`,
`clip`, `warn`, or `none`. Review the retained SNP counts in the log.

The default estimation mode is `--weight-mode he`. `--weight-mode ldsc` uses
an iterative weighted LD-score regression with a fixed univariate intercept
and a separately resolved overlap covariance. It is a different estimator;
its formulas and limits are in [Methods](Methods.md).

## Interpretation

With disjoint annotation bins, components describe each bin's contribution.
With overlapping annotations, a coefficient is conditional on the other
columns. Component rg values are ratios of those coefficient contributions;
they are not rg restricted to all SNPs carrying that annotation.

Raw estimates can be negative, and rg may be undefined when its h² denominator
is nonpositive. Do not turn those cases into zero or a bounded correlation.
Strongly collinear annotations may make components unidentifiable.

See [Batch analyses](Batch-analyses.md) for shared loading across many traits.
