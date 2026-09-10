# Heritability and genetic correlation

These analyses require GWAS summary statistics, reference LD scores, and
optional SNP annotations. See [Input files](Input-files.md) for column names.

## Heritability

```bash
summit --h2 trait.sumstats.gz --ldscores reference.gw.ldscore.gz \
  --out results/trait --njack chr --num-threads 4
```

For partitioned heritability, add `--annot annotations.tsv` and use the
corresponding multi-column LD scores. `--h2` also accepts a directory of traits
or chromosome-split paths.

Results are written to `<out>.results.tsv` and `<out>.log`. The table includes
component estimates, standard errors, and enrichment where applicable.

## Genetic correlation

For studies with no overlapping participants:

```bash
summit --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores reference.gw.ldscore.gz --overlap-covariance-rg 0 \
  --out results/trait1.trait2 --njack chr
```

For overlapping studies, either supply `--overlap-covariance-rg` or let SUMMIT
estimate the overlap covariance from summary statistics by omitting that option.
The supplied value is the standardized phenotype cross-product on the shared
samples divided by `sqrt(N1*N2)`. It is not the number or proportion of overlapping participants.

If you have participant-level inputs locally, `--pheno-rg a.tsv,b.tsv` and
optional `--pheno-rg-cov a.cov,b.cov` compute this value. Phenotype tables have
`FID IID` and the phenotype as their last column.

Partitioned analyses with summary-only overlap estimation need a scalar
regression LD score:

```bash
summit --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores partitioned.gw.ldscore.gz --annot annotations.tsv \
  --ldscores-reg total.gw.ldscore.gz --out results/pair --njack chr
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

`--njack chr` deletes one chromosome at a time. An integer requests SNP blocks.
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
