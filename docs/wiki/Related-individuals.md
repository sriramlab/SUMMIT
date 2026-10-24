# Related individuals

SUMMIT's h² and rg estimators assume an unrelated GWAS sample. Close relatives
share untagged genetic variation and often environment, so including them
inflates the estimates. Pruning to an unrelated set avoids this but discards
people, which matters most in small or admixed cohorts.

This workflow keeps every individual and removes the related pairs from the
estimating equations instead. Run the GWAS on everyone as usual. SUMMIT then
computes a per-SNP related-pair moment, `REL_M`, from the genotypes and
phenotypes, which is added to the summary statistics and subtracted by
`summit --h2`. The estimate
stays unbiased whatever the phenotypic covariance between relatives.
[Methods](Methods.md#related-individuals) gives the estimating equations.

## Inputs

- The BED genotypes used for the GWAS. `summit relatives moments` does not yet read PGEN.
- A phenotype table with `FID IID` followed by one or more phenotype columns,
  as given to the GWAS.
- The GWAS covariates, `FID IID` followed by numeric columns, without an
  intercept column. Use exactly the covariates of the GWAS, including genotype PCs.
- A pair table with `IID1 IID2`, optionally `FID1 FID2` and a kinship column.
- GWAS summary statistics from ordinary least squares on all individuals with
  those covariates, for example PLINK 2 `--glm`. Mixed-model statistics
  (BOLT-LMM, REGENIE, fastGWA) are not valid inputs.

```text
IID1 IID2 RELATEDNESS
SIM000001 SIM000002 0.51
SIM000002 SIM000417 0.06
```

Listed pairs are closed into families (connected components) by default, and
every within-family pair is excluded. `--no-closure` excludes only the listed
pairs. With `--kinship-col` and `--kinship-min`, only pairs above the threshold
in that column are used. For relatedness (twice the kinship coefficient), 0.0442
keeps fourth-degree and closer relatives and is a reasonable default.

SUMMIT's PC-adjusted GRM on LD-pruned SNPs and PC-Relate give practically the
same families. KING-robust overstates kinship for individuals of intermediate
ancestry and should not define pairs in recently admixed cohorts. Compute PCs
on unrelated individuals and project relatives onto them (PC-AiR) so that
families do not form their own components.

## Compute related-pair moments

```bash
summit relatives moments \
  --geno study.bed --pheno phenotypes.tsv --covar covariates.tsv \
  --pairs pairs.tsv --kinship-col RELATEDNESS --kinship-min 0.0442 \
  --out results/relmoments
```

The command writes:

- `results/relmoments.tsv.gz`: `CHR SNP BP A1 A2` and one `REL_M_<phenotype>`
  column per phenotype.
- `results/relmoments.json`: sample sizes, covariate counts, pairs used and
  each phenotype's relative-pair covariance in HE scale, `c_rel` (the mean of
  `REL_M`).
- `results/relmoments.log`.

Phenotypes are grouped by their complete-case samples, and each group is
projected on its own covariate basis, as in a GWAS that drops missing values.
`REL_M` does not depend on allele coding. Missing genotypes are mean-imputed,
which agrees with complete-case GWAS to first order in the missing fraction.
The cost is proportional to the number of related pairs times the number of
SNPs; genotypes are read once.

Add `--cross` to write `REL_M12_<a>__<b>` for every pair of phenotypes with the
same complete-case samples (see [Genetic correlation](#genetic-correlation)).
When the LD scores are computed in the study itself, the LD-score step adds the
relatives' share (see [LD scores](#ld-scores)).

## Estimate h²

Add the column to the summary statistics and fit as usual:

```bash
summit relatives append --sumstats trait.sumstats.gz \
  --rel results/relmoments.tsv.gz --pheno TRAIT --out trait.rel.sumstats.gz

summit --h2 trait.rel.sumstats.gz --ldscores reference.gw.ldscore.gz \
  --out results/trait
```

When a summary-statistics file has a `REL_M` column, SUMMIT subtracts it from
each SNP's HE-scale moment. Annotations, partitioned models and every jackknife
design work unchanged. The log reports `c_rel`; SNPs with a missing `REL_M` are
dropped. `--h2-batch-fast` does not accept `REL_M`.

## LD scores

| LD scores | Use when | Extra step |
|---|---|---|
| A matched reference, or the study's unrelated subset | The LD-score sample has the ancestry composition of the full GWAS sample | None |
| In-sample, on all GWAS individuals | Admixed or structured cohorts | Subtract the relatives' share |

In recently admixed cohorts, an unrelated subset chosen by kinship pruning can
have a different ancestry composition from the full cohort, and therefore
different admixture LD. Its LD scores then misstate the trace for the full
sample. In a UK Biobank admixed cohort this lowered h² by about 9%; in-sample
LD scores removed the bias.

In-sample LD scores include the relatives' pairs. Give the LD-score step the
same pairs and options as `moments`:

```bash
summit --geno study.bed --covar covariates.tsv \
  --relative-pairs pairs.tsv --kinship-col RELATEDNESS --kinship-min 0.0442 \
  --nvecs 20000 --block-size 16384 --num-threads 16 --out ld/study
```

`ld/study.gw.ldscore.gz` then has one `LD_REL_<column>` column per LD-score
column, holding the related pairs' share of that LD score. Fit with this file
as usual. `--h2` and `--rg` subtract LD_REL for summary
statistics with `REL_M` and use the LD scores unchanged otherwise; the log says
which. The LD-score step uses every sample with complete covariates, regardless
of phenotype missingness, and computes LD_REL with the same samples, covariates,
annotation and `V` weights as the LD scores. Partitioned LD scores need no
extra option, whatever the annotation. `--relative-pairs` reads BED or PGEN and
applies to genome-wide LD scores only.

An LD-score file without LD_REL columns is taken to come from unrelated
individuals.

## Genetic correlation

Pair exclusion must cover both h² denominators and the cross-trait moment:

```bash
summit relatives moments --geno study.bed --pheno phenotypes.tsv \
  --covar covariates.tsv --pairs pairs.tsv --cross --out results/relmoments
summit relatives append --sumstats a.sumstats.gz --rel results/relmoments.tsv.gz \
  --pheno A --out a.rel.sumstats.gz
summit relatives append --sumstats b.sumstats.gz --rel results/relmoments.tsv.gz \
  --pheno B --out b.rel.sumstats.gz
summit relatives cross-m12 --rel results/relmoments.tsv.gz \
  --pheno1 A --pheno2 B --out a_b.relm12.tsv

summit --rg a.rel.sumstats.gz,b.rel.sumstats.gz --rel-m12 a_b.relm12.tsv \
  --ldscores reference.gw.ldscore.gz \
  --pheno-rg a.pheno.tsv,b.pheno.tsv --pheno-rg-cov covariates.tsv,covariates.tsv \
  --out results/a.b
```

SUMMIT requires `REL_M` in both files and `--rel-m12`. With only some of them,
the pairs would be removed from some estimating equations and kept in others. With
in-sample LD scores that carry LD_REL, the same file serves h² and rg.
`REL_M12` exists only for phenotypes with identical complete-case samples.
The sample-overlap covariance describes each person's own two phenotypes and
needs no change. rg manifests do not accept `REL_M`.

## Interpretation

The estimate is the SNP heritability of the SNPs in the GRM, the quantity
SUMMIT estimates in unrelated samples. Covariance between relatives from
shared environment or untagged variants is removed with their pairs, whatever
its form. Pairs below the threshold remain; their contribution scales with their
squared relatedness and is small for distant relatives.

Jackknife standard errors are computed as for unrelated samples.

`REL_M` and `REL_M12` computed from few pairs carry information about those
individuals' genotypes and phenotypes. Keep per-SNP values in the secure
environment where they were computed. Binary traits (PCGC) are not covered.
