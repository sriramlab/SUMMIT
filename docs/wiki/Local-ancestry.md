# Local ancestry

In admixed individuals, causal variants whose allele frequencies differ between the
source populations are tagged by local ancestry as well as by nearby SNPs. A SNP-only
estimate of h² absorbs variance that only local ancestry tags, including that of a
single ancestry-differentiated locus of large effect such as the Duffy-null allele at
*ACKR1*. Because the SNPs tag local ancestry, the SNP-only estimate absorbs a multiple
of this variance.

SUMMIT treats local-ancestry calls as one more set of markers. Each marker is the
number of haplotypes of one ancestry at one position, with positions spaced along the
genome. A GWAS of these markers is an admixture-mapping scan. Their in-sample LD scores
with the SNPs then separate SNP heritability from heritability tagged only by local
ancestry, in the same `summit --h2` fit with the SNP annotation classes. Relatives can be
kept as described in [Related individuals](Related-individuals.md).

## Inputs

- Local-ancestry calls for each autosome. Two formats are read:
  - admix-kit `.lanc`, with the matching `.pvar` (or `.bim`) of the calling sites and
    the sample IDs, given with `--samples` or taken from the matching `.psam`;
  - RFMix 2 `.msp.tsv`.
- The PLINK 1 genotypes of the GWAS and their annotation, for example the MAF/LD bins
  of [LD scores](LD-scores.md).
- Phenotypes and covariates as for any SUMMIT analysis.

Calls of individuals with no admixture are often missing. `--fill CODE` gives
individuals without calls one ancestry on both haplotypes, so admixed and unadmixed
participants can be analyzed together.

## Make the markers

```bash
summit local-ancestry markers \
  --calls lanc/chr{chr}.lanc --format lanc --sites lanc/chr{chr}.pvar \
  --fam study.fam --ancestry 1 --spacing 250000 --out la/markers

summit local-ancestry merge \
  --geno study --markers la/markers --annot study.annot.tsv.gz --out la/joint
```

`markers` writes `la/markers.{bed,bim,fam}`, with the dosage of ancestry 1 at one
calling site per 250 kb and individuals in the order of `study.fam`. It also writes
`la/markers.theta.tsv`, each individual's global ancestry from the same calls. `merge`
interleaves SNPs and markers by position, which keeps the jackknife blocks genomic, and
writes `la/joint.annot.tsv.gz`, the SNP classes plus a class `LA`.

## Covariates

Add `THETA` from `markers.theta.tsv` to the covariates of the GWAS, the LD scores and
the fit, in addition to genotype PCs. Every dosage contains the individual's global
ancestry, and the local-ancestry kernel amplifies the global ancestry that the PCs leave
by a factor that grows with the sample size. Even PCs that explain 99.5% of the variance
of global ancestry can leave enough to create a spurious component. Adjusting for global
ancestry from the calls removes this. After the adjustment, individuals without admixture
have dosages close to zero and inform the SNP component only.

## Fit

```bash
plink2 --bfile la/joint --pheno pheno.tsv --covar covar_theta.tsv \
  --covar-variance-standardize --glm hide-covar no-firth --out gwas/joint

summit --geno la/joint.bed --annot la/joint.annot.tsv.gz --covar covar_theta.tsv \
  --out ld/joint

summit --h2 gwas/joint_sumstats.tsv --ldscores ld/joint.gw.ldscore.gz \
  --annot la/joint.annot.tsv.gz --out fit/joint
```

Convert the PLINK results to SUMMIT summary statistics as for any trait
([Input files](Input-files.md)). With relatives, add `--relative-pairs` to the LD scores
and the `REL_M` columns to the summary statistics as in
[Related individuals](Related-individuals.md). The `LA` row of the output is the
local-ancestry component. The SNP heritability is the sum of the other classes.

The local-ancestry component is on the scale of the dosages before covariate
adjustment. After the adjustment it explains about a third of this value in a typical
African–European admixed sample, because global ancestry accounts for about two thirds
of the variance of the dosages.

## Test and interval

A jackknife over SNP blocks or chromosomes cannot test the component when it comes from
one locus, because deleting the block that carries the locus removes the whole component.
`summit local-ancestry test` instead computes, exactly from the genotypes:

- a z statistic against the component's standard deviation under no local-ancestry
  effect, from the SNP-only fit;
- a standard error from a sandwich over families (related individuals form one unit),
  for intervals.

```bash
summit local-ancestry test --geno la/joint --annot la/joint.annot.tsv.gz \
  --pheno pheno.tsv --covar covar_theta.tsv --pairs pairs.tsv \
  --kinship-col RELATEDNESS --kinship-min 0.0442 --out fit/joint
```

The output `fit/joint.la_test.tsv` has, for each phenotype, the components `la` and
`snp`, their sandwich standard errors, `la_null_sd`, `la_z` and the one-sided `la_p`. It
holds one N × N matrix per annotation class in memory.

## Interpretation

The component measures variance that only local ancestry tags, after SNP tagging. A
local-ancestry-only model, by contrast, also includes variance shared with the SNPs. An
untagged variant loads on the component in proportion to its allele-frequency difference
between the source populations, so a small component does not show that the SNPs capture
all genetic variation. The model assumes two source populations and SNP effect sizes that
do not depend on ancestral differentiation. When effects concentrate on weakly
differentiated SNPs, or one large-effect SNP has nearly equal frequencies in both
ancestries, the component can be negative.
