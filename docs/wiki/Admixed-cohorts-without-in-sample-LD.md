# Admixed cohorts without in-sample LD (SUMMIT-MIX)

LD scores for an admixed GWAS sample are best computed in the sample itself, with the GWAS covariates
([LD scores](LD-scores.md#covariates-and-snp-variance-weights)). When the genotypes are not at hand, for example for a
cohort that releases only aggregates or for published summary statistics, SUMMIT-MIX (`summit admixed-ld`) composes
them from:

- unadmixed panels of the source populations, made once into a reference;
- the GWAS summary statistics, whose standard errors give the mean ancestry and the variance of local ancestry that the
  covariates leave;
- the decay of local-ancestry covariance with genetic distance, either from a curve that the cohort computes on its
  genotypes (**curve mode**, the default and recommended) or from a published admixture date (**date mode**).

The output is an ordinary `.gw.ldscore.gz` file with a `V` column, which `summit --h2` and `--rg` read unchanged. The
same composition also gives cross-population LD scores for genetic correlation with a European cohort, regional LD
matrices for fine-mapping, and LD scores for segments of each local ancestry.

For a runnable synthetic example, run `bash example/admixed_ld.sh` from the repository root (about a minute). For a
template with real inputs, see `example/admixed_ld_real_data.sh`.

## Who runs what

| Step | Command | Run by | Needs | Writes |
|---|---|---|---|---|
| 1 | `reference`, `clean-panels` | anyone, once per set of panels | panel genotypes, genetic map | reference directory |
| 2 | `curve` (curve mode only) | the cohort, where the genotypes are | QC'd genotypes, the GWAS covariates, related pairs | aggregates only |
| 3 | `compose` | anyone | reference, summary statistics, curve or date | LD scores |
| 4 | `summit --h2`, `--rg` | anyone | summary statistics, LD scores | estimates |

Optional steps: `sample-factors` (the cohort, for samples of a few hundred people), `derive` (a source without a
panel), `matrix` and `finemap-variance` (fine-mapping), `tractor-gwas`, `tractor-sumstats` and `track-rg` (correlation
of effects across local ancestries).

What the curve releases is aggregate: the kernel every 0.25 cM and between chromosomes (a few thousand numbers), the
moments of global ancestry, the background term every 0.5 cM, and a covariance matrix between 2-cM cells of the genome
(about 1,800 cells, a few MB). No row of it describes a person.

## Model

The GWAS sample descends from source populations s (two in the simple case, A and B). Each person has global ancestry
θ_i and local ancestry L_ij (0, 1 or 2 copies from A at SNP j). The covariance of the dosages of SNPs j and k, d cM
apart, is

```text
C(j,k) = Σ_s π_s(d) C_s(j,k) + δ_j δ_k κ(d),        π_s(d) = m_s exp(-λ_s d),
C(j,j) = Σ_s m_s v_s(j) + δ_j² κ(0),
```

with C_s and v_s the panels' covariances and variances, m_s the mean ancestry, δ_j = p_A(j) − p_B(j) the difference of
the sources' allele frequencies and κ(d) the covariance of local ancestry at distance d, the kernel. Within-source LD
survives only on segments of one ancestry that have not recombined since admixture. With more than two sources,
δ_j δ_k κ(d) becomes D_j' K(d) D_k, with D_j the frequency differences against the last source and K(d) a matrix.

The GWAS adjusts for PCs, which track each person's global ancestry. Everything therefore uses the adjusted genotypes
and the residual kernel κ_res(d): it starts at the variance of local ancestry, decays at a rate set by the time since
admixture, and is a small negative constant between chromosomes. The variance before adjustment adds back the
global-ancestry variance that the PCs remove.

The LD score of SNP j is the sum over all SNPs of the genome of the squared correlations (SUMMIT's genome-wide LD
scores). Pairs within 3 Mb use the panel covariances, with each squared panel covariance replaced by its unbiased
estimate. Pairs farther apart carry only admixture LD and are summed by a fast Fourier transform along each chromosome.
Pairs on different chromosomes use the between-chromosome value of the kernel.

A GWAS of N people adds a sample part to the expected squared correlation:

- the same-person term, because each person's genotype variance depends on their ancestry. It is of order 1/N per pair
  but is summed over M partners, so its share of the LD mass grows with M/N;
- a background term from people who share segments identical by descent, relatives included, measured by the curve;
- in curve mode, a cell matrix that holds the long-range structure that PCs leave on single chromosomes.

**Cohort quantities from the standard errors.** For a least-squares GWAS, 1/(N SE² + β̂²) is proportional to the
variance of the adjusted genotype, `m v_A + (1 - m) v_B + δ² κ_res(0)`. A regression across SNPs without an intercept
gives m and κ_res(0). SNPs with χ² ≥ 80 and outlying residuals are left out, and with several traits the medians are
used. Use least-squares or linear-probability statistics. Logistic and mixed-model standard errors weight people
differently and move m; compose names traits whose values stand apart.

## 1. The reference

```bash
summit admixed-ld reference --panel ref/afr --panel ref/eur --names AFR,EUR \
  --genetic-map maps/genetic_map_chr{chr}_combined_b37.txt --threads 16 --out ref/afr_eur
summit admixed-ld clean-panels --ref ref/afr_eur --panel ref/afr --panel ref/eur --names AFR,EUR \
  --threads 16 --out ref/afr_eur_clean
```

Each `--panel` is a PLINK 1 BED or PLINK 2 PGEN prefix of one source population, in the genome build of the cohort and
of the summary statistics. Use the cohort's SNP set or a superset. Variants are matched between panels by position and
alleles (`--match id` matches by ID). Strand flips are resolved except for A/T and C/G SNPs, which are matched as
listed; `--drop-ambiguous` leaves them out, and should be given to every step when the data sets may be on different
strands. Positions in cM come from `--genetic-map` (one file per chromosome with `{chr}`, or one file with a chromosome
column), otherwise from the CM column of the panel, otherwise 1.2 cM per Mb with a warning. Autosomes only.

The resource stores, for each chromosome, the frequencies, variances and covariances of all pairs within `--window-kb`
(default 3,000 kb), with the moments that give each covariance's sampling variance. For about 400,000 array SNPs and two
sources it is about 4 GB.

Panels carry LD of their own sample. Panels drawn from a biobank have a little admixture between the sources, and their
members are cryptic relatives. Inside the band the composition would add both to the cohort's own terms, so
`clean-panels` measures them with the cohort's estimators, run on each panel without covariates. It removes a panel's
admixture LD when its kernel at zero distance is at least 3 jackknife SEs above zero, and it records the panel's own
background, which compose removes inside the band. Run it on every reference; compose records a flag otherwise and
stops when the background term needs the records. `--from-fits` applies the fits of a cleaned reference to a
converted or subset copy made from the same panels, where the panel genotypes are not available.

## 2. The cohort's curve (curve mode)

```bash
summit admixed-ld curve --geno cohort --ref ref/afr_eur_clean --keep gwas_sample.txt --covar gwas.cov \
  --relative-pairs related_pairs.tsv --threads 16 --out cohort
```

Run this where the genotypes are, on the GWAS sample (`--keep`) with the GWAS's own covariate matrix (`--covar`,
including its PCs). It measures κ_res(d) as the weighted LD curve of ROLLOFF and ALDER on the adjusted genotypes, every
0.25 cM up to the longest chromosome and between chromosomes, the moments of global ancestry (mean, variance, third and
fourth moments), and the cell matrix, estimated in two halves of the people so that its square is unbiased.

- The background term is measured from all pairs of people among the least differentiated fifth of the SNPs, by
  distance and between chromosomes. It is part of curve mode; `--no-background` leaves it out. Each chromosome's
  background SNPs are held in memory (people x SNPs); above `--background-memory-gb` (default 32) a random subset of
  them is used, which keeps the term unbiased and is recorded.
- `--relative-pairs` lists related pairs (IID1 IID2, optionally FID1 FID2 and a kinship column), for example pairs with
  relatedness above 0.0442 in a GRM of PC-adjusted genotypes. They keep each family within one half of the cell matrix.
  `--kinship-col` and `--kinship-min` select pairs.

When the traits' GWAS samples differ, make the curve on a set of people that holds all of them; compose then scales the
background term by N (N − 1) / (n (n − 1)) and records a flag. A curve on each trait's own sample is better when the
samples differ much, for example when a trait is measured mostly in one subgroup.

The curve stops when the cohort's alleles disagree with the reference at many shared positions (another genome build)
and when strand-ambiguous SNPs have frequencies that fit the other strand (pass `--drop-ambiguous`). It records the share
of missing genotypes.

## 3. Compose the LD scores

Curve mode:

```bash
summit admixed-ld compose --ref ref/afr_eur_clean --sumstats gwas/ --annot annot.tsv.gz --curve cohort --out ld/cohort
```

Date mode, with only the summary statistics and a published date:

```bash
summit admixed-ld compose --ref ref/afr_eur_clean --sumstats gwas/ --annot annot.tsv.gz \
  --admixture-generations 6 --out ld/cohort_date
```

`--sumstats` takes SUMMIT summary statistics (files or a directory) with `BETA`, `SE`, `N` and `COV_RANK`. Variants are
matched by `CHR` and `BP`; if the summary statistics lack them, give the GWAS variant table with `--snps` (BIM, PVAR, or
a table with SNP CHR BP A1 A2). `--annot` is the annotation of the fit, such as MAF/LD bins.

In curve mode the kernel beyond 2 cM, the decay of the within-source weights and the moments of global ancestry come from
the curve. Below 2 cM the curve also holds within-source LD, so the kernel there is an exponential from κ_res(0) of the
standard errors to the curve's value at 2 cM. The background term is added at every distance, smoothed as a constant
plus decaying exponentials fitted from 3 cM on, and the panels' own background is removed inside the band. The cell
matrix is used for pairs of 2-cM cells at least 10 cM apart and between chromosomes. Both are used whenever the curve
has them; `--relatedness none` and `--no-cell-matrix` leave them out. Compose stops when the curve at 2 cM is above
κ_res(0) of the standard errors, because then the two inputs describe different samples or covariates.

In date mode the kernel decays at T/100 per cM from κ_res(0) to the between-chromosome constant −2 κ_res(0)/(T G), G the
genetic length of the genome. The dates we used were 6 generations for African American and African Caribbean groups and
11 generations for US Latino groups. The same-person term needs the standard deviation of global ancestry: give a
published value with `--ancestry-sd`; otherwise compose uses the value that m and κ_res(0) imply under random mating,
(2 m (1 − m) − κ_res(0))/4, which assortative mating makes too small. There is no background term or cell matrix.

`--sources AFR,EUR,NAT` composes three or more sources (the last is the baseline); give the same `--sources` to every
step. `--mean-ancestry` and `--kappa0` replace the values from the standard errors.

**One file per GWAS sample size.** The same-person and background terms belong to each GWAS's sample. When the traits'
median N differ by more than 1%, compose writes one file per group of traits (`ld/cohort.n<N>.gw.ldscore.gz`, listed
with their traits in `ld/cohort.n_groups.tsv`); fit each trait with its group's file. `ld/cohort.gw.ldscore.gz` is
the file for the median N. When no group's LD mass differs from it by 0.1% (large samples), it serves every trait and
no group files are written. `--single-n` writes only that file.

**Checks.** Compose stops when the GWAS's alleles disagree with the reference at many shared positions. It records
flags in `ld/cohort.admixed_ld.json` (and logs a warning) when a panel was not cleaned, when the ancestry SD is the
derived one, when the curve was made on other people or with another number of covariates than the GWAS reports, and
when the standard errors of some traits give cohort quantities that stand apart from the others. The json also records
every parameter, m and κ_res(0) per trait, and the matching counts.

Fit as usual:

```bash
summit --h2 gwas/ --ldscores ld/cohort.gw.ldscore.gz --annot annot.tsv.gz --out results/h2
summit --rg gwas/height.tsv,gwas/bmi.tsv --ldscores ld/cohort.gw.ldscore.gz --annot annot.tsv.gz --out results/rg
```

**Dense partners.** By default the partners of each LD score are the GWAS SNPs. When the causal variants are denser
than the GWAS SNPs, `--partners reference` takes every reference variant listed in `--annot` as a partner, with the
GWAS SNPs as rows. Build the reference on the dense set; `.gw.M` then holds the partners' counts. The cell matrix and
sample factors are measured on the GWAS SNPs and cannot be combined with it.

## Small samples: the sample statistic

The same-person term assumes that each person's genotype variance follows their global ancestry. In samples of a few
hundred people it is carried as much by the leverage of the covariates, and in first-generation groups by the parents'
ancestries. The cohort can then release a sample statistic made on the GWAS sample with its covariates:

```bash
summit admixed-ld sample-factors --geno cohort --ref ref/afr_eur_clean --keep gwas_sample.txt --covar gwas.cov \
  --threads 8 --out cohort
summit admixed-ld compose ... --curve cohort --sample-factors cohort --out ld/cohort
```

The statistic is a small covariance matrix of person factors (a level, the ancestry proportions and the local-ancestry
variances), estimated from odd and even chromosomes separately so that each person's noise cancels, plus the sample's
realized LD mass in total and between chromosomes, from per-chromosome Gram matrices (the cost of a GRM; `--no-ld-mass`
skips it). Compose scales the factor term to the realized between-chromosome same-person mass and flags a difference of
more than 5% between its LD mass and the realized one. `--calibrate-between-chromosomes` puts that difference on pairs
of SNPs on different chromosomes, equally per pair, which makes the composed LD mass equal to the realized one.

## Sources without a panel

When a source has no unadmixed panel of at least about 100 people, `derive` estimates its frequencies from an admixed
cohort that carries it, by supervised admixture with the other sources' frequencies fixed:

```bash
summit admixed-ld derive --geno cohort --ref ref/afr_eur_clean --name NAT --panel ref/nat_small --covar gwas.cov \
  --ld panel --threads 16 --out ref/afr_eur_nat
summit admixed-ld clean-panels --ref ref/afr_eur_nat --panel ref/nat_small --names NAT --out ref/afr_eur_nat_clean
```

Members of the small panel (`--panel`) anchor the frequencies, which supervised admixture alone does not identify.
With `--ld panel` the source's window LD is the anchor panel's, on the derived source's variances. Every source in the
cohort should be included: a missing source makes the background term take up its admixture LD, in total but along the
wrong frequency differences. A proxy panel from a related population attenuates the admixture LD by r⁴, with r the
correlation of the true and proxy frequency differences, so we do not recommend proxies.

## Cross-population LD

`compose --cross-source EUR` also writes `ld/cohort.cross_EUR.gw.ldscore.gz`, the cross LD scores of this GWAS with a
GWAS of other people whose LD is that of the EUR panel: the products of the two samples' correlations within the band,
signed, with each sample's own V. Fit the genetic covariance with SUMMIT's cross mode, each sample's own LD scores for
its heritability, and the intercept fixed at zero for disjoint samples.

## Regional LD matrices and fine-mapping

```bash
summit admixed-ld matrix --ref ref/afr_eur_clean --params ld/cohort.admixed_ld.json --curve cohort \
  --region 4:1000000-4000000 --sumstats gwas/height.tsv --error-variance --out fm/region1
Rscript example/finemap_two_stage.R fm/region1
```

`matrix` writes the correlation matrix of the adjusted genotypes of the GWAS sample over a region (`fm/region1.ld`,
`.ld.npz`) and the SNPs with the counted allele of each (`fm/region1.snps.tsv`). With `--sumstats` it keeps the region's
SNPs in the summary statistics and writes their z-scores on that allele (`fm/region1.z.tsv`; flipped where the GWAS
counts the other allele, strand-ambiguous SNPs left out unless `--keep-ambiguous`). The admixture term enters linearly
and is computed for every pair; if the kernel over the region's positions is not positive semi-definite, it is
projected onto that cone. The panel term carries sampling noise at every pair, so by default (`--taper auto`) it is
tapered exponentially with distance, at a scale chosen from the panels alone by an unbiased estimate of its squared
error. Use the same mode (`--curve` or `--admixture-generations`) as compose.

The error of a composed matrix has the shape of the matrix itself, so the RSS-λ model of SuSiE, which treats it as
independent noise, finds almost none. Fine-mapping with SUMMIT-MIX matrices therefore uses two SuSiE fits of
z = R b + e with e ~ N(0, s² (R + λI)) and λ = 10⁻⁴: the first with s² = 1, the second with
s² = 1 + mean_j Σ_{c≠j} V_jc E(b_c²), from the first fit's posterior second moments (on the z scale) and V, the
predicted variance of the error of each entry of the matrix. `--error-variance` writes V (`fm/region1.ld_var`,
`.ld_var.npz`) for a GWAS of `--n` people (default: the median N of `--sumstats`): the panels' sampling with
Hardy-Weinberg moments, the taper's squared bias, and the sampling variance of the correlation in the GWAS sample
itself. V depends only on the panels and the composition, never on in-sample LD. `example/finemap_two_stage.R`
(susieR) runs both fits; `summit admixed-ld finemap-variance --matrix fm/region1 --posterior FIT1.tsv --out fm/region1`
gives s² from any first fit's posterior (columns SNP and EB2). At strongly differentiated loci with very strong
signals, and at loci that define how the sample was selected, composed matrices carry error beyond this prediction.

## Correlation of effects across local ancestries

In an admixed person, alleles on segments of each ancestry can have different causal effects. Hou et al. (2023) called
the correlation of these per-allele effects r_admix. SUMMIT-MIX estimates it from the summary statistics of a
local-ancestry-aware GWAS such as Tractor, which regresses the phenotype at each SNP on the covariates, the
local-ancestry dosage and the allele counts on each ancestry's segments. With the local-ancestry dosage in the model,
the variable of each track is the allele's deviation within that track, which carries no admixture LD and is
uncorrelated with the other track's alleles. The track LD scores are therefore short-range and the two tracks' noise
covariance is zero.

```bash
summit admixed-ld tractor-gwas --pgen phased/chr{chr} --lanc lanc/chr{chr}.lanc --pheno pheno.tsv --covar gwas.cov \
  --sources AFR,EUR --threads 16 --out tractor/gwas
summit admixed-ld tractor-sumstats --tractor tractor/gwas.height.tractor.tsv --sources AFR,EUR --out tracks/height
summit admixed-ld compose --ref ref/afr_eur_clean --sumstats tracks/height.AFR.sumstats --annot maf5.tsv.gz \
  --curve cohort --mean-ancestry 0.59,0.41 --ancestry-sd 0.29 --tracks --cross-source EUR --out ld/tracks
summit admixed-ld track-rg --tracks ld/tracks --sumstats tracks/height tracks/bmi tracks/ldl --annot maf5.tsv.gz \
  --out results/pooled
```

**Standard errors.** The estimate needs sandwich standard errors whose meat uses the residuals of the covariates-only
model, σ_i² = e0_i²/(1 − h_ii), with h_ii the person's hat value under the covariates. Least-squares SEs assume equal
residual variance for everyone, but the track variable's variance follows each person's local ancestry and the residual
variance of many traits changes with ancestry, so their own-track moments hold same-person terms that the LD scores do
not model. The usual HC1 sandwich takes σ_i² from the residuals of the full model, which the fit shrinks at alleles
carried by few people on one ancestry's segments, and its estimates are biased low. Tractor writes only OLS SEs.
`tractor-gwas` fits Tractor's model (the phenotype on the covariates, the local-ancestry dosage and the allele counts on
each ancestry's haplotypes, from phased PLINK 2 genotypes and admix-kit local-ancestry calls) on each trait's complete
cases and writes Tractor's columns with these SEs beside the OLS ones (`se_hcn_anc0`, ...; `--hc1` adds HC1 for
comparison). It runs where the genotypes are and writes summary statistics only. With another GWAS tool, compute the
SEs alongside the regression (a sum over people of x_a x_b σ_i²). `tractor-sumstats` reads them as `se_hcn_anc0`, ...,
stops without a sandwich column, and records a flag when only full-model sandwich columns (HC0-HC3) are present. Leave out rare binary traits, whose sandwich z-scores are heavy-tailed.

`compose --tracks` writes each track's own LD scores (`ld/tracks.track_AFR`, `.track_EUR`), their cross scores
(`.track_AFR_EUR`) and, with `--cross-source EUR`, each track against a GWAS of other people from that population. They
are on the per-allele scale, with effect variance proportional to one over the cohort's genotype variance. `track-rg`
fits each pair of tracks with SUMMIT's cross-population equations with the noise covariance fixed at zero, and
combines the classes as r = Σ_c τ_AE,c n_c / √(Σ_c τ_A,c n_c · Σ_c τ_E,c n_c), n_c the partners in class c. Partition by
five MAF classes (edges 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), which take up the dependence of the effect variance on
frequency between classes. One trait is noisy at biobank sizes, so `track-rg` with several traits' prefixes pools them
as the ratio of their summed moments (over traits and classes, τ_AE,c n_c over the root of the two tracks' sums) on
shared jackknife blocks, which weighs traits by their track heritability as a pooled likelihood does. Each MAF class
can also be split at the median LD score of its SNPs (classes named `maf1_ld1`, `maf1_ld2`, ...): with
`--merge-ld-halves` a class keeps its split only when all four of its own-track moments pooled over the traits have a
jackknife z of at least 2 (`--merge-z`), and the traits are fitted again on the merged classes. `--external` (one trait
at a time) adds each track's correlation with a GWAS of other people from the cross-source population. Local-ancestry
errors put alleles on the wrong track and push r up, so read r next to the caller's accuracy.

## Compute

On the UK Biobank development data (3,787 people, 405,807 array SNPs, two sources), `curve` with 592 related pairs
took about 10 minutes on 8 cores of a busy machine with a peak of 2.2 GB. `compose` in curve mode took about 6 minutes on one
core with a peak of 3.3 GB, about half of it writing the per-sample-size files. The curve's main pass is linear in
people and SNPs. The background is linear in people and quadratic in the least differentiated SNPs of each chromosome,
and at biobank scale it is the largest cost. Compose is linear in the number of SNP pairs within the band and does not
read genotypes.

## Accuracy and limits

In two admixed UK Biobank samples (3,787 and 6,111 people, 405,807 array SNPs, 32 traits), curve mode gave a sum of h²
over traits 0.996 and 0.994 times that with in-sample LD scores (eight MAF/LD bins; 1.001 and 0.993 with two MAF
classes). The panels were African-ancestry (4,905 and 2,452 people) and white British (12,148) participants outside the
samples.

The method assumes that the cohort's sources have panels whose LD is that of the sources, and correct local-ancestry
calls for the track correlation. It covers the LD scores and matrices only. Close relatives also share environment,
which enters the summary statistics themselves, so prune them before the GWAS or use
[Related individuals](Related-individuals.md) with in-sample LD scores. A GWAS sample selected on the phenotype carries
LD at the loci that drive the selection that no composition holds; genome-wide estimates are little affected, but
fine-mapping at those loci needs in-sample LD.
