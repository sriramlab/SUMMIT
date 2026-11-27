# SUMMIT-MIX details

This page has the model behind `summit admixed-ld`, the options of each step, the flags that compose records and the
optional analyses. Start with [Admixed cohorts without in-sample LD](Admixed-cohorts-without-in-sample-LD.md) for what
the method does and the two quick starts.

## Terms

| Term | Meaning |
|---|---|
| source, baseline source | A population that the cohort descends from, such as African or European, with a reference panel. The frequency differences are taken against the last source given, the baseline. |
| curve | The covariance of local ancestry between two SNPs of the cohort as a function of their genetic distance (the weighted LD curve). |
| same-person term | The part of the in-sample LD that comes from each person paired with themselves, because a person's genotype variance depends on their ancestry. It is of order 1/N per pair of SNPs and summed over all partners. |
| background term | LD that pairs of people add when they share segments of the genome identical by descent, relatives included. |
| cell matrix | The covariance of the cohort's ancestry between 2-cM segments (cells) of the genome, which holds long-range structure that one value per distance cannot. |
| sample statistic | A small covariance matrix of person-level factors (a level, the ancestry proportions and the local-ancestry variances) of a GWAS sample, with its realized same-person mass between chromosomes. Compose takes the same-person term from it. |
| release | The folder that the cohort shares: the curve, the background term, the cell matrix, the sample statistic and the MAF and LD bins, all aggregates. |
| LD mass | The sum of squared correlations over all pairs of SNPs, weighted by the share of each SNP's variance that the covariates leave. It is the total of the LD scores and the scale of heritability. |
| ν (nu) | The GWAS sample size less its covariates, N − COV_RANK − 1. |
| band | Pairs of SNPs within 3 Mb, whose covariances the reference stores. |

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
survives only on segments of one ancestry that have not recombined since admixture, hence the weight π_s(d). With more
than two sources, δ_j δ_k κ(d) becomes D_j' K(d) D_k, with D_j the frequency differences against the baseline source
and K(d) a matrix.

The GWAS adjusts for principal components, which track each person's global ancestry. Everything therefore uses the
adjusted genotypes and the residual kernel κ_res(d). It starts at the variance of local ancestry, decays at a rate set
by the time since admixture, and is a small negative constant between chromosomes.

The LD score of SNP j is the sum over all SNPs of the genome of the squared correlations (SUMMIT's genome-wide LD
scores). Pairs in the band use the panel covariances, with each squared panel covariance replaced by its unbiased
estimate. Pairs farther apart carry only admixture LD and are summed by a fast Fourier transform along each chromosome.
Pairs on different chromosomes use the between-chromosome value of the kernel.

A GWAS of N people adds the same-person term and the background term to the expected squared correlation. In curve
mode the same-person term comes from the sample statistic and the background term from the curve, and the cell matrix
replaces the kernel for pairs of SNPs far apart. Date mode composes the same-person term under random mating and has
neither of the other two.

**Cohort quantities from the standard errors.** For a least-squares GWAS, 1/(N SE² + β̂²) is proportional to the
variance of the adjusted genotype, `m v_A + (1 - m) v_B + δ² κ_res(0)`. A regression across SNPs without an intercept
gives m and κ_res(0). SNPs with χ² ≥ 80 and outlying residuals are left out, and with several traits the medians are
used. Logistic and mixed-model standard errors weight people differently and move m, so use least-squares or
linear-probability statistics.

## The reference

`reference` reads one PLINK 1 BED or PLINK 2 PGEN prefix per source (`--panel`, with the names in `--names`), in the
genome build of the cohort and of the summary statistics. Use the cohort's SNP set or a superset. Autosomes only.

- Variants are matched between panels by position and alleles (`--match id` matches by ID). Strand flips are resolved
  except for A/T and C/G SNPs, which are matched as listed. `--drop-ambiguous` leaves them out of the reference.
- Positions in cM come from `--genetic-map` (one file per chromosome with `{chr}`, or one file with a chromosome
  column), otherwise from the CM column of the panel, otherwise 1.2 cM per Mb with a warning. A map whose rates are not
  0.2 to 6 cM per Mb (a map in Morgans, or base pairs in the CM column) stops the step.
- The reference stores, for each chromosome, the frequencies, variances and covariances of all pairs in the band
  (`--window-kb`, default 3,000), with the moments that give each covariance's sampling variance. For about 400,000
  array SNPs and two sources it is about 4 GB.

Panels drawn from a biobank have a little admixture between the sources, and their members include cryptic relatives.
`clean-panels` measures both with the cohort's estimators, run on each panel without covariates. It removes a panel's
admixture LD when its kernel at zero distance is at least 3 jackknife standard errors above zero, and it records the
panel's own background, which compose removes inside the band. Run it on every reference. `--from-fits` applies the
fits of a cleaned reference to a converted or subset copy made from the same panels, where the panel genotypes are not
available. The reference holds only panel aggregates and can be shared with the analysts.

**Panel choice.** Each source needs a panel whose LD is that of the source. In All of Us the choice among the panels
available to us moved the heritability ratio of curve mode by at most 1.7 points at array SNPs, while the same panels
used directly as reference LD missed the in-sample sum by up to 63%. Use unadmixed panels of the populations closest to
the cohort's sources (for African American cohorts, West African panels such as the Yoruba and Esan of 1000 Genomes or
a panel of several West African populations), and a source panel of at least about 100 people. A source without such a
panel can be derived ([Sources without a panel](#sources-without-a-panel)). Leaving a source out puts its admixture LD
on the wrong SNPs.

## The release

`release` is the cohort's step. It runs `curve` and `bins` on the same people, with the same options, and writes one
folder:

| File | What it holds |
|---|---|
| `cohort.curve.tsv`, `cohort.curve.json` | the curve every 0.25 cM up to the longest chromosome and between chromosomes, the decay fits, the moments of global ancestry, and the counts of people, SNPs and related pairs |
| `cohort.background.tsv` | the background term by distance and between chromosomes |
| `cohort.cells.npz` | the cell matrix, estimated in two halves of the people so that its square is unbiased (about 27 MB for two sources) |
| `cohort.sample_factors.json` | the sample statistic |
| `cohort.bins8.annot.gz`, `cohort.maf2.annot.gz` | the eight MAF and LD bins and the two MAF classes (0/1 columns per SNP) |
| `release.json`, `README.txt` | what was made, from how many people and covariates, with which reference, and the list of what to share and what never to share |
| `release.log` | the log of the run |

Every file holds aggregates over the release's people, and none has a row about a person. Make the release on everyone
who enters any GWAS of the cohort (`--keep`), with the covariates that every GWAS uses, principal components included
(`--covar`). `--relative-pairs` is a file of related pairs with columns IID1 and IID2 (optionally FID1, FID2 and a
kinship column), for example pairs with relatedness above 0.0442 in a GRM of PC-adjusted genotypes. They keep each
family within one half of the cell matrix (`--kinship-col` and `--kinship-min` select pairs). `--out` must be a new or empty folder, or an earlier
release, which is replaced.

The release refuses a sample of fewer than 50 people. Its curve stops when the cohort's alleles disagree with the
reference at many shared positions (another genome build) and when strand-ambiguous SNPs have frequencies that fit the
other strand (pass `--drop-ambiguous` to `reference`, `release` and `compose`). It records the share of missing genotypes.

`curve` and `bins` can also be run alone. `curve` writes the same curve files under its `--out` prefix.
`--no-sample-factors` leaves out the statistic, and `--no-background` the background term (with `--relative-pairs` it
then writes the listed pairs' own profile, `.relatedness.tsv`). The curve's two halves of the statistic are the odd and
the even chromosomes, so a curve on one chromosome has no statistic (the flag `sample_factors_not_made`). Each
chromosome's background SNPs are held in memory (people × SNPs); above `--background-memory-gb` (default 32) a random
subset of them is used, which keeps the term unbiased.

## The MAF and LD bins

Heritability is fitted with eight bins of MAF and LD score. SNPs are placed in six MAF bins, cut at 0.02, 0.05, 0.1,
0.2 and 0.3, and in the quartiles of a local in-sample LD score within each. The bins are then pooled by quartile into
two MAF classes, below 0.05 (`maf_low_ld1` to `maf_low_ld4`) and at least 0.05 (`maf_high_ld1` to `maf_high_ld4`). The
local score of a SNP is the sum of the unbiased squared correlations, r² − (1 − r²)/(n − 2) in n people, of its
genotypes with those of the SNPs within 1 Mb on either side, without covariates. MAF is computed among the called
genotypes of the same people.

A score summed over the whole genome holds the sample's own sampling noise, mostly its same-person term, and quartiles
cut on it select SNPs on that noise, which no composed LD can reproduce. Cutting the bins on composed LD scores selects
on the panels' noise instead (in the admixed UK Biobank sample the composed LD score per bin was then 0.96 to 4.79 times
in-sample). A local score carries almost none of either, but only the cohort can compute it, so the release includes
the bins.

```bash
summit admixed-ld bins --geno cohort --keep gwas_people.txt --threads 8 --out bins/cohort
```

writes `bins/cohort.bins8.annot.gz`, `bins/cohort.maf2.annot.gz` and `bins/cohort.bins.json`. `--scores` also writes
each SNP's MAF and local score (`.local_ld.tsv.gz`), which the fit does not need.

**Two MAF classes.** Without the cohort's bins, fit two MAF classes cut at 0.05 (`maf_low`, `maf_high`). They need only
the MAF, from an allele-frequency column of the summary statistics or from the panels at the GWAS's mean ancestry (the
`mean_ancestry` of compose's plan):

```bash
summit admixed-ld bins --sumstats gwas/height.tsv --out bins/gwas
summit admixed-ld bins --ref ref/afr_eur_clean --mean-ancestry 0.80,0.20 --out bins/panels
```

Two MAF classes are also the more stable readout in small samples, where the heritability of eight bins is poorly
determined. In the admixed UK Biobank sample curve mode gave 0.996 with two MAF classes and 0.986 with eight bins.

## Compose

`--sumstats` takes SUMMIT summary statistics with `BETA`, `SE`, `N` and `COV_RANK` (the number of covariates of the
GWAS, principal components included), as files or as a directory, of which it reads every file whose name does not
start with a dot, as `summit --h2` does. Variants are matched by `CHR` and `BP`; if the summary statistics lack them,
give the GWAS variant table with `--snps` (BIM, PVAR, or a table with SNP CHR BP A1 A2). `--annot` is the annotation
of the fit. Its 0/1 columns may overlap, for example the eight bins with coding or conserved regions as further columns.
Compose then works on their disjoint combinations and writes the columns as given. Its memory and time grow with the
number of combinations, so it stops when the columns form more than 64 of them on the GWAS SNPs
(`--max-annotation-atoms`).

**COV_RANK.** ν = N − COV_RANK − 1 sets the size of the same-person term. Summary statistics without a COV_RANK column
take `--cov-rank` if given, else the covariate rank of the release's curve, which compose logs with the flag
`cov_rank_from_curve`. In date mode without either, ν = N − 1 and compose records `cov_rank_missing`. A COV_RANK that
differs from the curve's covariate rank records `covariates_differ_from_curve`, because the curve must use the GWAS's
own covariates.

**Curve mode** (`--release DIR`, or `--curve PREFIX` for a curve alone). The kernel beyond 2 cM, the decay of the
within-source weights and the moments of global ancestry come from the curve. Below 2 cM the curve also holds
within-source LD, so the kernel there is an exponential from κ_res(0) of the standard errors to the curve's value at
2 cM. Compose stops when the curve at 2 cM is above κ_res(0), because then the two inputs describe different samples or
covariates. The background term is added at every distance, smoothed as a constant plus decaying exponentials fitted
from 3 cM on, and the panels' own background is removed inside the band. The cell matrix takes the place of the kernel
for pairs of SNPs whose 2-cM cells have midpoints at least 10 cM apart on a chromosome, and for pairs on different
chromosomes. The same-person term comes from the release's sample statistic. `--relatedness none`, `--no-cell-matrix`
and `--no-sample-factors` leave out the three cohort terms, and `--sample-factors FILE` gives the statistic of a GWAS
sample of its own ([Traits measured in part of the cohort](#traits-measured-in-part-of-the-cohort)).

**Date mode** (`--admixture-generations T`). The kernel decays at T/100 per cM from κ_res(0) to the between-chromosome
constant −2 κ_res(0)/(T G), with G the genetic length of the genome in Morgans. We used 6 generations for African
American and African Caribbean groups and 11 generations for US Latino groups. In the admixed UK Biobank sample dates of
5 and 7 generations moved the heritability ratio by 5% either way. The same-person term needs the standard deviation of
global ancestry. Give a published value with `--ancestry-sd`; otherwise compose uses the value that m and κ_res(0) imply
under random mating, (2 m (1 − m) − κ_res(0))/4, which assortative mating makes too small. Date mode has no background
term and no cell matrix.

**Several sources.** `--sources AFR,EUR,NAT` composes three or more sources (the last is the baseline). Give the same
`--sources` to `release` (or `curve`), `compose`, `sample-factors` and `matrix`; `reference` and `clean-panels` take
the panels' `--names`, and `derive` the kept sources' `--known`. `--mean-ancestry` and `--kappa0` replace the values
from the standard errors.

**One file per GWAS sample size.** The same-person and background terms belong to each GWAS's sample. When the traits'
median N differ by more than 1%, compose writes one file per group of traits (`ld/cohort.n<N>.gw.ldscore.gz`, listed
with their traits and N in `ld/cohort.n_groups.tsv`, next to the files, so the folder can be moved). `summit --h2` with
`--ldscores ld/cohort` (or `ld/cohort.gw.ldscore.gz`) reads that list and fits each trait with its own group's file. A
trait not in the list stops the run, and so does a trait whose summary statistics have a median N more than 1% from the
list's. With a directory of summary statistics, `--h2` writes the file each trait used to `OUT.ldscore_groups.tsv`, the
results to `OUT.results.tsv` and each group's own fit to `OUT.ldgroups/<k>/`. `--ldscore-groups off` fits every trait
with the file given. `ld/cohort.gw.ldscore.gz` is the file for the median N. When no group's LD mass differs from it by
0.1% (large samples), it serves every trait and no group files are written. `--single-n` writes only that file.

**The plan.** Before composing, compose logs what it will use. The plan names the mode and the source of the decay,
the source of the same-person term and of the background term, the cell matrix, the ancestry, the sample sizes and the
flags so far. It is stored as `plan` in `ld/cohort.admixed_ld.json`, with the LD-score file each trait was given.
`--dry-run` writes it to `ld/cohort.plan.json` and stops. `--require statistic,cells` (any of curve, statistic, cells, relatives,
background) stops with an error when one of them would be missing, for example in a pipeline.

**Genetic correlation.** Fit traits of the same GWAS sample with `--keep-ambiguous`, because `summit --rg` otherwise
drops A/T and C/G SNPs, which need no alignment between two GWAS of the same people. Give each GWAS's phenotype and
covariate files where they are at hand (`--pheno-rg`, `--pheno-rg-cov`), so that the covariance of the two GWAS
samples is computed from them; without them SUMMIT estimates it from the summary statistics. `--rg` uses the LD-score
file given for both traits, and `--ldscores ld/cohort` gives the file for the traits' median N.

## Traits measured in part of the cohort

One release serves every GWAS of the cohort, and compose scales the same-person and background terms to each trait's N.
This assumes that each GWAS sample is a random subset of the release's people. A trait measured in a selected part of
the cohort can also differ in ancestry and relatedness. In the UK Biobank samples, participants with a measurement of
heel bone mineral density had a mean African ancestry 0.06 and 0.05 below that of the whole sample. Compose therefore
flags traits whose GWAS N is below 85% of the release's people (`gwas_sample_smaller_than_release`, threshold
`--min-release-share`). For such a trait the cohort also releases the statistic of the trait's people, made with the
same covariates, and the analyst composes the trait alone with it:

```bash
summit admixed-ld sample-factors --geno cohort --ref ref/afr_eur_clean --keep fluid_intelligence_people.txt \
  --covar gwas_covariates.txt --threads 8 --out release_extra/fluid_intelligence
summit admixed-ld compose --ref ref/afr_eur_clean --release release --sumstats gwas/fluid_intelligence.tsv \
  --sample-factors release_extra/fluid_intelligence --annot release/cohort.bins8.annot.gz --out ld/fluid_intelligence
```

For fluid intelligence, measured in half of each UK Biobank sample, the release alone gave heritability ratios of 1.10
and 1.02, and the release with the trait's statistic 1.02 and 1.01. The statistic carries the trait's people into the
same-person term, but the admixture LD within chromosomes and the moments of global ancestry still come from the curve
of all participants. For the 40
traits of All of Us participants of African ancestry that were measured in fewer than 85% of them, the release with
each trait's statistic gave heritability ratios of 1.016 to 1.034.

Do not release a trait's statistic when the trait's people, or the release's people without the trait, number fewer
than 50. Two statistics whose samples differ by a few people describe those people. `sample-factors` refuses fewer than
50 people.

## The sample statistic

Under random mating each person's genotype variance follows their global ancestry. In real samples the leverage of the
covariates and, in first-generation groups, the parents' ancestries also matter, most in samples of a few hundred
people. The sample statistic measures this on the GWAS sample with its covariates. It is estimated from the odd and the
even chromosomes separately, so that each person's noise cancels, and holds the realized same-person mass between
chromosomes, to which compose scales the factor term. It also records the mean, standard deviation and upper quantiles
of the covariates' leverage. Under random mating, as date mode composes it, the same-person term was 5% to 6% larger
than with the statistic in the two UK Biobank samples and 26% to 31% larger in a multiracial group of 2,248 All of Us
participants.

`sample-factors` makes the statistic alone, for example for one trait's people; give it another `--out` than the
release's. It can also record the sample's realized LD mass, exactly from per-chromosome Gram matrices (the cost of a
GRM; `--no-ld-mass` skips it) or with B random probes (`--ld-mass-probes B`). A few hundred probes give the total to
about 1%; the between-chromosome part is a small difference of large sums and much less precise. With it, compose flags
a difference of more than 5% between its LD mass and the realized one, and `--calibrate-between-chromosomes` puts the
difference on pairs of SNPs on different chromosomes, equally per pair.

## Flags

Compose records a flag in `ld/cohort.admixed_ld.json` (`flags`) for every departure from the standard recipe that it
allows, and logs a warning or a note. Inputs that disagree stop the step instead. The curve records its flags in
`cohort.curve.json`, and the track steps in their own json. Flags marked "expected" are normal when one release serves
every GWAS of the cohort.

| Flag | Step | Meaning | What to do |
|---|---|---|---|
| `relatedness_curve_sample_differs` | compose | The GWAS's N differs from the release's people by more than 1%, and the background term is scaled by N (N − 1)/(n (n − 1)). Expected. | Nothing, unless the trait was measured in a selected part of the cohort (see `gwas_sample_smaller_than_release`). |
| `sample_factors_from_release` | compose | The same-person term comes from the release's statistic, scaled to the GWAS's N. Expected. | Nothing. |
| `cov_rank_from_curve` | compose | Summary statistics without COV_RANK took the covariate rank of the release's curve. Expected when every GWAS used the release's covariates. | Pass `--cov-rank` if a GWAS used other covariates. |
| `sample_factors_snps_differ` | compose | The GWAS has more than 2% fewer or more SNPs than the statistic, whose realized mass is rescaled by the squared ratio of SNP counts. Expected when the GWAS uses a subset of the release's SNPs. | Nothing for a random subset; make the release on the GWAS's SNPs if the subset is selected (for example by MAF). |
| `gwas_sample_smaller_than_release` | compose | A trait's GWAS N is below 85% of the release's people (of the given statistic's people with `--sample-factors`). | Compose the trait alone with the statistic of its own people ([Traits measured in part of the cohort](#traits-measured-in-part-of-the-cohort)). |
| `se_regression_outlier_traits` | compose | A trait's standard errors give a mean ancestry or κ_res(0) far from the other traits'. Logistic or mixed-model SEs, other covariates or another sample of people do this. | Use least-squares or linear-probability statistics; compose a trait of another sample alone with its own statistic. |
| `covariates_differ_from_curve` | compose | A GWAS's COV_RANK (or `--cov-rank`) differs from the covariate rank of the curve. | Make the release with the GWAS's own covariates, the same for every GWAS. |
| `cov_rank_missing` | compose | Summary statistics without COV_RANK and nothing else that gives it (date mode); ν = N − 1. | Pass `--cov-rank` or add a COV_RANK column. |
| `curve_without_sample_factors` | compose | The curve lists no sample statistic, so the same-person term is under random mating. | Make the release (it always has one) or pass `--sample-factors`. |
| `sample_factors_sample_differs` | compose | A statistic given with `--sample-factors` is of more than 1% more or fewer people than the GWAS's N. | Make the statistic on the GWAS's own people. |
| `sample_factors_calibration_large` | compose | The statistic's term needed a factor outside 0.8 to 1.25 to match its realized mass. | Check that the statistic and the GWAS have the same people and covariates. |
| `sample_factors_not_calibrated` | compose | The term could not be scaled to the realized mass (a mass that is not positive); it is used as estimated. | As for `sample_factors_calibration_large`. |
| `sample_factors_not_psd` | compose | The split-half estimate of the factors' covariance has a negative eigenvalue; it is used as estimated. | Nothing; it happens in small samples. |
| `ld_mass_differs_from_cohort` | compose | The composed LD mass differs by more than 5% from the realized mass that the statistic records. | Look for a missing source, other covariates or another sample; `--calibrate-between-chromosomes` moves the difference between chromosomes. |
| `between_chromosome_shift_skipped` | compose | `--calibrate-between-chromosomes` was not applied: no realized LD mass, or one too noisy. | Make the statistic with its LD mass, or with more probes. |
| `nu_differs_from_gwas_n` | compose | `--nu` was given and differs from N − COV_RANK − 1 by more than 2%. | Leave out `--nu`. |
| `panel_admixture_unmeasured` | compose | A panel of the reference was not cleaned. | Run `clean-panels`. |
| `ancestry_sd_from_identity` | compose | Date mode without `--ancestry-sd`: the SD of global ancestry is the one implied under random mating, a lower bound. | Pass `--ancestry-sd` with a published value. |
| `kernel_flat_join` | compose, matrix | With `--allow-flat-kernel-join`: the curve at 2 cM is not below κ_res(0), and the kernel is held at the curve's value below 2 cM. | Check that the curve and the summary statistics come from the same people and covariates. |
| `relatedness_no_pairs` | compose | `--relatedness curve` with a curve without the background term whose listed relative pairs are not in the sample; the term is zero. | Nothing. |
| `partners_reference` | compose | The development option `--partners reference` was used; `summit --h2` and `--rg` refuse these LD scores. | Compose with the default partners for h² and rg. |
| `kappa0_from_ancestry_sd` | compose | Track summary statistics with `--ancestry-sd`: κ_res(0) = 2 m (1 − m) − 4 SD². | Nothing. |
| `track_scores_without_relatives` | compose | `--tracks` in curve mode: the track LD scores leave out the background term. | Nothing; it is small in the own-track moments. |
| `missing_genotypes` | curve | More than 5% of the genotype calls are missing; they are imputed with the SNP's mean. | Filter SNPs on missingness before the release. |
| `sample_factors_not_made` | curve | No sample statistic could be made (SNPs on one chromosome). | Make the release on SNPs of two or more chromosomes. |
| `cov_rank_without_covariates` | tractor-sumstats | COV_RANK is 2 (K − 1), because the number of covariates was not given. | Pass `--covariates`. |
| `se_rounded` | tractor-sumstats | Standard errors with four decimals and no t statistics. | Give the t statistics or full-precision SEs. |
| `sandwich_from_full_model_residuals` | tractor-sumstats | Sandwich SEs from the full model's residuals, which are biased low at alleles rare on a track. | Use `tractor-gwas`, which writes the covariates-only sandwich SEs. |
| `ols_standard_errors` | tractor-sumstats, track-rg | OLS SEs, accepted with `--allow-ols`; the own-track moments are biased. | Use sandwich SEs. |
| `local_ancestry_shares_do_not_sum_to_one` | tractor-sumstats | The local-ancestry shares of a SNP deviate from 1 by more than 0.02. | Check the local-ancestry calls. |
| `se_type_unknown` | track-rg | Track summary statistics without an SE_TYPE column; sandwich SEs are assumed. | Make them with `tractor-sumstats`. |

## Sources without a panel

When a source has no unadmixed panel of at least about 100 people, `derive` estimates its frequencies from an admixed
cohort that carries it, by supervised admixture with the other sources' frequencies fixed:

```bash
summit admixed-ld derive --geno cohort --ref ref/afr_eur_clean --name NAT --panel ref/nat_small \
  --covar gwas_covariates.txt --ld panel --threads 16 --out ref/afr_eur_nat
summit admixed-ld clean-panels --ref ref/afr_eur_nat --panel ref/nat_small --names NAT --out ref/afr_eur_nat_clean
```

Members of the small panel (`--panel`) anchor the frequencies, which supervised admixture alone does not identify.
With `--ld panel` the source's window LD is the anchor panel's, on the derived source's variances. Include every source
of the cohort. A missing source makes the background term take up its admixture LD, in total but along the wrong
frequency differences. A proxy panel from a related population attenuates the admixture LD by r⁴, with r the
correlation of the true and proxy frequency differences, so we do not recommend proxies. `derive` reads the cohort's
genotypes, and its output contains the cohort's allele frequencies of the derived source.

## Cross-population LD

The genetic correlation of a trait between the admixed GWAS and a GWAS of other people from an unadmixed population,
for example a European cohort, needs cross LD scores. `compose --cross-source EUR` also writes
`ld/height.cross_EUR.gw.ldscore.gz`, the products of the two samples' correlations within the band (signed, with each
sample's V), for the SNPs polymorphic in the EUR panel of the reference. The two samples are disjoint, so there is no
same-person or background term. With it compose writes `ld/height.cross_EUR.annot.gz` and `ld/height.cross_EUR.snps`,
the annotation and the IDs of these SNPs. SUMMIT's cross mode fits the genetic covariance with the cross LD scores and
each trait's heritability with its own sample's LD scores, which must list the same SNPs in the same order, with the
overlap covariance fixed at zero:

```bash
summit admixed-ld compose --ref ref/afr_eur_clean --release release --sumstats gwas/height.tsv \
  --annot release/cohort.maf2.annot.gz --cross-source EUR --out ld/height
summit admixed-ld compose --ref ref/afr_eur_clean --release release --sumstats gwas/height.tsv \
  --annot ld/height.cross_EUR.annot.gz --out ld/height_own
plink2 --bfile ref/eur --extract ld/height.cross_EUR.snps --make-bed --out ld/eur_panel
summit --geno ld/eur_panel.bed --annot ld/height.cross_EUR.annot.gz --covar ref/eur_pcs.txt --nvecs 200 \
  --num-threads 8 --out ld/eur_own
summit --rg gwas/height.tsv,eur_gwas/height.tsv --ldscores ld/height.cross_EUR \
  --ldscores-h2-1 ld/height_own --ldscores-h2-2 ld/eur_own --annot ld/height.cross_EUR.annot.gz \
  --overlap-covariance-rg 0 --out results/rg_cross
```

The second compose gives the admixed GWAS's own LD scores on the SNPs of the cross file. The European LD scores
(`ld/eur_own`) are best the in-sample LD scores of the European GWAS with its covariates, which its cohort computes with
`summit --geno` on its genotypes and the same annotation. Without them, compute them as above from the genotypes of a
European panel, here the panel of `--cross-source` kept to the same SNPs, with principal components of the panel as
`--covar` (`ref/eur_pcs.txt`, FID IID and the PCs; the fit needs the V column that covariates give). The two GWAS are
aligned on their alleles, and A/T and C/G SNPs are left out, as `--rg` does by default for GWAS of different cohorts.

## Regional LD matrices and fine-mapping

```bash
summit admixed-ld matrix --ref ref/afr_eur_clean --params ld/cohort.admixed_ld.json --curve release/cohort \
  --region 4:1000000-4000000 --sumstats gwas/height.tsv --error-variance --out fm/region1
Rscript example/finemap_two_stage.R fm/region1
```

`matrix` writes the correlation matrix of the adjusted genotypes of the GWAS sample over a region (`fm/region1.ld`,
`.ld.npz`, exactly symmetric) and the SNPs with the counted allele of each (`fm/region1.snps.tsv`). Use the same mode
(`--curve` with the release's curve, or `--admixture-generations`) as compose.

- With `--sumstats` it keeps the region's SNPs in the summary statistics and writes their z-scores on the matrix's
  counted allele (`fm/region1.z.tsv`; flipped where the GWAS counts the other allele). It leaves out A/T and C/G SNPs,
  whose strand cannot be resolved, unless `--keep-ambiguous`.
- The admixture term is computed for every pair. If the kernel over the region's positions is not positive
  semi-definite, it is projected onto that cone.
- The panel term carries sampling noise at every pair, so by default (`--taper auto`) it is tapered exponentially with
  distance, at a scale chosen from the panels alone by an unbiased estimate of its squared error.

**Two-stage fine-mapping.** The error of a composed matrix has the shape of the matrix itself, so the RSS-λ model of
SuSiE, which treats it as independent noise, finds almost none. Fine-mapping with a composed matrix therefore uses two
SuSiE fits of z = R b + e with e ~ N(0, s² (R + λI)) and λ = 10⁻⁴:

1. the first with s² = 1;
2. the second with s² = 1 + mean_j Σ_{c≠j} V_jc E(b_c²), from the first fit's posterior second moments (on the z
   scale) and V, the predicted variance of the error of each entry of the matrix.

`--error-variance` writes V (`fm/region1.ld_var`, `.ld_var.npz`) for a GWAS of `--n` people (by default the median N
of `--sumstats`). It has three parts, the panels' sampling with Hardy-Weinberg moments, the taper's squared bias, and the
sampling variance of the correlation in the GWAS sample itself. V depends only on the panels and the composition, never
on in-sample LD. `example/finemap_two_stage.R` (susieR) runs both fits;
`summit admixed-ld finemap-variance --matrix fm/region1 --posterior FIT1.tsv --out fm/region1` gives s² from any first
fit's posterior (columns SNP and EB2). At strongly differentiated loci with very strong signals, and at loci that
define how the sample was selected, composed matrices carry error beyond this prediction.

## Correlation of effects across local ancestries

In an admixed person, alleles on segments of each ancestry can have different causal effects. Hou et al. (2023) called
the correlation of these per-allele effects r_admix. SUMMIT-MIX estimates it from the summary statistics of a
local-ancestry-aware GWAS such as Tractor, which regresses the phenotype at each SNP on the covariates, the
local-ancestry dosage and the allele counts on each ancestry's segments. With the local-ancestry dosage in the model,
the variable of each track is the allele's deviation within that track, which carries no admixture LD and is
uncorrelated with the other track's alleles. The track LD scores are therefore short-range and the two tracks' noise
covariance is zero.

```bash
summit admixed-ld tractor-gwas --pgen phased/chr{chr} --lanc lanc/chr{chr}.lanc --pheno pheno.tsv \
  --covar gwas_covariates.txt --sources AFR,EUR --threads 16 --out tractor/gwas
summit admixed-ld tractor-sumstats --tractor tractor/gwas.height.tractor.tsv --sources AFR,EUR --out tracks/height
summit admixed-ld compose --ref ref/afr_eur_clean --sumstats tracks/height.AFR.sumstats --annot maf5.tsv.gz \
  --curve release/cohort --mean-ancestry 0.59,0.41 --ancestry-sd 0.29 --tracks --cross-source EUR --out ld/tracks
summit admixed-ld track-rg --tracks ld/tracks --sumstats tracks/height tracks/bmi tracks/ldl --annot maf5.tsv.gz \
  --out results/pooled
```

`tractor-gwas` reads phased genotypes and local-ancestry calls and runs where the cohort's data are; it writes summary
statistics only.

**Standard errors.** The estimate needs sandwich standard errors whose meat uses the residuals of the covariates-only
model, σ_i² = e0_i²/(1 − h_ii), with h_ii the person's hat value under the covariates.

- Least-squares SEs assume equal residual variance for everyone. The track variable's variance follows each person's
  local ancestry and the residual variance of many traits changes with ancestry, so their own-track moments hold
  same-person terms that the LD scores do not model.
- The usual HC1 sandwich takes σ_i² from the residuals of the full model, which the fit shrinks at alleles carried by
  few people on one ancestry's segments, and its estimates are biased low.
- Tractor writes only OLS SEs. `tractor-gwas` fits Tractor's model (from phased PLINK 2 genotypes and admix-kit
  local-ancestry calls) on each trait's complete cases and writes Tractor's columns with these SEs beside the OLS ones
  (`se_hcn_anc0`, ...; `--hc1` adds HC1 for comparison). With another GWAS tool, compute the SEs alongside the
  regression (a sum over people of x_a x_b σ_i²).
- `tractor-sumstats` reads them as `se_hcn_anc0`, ..., stops without a sandwich column, and records a flag when only
  full-model sandwich columns (HC0-HC3) are present. Leave out rare binary traits, whose sandwich z-scores are
  heavy-tailed.

**Track LD scores and the fit.** `compose --tracks` writes each track's own LD scores (`ld/tracks.track_AFR`,
`.track_EUR`), their cross scores (`.track_AFR_EUR`) and, with `--cross-source EUR`, each track against a GWAS of other
people from that population. They are on the per-allele scale, with effect variance proportional to one over the
cohort's genotype variance. `track-rg` fits each pair of tracks with SUMMIT's cross-population equations with the noise
covariance fixed at zero, and combines the classes as r = Σ_c τ_AE,c n_c / √(Σ_c τ_A,c n_c · Σ_c τ_E,c n_c), n_c the
partners in class c.

- Partition by five MAF classes (edges 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), which take up the dependence of the effect
  variance on frequency between classes.
- One trait is noisy at biobank sizes. With several traits' prefixes, `track-rg` pools them as the ratio of their
  summed moments (over traits and classes, τ_AE,c n_c over the root of the two tracks' sums) on shared jackknife
  blocks, which weighs traits by their track heritability as a pooled likelihood does.
- Each MAF class can also be split at the median LD score of its SNPs (classes named `maf1_ld1`, `maf1_ld2`, ...). With
  `--merge-ld-halves` a class keeps its split only when all four of its own-track moments pooled over the traits have a
  jackknife z of at least 2 (`--merge-z`), and the traits are fitted again on the merged classes.
- `--external` (one trait at a time) adds each track's correlation with a GWAS of other people from the cross-source
  population.

Local-ancestry errors put alleles on the wrong track and push r up, so read r next to the caller's accuracy.

**Partners other than the GWAS SNPs.** By default the partners of each LD score are the GWAS SNPs. `compose --partners
reference` takes every reference variant of `--annot` as a partner, with the GWAS SNPs as rows, and `.gw.M` then holds
the partners' counts, which `track-rg` reads. It is a development option. SUMMIT's h² and rg fits take each class's
size from the regression SNPs and symmetrize the matrix of class traces, which is exact only when the rows are the
partners, so `summit --h2` and `--rg` refuse these LD scores (each has a marker `.partners.json` next to it). With one
class `track-rg` is exact; with several classes the symmetrized matrix is an approximation.

## Compute and logs

On the UK Biobank development data (3,800 to 6,100 people, 405,807 array SNPs, two sources):

| Step | Time | Peak memory |
|---|---|---|
| `reference` (17,000 panel members) | about 18 minutes on 16 cores | a few GB; the reference is about 4 GB on disk |
| `clean-panels` | about 15 minutes on 4 cores | 2.7 GB |
| `curve`, the main part of `release` | 9 to 16 minutes on 8 cores | 2.9 to 3.4 GB |
| `bins`, the rest of `release` (3,781 people) | about 2 minutes on 8 cores | 1.1 GB |
| `compose`, curve mode | about 6 minutes on one core, half of it writing one file per GWAS sample size | 3.3 GB |

The curve's main pass is linear in people and SNPs. The background term is linear in people and quadratic in the least
differentiated SNPs of each chromosome, and at biobank scale it is the largest cost. Plan for hours and tens of GB at
10⁵ people at HapMap3 density. The bins are linear in people and in the SNP pairs within 1 Mb. Compose is linear in the
number of SNP pairs in the band, single-threaded, and does not read genotypes.

Each step writes its log next to its output. `reference`, `clean-panels`, `derive` and `release` write
`reference.log`, `clean-panels.log`, `derive.log` and `release.log` inside their output folder, the other steps
`OUT.<step>.log` (for example `ld/cohort.compose.log`), and `summit --h2` and `--rg` write `OUT.log`.
