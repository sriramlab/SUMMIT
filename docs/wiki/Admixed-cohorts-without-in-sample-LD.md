# Admixed cohorts without in-sample LD

LD scores for an admixed GWAS sample are best computed in the sample itself, with the
GWAS covariates ([LD scores](LD-scores.md#covariates-and-snp-variance-weights)). When
the genotypes are not at hand, for example for published summary statistics or a cohort
that releases only aggregates, `summit admixed-ld` composes them from three inputs:

- unadmixed panels of the two source populations, made once into a reference resource;
- the GWAS summary statistics, whose standard errors give the mean ancestry and the
  variance of local ancestry that the covariates leave;
- the decay of local-ancestry covariance with genetic distance, as the cohort's measured
  curve, a published admixture date, or a decay rate.

The sum of squared relatedness of relatives in the sample is an optional fourth input. The
output is an ordinary `.gw.ldscore.gz` file with a `V` column, which `summit --h2` and
`--rg` read unchanged.

For a runnable synthetic example, run `bash example/admixed_ld.sh` from the repository
root. It composes LD scores in two ways and fits h² with them and with in-sample LD
scores.

## Model

The composition rests on these assumptions.

1. The cohort descends from two source populations, and each person's genome is a
   mosaic of segments of one ancestry.
2. On a stretch of one ancestry with no recombination since admixture, LD between two
   SNPs is that of the source population, as measured in its panel. Recombination since
   admixture, at rate λ per cM, breaks this LD.
3. Local ancestry makes SNPs whose allele frequencies differ between the sources
   correlated over long distances. After the covariates, the covariance of local ancestry
   at distance d is κ(d), which decays with d.
4. The GWAS used least squares with the covariates of the cohort, including genotype
   PCs or global ancestry.

The covariance of the dosages of SNPs j and k after the covariates is then

```text
C(j,k) = m exp(-λ d) C_A(j,k) + (1 - m) exp(-λ d) C_E(j,k) + δ_j δ_k κ(d),
```

where m is the mean ancestry from the first source, C_A and C_E are the panel
covariances, and δ is the difference of allele frequencies between the sources. On
different chromosomes only the last term remains, with the between-chromosome value of κ.
The LD scores follow SUMMIT's definition with these covariances. They include each
person's pairing with themselves, which SUMMIT's noise correction leaves when
heterozygosity follows ancestry, and V weights from the same model.
[Methods](Methods.md) gives the in-sample definition.

Mean ancestry and κ(0) come from the standard errors. For least squares,
`N SE² + β²` is proportional to one over the dosage variance after the covariates, which
the model gives as `m v_A + (1 - m) v_E + δ² κ(0)`. A regression across SNPs without an
intercept gives both. SNPs with χ² ≥ 80 and outlying residuals are left out, and with
several traits the medians are used.

## Build the reference

```bash
summit admixed-ld reference --panel ref/afr --panel ref/eur --names AFR,EUR \
  --genetic-map maps/genetic_map_chr{chr}_combined_b37.txt --threads 16 --out ref/afr_eur
```

Each `--panel` is a PLINK 1 BED or PLINK 2 PGEN prefix of one source population. Use the
SNP set of the GWAS, or a superset of it. Variants are matched between panels by position
and alleles (`--match id` matches by ID). Strand flips are resolved except for A/T and
C/G SNPs, which are matched as listed. For each chromosome the resource stores allele
frequencies, dosage variances and the covariances of all pairs within `--window-kb`
(default 3,000 kb), with the moments that give their sampling variance. Positions in cM
come from `--genetic-map` (one file per chromosome with `{chr}`, or one file with a
chromosome column), otherwise from the CM column of the panel. Without either, 1.2 cM
per Mb is assumed with a warning.

The resource is built once and serves every cohort with these sources. The covariance
products use SUMMIT's native protected matrix products. `compose` replaces each squared
panel covariance by its unbiased estimate (a U-statistic), so small panels do not inflate
the LD scores. Rare variants are the least accurate in a small panel, since only a few
people carry them. In the check below, a random 85 of the 4,905 African-ancestry panel
members gave mean h² within 2% of the in-sample value.

Panels drawn from a biobank carry LD of their own sample. Their members have a little
admixture between the sources, and they are cryptic relatives and a structured sample, so
their unbiased squared covariances hold the cross-person LD of their members on top of
the population's LD (for a panel of n, v_j v_k S(d) / (n (n - 3)), with S the panel's
background below). The cohort's curve and its own background measurement already hold the
cohort's versions of both, so the panels' copies are counted twice inside the band.

```bash
summit admixed-ld clean-panels --ref ref/afr_eur --panel ref/afr --panel ref/eur --names AFR,EUR \
  --threads 16 --out ref/afr_eur_clean
```

`clean-panels` runs the curve on each panel. It removes the panel's admixture LD from its
covariances and variances when the panel's kernel at zero is at least 3 jackknife SEs above
zero, and it records the panel's own background LD (the cohort estimator of
`curve --background`, run on the panel without covariates; aggregates only) in the
manifest whether or not anything is removed. `--from-fits` applies both to a converted or
subset copy of the same panels' reference where the genotypes are not available.

## The cohort's curve

```bash
summit admixed-ld curve --geno cohort --ref ref/afr_eur --covar gwas.cov --out cohort
```

Run this where the genotypes are. It measures κ(d) as the weighted-LD curve of ROLLOFF
and ALDER on the genotypes adjusted for the GWAS covariates, in cells of `--cell-cm`
(default 0.25 cM), along with its between-chromosome value and the moments of global
ancestry. With `--relative-pairs pairs.tsv`, it also computes the relatedness of each pair
in the genotypes adjusted for the covariates and writes the sum of its square. The listed
pairs are used, selected with `--kinship-col` and `--kinship-min`. `--closure` adds every
pair within a family (connected component), which helps when the list holds only some
pairs of each family. Do not use it with a dense list of distant relatives. The relatedness after adjustment is what enters the in-sample LD scores,
and it can differ greatly from the relatedness in a pair file. In a cohort of a few
thousand people, a large family can take up a PC of its own, and first-degree pairs with
a GRM of 0.5 then have a relatedness near zero after the covariates.

Relatives share segments identical by descent, so their LD on one chromosome is far larger
than between chromosomes; for distant relatives, per SNP pair at a few cM, it is ten to
twenty times their g². The curve therefore also writes `cohort.relatedness.tsv`, the
listed pairs' products of local relatedness by distance on one chromosome (cells of
`--relatedness-cell-cm`, default 0.5 cM), and their product between chromosomes, both
net of what random unlisted pairs (`--relatedness-controls`) contribute to the same sums.
The controls remove the LD that two unrelated people share and the covariates' projection,
which the population part of the LD score already counts, and the products leave out each
SNP with itself, so the noise floor of g² (about the mean LD score over the number of SNPs
per pair) is gone too. `--no-relatedness-profile` restores the sum of g² alone.

Pair lists miss most of the relatedness that matters: distant and cryptic relatives, whose
LD per SNP pair is a property of the population rather than of the cohort's families.
`--background` measures the relatedness LD of all pairs of people without a list, from the
cross-person LD among the least differentiated SNPs (`--background-quantile`, default the
lowest fifth of the source contrast), less its admixture part and SUMMIT's null, by
distance and between chromosomes (`cohort.background.tsv`). Beyond a few cM this is what
relatives add; closer, the same products also hold within-source LD. The outputs,
`cohort.curve.tsv`, `cohort.curve.json` and these tables, hold aggregates only and can be
released with the summary statistics.

## Compose the LD scores

```bash
summit admixed-ld compose --ref ref/afr_eur --sumstats gwas/ --annot annot.tsv.gz \
  --curve cohort --out ld/cohort
```

`--sumstats` takes SUMMIT summary statistics (files or a directory) with `BETA`, `SE`
and `N`. Variants are matched by `CHR` and `BP`; if the summary statistics lack them,
give the GWAS variant table with `--snps` (BIM, PVAR, or a table with SNP CHR BP A1 A2).
`--annot` is the annotation of the fit, such as MAF/LD bins.

The decay is given in one of three ways.

| Option | Decay |
|---|---|
| `--curve PREFIX` | the cohort's measured curve; recommended |
| `--admixture-generations T` | a published admixture date; λ = T/100 per cM |
| `--decay-per-cm λ` | a decay rate per cM |

With a date or a rate, κ(d) is `far + (κ(0) - far) exp(-λ d)`, where
`far = -2 κ(0) / (λ G)` is the constant that adjustment for each person's realized
ancestry leaves and G is the genetic length of the genome. The SD of global ancestry
enters only the term of each person's pairing with themselves. It comes from the curve,
from `--ancestry-sd`, or defaults to 0.12. `--mean-ancestry` and `--kappa0` replace the
values from the standard errors. `--relatedness curve` adds the relatives' part of the
LD scores from the curve: B between chromosomes and φ(d) along each chromosome, inside the
reference window too. With a curve made with `--background`, the term is the measured LD
of all pairs of the cohort at every distance, inside the reference band too. It is smoothed
as B plus a sum of decaying exponentials fitted from `--background-min-cm` (default 3) on,
with rates up to one over that distance, the components that the fitting range resolves;
closer, the measurement also holds within-source LD, which the panels supply. Inside the
band the panels' own background LD, recorded by `clean-panels`, is removed with its
weight π_s(d)² and the panel's variances. This needs a reference from `clean-panels`
(compose stops otherwise). `--background-window zero` and `flat` keep the rules of 5 Nov
2026, which used the measurement only from `--background-min-cm` on, with the listed
pairs' profile below. `--relatedness uniform` adds the curve's sum of g² to every SNP pair,
as before 5 Nov 2026, and `--relatedness X` gives that sum directly.

The outputs are `ld/cohort.gw.ldscore.gz`, `ld/cohort.gw.M` and
`ld/cohort.admixed_ld.json`, which records every parameter used. Fit as usual:

```bash
summit --h2 gwas/ --ldscores ld/cohort.gw.ldscore.gz --annot annot.tsv.gz --out results/h2
```

By default the partner SNPs of each LD score are the GWAS SNPs themselves. When the causal
variants are denser than the GWAS SNPs (an array GWAS, or HapMap3 SNPs from imputed data),
LD scores over the GWAS SNPs miss the LD with untyped causal variants, and the estimates
are biased by an amount that differs between LD scores of different kinds. With
`--partners reference`, every reference variant listed in `--annot` is a partner and the
GWAS SNPs are the rows, as in LDSC's LD scores from 1000 Genomes. Build the reference on
the dense variant set (it must contain the GWAS SNPs), and list every partner in the
annotation. `.gw.M` then holds the partners' counts per class. The cohort's cell matrix
and sample factors are measured on the GWAS SNPs and cannot be combined with this option.

## Correlation of effects across ancestry tracks

In an admixed person, the alleles on African-ancestry segments and those on
European-ancestry segments can have different causal effects. Hou et al. (2023) called the
correlation of these per-allele effects r_admix and estimated it from individual genotypes.
`summit admixed-ld` estimates it from summary statistics of a local-ancestry-aware GWAS
such as Tractor, which regresses the phenotype at each SNP on the covariates, the
local-ancestry dosage and the allele counts on each ancestry's segments. Because the
local-ancestry dosage is in the model, the variable each track's effect is estimated on is
the allele's deviation within that track. This deviation carries no admixture LD and is
uncorrelated with the other track's alleles. So the track LD scores are short-range, and
the two tracks' estimates have zero noise covariance (at one SNP the two alleles lie on
different haplotypes).

The estimate needs sandwich (HC) standard errors. With ordinary least squares, the
residual variance is assumed equal for everyone, but the track variable's variance follows
each person's local ancestry, and so does the phenotype's variance when environment or
scale differ by ancestry. The own-track moments then hold terms that the LD scores do not
model, and these are not small (in simulations with variance that rises with ancestry,
r_admix came out at 1.4 for a truth of 0.6). Sandwich SEs remove them exactly. Tractor
itself writes only OLS SEs, so add HC1 SEs to the regression (a sum over people of
x_a x_b e², cheap to compute alongside the fit).

```bash
summit admixed-ld tractor-sumstats --tractor tractor/height.tsv --sources AFR,EUR \
  --covariates 13 --out tracks/height
summit admixed-ld compose --ref ref/afr_eur_dense --sumstats tracks/height.AFR.sumstats \
  --annot annot.tsv.gz --admixture-generations 6 --mean-ancestry 0.59,0.41 --ancestry-sd 0.29 \
  --tracks --partners reference --cross-source EUR --out ld/tracks
summit admixed-ld track-rg --tracks ld/tracks --sumstats tracks/height --annot annot.tsv.gz \
  --external eur/height.sumstats --out results/height
```

`tractor-sumstats` reads Tractor's table (`beta_anc0`, `se_anc0` or `tval_anc0`,
`AF_anc0`, `LAprop_anc0`, ... and the sandwich SEs `se_hc_anc0`, ...) and writes one
summary-statistic file per track, with the ALT allele counted. It stops when the sandwich
SEs are missing; `--se ols --allow-ols` proceeds with a recorded flag. Its
`tracks.json` lists the mean local-ancestry share of each track, which `compose` takes as
`--mean-ancestry`. Because a track's standard errors do not give the cohort's kappa(0),
give `--kappa0` or `--ancestry-sd` (from which kappa(0) is taken as 2 m (1 - m) - 4 SD²).

`compose --tracks` writes the own LD scores of each track (`ld/tracks.track_AFR`,
`.track_EUR`), their cross scores (`.track_AFR_EUR`) and, with `--cross-source EUR`, the
scores of each track against a GWAS of other people from that population
(`.track_AFR_vs_EUR`, `.track_EUR_vs_EUR`) with that population's own LD scores
(`.cohort_EUR`). They are on the per-allele scale: effects whose variance is proportional
to one over the cohort's genotype variance (SUMMIT's own scale), or to
[2 f (1 - f)]^α with `--track-alpha α` (Hou et al.'s model, α = -0.38).

`track-rg` fits each pair of tracks with SUMMIT's cross-population equations and the noise
covariance fixed at zero, the same fit as

```bash
summit --rg tracks/height.AFR.sumstats,tracks/height.EUR.sumstats \
  --ldscores ld/tracks.track_AFR_EUR.gw.ldscore.gz \
  --ldscores-h2-1 ld/tracks.track_AFR.gw.ldscore.gz --ldscores-h2-2 ld/tracks.track_EUR.gw.ldscore.gz \
  --overlap-covariance-rg 0 --annot annot.tsv.gz
```

and combines the classes of the annotation as
r = Σ_c τ_AE,c n_c / √(Σ_c τ_A,c n_c · Σ_c τ_E,c n_c), with τ the per-allele variances and
covariance in class c and n_c the number of partner variants in it. With the GWAS SNPs as
partners this is the rg of the `summit --rg` call above. It also reports r in each class,
over groups of classes (`--group low=maf1_ld1,maf1_ld2,...`), and each track's heritability.
With `--external`, it adds r of each track with the other population's GWAS. In the same
people, r_admix separates ancestry from everything else, and with the ancestry held fixed
r(EUR track, EUR cohort) separates the cohorts. If the two effects multiply,
r(AFR track, EUR cohort) is about r_admix · r(EUR track, EUR cohort), and the output
reports the difference with its jackknife SE.

Partition the fit by MAF and LD, as for h². The per-class variances then take up how the
effect variance depends on frequency between classes, but not within them. When the true
dependence differs from the weights, what is left is a few per cent with SUMMIT's eight
MAF/LD bins, whose MAF classes split only at 0.1, and under 1% with five MAF classes
(see the numbers below). Local-ancestry errors put alleles on the wrong track and push r
up, by about 2.5 times the share of miscalled positions in simulations, so interpret r
next to the caller's accuracy.

## Accuracy and limits

![The cohort's curve with the kernel from a 5-generation admixture date, per-SNP composed against in-sample LD scores, and h² of 32 traits from composed against in-sample LD scores in two admixed UK Biobank cohorts.](assets/admixed-ld-ukb.png)

(a) The curve measured in cohort 1, the kernel from an admixture date of 5 generations,
and the curve's between-chromosome value. (b) Composed LD scores, with the curve and the
relatives' term, against in-sample LD scores of the same SNPs in cohort 1, summed over
the eight classes. (c) h² of 32 traits fitted with composed and with in-sample LD scores,
from the same summary statistics.

We compared composed with in-sample LD scores in two cohorts of admixed UK Biobank
participants of African and European ancestry (3,787 and 6,111 people, mean African
ancestry 0.61 and 0.76), with 32 traits, 405,807 array SNPs and eight MAF/LD classes. The
references were panels of African-ancestry (4,905 and 2,452 people) and white British
(12,148) participants outside the cohorts. With the curve and the relatives' term, mean h²
over traits was 0.8% and 1.1% higher than with in-sample LD scores. The mean absolute
difference was 0.03 standard errors, and standard errors were within 1%. Without the
relatives' term the differences were 1.7% and 1.4%. With an admixture date of 5
generations (the decay that the curves measured) in place of the curve, and without the
relatives' term, they were 1.9% and 3.5%.

On 16 cores, the reference took about 18 minutes for 17,000 panel members. The curve took
about a minute for 3,787 people, and the composition about 90 seconds, most of it spent
reading the 32 summary-statistic files.

The model covers two-way admixture. It does not cover three or more sources, or panels
whose LD differs from that of the cohort's sources. The relatives' term covers the LD
scores only. Close relatives also share environment, which enters the summary statistics
themselves, so prune them before the GWAS or use [Related individuals](Related-individuals.md)
with in-sample LD scores. Resource size grows with the number of SNP pairs within the
window. For about 400,000 array SNPs and a 3,000 kb window it is about 4 GB for two
sources.
