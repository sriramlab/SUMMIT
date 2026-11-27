# Commands and options

Use `summit --help` for analysis options, or add `--help` to a command such
as `summit pgs fit`.

## Commands

| Task | Command |
|---|---|
| Genome-wide or windowed LD scores | `summit --geno ...` |
| Heritability or genetic correlation | `summit --h2 ...` or `summit --rg ...` |
| Related-pair moments for h²/rg with relatives | `summit relatives {moments,append,cross-m12}` |
| Long-range LD correction of h²/rg standard errors | `summit relatives longrange`, then `--long-range-se` |
| Heritability due to direct genetic effects from sibships | `summit relatives direct` |
| Local-ancestry markers, merged with the SNPs, and the test of their component | `summit local-ancestry {markers,merge,test}` |
| LD scores, regional matrices and track correlations of an admixed cohort without its genotypes (SUMMIT-MIX) | `summit admixed-ld {reference,clean-panels,release,curve,bins,compose,sample-factors,derive,matrix,finemap-variance,tractor-gwas,tractor-sumstats,track-rg}` |
| Additive and generalized G×E PCGC | `summit --binary-method pcgc ...` |
| Generalized cross-trait PCGC | `summit pcgc-cross {prepare,fit}` |
| One-environment G×E | `summit --geno ... --env ...`, `--gxe-score-reference`, `--gxe-fit`, or `--gxe-fit-batch` |
| Polygenic scores | `summit pgs {plan,fit,score,scale,inspect}` |
| Generalized G×E reference planning and inspection | `summit reference {plan,inspect}` |
| Cross-trait reference Z moments | `summit reference zpass` |
| Quantitative epistasis | `summit epistasis {make-inputs,train-direction,prepare,prepare-traits,fit,combine,followup,scale-free}` |

Quantitative generalized G×E fitting and cross-trait response models use
Python APIs. See [Multiple environments](Multiple-environments.md) and
[Cross-trait analysis](Cross-trait-analysis.md) for their workflows.

PCGC preparation (`--make-binary-sumstats`) writes `.binary.sumstats.npz` and
`.binary.ldscores.npz` files. Fit with `--h2` and `--ldscores`; SUMMIT checks that
they match. `--binary-output-format combined` retains the older single-file
output, which is fitted through `--h2` alone. [Input files](Input-files.md#files-used-for-inference)
compares the formats used by each workflow.

[Generalized cross-trait PCGC](Generalized-cross-trait-PCGC.md) writes matched
`.cross.ldscores.npz` and `.cross.sumstats.npz` files. Its selection assumptions
and experimental joint uncertainty are described in that guide.

## SUMMIT-MIX (`summit admixed-ld`)

LD scores of an admixed GWAS sample composed from source panels, the GWAS summary statistics and either the cohort's
release (curve mode, the default) or a published admixture date (date mode). See
[Admixed cohorts without in-sample LD](Admixed-cohorts-without-in-sample-LD.md) for the two quick starts, and
[SUMMIT-MIX details](SUMMIT-MIX-details.md) for every option.

| Command | Needs | Writes |
|---|---|---|
| `reference` | unadmixed panels of the sources, a genetic map | the reference (once per set of panels) |
| `clean-panels` | the reference and its panels | a reference without the panels' own admixture LD, with their background |
| `release` | the cohort's genotypes, its GWAS people and covariates, related pairs | one folder of aggregates: the curve with its background term, cell matrix and sample statistic, the MAF/LD bins, and a list of what to share |
| `curve` | as `release` | the curve files of the release alone |
| `bins` | the cohort's genotypes, or summary statistics with allele frequencies, or the reference and a mean ancestry | the eight MAF/LD bins, the 24 unpooled bins and the two MAF classes (other cuts with `--maf-cuts`, `--ld-quantiles`, `--maf-class-cut`), or the two MAF classes alone |
| `compose` | reference, summary statistics, annotation, `--release` or `--admixture-generations` | `.gw.ldscore.gz`, `.gw.M`, `.admixed_ld.json` |
| `sample-factors` | the genotypes and covariates of one GWAS sample (for example a trait measured in part of the cohort) | its sample statistic alone (aggregates only) |
| `derive` | an admixed cohort that carries a source without a panel, and anchor members | a reference with that source |
| `matrix` | reference, compose's `.admixed_ld.json`, a region, optionally the GWAS | a regional LD matrix, the aligned z-scores and the predicted error of its entries |
| `finemap-variance` | a matrix with `--error-variance` and a first SuSiE fit's posterior | the residual variance of the second fit |
| `tractor-gwas` | phased genotypes, local-ancestry calls, phenotypes and covariates of the GWAS sample | a Tractor-style GWAS with the covariates-only sandwich SEs |
| `tractor-sumstats` | a Tractor-style GWAS with sandwich SEs | one summary-statistic file per local-ancestry track |
| `track-rg` | `compose --tracks` LD scores and the track summary statistics | the correlation of effects across tracks |

`release`, `curve`, `bins --geno`, `sample-factors`, `derive` and `tractor-gwas` read the cohort's individual-level
data and run where the cohort's data are; they write aggregates or summary statistics only. `reference` and
`clean-panels` read the panels' genotypes, and the other steps need no individual-level data.

`reference`, `release`, `curve`, `compose`, `sample-factors` and `derive` take `--drop-ambiguous`, which leaves out A/T
and C/G SNPs; give it to each of them when the data sets may be on different strands. `matrix --sumstats` leaves these
SNPs out of the z-scores unless `--keep-ambiguous`, and `clean-panels` uses the reference's SNPs. `release`, `curve`,
`compose`, `sample-factors` and `matrix` take `--sources` (the last source is the baseline); `reference` and
`clean-panels` name the panels with `--names`, and `derive` the kept sources with `--known`. Each step logs its
matching counts, and the steps that read genotypes take `--threads`.

## Shared controls

| Option | Meaning |
|---|---|
| `--geno` | BED or PGEN input, with companion files |
| `--annot` | SNP annotations |
| `--out` | Output prefix or directory, as specified by the command |
| `--nvecs` | Number of random vectors for reference estimation |
| `--seed` | Random seed |
| `--block-size` | Number of genotype variants processed together |
| `--num-threads` | Compute thread count |
| `--memory-gib` | Working-memory budget in GiB |
| `--njack` | Jackknife block count or scheme |
| `--snp-var-weights-off` | LD scores without the `V` column; h²/rg refuse `V` files |
| `--relative-pairs` | Add `LD_REL_<column>` for related pairs in the LD sample |

Use complete option names. `--block-size` controls computation; `--njack`
controls uncertainty estimation. Windowed LD scores do not use random vectors.
`--block-size auto` is available for one-environment G×E reference generation.
Its resolved width is saved with the reference because it affects the
finite-vector estimate.

| Workflow | Random vectors | Seed | Block size | Memory budget |
|---|---:|---|---:|---|
| LD/G×E preparation | 1,000 | Unspecified | 1,000 | `auto` |
| Binary preparation | 256 | 0 | 256 | 1 GiB |
| PGS | — | — | 512 | 16 GiB |
| Generalized reference plan | Required | — | 4,096 | Required |
| Reference Z moments | — | — | 128 | — |

h², rg and binary inference use 200 contiguous SNP blocks by default;
`--njack chr` deletes one chromosome at a time instead. One-environment G×E
fits default to chromosome deletion. In admixed and other structured samples,
`--long-range-se` corrects h² and rg jackknife SEs for covariance between
blocks ([Related individuals](Related-individuals.md#long-range-ld-correction-of-standard-errors)).
Input formats and method-specific settings are described in the individual
analysis guides.

For binary G×E PCGC, `--binary-crossblock-blocks N` at preparation (opt-in,
for large N/M) records the participant rows that correct the SNP-block
jackknife for covariance between blocks through the same participants. Fit
with the same block count: omit `--njack` to use the prepared blocks. See
[Generalized G×E PCGC](Generalized-GxE-PCGC.md#standard-errors).

## Memory

For LD/G×E, `--memory-gib` budgets the sketch panels or windowed-LD workspace.
For binary preparation it budgets reference workspace; for PGS it supplies
the fit or scoring memory plan. It is not a limit on process resident memory.
Allow additional memory for inputs, libraries, and output arrays.

G×E also provides `--gxe-native-workspace-gib` for each direct native call
(default 16 GiB), and `--gxe-total-memory-gib` for the estimated process peak
(default `auto`). `--win-cache-mb` controls the windowed-LD cache in MiB.
These limits cover different allocations.

## Genome build

Provide build-matched genotypes, annotations, and reference files. SUMMIT does
not convert coordinates or infer the genome build.

`--genome-build` optionally records a label during binary or PGS scale
preparation. The PGS specification accepts the equivalent `genome_build`
field. Labeled saved scales must be reused with the same label. PGS scoring
rejects conflicting labels when both are supplied and always checks SNP
positions and allele pairs, including when labels are omitted.
