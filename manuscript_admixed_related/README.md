# Heritability from summary statistics of admixed and related biobank participants

Source data and code for the paper by Moonseong Jeong, Sungjoo Han, Prateek Anand and Sriram Sankararaman.

`data/` holds the aggregate results behind every figure, and `scripts/` draws the figures from them. `analysis/` holds
the scripts that produced the results from UK Biobank genotypes and phenotypes, and the cells run in the All of Us
Researcher Workbench. The methods themselves are SUMMIT commands: `summit relatives` for related individuals and direct
genetic effects, and `summit local-ancestry` for local ancestry (see the
[Related individuals](../docs/wiki/Related-individuals.md) and [Local ancestry](../docs/wiki/Local-ancestry.md) pages).
No individual-level data are included.

## Reproduce the figures

From this folder, with Python 3, NumPy, pandas, SciPy and matplotlib:

```bash
python scripts/make_fig1.py
python scripts/make_figures.py
python scripts/make_fig34.py
python scripts/make_fig_ancestry_groups.py
```

The figures are written to `output/`, together with `output/numbers.txt`, which lists the plotted summaries of Figs. 2,
5 and 6. Use `--out DIR` to write them elsewhere. Drawing all figures takes about a minute on one CPU. The All of Us
results (Fig. 3B-D, Table 5) are aggregate statistics from the All of Us Researcher Workbench and are not included in
this folder; `make_fig34.py --aou FILE` draws Fig. 3B-D from that table.

| Figure | Script | Data |
|---|---|---|
| 1 | `make_fig1.py` | none (schematic) |
| 2A-C | `make_figures.py` | `simulations.tsv.gz`, `simulations_reml.tsv.gz`, `simulations_covldsc.tsv.gz` |
| 2D | `make_figures.py` | `local_ancestry_simulations.tsv.gz` |
| 2E | `make_figures.py` | `wider_group_simulations.tsv.gz` |
| 3A | `make_fig34.py` | `white_british_traits.tsv` |
| 3B-D | `make_fig34.py` | All of Us aggregate table (not included) |
| 4A | `make_fig34.py` | `cohort_tests.tsv` |
| 4B, 4D | `make_fig34.py` | `genetic_correlation_cohorts.tsv.gz`, `genetic_correlation_reference.tsv` |
| 4C | `make_fig34.py` | `cohort_traits.tsv` |
| 4E | `make_fig34.py` | `local_ancestry_traits.tsv` |
| 5 | `make_figures.py` | `genetic_correlation_cohorts.tsv.gz` |
| 6 | `make_figures.py` | `direct_effects_simulations.tsv.gz`, `direct_effects_traits*.tsv` |
| 7 | `make_fig_ancestry_groups.py` | `ancestry_groups_traits.tsv`, `ancestry_groups_simulations.tsv.gz` |

## Data

Cohort codes: EUR, SAS and AFR are the white British, South Asian and admixed African-European groups with their
relatives (14,558, 8,566 and 4,324 participants); AFRW is the wider group of all 9,647 participants of African ancestry,
AFRWA its 4,202 admixed members, and AFRWX the wider group with indicators of unadmixed ancestry and the squared African
proportion among the covariates. Analysis arms: U, the unrelated subset; K, everyone with related pairs kept in the
estimating equations; XA, everyone with related pairs excluded. Models: `single` (one component) and `8bin` (8 MAF-LD
bins). Heritability of binary traits in `cohort_traits.tsv` and `white_british_traits.tsv` is on the liability scale
where a column says so (`h2L_*`, `liab_mult`).

| File | Contents |
|---|---|
| `simulations.tsv.gz` | Estimates of each replicate of the simulation grid (cohort, setting, analysis arm, model; h² or rg with SE and truth) |
| `simulations_reml.tsv.gz`, `simulations_covldsc.tsv.gz` | Two-kernel REML and cov-LDSC (genome-wide LD scores) on the same replicates |
| `local_ancestry_simulations.tsv.gz` | SNP-only and joint SNP and local-ancestry fits of the local-ancestry settings in the admixed group |
| `wider_group_simulations.tsv.gz` | Replicates in the wider group and its admixed members (`task` = cohort and setting) |
| `ancestry_environment_simulations.tsv.gz` | Environments that depend nonlinearly on global ancestry, with and without the ancestry covariates |
| `white_british_traits.tsv` | 40 traits in 406,657 white British participants with relatives against the 337,845 unrelated participants |
| `cohort_traits.tsv` | 40 traits in each cohort and arm, with origin covariates (`set = origin`) and without (`base`) |
| `cohort_tests.tsv` | Differences of each cohort's heritability from that of the 337,845 unrelated white British participants (`ref = EUR338k`), fitted with the same covariate set, with Bonferroni and FDR over the 40 traits of each cohort (`bonf_pooled`, Bonferroni over the cohorts of a covariate set) |
| `local_ancestry_traits.tsv` | Local-ancestry and SNP components of the 40 traits, with the test against the null variance and sandwich SEs |
| `direct_effects_simulations.tsv.gz` | Direct and population estimators on the siblings' genotypes, by setting |
| `direct_effects_traits*.tsv` | Direct-effect and population heritability of the traits in 35,365 white British siblings |
| `genetic_correlation_cohorts.tsv.gz` | Genetic correlation of the 780 trait pairs in each cohort and arm |
| `genetic_correlation_reference.tsv` | The same pairs in the 337,845 unrelated white British participants |
| `ancestry_groups_traits.tsv` | European group and wider group in one analysis: heritability of each group, genetic covariance and correlation between them, local-ancestry component and one-kernel heritability (origin covariates) |
| `ancestry_groups_simulations.tsv.gz` | Simulations of the analysis of the two groups, one row per replicate (`scenario` gives the setting) |

## Analysis scripts

`analysis/README.md` maps each script to the figures and tables, lists the order of the steps and their resources, and
notes where the scripts differ from the runs in the paper. The scripts need UK Biobank data under an approved
application and take every input location as an argument. The All of Us cells in `analysis/aou/` run inside the
Researcher Workbench and export aggregate statistics only.

This research was conducted using the UK Biobank Resource under application 33127.
