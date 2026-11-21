# Admixed and related participants

Scripts for the analyses of "Heritability from summary statistics of admixed and related biobank participants". They need
UK Biobank data under an approved application, so they record the analyses rather than run as a self-contained example.
Every input location is a command-line argument; each script's docstring lists its inputs and outputs. The methods are SUMMIT
commands: `summit relatives moments/append/cross-m12/longrange`, the LD-score step with `--relative-pairs`,
`summit relatives direct`, and `summit local-ancestry markers/merge/test` (see the
[Related individuals](../../docs/wiki/Related-individuals.md) and [Local ancestry](../../docs/wiki/Local-ancestry.md) wiki pages).

| Analysis | Script | Paper |
|---|---|---|
| Cohort inputs: genotypes, PCs, related pairs, families, unrelated subset, covariates, 8 MAF-LD bins, in-sample LD scores | `prepare_cohort.py` | all |
| UK Biobank traits (40), age and sex covariates, origin covariates | `traits.py` | all real-trait results |
| Simulation settings | `designs.py` | Fig. 2 |
| One simulation task (phenotypes, GWAS, related-pair parts, fits of U, K and XA) | `simulate.py` | Fig. 2A-C, E; Tables 2-3 |
| Two-kernel REML and cov-LDSC for a task | `compare_methods.py` | Fig. 2A-C; Tables 2-3; Appendix E |
| Summaries of the simulations | `summarize.py` | Fig. 2; Tables 2-3 |
| Real traits in the cohorts (U, K, XA; long-range SE correction) | `real_traits.py` | Fig. 4C; Results |
| Heritability against a reference cohort | `compare_groups.py` | Fig. 4A |
| Local-ancestry markers, simulations, real traits and the local-ancestry test | `local_ancestry.py` | Fig. 2D, 4E; Appendix B |
| White British participants with relatives against the unrelated subset; cov-LDSC | `white_british.py` | Fig. 3A; Appendix E |
| Sibships, direct-effect simulations and real traits | `direct_effects.py` | Fig. 6; Appendix F |
| Genetic correlation in the cohorts and agreement with a reference | `genetic_correlation.py` | Table 4; Fig. 4B, D and Fig. 5 |
| Factors of the long-range LD correction | `standard_errors.py` | Table 6 (LD-score columns) |
| European and African-ancestry participants in one analysis | `ancestry_groups.py` | Fig. 7; Appendix G |
| All of Us | `aou/` | Results, All of Us |

`common.py` holds the helpers the scripts share (GWAS to SUMMIT summary statistics, transforms).

## Order of the steps

1. Cohorts, with `prepare_cohort.py` on the quality-controlled genotypes: EUR, SAS and AFR (the three simulation cohorts,
   participants with their relatives), AFRW (all participants of African ancestry who pass quality control), AFRWA (its admixed
   members) and AFRWX (AFRW with `--ancestry`, which adds an indicator of unadmixed participants and the squared African
   proportion to the covariates). Folder names are used as cohort labels by the summaries and figures.
2. Traits with `traits.py`, for each cohort's samples named `all` (`full.fam`) and `unrel` (`unrel.fam`), with `--origin` for
   the origin covariates; for the white British analysis, samples named `rel` and `unrel`.
3. Simulations: `simulate.py` for every setting of `designs.py` in EUR, SAS and AFR, the main grid and MAF-LD settings in AFRW
   and AFRWA (Fig. 2E lists them in `designs.WIDER_GROUP`), and the ancestry settings in AFRW and AFRWX (with `--ancestry`);
   genetic-correlation settings with `--long-range`. Then `compare_methods.py --ld genome` (and `--ld window` for Appendix E)
   on the grid and MAF-LD tasks, and `summarize.py`.
4. Real traits with `real_traits.py` (and `--origin`), `compare_groups.py` against the unrelated white British reference, and
   `genetic_correlation.py`.
5. Local ancestry: `local_ancestry.py build` for the admixed group (AFR), the wider group (AFRW, with `--fill`) and its admixed
   members (AFRWA), then `simulate` (settings null, model, untagged, untagged2 in AFR) and `real`.
6. `white_british.py sample`, `genotypes` and `fit`; `direct_effects.py sibships`, `simulate` and `real` on the white
   British siblings; `standard_errors.py` on the `summit relatives longrange` outputs.
7. `ancestry_groups.py` for the European group and the wider group in one analysis (its docstring gives the options of
   the paper's runs).
8. The figures are drawn from the aggregate results by the scripts in `../scripts/`.

## Differences from the runs in the paper

The scripts reproduce the paper's computations with these exceptions.

- Simulation replicates use different random seeds, so individual replicates differ from the paper's draws.
- Real-trait standard errors use `--long-range-se`. The paper's trait tables used the exact rescaling factors, which differ by at
  most 2%. Table 6's exact columns need block-pair traces and are not reproduced; its LD-score columns are.
- The white British fits compute LD scores and bins in the unrelated subset unless `--ld1`, `--ld8` and `--annot8` supply the
  panel LD scores used in the paper.
- Local-ancestry simulations add the global ancestry THETA to the covariates unless built and run with `--without-theta`, as
  in Fig. 2D. Untagged variants are drawn from the local ancestry at the nearest marker of a 600-kb grid, not at the
  variant's own position.
- Origin covariates drop rare and collinear categories with frequency and rank filters, not the per-trait variance-inflation
  screen of the paper's runs.
- The white British reference of Fig. 4A, B and Table 4 is the unrelated subset of `white_british.py` (337,845
  participants), fitted like the cohorts (8 MAF-LD bins, 200 blocks, no `--max-chisq`) on the panel LD scores given to
  `white_british.py` (`--ld8`, `--annot8`), with the origin covariates of `traits.py` added for the comparisons with
  origin covariates, and without the long-range correction of its standard errors. cov-LDSC is an external dependency.

## Resources

For the 14,558-participant cohort, the GWAS of a simulation task (100 replicates) took about 8 minutes on 16 CPUs, and
the two-kernel REML of `compare_methods.py` about an hour (about 35 s per fit, 8.4 GB of memory); REML costs about N^3
operations per iteration and a few N x N matrices of memory. The
direct-effect estimator stores eight N x N class GRMs of the siblings (about 40 GB for 35,365 siblings). The white British
GWAS of 406,657 participants and their related-pair pass take several hours each on 16 CPUs.

## All of Us

`aou/` holds the cells run in the All of Us Researcher Workbench: `relatives_cells.py` with `relatives_helpers.py`
(stage 1 simulations on the unrelated panel; stage 3 real traits for the unrelated panel, everyone with related pairs kept
and everyone with related pairs excluded; the unrelated panel with its own related pairs excluded and paired jackknife SEs),
and `direct_cells.py` with `direct_helpers.py` and `sib_direct.py` (the direct-effect estimator as run there, which
`summit relatives direct` implements). They start from a foundation folder of quality-controlled group genotypes, unrelated
panels, PCs and participant data (described in their first cell), and export aggregate statistics only: counts of 1 to 20
are masked, and breakdowns whose totals would reveal masked counts are left out.
