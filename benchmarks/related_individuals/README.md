# Related-individual benchmark

Scripts for the simulations on the [Related individuals](../../docs/wiki/Related-individuals.md#simulations)
wiki page. They need UK Biobank genotypes, so they record the analysis rather
than run as a self-contained example. Paths are relative to the working
directory used for the runs.

| Step | Script |
|---|---|
| Simulate phenotypes from real genotypes (scenarios A–E) | `sim_bench.py` (with `bedio.py`) |
| Covariates, unrelated subset, phenotype and pair files | `setup_cohort.py` |
| In-sample LD scores with V (and LD_REL for everyone) | `run_ld.sh`, `run_ld_unrel.sh` |
| PLINK 2 GWAS, `summit relatives moments`, sumstats | `gwas_and_moments.sh`, `convert_sumstats.py` |
| h² fits for each analysis and jackknife | `fit_arms.sh` |
| Genetic correlation (bivariate phenotypes, REL_M12) | `sim_bivariate_final.py`, `rg_stage.sh`, `convert_rg.py` |
| Summary tables and wiki figures | `summarize_final.py` |

`results/` holds the summary tables behind the figures.
