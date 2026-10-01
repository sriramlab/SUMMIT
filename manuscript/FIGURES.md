# Figure scripts and inputs

Run `python reproduce.py --figure ID`. Paths below are relative to this folder.
Generated figures and companion result tables are written under `output/` by default.

| Figure | Output | Script |
|---|---|---|
| 1 | `figs/main/fig01_ld_and_h2_benchmarks.png` | [scripts/figure1/render.py](scripts/figure1/render.py) |
| 2 | `figs/main/fig02_enrichment_benchmarks.png` | [scripts/figure2/render.py](scripts/figure2/render.py) |
| 3 | `figs/main/fig03_genetic_correlation_benchmarks.png` | [scripts/figure3/render.py](scripts/figure3/render.py) |
| 4 | `figs/main/fig04_real_trait_heritability.png` | [scripts/real_h2/figure4.py](scripts/real_h2/figure4.py) |
| 5 | `figs/main/fig05_baseline_enrichment.png` | [scripts/baseline/render.py](scripts/baseline/render.py) |
| 6 | `figs/main/fig06_genetic_correlation_heatmaps.png` | [scripts/real_rg/render_heatmap.py](scripts/real_rg/render_heatmap.py) |
| 7 | `figs/main/fig07_genetic_correlation_agreement.png` | [scripts/real_rg/figure7.py](scripts/real_rg/figure7.py) |
| 8 | `figs/main/fig08_external_enrichment.png` | [scripts/external/figure8.py](scripts/external/figure8.py) |
| S1 | `figs/supplementary/fig_s01_snp_sets.png` | [scripts/qc/snp_overlap.py](scripts/qc/snp_overlap.py) |
| S2 | `figs/supplementary/fig_s02_ld_scores.pdf` | [scripts/ld/population_ld.py](scripts/ld/population_ld.py) |
| S3 | `figs/supplementary/fig_s03_pc_sensitivity_ld.png` | [scripts/ld/pc_ld.py](scripts/ld/pc_ld.py) |
| S4 | `figs/supplementary/fig_s04_pc_sensitivity_h2.png` | [scripts/ld/pc_h2.py](scripts/ld/pc_h2.py) |
| S5 | `figs/supplementary/fig_s05_h2_window_sensitivity.pdf` | [scripts/sim_h2/total_h2.py](scripts/sim_h2/total_h2.py) |
| S6 | `figs/supplementary/fig_s06_h2_simulations.pdf` | [scripts/sim_h2/total_h2.py](scripts/sim_h2/total_h2.py) |
| S7 | `figs/supplementary/fig_s07_h2_simulations_eur300k.pdf` | [scripts/sim_h2/eur300k.py](scripts/sim_h2/eur300k.py) |
| S8 | `figs/supplementary/fig_s08_binary_h2_simulations.pdf` | [scripts/sim_h2/binary_h2.py](scripts/sim_h2/binary_h2.py) |
| S9 | `figs/supplementary/fig_s09_pc_adjustment.pdf` | [scripts/sim_h2/no_pc.py](scripts/sim_h2/no_pc.py) |
| S10 | `figs/supplementary/fig_s10_pc_stratified_simulations.pdf` | [scripts/sim_h2/pc_stratified.py](scripts/sim_h2/pc_stratified.py) |
| S11 | `figs/supplementary/fig_s11_missing_snp_sensitivity.png` | [scripts/sim_h2/missing_snps.py](scripts/sim_h2/missing_snps.py) |
| S12 | `figs/supplementary/fig_s12_mafld_h2_mse.pdf` | [scripts/sim_h2/render_mafld.py](scripts/sim_h2/render_mafld.py) |
| S13 | `figs/supplementary/fig_s13_mafld_h2_error.pdf` | [scripts/sim_h2/render_mafld.py](scripts/sim_h2/render_mafld.py) |
| S14 | `figs/supplementary/fig_s14_coding_power.pdf` | [scripts/sim_h2/render_coding_curves.py](scripts/sim_h2/render_coding_curves.py) |
| S15 | `figs/supplementary/fig_s15_coding_calibration.png` | [scripts/sim_h2/coding_calibration.py](scripts/sim_h2/coding_calibration.py) |
| S16 | `figs/supplementary/fig_s16_partitioned_rg_error.pdf` | [scripts/sim_rg/render_partitioned.py](scripts/sim_rg/render_partitioned.py) |
| S17 | `figs/supplementary/fig_s17_total_rg_error.pdf` | [scripts/sim_rg/render_partitioned.py](scripts/sim_rg/render_partitioned.py) |
| S18 | `figs/supplementary/fig_s18_rg_architecture_differences.png` | [scripts/real_rg/gradient.py](scripts/real_rg/gradient.py) |
| S19 | `figs/supplementary/fig_s19_sumcore_overlap_modes.png` | [scripts/sim_rg/overlap_modes.py](scripts/sim_rg/overlap_modes.py) |
| S20 | `figs/supplementary/fig_s20_ld_projection_error.png` | [scripts/ld/numvec_error.py](scripts/ld/numvec_error.py) |
| S21 | `figs/supplementary/fig_s21_ld_projection_distributions.png` | [scripts/ld/numvec_histogram.py](scripts/ld/numvec_histogram.py) |
| S22 | `figs/supplementary/fig_s22_h2_projection_sensitivity.png` | [scripts/ld/numvec_h2.py](scripts/ld/numvec_h2.py) |
| S23 | `figs/supplementary/fig_s23_runtime_scaling.png` | [scripts/ld/runtime_scaling.py](scripts/ld/runtime_scaling.py) |
| S24 | `figs/supplementary/fig_s24_real_h2_24bins.pdf` | [scripts/real_h2/total_h2.py](scripts/real_h2/total_h2.py) |
| S25 | `figs/supplementary/fig_s25_real_h2_8bins.pdf` | [scripts/real_h2/total_h2.py](scripts/real_h2/total_h2.py) |
| S26 | `figs/supplementary/fig_s26_real_h2_annotation_comparison.pdf` | [scripts/real_h2/annotation_comparison.py](scripts/real_h2/annotation_comparison.py) |
| S27 | `figs/supplementary/fig_s27_heritability_population_differences.pdf` | [scripts/real_h2/population_differences.py](scripts/real_h2/population_differences.py) |
| S28 | `figs/supplementary/fig_s28_baseline_enrichment_portability.png` | [scripts/baseline/render.py](scripts/baseline/render.py) |
| S29 | `figs/supplementary/fig_s29_baseline_tau.png` | [scripts/baseline/tau_all_pops.py](scripts/baseline/tau_all_pops.py) |
| S30 | `figs/supplementary/fig_s30_baseline_gazal_comparison.png` | [scripts/external/gazal.py](scripts/external/gazal.py) |
| S31 | `figs/supplementary/fig_s31_array_imputed_h2.png` | [scripts/array_imputed/render.py](scripts/array_imputed/render.py) |
| S32 | `figs/supplementary/fig_s32_array_imputed_enrichment.png` | [scripts/array_imputed/render.py](scripts/array_imputed/render.py) |
| S33 | `figs/supplementary/fig_s33_rg_heatmap_eur300k.png` | [scripts/real_rg/render_heatmap.py](scripts/real_rg/render_heatmap.py) |
| S34 | `figs/supplementary/fig_s34_rg_heatmap_eur.png` | [scripts/real_rg/render_heatmap.py](scripts/real_rg/render_heatmap.py) |
| S35 | `figs/supplementary/fig_s35_rg_heatmap_sas.png` | [scripts/real_rg/render_heatmap.py](scripts/real_rg/render_heatmap.py) |
| S36 | `figs/supplementary/fig_s36_rg_heatmap_afr.png` | [scripts/real_rg/render_heatmap.py](scripts/real_rg/render_heatmap.py) |
| S37 | `figs/supplementary/fig_s37_rg_bonferroni_overlap.png` | [scripts/real_rg/render_venn.py](scripts/real_rg/render_venn.py) |
| S38 | `figs/supplementary/fig_s38_rg_fdr_overlap.png` | [scripts/real_rg/render_venn.py](scripts/real_rg/render_venn.py) |
| S39 | `figs/supplementary/fig_s39_rg_annotation_comparison_eur300k.png` | [scripts/real_rg/annotation_comparison.py](scripts/real_rg/annotation_comparison.py) |
| S40 | `figs/supplementary/fig_s40_rg_annotation_comparison_eur.png` | [scripts/real_rg/annotation_comparison.py](scripts/real_rg/annotation_comparison.py) |
| S41 | `figs/supplementary/fig_s41_rg_annotation_comparison_sas.png` | [scripts/real_rg/annotation_comparison.py](scripts/real_rg/annotation_comparison.py) |
| S42 | `figs/supplementary/fig_s42_rg_annotation_comparison_afr.png` | [scripts/real_rg/annotation_comparison.py](scripts/real_rg/annotation_comparison.py) |
| S43 | `figs/supplementary/fig_s43_external_heritability.png` | [scripts/external/total_h2.py](scripts/external/total_h2.py) |
| S44 | `figs/supplementary/fig_s44_external_reference_mismatch.png` | [scripts/external/reference_diagnostics.py](scripts/external/reference_diagnostics.py) |

## Figure 1

Inputs:

- [data/sim_h2/figure1/panel_a_source.tsv](data/sim_h2/figure1/panel_a_source.tsv)
- [data/sim_h2/figure1/panel_b_source.tsv](data/sim_h2/figure1/panel_b_source.tsv)
- [data/sim_h2/figure1/panel_c_relmse_summary.tsv](data/sim_h2/figure1/panel_c_relmse_summary.tsv)
- [data/sim_h2/figure1/panel_d_bias_source.tsv](data/sim_h2/figure1/panel_d_bias_source.tsv)

## Figure 2

Inputs:

- [data/sim_h2/figure2/calibration.tsv](data/sim_h2/figure2/calibration.tsv)
- [data/sim_h2/figure2/overlap_curves.tsv](data/sim_h2/figure2/overlap_curves.tsv)
- [data/sim_h2/figure2/discrimination.tsv](data/sim_h2/figure2/discrimination.tsv)

## Figure 3

Inputs:

- [data/sim_rg/figure3/calibration.tsv](data/sim_rg/figure3/calibration.tsv)
- [data/sim_rg/figure3/normalization.tsv](data/sim_rg/figure3/normalization.tsv)
- [data/sim_rg/figure3/panelB.tsv](data/sim_rg/figure3/panelB.tsv)
- [data/sim_rg/figure3/panelC.tsv](data/sim_rg/figure3/panelC.tsv)

## Figure 4

Inputs:

- [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv)
- [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv)
- [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv)
- [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)
- [data/traits/traits.txt](data/traits/traits.txt)
- [data/traits/wgs_heritability.csv](data/traits/wgs_heritability.csv)

## Figure 5

Inputs:

- [data/baseline/figure5/panelA_pairwise_portability.csv](data/baseline/figure5/panelA_pairwise_portability.csv)
- [data/baseline/figure5/ratio_meta.csv](data/baseline/figure5/ratio_meta.csv)

## Figure 6

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)

## Figure 7

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure 8

Inputs:

- [data/external/ld_components.tsv](data/external/ld_components.tsv)
- [data/external/reference_swap.tsv](data/external/reference_swap.tsv)
- [data/external/enrichment_values.csv](data/external/enrichment_values.csv)
- [data/external/enrichment_jackknife.tsv](data/external/enrichment_jackknife.tsv)

## Figure S1

Inputs:

- [data/qc/snp_overlap.json](data/qc/snp_overlap.json)

## Figure S2

Inputs:

- [data/ld/summary_nocov.csv](data/ld/summary_nocov.csv)
- [data/ld/summary_pc.csv](data/ld/summary_pc.csv)

## Figure S3

Inputs:

- [data/ld/pc_ld_summary.csv](data/ld/pc_ld_summary.csv)

## Figure S4

Inputs:

- [data/ld/pc_sensitivity_h2.csv](data/ld/pc_sensitivity_h2.csv)

## Figure S5

Inputs:

- [data/sim_h2/total/EUR/estimates.csv](data/sim_h2/total/EUR/estimates.csv)
- [data/sim_h2/total/SAS/estimates.csv](data/sim_h2/total/SAS/estimates.csv)
- [data/sim_h2/total/AFR/estimates.csv](data/sim_h2/total/AFR/estimates.csv)

## Figure S6

Inputs:

- [data/sim_h2/total/EUR/estimates.csv](data/sim_h2/total/EUR/estimates.csv)
- [data/sim_h2/total/SAS/estimates.csv](data/sim_h2/total/SAS/estimates.csv)
- [data/sim_h2/total/AFR/estimates.csv](data/sim_h2/total/AFR/estimates.csv)

## Figure S7

Inputs:

- [data/sim_h2/eur300k_h2.tsv](data/sim_h2/eur300k_h2.tsv)

## Figure S8

Inputs:

- [data/sim_h2/binary/EUR/estimates.csv](data/sim_h2/binary/EUR/estimates.csv)
- [data/sim_h2/binary/SAS/estimates.csv](data/sim_h2/binary/SAS/estimates.csv)
- [data/sim_h2/binary/AFR/estimates.csv](data/sim_h2/binary/AFR/estimates.csv)

## Figure S9

Inputs:

- [data/sim_h2/no_pc_comparison.tsv](data/sim_h2/no_pc_comparison.tsv)

## Figure S10

Inputs:

- [data/sim_h2/pc_stratified.tsv](data/sim_h2/pc_stratified.tsv)

## Figure S11

Inputs:

- [data/sim_h2/missing_snp_summary.csv](data/sim_h2/missing_snp_summary.csv)

## Figure S12

Inputs:

- [data/sim_h2/mafld/EUR/estimates.csv](data/sim_h2/mafld/EUR/estimates.csv)
- [data/sim_h2/mafld/SAS/estimates.csv](data/sim_h2/mafld/SAS/estimates.csv)
- [data/sim_h2/mafld/AFR/estimates.csv](data/sim_h2/mafld/AFR/estimates.csv)

## Figure S13

Inputs:

- [data/sim_h2/mafld/EUR/estimates.csv](data/sim_h2/mafld/EUR/estimates.csv)
- [data/sim_h2/mafld/SAS/estimates.csv](data/sim_h2/mafld/SAS/estimates.csv)
- [data/sim_h2/mafld/AFR/estimates.csv](data/sim_h2/mafld/AFR/estimates.csv)

## Figure S14

Inputs:

- [data/sim_h2/coding/EUR/estimates.csv](data/sim_h2/coding/EUR/estimates.csv)
- [data/sim_h2/coding_null/EUR/estimates.csv](data/sim_h2/coding_null/EUR/estimates.csv)
- [data/sim_h2/coding/SAS/estimates.csv](data/sim_h2/coding/SAS/estimates.csv)
- [data/sim_h2/coding_null/SAS/estimates.csv](data/sim_h2/coding_null/SAS/estimates.csv)
- [data/sim_h2/coding/AFR/estimates.csv](data/sim_h2/coding/AFR/estimates.csv)
- [data/sim_h2/coding_null/AFR/estimates.csv](data/sim_h2/coding_null/AFR/estimates.csv)
- [data/sim_h2/coding_curves.json](data/sim_h2/coding_curves.json)

Calculated summaries: `data/sim_h2/coding_curves.json`.

## Figure S15

Inputs:

- [data/sim_h2/coding_null/EUR/estimates.csv](data/sim_h2/coding_null/EUR/estimates.csv)
- [data/sim_h2/coding_null/SAS/estimates.csv](data/sim_h2/coding_null/SAS/estimates.csv)
- [data/sim_h2/coding_null/AFR/estimates.csv](data/sim_h2/coding_null/AFR/estimates.csv)
- [data/sim_h2/coding_calibration.tsv](data/sim_h2/coding_calibration.tsv)

Calculated summaries: `data/sim_h2/coding_calibration.tsv`.

## Figure S16

Inputs:

- [data/sim_rg/partitioned/EUR_300k/estimates.csv](data/sim_rg/partitioned/EUR_300k/estimates.csv)
- [data/sim_rg/partitioned/EUR/estimates.csv](data/sim_rg/partitioned/EUR/estimates.csv)
- [data/sim_rg/partitioned/SAS/estimates.csv](data/sim_rg/partitioned/SAS/estimates.csv)
- [data/sim_rg/partitioned/AFR/estimates.csv](data/sim_rg/partitioned/AFR/estimates.csv)
- [data/sim_rg/partitioned/truth.csv](data/sim_rg/partitioned/truth.csv)

## Figure S17

Inputs:

- [data/sim_rg/partitioned/EUR_300k/estimates.csv](data/sim_rg/partitioned/EUR_300k/estimates.csv)
- [data/sim_rg/partitioned/EUR/estimates.csv](data/sim_rg/partitioned/EUR/estimates.csv)
- [data/sim_rg/partitioned/SAS/estimates.csv](data/sim_rg/partitioned/SAS/estimates.csv)
- [data/sim_rg/partitioned/AFR/estimates.csv](data/sim_rg/partitioned/AFR/estimates.csv)
- [data/sim_rg/partitioned/truth.csv](data/sim_rg/partitioned/truth.csv)

## Figure S18

Inputs:

- [data/real_rg/gradient/pair_summary.csv](data/real_rg/gradient/pair_summary.csv)
- [data/real_rg/gradient/pair_hits_directional_nominal.csv](data/real_rg/gradient/pair_hits_directional_nominal.csv)
- [data/real_rg/gradient/parsed.csv](data/real_rg/gradient/parsed.csv)

## Figure S19

Inputs:

- [data/sim_rg/total/EUR_300k/estimates.csv](data/sim_rg/total/EUR_300k/estimates.csv)
- [data/sim_rg/total/EUR/estimates.csv](data/sim_rg/total/EUR/estimates.csv)
- [data/sim_rg/total/SAS/estimates.csv](data/sim_rg/total/SAS/estimates.csv)
- [data/sim_rg/total/AFR/estimates.csv](data/sim_rg/total/AFR/estimates.csv)

## Figure S20

Inputs:

- [data/ld/numvec_error.csv](data/ld/numvec_error.csv)

## Figure S21

Inputs:

- [data/ld/numvec_histogram.csv](data/ld/numvec_histogram.csv)

## Figure S22

Inputs:

- [data/ld/numvec_h2.csv](data/ld/numvec_h2.csv)

## Figure S23

Inputs:

- [data/ld/runtime_scaling.csv](data/ld/runtime_scaling.csv)

## Figure S24

Inputs:

- [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv)
- [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv)
- [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv)
- [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)
- [data/traits/traits.txt](data/traits/traits.txt)

## Figure S25

Inputs:

- [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv)
- [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv)
- [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv)
- [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)
- [data/traits/traits.txt](data/traits/traits.txt)

## Figure S26

Inputs:

- [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv)
- [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv)
- [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv)
- [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)
- [data/traits/traits.txt](data/traits/traits.txt)

## Figure S27

Inputs:

- [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv)
- [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv)
- [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv)
- [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)
- [data/traits/population_comparison_traits.txt](data/traits/population_comparison_traits.txt)
- [data/traits/traits.txt](data/traits/traits.txt)

## Figure S28

Inputs:

- [data/baseline/s28/panelA_pairwise_portability.csv](data/baseline/s28/panelA_pairwise_portability.csv)
- [data/baseline/s28/trait_detail.csv](data/baseline/s28/trait_detail.csv)
- [data/baseline/s28/annotation_pairs.csv](data/baseline/s28/annotation_pairs.csv)
- [data/baseline/s28/annotation_summary.csv](data/baseline/s28/annotation_summary.csv)

## Figure S29

Inputs:

- [data/baseline/tau_all_pops.csv](data/baseline/tau_all_pops.csv)

## Figure S30

Inputs:

- [data/external/gazal_comparison.tsv](data/external/gazal_comparison.tsv)

## Figure S31

Inputs:

- [data/array_imputed/total_heritability.csv](data/array_imputed/total_heritability.csv)
- [data/array_imputed/annotation_comparison.csv](data/array_imputed/annotation_comparison.csv)
- [data/array_imputed/annotation_variant_counts.csv](data/array_imputed/annotation_variant_counts.csv)
- [data/array_imputed/variant_count_summary.csv](data/array_imputed/variant_count_summary.csv)

## Figure S32

Inputs:

- [data/array_imputed/total_heritability.csv](data/array_imputed/total_heritability.csv)
- [data/array_imputed/annotation_comparison.csv](data/array_imputed/annotation_comparison.csv)
- [data/array_imputed/annotation_variant_counts.csv](data/array_imputed/annotation_variant_counts.csv)
- [data/array_imputed/variant_count_summary.csv](data/array_imputed/variant_count_summary.csv)

## Figure S33

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)

## Figure S34

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)

## Figure S35

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)

## Figure S36

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)
- [data/traits/trait_categories.tsv](data/traits/trait_categories.tsv)

## Figure S37

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S38

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S39

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S40

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S41

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S42

Inputs:

- [data/real_rg/estimates.csv](data/real_rg/estimates.csv)

## Figure S43

Inputs:

- [data/external/total_h2.tsv](data/external/total_h2.tsv)

## Figure S44

Inputs:

- [data/external/reference_swap.tsv](data/external/reference_swap.tsv)
- [data/external/ld_hexbin.json](data/external/ld_hexbin.json)
