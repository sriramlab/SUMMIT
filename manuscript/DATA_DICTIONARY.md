# Data dictionary

Rows describe simulation replicates, estimates, trait pairs, annotations, jackknife blocks, runtime measurements or aggregate LD bins. Metadata workbooks contain cohort and trait summaries.

The JSON dictionary records column names and row counts. Annotation-bin indices follow the parsed estimator outputs. Plotting scripts apply the stated finite-value filters.

| File / worksheet | Rows | Schema |
|---|---:|---|
| [data/array_imputed/annotation_comparison.csv](data/array_imputed/annotation_comparison.csv) | 48 | T1 |
| [data/array_imputed/total_heritability.csv](data/array_imputed/total_heritability.csv) | 40 | T2 |
| [data/array_imputed/variant_count_summary.csv](data/array_imputed/variant_count_summary.csv) | 2 | T3 |
| [data/array_imputed/annotation_variant_counts.csv](data/array_imputed/annotation_variant_counts.csv) | 13 | T4 |
| [data/baseline/figure5/panelA_pairwise_portability.csv](data/baseline/figure5/panelA_pairwise_portability.csv) | 36 | T5 |
| [data/baseline/figure5/ratio_meta.csv](data/baseline/figure5/ratio_meta.csv) | 40 | T6 |
| [data/baseline/s28/annotation_pairs.csv](data/baseline/s28/annotation_pairs.csv) | 288 | T7 |
| [data/baseline/s28/annotation_summary.csv](data/baseline/s28/annotation_summary.csv) | 6 | T8 |
| [data/baseline/s28/panelA_pairwise_portability.csv](data/baseline/s28/panelA_pairwise_portability.csv) | 60 | T5 |
| [data/baseline/s28/trait_detail.csv](data/baseline/s28/trait_detail.csv) | 109 | T9 |
| [data/baseline/tau_all_pops.csv](data/baseline/tau_all_pops.csv) | 192 | T10 |
| [data/external/enrichment_jackknife.tsv](data/external/enrichment_jackknife.tsv) | 78 | T11 |
| [data/external/enrichment_values.csv](data/external/enrichment_values.csv) | 2,041 | T12 |
| [data/external/gazal_comparison.tsv](data/external/gazal_comparison.tsv) | 192 | T13 |
| [data/external/ld_components.tsv](data/external/ld_components.tsv) | 94 | T14 |
| [data/external/reference_swap.tsv](data/external/reference_swap.tsv) | 1,218 | T15 |
| [data/external/total_h2.tsv](data/external/total_h2.tsv) | 78 | T16 |
| [data/ld/numvec_error.csv](data/ld/numvec_error.csv) | 385 | T17 |
| [data/ld/numvec_h2.csv](data/ld/numvec_h2.csv) | 60 | T18 |
| [data/ld/numvec_histogram.csv](data/ld/numvec_histogram.csv) | 2,560 | T19 |
| [data/ld/pc_ld_summary.csv](data/ld/pc_ld_summary.csv) | 28 | T20 |
| [data/ld/pc_sensitivity_h2.csv](data/ld/pc_sensitivity_h2.csv) | 84 | T21 |
| [data/ld/runtime_scaling.csv](data/ld/runtime_scaling.csv) | 16 | T22 |
| [data/ld/summary_nocov.csv](data/ld/summary_nocov.csv) | 12 | T23 |
| [data/ld/summary_pc.csv](data/ld/summary_pc.csv) | 12 | T23 |
| [data/real_h2/AFR/estimates.csv](data/real_h2/AFR/estimates.csv) | 400 | T24 |
| [data/real_h2/EUR/estimates.csv](data/real_h2/EUR/estimates.csv) | 400 | T24 |
| [data/real_h2/EUR_300k/estimates.csv](data/real_h2/EUR_300k/estimates.csv) | 400 | T24 |
| [data/real_h2/SAS/estimates.csv](data/real_h2/SAS/estimates.csv) | 400 | T24 |
| [data/real_rg/gradient/pair_hits_directional_nominal.csv](data/real_rg/gradient/pair_hits_directional_nominal.csv) | 18 | T25 |
| [data/real_rg/gradient/pair_summary.csv](data/real_rg/gradient/pair_summary.csv) | 760 | T25 |
| [data/real_rg/gradient/parsed.csv](data/real_rg/gradient/parsed.csv) | 9,360 | T26 |
| [data/real_rg/estimates.csv](data/real_rg/estimates.csv) | 46,800 | T27 |
| [data/sim_h2/binary/AFR/estimates.csv](data/sim_h2/binary/AFR/estimates.csv) | 800 | T28 |
| [data/sim_h2/binary/EUR/estimates.csv](data/sim_h2/binary/EUR/estimates.csv) | 800 | T28 |
| [data/sim_h2/binary/SAS/estimates.csv](data/sim_h2/binary/SAS/estimates.csv) | 800 | T28 |
| [data/sim_h2/coding/AFR/estimates.csv](data/sim_h2/coding/AFR/estimates.csv) | 33,600 | T29 |
| [data/sim_h2/coding/EUR/estimates.csv](data/sim_h2/coding/EUR/estimates.csv) | 33,600 | T29 |
| [data/sim_h2/coding/SAS/estimates.csv](data/sim_h2/coding/SAS/estimates.csv) | 33,600 | T29 |
| [data/sim_h2/coding_calibration.tsv](data/sim_h2/coding_calibration.tsv) | 147 | T30 |
| [data/sim_h2/coding_null/AFR/estimates.csv](data/sim_h2/coding_null/AFR/estimates.csv) | 11,700 | T31 |
| [data/sim_h2/coding_null/EUR/estimates.csv](data/sim_h2/coding_null/EUR/estimates.csv) | 11,700 | T31 |
| [data/sim_h2/coding_null/SAS/estimates.csv](data/sim_h2/coding_null/SAS/estimates.csv) | 11,700 | T31 |
| [data/sim_h2/eur300k_h2.tsv](data/sim_h2/eur300k_h2.tsv) | 2,700 | T32 |
| [data/sim_h2/figure1/panel_a_source.tsv](data/sim_h2/figure1/panel_a_source.tsv) | 18 | T33 |
| [data/sim_h2/figure1/panel_b_source.tsv](data/sim_h2/figure1/panel_b_source.tsv) | 9 | T34 |
| [data/sim_h2/figure1/panel_c_relmse_summary.tsv](data/sim_h2/figure1/panel_c_relmse_summary.tsv) | 18 | T35 |
| [data/sim_h2/figure1/panel_d_bias_source.tsv](data/sim_h2/figure1/panel_d_bias_source.tsv) | 72 | T36 |
| [data/sim_h2/figure2/calibration.tsv](data/sim_h2/figure2/calibration.tsv) | 9 | T37 |
| [data/sim_h2/figure2/discrimination.tsv](data/sim_h2/figure2/discrimination.tsv) | 9 | T38 |
| [data/sim_h2/figure2/overlap_curves.tsv](data/sim_h2/figure2/overlap_curves.tsv) | 378 | T39 |
| [data/sim_h2/mafld/AFR/estimates.csv](data/sim_h2/mafld/AFR/estimates.csv) | 4,800 | T40 |
| [data/sim_h2/mafld/EUR/estimates.csv](data/sim_h2/mafld/EUR/estimates.csv) | 4,800 | T40 |
| [data/sim_h2/mafld/SAS/estimates.csv](data/sim_h2/mafld/SAS/estimates.csv) | 4,800 | T40 |
| [data/sim_h2/missing_snp_summary.csv](data/sim_h2/missing_snp_summary.csv) | 108 | T41 |
| [data/sim_h2/no_pc_comparison.tsv](data/sim_h2/no_pc_comparison.tsv) | 102 | T42 |
| [data/sim_h2/pc_stratified.tsv](data/sim_h2/pc_stratified.tsv) | 1,800 | T43 |
| [data/sim_h2/total/AFR/estimates.csv](data/sim_h2/total/AFR/estimates.csv) | 17,100 | T44 |
| [data/sim_h2/total/EUR/estimates.csv](data/sim_h2/total/EUR/estimates.csv) | 17,100 | T44 |
| [data/sim_h2/total/SAS/estimates.csv](data/sim_h2/total/SAS/estimates.csv) | 17,100 | T44 |
| [data/sim_rg/figure3/calibration.tsv](data/sim_rg/figure3/calibration.tsv) | 9 | T45 |
| [data/sim_rg/figure3/normalization.tsv](data/sim_rg/figure3/normalization.tsv) | 685 | T46 |
| [data/sim_rg/figure3/panelB.tsv](data/sim_rg/figure3/panelB.tsv) | 9 | T47 |
| [data/sim_rg/figure3/panelC.tsv](data/sim_rg/figure3/panelC.tsv) | 18 | T48 |
| [data/sim_rg/partitioned/AFR/estimates.csv](data/sim_rg/partitioned/AFR/estimates.csv) | 7,300 | T49 |
| [data/sim_rg/partitioned/EUR/estimates.csv](data/sim_rg/partitioned/EUR/estimates.csv) | 7,300 | T49 |
| [data/sim_rg/partitioned/EUR_300k/estimates.csv](data/sim_rg/partitioned/EUR_300k/estimates.csv) | 7,200 | T50 |
| [data/sim_rg/partitioned/SAS/estimates.csv](data/sim_rg/partitioned/SAS/estimates.csv) | 7,300 | T49 |
| [data/sim_rg/partitioned/truth.csv](data/sim_rg/partitioned/truth.csv) | 24 | T51 |
| [data/sim_rg/total/AFR/estimates.csv](data/sim_rg/total/AFR/estimates.csv) | 1,010 | T52 |
| [data/sim_rg/total/EUR/estimates.csv](data/sim_rg/total/EUR/estimates.csv) | 1,300 | T52 |
| [data/sim_rg/total/EUR_300k/estimates.csv](data/sim_rg/total/EUR_300k/estimates.csv) | 300 | T52 |
| [data/sim_rg/total/SAS/estimates.csv](data/sim_rg/total/SAS/estimates.csv) | 1,300 | T52 |
| [data/traits/wgs_heritability.csv](data/traits/wgs_heritability.csv) | 30 | T53 |
| [metadata/supplementary_data.xlsx / Trait sample sizes](metadata/supplementary_data.xlsx) | 160 | T54 |
| [metadata/supplementary_data.xlsx / Cohorts](metadata/supplementary_data.xlsx) | 4 | T55 |
| [metadata/supplementary_data.xlsx / External GWAS](metadata/supplementary_data.xlsx) | 43 | T56 |
| [metadata/supplementary_data.xlsx / Definitions](metadata/supplementary_data.xlsx) | 6 | T57 |

<details><summary>T1: 22 columns</summary>

`annotation`, `array_tau_star_meta`, `imputed_tau_star_meta`, `array_tau_star_se_meta`, `imputed_tau_star_se_meta`, `array_enrichment_meta`, `imputed_enrichment_meta`, `array_log_enrichment_meta`, `imputed_log_enrichment_meta`, `array_log_enrichment_se_meta`, `imputed_log_enrichment_se_meta`, `delta_tau_star`, `delta_log_enrichment`, `array_log2_enrichment_meta`, `imputed_log2_enrichment_meta`, `array_log2_enrichment_se_meta`, `imputed_log2_enrichment_se_meta`, `delta_log2_enrichment`, `array_enrichment_rank`, `imputed_enrichment_rank`, `best_enrichment_rank`, `mean_enrichment_rank`

</details>

<details><summary>T2: 9 columns</summary>

`phen`, `phen_name_array`, `array_h2`, `array_h2_se`, `phen_name_imp`, `imp_h2`, `imp_h2_se`, `delta_h2`, `plot_label`

</details>

<details><summary>T3: 4 columns</summary>

`genotype`, `n_variants`, `n_maf_lt_0.01`, `n_maf_ge_0.01`

</details>

<details><summary>T4: 5 columns</summary>

`annotation`, `label`, `n_variants`, `n_maf_lt_0.01`, `frac_maf_lt_0.01`

</details>

<details><summary>T5: 14 columns</summary>

`method`, `method_label`, `pop_pair`, `n_phens`, `pearson_r_agg`, `pearson_r_se`, `spearman_rho_agg`, `spearman_rho_se`, `pair_lo`, `pair_hi`, `pair`, `pair_pretty`, `estimand`, `n_units`

</details>

<details><summary>T6: 26 columns</summary>

`contrast_set`, `pop`, `pair_name`, `display_order`, `numerator_annotation`, `denominator_annotation`, `numerator_label`, `denominator_label`, `estimate`, `ci_lo`, `ci_hi`, `se_log`, `n_obs`, `meta_model`, `n_aggregate_jackknife`, `min_traits_in_aggregate_replicate`, `max_traits_in_aggregate_replicate`, `z_log_ratio`, `p_one_sided_gt1`, `expected_gt1`, `bh_fdr_one_sided_all_panel_b`, `bonferroni_one_sided_all_panel_b`, `bh_fdr_one_sided_by_pop`, `bonferroni_one_sided_by_pop`, `bh_fdr_one_sided_expected_gt1`, `bonferroni_one_sided_expected_gt1`

</details>

<details><summary>T7: 8 columns</summary>

`reference_pop`, `target_pop`, `metric`, `annotation`, `annotation_label`, `annotation_family`, `reference_estimate`, `target_estimate`

</details>

<details><summary>T8: 6 columns</summary>

`reference_pop`, `target_pop`, `metric`, `n_annotations`, `pearson`, `spearman`

</details>

<details><summary>T9: 9 columns</summary>

`reference_pop`, `target_pop`, `phen`, `n_annotations`, `pearson`, `spearman`, `mean_abs_log2_diff`, `median_abs_log2_diff`, `mean_signed_log2_diff`

</details>

<details><summary>T10: 18 columns</summary>

`pop`, `annotation`, `annotation_label`, `annotation_family`, `n_traits`, `tau_star_meta`, `tau_star_ci_lo`, `tau_star_ci_hi`, `tau_star_se`, `enrichment_meta`, `enrichment_ci_lo`, `enrichment_ci_hi`, `log2_enrichment_meta`, `log2_enrichment_ci_lo`, `log2_enrichment_ci_hi`, `annotation_group`, `annotation_label_plot`, `n_jackknife`

</details>

<details><summary>T11: 28 columns</summary>

`dataset_kind`, `pop`, `pop_label`, `annotation`, `annotation_label`, `bin_index`, `metric`, `estimate`, `se`, `ci_lo`, `ci_hi`, `jackknife_p025`, `jackknife_p975`, `n_traits`, `n_total_traits`, `n_jackknife_blocks`, `n_nonfinite_jackknife_medians`, `median_finite_traits_per_jackknife`, `min_finite_traits_per_jackknife`, `n_traits_with_any_nonfinite_jackknife`, `n_negative_trait_enrichment`, `negative_fraction`, `n_nonpositive_total_h2`, `n_nonpositive_bin_h2`, `trait_ids`, `annotation_order`, `pop_order`, `kind_order`

</details>

<details><summary>T12: 42 columns</summary>

`pop`, `phen`, `phen_name`, `phen_short`, `trait_group`, `method`, `selected_window`, `bin_index`, `annotation`, `annotation_label`, `h2bin`, `h2bin_se`, `enrichment`, `enrichment_se`, `tau`, `tau_se`, `tau_star`, `tau_star_se`, `is_base`, `is_maf`, `dataset_id`, `dataset_kind`, `cohort`, `source`, `source_group`, `external_trait`, `local_match_phen`, `h2`, `h2_se`, `manifest_n_median`, `matched_snps`, `log2_enrichment`, `log2_enrichment_se`, `enrichment_normalizer_raw`, `enrichment_normalizer`, `normalizer_source`, `signed_enrichment`, `signed_enrichment_plot`, `is_negative_enrichment`, `is_undefined_enrichment`, `is_nonpositive_total_h2`, `group_label`

</details>

<details><summary>T13: 35 columns</summary>

`dataset_kind`, `pop`, `annotation`, `annotation_label`, `summit_tau_star`, `summit_tau_star_se`, `summit_tau_star_n`, `summit_tau_star_ivw_model_se`, `summit_tau_star_ivw_jackknife_se`, `summit_tau_star_gls_se`, `n_traits_ivw`, `n_traits_gls`, `summit_log2_enrichment`, `summit_log2_enrichment_se`, `summit_enrichment_n`, `summit_log2_enrichment_ivw_model_se`, `summit_log2_enrichment_ivw_jackknife_se`, `summit_log2_enrichment_gls_se`, `gazal_annotation`, `gazal_enrichment`, `gazal_enrichment_se`, `gazal_tau_star`, `gazal_tau_star_se`, `gazal_tau_star_p`, `gazal_log2_enrichment`, `gazal_log2_enrichment_se`, `summit_enrichment`, `summit_enrichment_se`, `summit_tau_star_ci_lo`, `summit_tau_star_ci_hi`, `summit_enrichment_ci_lo`, `summit_enrichment_ci_hi`, `enrichment_ratio_summit_over_gazal`, `log2_enrichment_diff_summit_minus_gazal`, `tau_star_diff_summit_minus_gazal`

</details>

<details><summary>T14: 23 columns</summary>

`panel_id`, `annotation_model`, `row_type`, `metric_name`, `n_components`, `n_valid_pairwise`, `kappa_sas_on_eur_no_intercept`, `r2_no_intercept`, `pearson_r`, `spearman_r`, `eur_sum`, `sas_sum`, `sas_over_eur_sum`, `eur_mean`, `sas_mean`, `n_positive`, `median_sas_over_eur`, `median_abs_log_ratio`, `log_ratio_sd`, `eur_mean_block_jackknife_se`, `sas_mean_block_jackknife_se`, `sas_over_eur_mean`, `component_label`

</details>

<details><summary>T15: 19 columns</summary>

`trait`, `bin_index`, `annotation`, `annotation_group`, `annotation_prop`, `eur_h2`, `sas_h2`, `eur_h2bin`, `sas_h2bin`, `eur_h2bin_se`, `sas_h2bin_se`, `eur_enrichment`, `sas_enrichment`, `h2_ratio_sas_over_eur`, `h2bin_ratio_sas_over_eur`, `enrichment_ratio_sas_over_eur`, `log2_enrichment_diff_sas_minus_eur`, `eur_h2bin_z`, `sas_h2bin_z`

</details>

<details><summary>T16: 14 columns</summary>

`cohort`, `method`, `source`, `group`, `source_label`, `trait`, `local_phen`, `external_h2`, `external_h2_se`, `ukb_h2`, `ukb_h2_se`, `exclude_reason`, `included`, `finite_for_plot`

</details>

<details><summary>T17: 11 columns</summary>

`pop`, `cohort`, `cov_label`, `numvec`, `seed`, `ref_numvec`, `n_ref_seeds`, `n_snps`, `rel_rmse_pct`, `rel_mae_pct`, `rel_sum_error_pct`

</details>

<details><summary>T18: 8 columns</summary>

`pop`, `phen`, `phen_label`, `numvec`, `h2`, `h2_se`, `ld_source`, `ld_components`

</details>

<details><summary>T19: 4 columns</summary>

`pop`, `numvec`, `center`, `density`

</details>

<details><summary>T20: 7 columns</summary>

`pop`, `pop_label`, `pc`, `num_pcs`, `mean_ldscore`, `num_snps`, `num_ld_cols`

</details>

<details><summary>T21: 8 columns</summary>

`pop`, `phen`, `phen_label`, `pc`, `num_pcs`, `num_bins`, `h2`, `h2_se`

</details>

<details><summary>T22: 15 columns</summary>

`experiment`, `setting`, `pop`, `n`, `m`, `k`, `numvec`, `n_runs`, `elapsed_mean`, `elapsed_median`, `elapsed_sd`, `elapsed_min`, `elapsed_max`, `max_rss_kb_mean`, `max_rss_kb_max`

</details>

<details><summary>T23: 8 columns</summary>

`pop`, `case`, `pc_case`, `window`, `group`, `mean`, `se`, `n_chr`

</details>

<details><summary>T24: 207 columns</summary>

`pop`, `phen`, `phen_name`, `method`, `pc`, `window`, `num_bins`, `h2`, `h2_se`, `h2bin_0`, `h2bin_se_0`, `enr_0`, `enr_se_0`, `tau_0`, `tau_se_0`, `tau_star_0`, `tau_star_se_0`, `h2bin_1`, `h2bin_se_1`, `enr_1`, `enr_se_1`, `tau_1`, `tau_se_1`, `tau_star_1`, `tau_star_se_1`, `h2bin_2`, `h2bin_se_2`, `enr_2`, `enr_se_2`, `tau_2`, `tau_se_2`, `tau_star_2`, `tau_star_se_2`, `h2bin_3`, `h2bin_se_3`, `enr_3`, `enr_se_3`, `tau_3`, `tau_se_3`, `tau_star_3`, `tau_star_se_3`, `h2bin_4`, `h2bin_se_4`, `enr_4`, `enr_se_4`, `tau_4`, `tau_se_4`, `tau_star_4`, `tau_star_se_4`, `h2bin_5`, `h2bin_se_5`, `enr_5`, `enr_se_5`, `tau_5`, `tau_se_5`, `tau_star_5`, `tau_star_se_5`, `h2bin_6`, `h2bin_se_6`, `enr_6`, `enr_se_6`, `tau_6`, `tau_se_6`, `tau_star_6`, `tau_star_se_6`, `h2bin_7`, `h2bin_se_7`, `enr_7`, `enr_se_7`, `tau_7`, `tau_se_7`, `tau_star_7`, `tau_star_se_7`, `h2bin_8`, `h2bin_se_8`, `enr_8`, `enr_se_8`, `tau_8`, `tau_se_8`, `tau_star_8`, `tau_star_se_8`, `h2bin_9`, `h2bin_se_9`, `enr_9`, `enr_se_9`, `tau_9`, `tau_se_9`, `tau_star_9`, `tau_star_se_9`, `h2bin_10`, `h2bin_se_10`, `enr_10`, `enr_se_10`, `tau_10`, `tau_se_10`, `tau_star_10`, `tau_star_se_10`, `h2bin_11`, `h2bin_se_11`, `enr_11`, `enr_se_11`, `tau_11`, `tau_se_11`, `tau_star_11`, `tau_star_se_11`, `h2bin_12`, `h2bin_se_12`, `enr_12`, `enr_se_12`, `tau_12`, `tau_se_12`, `tau_star_12`, `tau_star_se_12`, `h2bin_13`, `h2bin_se_13`, `enr_13`, `enr_se_13`, `tau_13`, `tau_se_13`, `tau_star_13`, `tau_star_se_13`, `h2bin_14`, `h2bin_se_14`, `enr_14`, `enr_se_14`, `tau_14`, `tau_se_14`, `tau_star_14`, `tau_star_se_14`, `h2bin_15`, `h2bin_se_15`, `enr_15`, `enr_se_15`, `tau_15`, `tau_se_15`, `tau_star_15`, `tau_star_se_15`, `h2bin_16`, `h2bin_se_16`, `enr_16`, `enr_se_16`, `tau_16`, `tau_se_16`, `tau_star_16`, `tau_star_se_16`, `h2bin_17`, `h2bin_se_17`, `enr_17`, `enr_se_17`, `tau_17`, `tau_se_17`, `tau_star_17`, `tau_star_se_17`, `h2bin_18`, `h2bin_se_18`, `enr_18`, `enr_se_18`, `tau_18`, `tau_se_18`, `tau_star_18`, `tau_star_se_18`, `h2bin_19`, `h2bin_se_19`, `enr_19`, `enr_se_19`, `tau_19`, `tau_se_19`, `tau_star_19`, `tau_star_se_19`, `h2bin_20`, `h2bin_se_20`, `enr_20`, `enr_se_20`, `tau_20`, `tau_se_20`, `tau_star_20`, `tau_star_se_20`, `h2bin_21`, `h2bin_se_21`, `enr_21`, `enr_se_21`, `tau_21`, `tau_se_21`, `tau_star_21`, `tau_star_se_21`, `h2bin_22`, `h2bin_se_22`, `enr_22`, `enr_se_22`, `tau_22`, `tau_se_22`, `tau_star_22`, `tau_star_se_22`, `h2bin_23`, `h2bin_se_23`, `enr_23`, `enr_se_23`, `tau_23`, `tau_se_23`, `tau_star_23`, `tau_star_se_23`, `sample_prev`, `pop_prev`, `liab_mult`, `n_case`, `n_control`, `n_cc`

</details>

<details><summary>T25: 60 columns</summary>

`pop`, `method`, `a`, `b`, `total_rg`, `total_rg_se`, `total_rg_z`, `total_rg_p_two_sided`, `total_rg_sign`, `n_valid_panels`, `valid_panel_labels`, `expected_panel_labels`, `n_positive_panels`, `n_negative_panels`, `n_zero_panels`, `frac_positive_panels`, `all_panel_deltas_positive`, `min_panel_delta`, `max_panel_delta`, `pair_gls_mean`, `pair_gls_se`, `pair_gls_z`, `pair_gls_p_one_sided`, `pair_gls_p_two_sided`, `pair_omnibus_Q`, `pair_omnibus_df`, `pair_omnibus_p`, `jack_reps_total`, `jack_reps_complete`, `jack_complete_fraction`, `pair_gls_family_m`, `pair_gls_bonf_alpha`, `pair_gls_p_bonf`, `pair_gls_p_holm`, `pair_gls_q_bh`, `pair_gls_sig_bonf`, `pair_gls_sig_holm`, `pair_gls_sig_bh`, `pair_omnibus_family_m`, `pair_omnibus_bonf_alpha`, `pair_omnibus_p_bonf`, `pair_omnibus_p_holm`, `pair_omnibus_q_bh`, `pair_omnibus_sig_bonf`, `pair_omnibus_sig_holm`, `pair_omnibus_sig_bh`, `pair_gls_hit_nominal`, `pair_omnibus_hit_nominal`, `primary_hit_nominal`, `primary_hit_fdr_bh`, `primary_hit_bonf`, `primary_hit_holm`, `relaxed_hit_nominal`, `relaxed_hit_fdr_bh`, `relaxed_hit_bonf`, `relaxed_hit_holm`, `directional_hit_nominal`, `directional_hit_fdr_bh`, `directional_hit_bonf`, `directional_hit_holm`

</details>

<details><summary>T26: 93 columns</summary>

`pop`, `method`, `annot_type`, `phen1`, `phen2`, `phen1_name`, `phen2_name`, `rg`, `rg_se`, `intercept_c`, `intercept_c_se`, `gencov_total`, `gencov_total_se`, `h2_1_total`, `h2_1_total_se`, `h2_2_total`, `h2_2_total_se`, `n_bins`, `n_bins_parsed`, `parse_ok`, `parse_error`, `bin_name_0`, `h2_1_bin_0`, `h2_1_bin_se_0`, `h2_2_bin_0`, `h2_2_bin_se_0`, `gencov_bin_0`, `gencov_bin_se_0`, `rg_bin_0`, `rg_bin_se_0`, `bin_name_1`, `h2_1_bin_1`, `h2_1_bin_se_1`, `h2_2_bin_1`, `h2_2_bin_se_1`, `gencov_bin_1`, `gencov_bin_se_1`, `rg_bin_1`, `rg_bin_se_1`, `bin_name_2`, `h2_1_bin_2`, `h2_1_bin_se_2`, `h2_2_bin_2`, `h2_2_bin_se_2`, `gencov_bin_2`, `gencov_bin_se_2`, `rg_bin_2`, `rg_bin_se_2`, `bin_name_3`, `h2_1_bin_3`, `h2_1_bin_se_3`, `h2_2_bin_3`, `h2_2_bin_se_3`, `gencov_bin_3`, `gencov_bin_se_3`, `rg_bin_3`, `rg_bin_se_3`, `bin_name_4`, `h2_1_bin_4`, `h2_1_bin_se_4`, `h2_2_bin_4`, `h2_2_bin_se_4`, `gencov_bin_4`, `gencov_bin_se_4`, `rg_bin_4`, `rg_bin_se_4`, `bin_name_5`, `h2_1_bin_5`, `h2_1_bin_se_5`, `h2_2_bin_5`, `h2_2_bin_se_5`, `gencov_bin_5`, `gencov_bin_se_5`, `rg_bin_5`, `rg_bin_se_5`, `bin_name_6`, `h2_1_bin_6`, `h2_1_bin_se_6`, `h2_2_bin_6`, `h2_2_bin_se_6`, `gencov_bin_6`, `gencov_bin_se_6`, `rg_bin_6`, `rg_bin_se_6`, `bin_name_7`, `h2_1_bin_7`, `h2_1_bin_se_7`, `h2_2_bin_7`, `h2_2_bin_se_7`, `gencov_bin_7`, `gencov_bin_se_7`, `rg_bin_7`, `rg_bin_se_7`

</details>

<details><summary>T27: 20 columns</summary>

`pop`, `method`, `annot_type`, `phen1`, `phen2`, `phen1_name`, `phen2_name`, `rg`, `rg_se`, `intercept_c`, `intercept_c_se`, `gencov_total`, `gencov_total_se`, `h2_1_total`, `h2_1_total_se`, `h2_2_total`, `h2_2_total_se`, `n_bins_parsed`, `parse_ok`, `parse_error`

</details>

<details><summary>T28: 14 columns</summary>

`method`, `run`, `window`, `h2_true`, `tag`, `K_pop`, `P_samp`, `scale_factor`, `h2_obs`, `se_obs`, `h2_liab`, `se_liab`, `h2_liab_reported`, `se_liab_reported`

</details>

<details><summary>T29: 19 columns</summary>

`method`, `run`, `window`, `true_h2`, `p_causal`, `arch`, `scenario`, `setting`, `num_bins`, `h2`, `h2_se`, `h2bin_0`, `h2bin_se_0`, `enr_0`, `enr_se_0`, `h2bin_1`, `h2bin_se_1`, `enr_1`, `enr_se_1`

</details>

<details><summary>T30: 10 columns</summary>

`pop`, `method_label`, `alpha`, `n_sig`, `n_noncausal`, `empirical`, `emp_se`, `emp_lo`, `emp_hi`, `n_clusters`

</details>

<details><summary>T31: 17 columns</summary>

`method`, `run`, `window`, `true_h2`, `p_causal`, `setting`, `num_bins`, `h2`, `h2_se`, `h2bin_0`, `h2bin_se_0`, `enr_0`, `enr_se_0`, `h2bin_1`, `h2bin_se_1`, `enr_1`, `enr_se_1`

</details>

<details><summary>T32: 10 columns</summary>

`pop`, `h2`, `pcausal`, `method`, `method_window`, `pretty`, `run`, `window`, `estimate`, `se`

</details>

<details><summary>T33: 19 columns</summary>

`pop`, `case`, `case_label`, `group`, `estimator`, `target_mean`, `baseline_mean`, `ratio`, `deficit_pct`, `n_snps`, `summit_nvecs`, `summit_seed`, `baseline_nvecs`, `baseline_seed`, `baseline_mean_ldscore`, `target_nvecs`, `target_seed`, `target_mean_ldscore`, `target_file_count`

</details>

<details><summary>T34: 8 columns</summary>

`pop`, `method`, `n_settings`, `median`, `q25`, `q75`, `min`, `max`

</details>

<details><summary>T35: 7 columns</summary>

`pop`, `architecture`, `method`, `median_rel_mse`, `q25_rel_mse`, `q75_rel_mse`, `n_settings`

</details>

<details><summary>T36: 12 columns</summary>

`pop`, `architecture`, `setting`, `p_causal`, `scenario`, `method`, `mean_error`, `n_replicates`, `true_h2`, `absolute_relative_bias`, `maf_support`, `design_category`

</details>

<details><summary>T37: 8 columns</summary>

`pop`, `method`, `N_valid`, `N_clusters`, `fpr`, `fpr_se`, `fpr_lo`, `fpr_hi`

</details>

<details><summary>T38: 9 columns</summary>

`pop`, `method`, `auroc`, `aupr`, `n_scenarios`, `min_n_positive`, `max_n_positive`, `min_n_null`, `max_n_null`

</details>

<details><summary>T39: 9 columns</summary>

`pop`, `method`, `scenario`, `metric`, `alpha`, `estimate`, `ci_lo`, `ci_hi`, `n`

</details>

<details><summary>T40: 9 columns</summary>

`method`, `run`, `window`, `true_h2`, `p_causal`, `setting`, `num_bins`, `h2`, `h2_se`

</details>

<details><summary>T41: 10 columns</summary>

`pop`, `true_h2`, `pcausal`, `missing_fraction`, `mean_h2`, `mean_complete_h2`, `proportion`, `ci95_low`, `ci95_high`, `n`

</details>

<details><summary>T42: 24 columns</summary>

`pop`, `category`, `category_label`, `setting`, `architecture`, `base_setting`, `p_causal`, `true_h2`, `n_no_pc_total`, `n_no_pc_valid`, `valid_rate_no_pc`, `n_matched`, `mse_pc`, `mse_no_pc`, `mse_ratio_no_pc_over_pc`, `ldak_source`, `comparison`, `comparison_label`, `mse_ratio`, `n_reference_total`, `n_reference_valid`, `valid_rate_reference`, `mse_reference`, `mse_ratio_no_pc_over_reference`

</details>

<details><summary>T43: 6 columns</summary>

`pop`, `method`, `rep`, `estimate`, `se`, `window_kb`

</details>

<details><summary>T44: 7 columns</summary>

`estimate`, `se`, `method`, `run`, `window`, `h2`, `pcausal`

</details>

<details><summary>T45: 26 columns</summary>

`h2_cfg`, `pop`, `method_plot`, `N_valid`, `bias`, `var`, `mse`, `true_se`, `mean_reported_se`, `median_reported_se`, `min_reported_se`, `max_reported_se`, `rej_0p001`, `rej_0p01`, `rej_0p05`, `rej_0p1`, `rej_0p2`, `fpr`, `fpr_lo`, `fpr_hi`, `fpr_se`, `N_total`, `N_calib`, `fail_rate`, `method`, `fpr_interval`

</details>

<details><summary>T46: 36 columns</summary>

`pop`, `method`, `arch`, `h2_cfg`, `rg_cfg`, `nbins`, `i`, `h2_1`, `h2_2`, `gamma_g`, `rg`, `parse_ok`, `method_source_label`, `true_h2_1`, `true_h2_2`, `true_gamma`, `true_rg`, `finite_constituents`, `positive_estimated_h2_product`, `included_in_decomposition`, `covariance_contribution`, `denominator_contribution`, `total_rg_error`, `rg_derived`, `rg_logged_minus_derived`, `decomposition_additivity_error`, `opposing_contributions`, `combined_smaller_than_both_contributions`, `rg_abs_gt_1p2`, `covariance_contribution_pct`, `denominator_contribution_pct`, `net_rg_error_pct`, `denominator`, `true_denominator`, `covariance_factor`, `denominator_factor`

</details>

<details><summary>T47: 16 columns</summary>

`metric`, `method`, `method_label`, `nbins`, `n_expected`, `n_finite`, `mean`, `median`, `sd`, `rmse`, `q025`, `q25`, `q75`, `q975`, `axis_low`, `axis_high`

</details>

<details><summary>T48: 16 columns</summary>

`metric`, `pop`, `method`, `method_label`, `profile`, `profile_label`, `n_expected`, `n_finite`, `mean`, `mean_ci95_low`, `mean_ci95_high`, `median`, `sd`, `rmse`, `q025`, `q975`

</details>

<details><summary>T49: 213 columns</summary>

`pop`, `method`, `arch`, `h2_cfg`, `rg_cfg`, `rho_g`, `nbins`, `i`, `h2_1`, `h2_se_1`, `h2_2`, `h2_se_2`, `intercept_c`, `intercept_se`, `gamma_g`, `gamma_se`, `rg`, `rg_se`, `n_bins_parsed`, `parse_ok`, `parse_error`, `h2_1_bin_0`, `h2_1_bin_se_0`, `h2_2_bin_0`, `h2_2_bin_se_0`, `gamma_bin_0`, `gamma_bin_se_0`, `rg_bin_0`, `rg_bin_se_0`, `h2_1_bin_1`, `h2_1_bin_se_1`, `h2_2_bin_1`, `h2_2_bin_se_1`, `gamma_bin_1`, `gamma_bin_se_1`, `rg_bin_1`, `rg_bin_se_1`, `h2_1_bin_2`, `h2_1_bin_se_2`, `h2_2_bin_2`, `h2_2_bin_se_2`, `gamma_bin_2`, `gamma_bin_se_2`, `rg_bin_2`, `rg_bin_se_2`, `h2_1_bin_3`, `h2_1_bin_se_3`, `h2_2_bin_3`, `h2_2_bin_se_3`, `gamma_bin_3`, `gamma_bin_se_3`, `rg_bin_3`, `rg_bin_se_3`, `h2_1_bin_4`, `h2_1_bin_se_4`, `h2_2_bin_4`, `h2_2_bin_se_4`, `gamma_bin_4`, `gamma_bin_se_4`, `rg_bin_4`, `rg_bin_se_4`, `h2_1_bin_5`, `h2_1_bin_se_5`, `h2_2_bin_5`, `h2_2_bin_se_5`, `gamma_bin_5`, `gamma_bin_se_5`, `rg_bin_5`, `rg_bin_se_5`, `h2_1_bin_6`, `h2_1_bin_se_6`, `h2_2_bin_6`, `h2_2_bin_se_6`, `gamma_bin_6`, `gamma_bin_se_6`, `rg_bin_6`, `rg_bin_se_6`, `h2_1_bin_7`, `h2_1_bin_se_7`, `h2_2_bin_7`, `h2_2_bin_se_7`, `gamma_bin_7`, `gamma_bin_se_7`, `rg_bin_7`, `rg_bin_se_7`, `h2_1_bin_8`, `h2_1_bin_se_8`, `h2_2_bin_8`, `h2_2_bin_se_8`, `gamma_bin_8`, `gamma_bin_se_8`, `rg_bin_8`, `rg_bin_se_8`, `h2_1_bin_9`, `h2_1_bin_se_9`, `h2_2_bin_9`, `h2_2_bin_se_9`, `gamma_bin_9`, `gamma_bin_se_9`, `rg_bin_9`, `rg_bin_se_9`, `h2_1_bin_10`, `h2_1_bin_se_10`, `h2_2_bin_10`, `h2_2_bin_se_10`, `gamma_bin_10`, `gamma_bin_se_10`, `rg_bin_10`, `rg_bin_se_10`, `h2_1_bin_11`, `h2_1_bin_se_11`, `h2_2_bin_11`, `h2_2_bin_se_11`, `gamma_bin_11`, `gamma_bin_se_11`, `rg_bin_11`, `rg_bin_se_11`, `h2_1_bin_12`, `h2_1_bin_se_12`, `h2_2_bin_12`, `h2_2_bin_se_12`, `gamma_bin_12`, `gamma_bin_se_12`, `rg_bin_12`, `rg_bin_se_12`, `h2_1_bin_13`, `h2_1_bin_se_13`, `h2_2_bin_13`, `h2_2_bin_se_13`, `gamma_bin_13`, `gamma_bin_se_13`, `rg_bin_13`, `rg_bin_se_13`, `h2_1_bin_14`, `h2_1_bin_se_14`, `h2_2_bin_14`, `h2_2_bin_se_14`, `gamma_bin_14`, `gamma_bin_se_14`, `rg_bin_14`, `rg_bin_se_14`, `h2_1_bin_15`, `h2_1_bin_se_15`, `h2_2_bin_15`, `h2_2_bin_se_15`, `gamma_bin_15`, `gamma_bin_se_15`, `rg_bin_15`, `rg_bin_se_15`, `h2_1_bin_16`, `h2_1_bin_se_16`, `h2_2_bin_16`, `h2_2_bin_se_16`, `gamma_bin_16`, `gamma_bin_se_16`, `rg_bin_16`, `rg_bin_se_16`, `h2_1_bin_17`, `h2_1_bin_se_17`, `h2_2_bin_17`, `h2_2_bin_se_17`, `gamma_bin_17`, `gamma_bin_se_17`, `rg_bin_17`, `rg_bin_se_17`, `h2_1_bin_18`, `h2_1_bin_se_18`, `h2_2_bin_18`, `h2_2_bin_se_18`, `gamma_bin_18`, `gamma_bin_se_18`, `rg_bin_18`, `rg_bin_se_18`, `h2_1_bin_19`, `h2_1_bin_se_19`, `h2_2_bin_19`, `h2_2_bin_se_19`, `gamma_bin_19`, `gamma_bin_se_19`, `rg_bin_19`, `rg_bin_se_19`, `h2_1_bin_20`, `h2_1_bin_se_20`, `h2_2_bin_20`, `h2_2_bin_se_20`, `gamma_bin_20`, `gamma_bin_se_20`, `rg_bin_20`, `rg_bin_se_20`, `h2_1_bin_21`, `h2_1_bin_se_21`, `h2_2_bin_21`, `h2_2_bin_se_21`, `gamma_bin_21`, `gamma_bin_se_21`, `rg_bin_21`, `rg_bin_se_21`, `h2_1_bin_22`, `h2_1_bin_se_22`, `h2_2_bin_22`, `h2_2_bin_se_22`, `gamma_bin_22`, `gamma_bin_se_22`, `rg_bin_22`, `rg_bin_se_22`, `h2_1_bin_23`, `h2_1_bin_se_23`, `h2_2_bin_23`, `h2_2_bin_se_23`, `gamma_bin_23`, `gamma_bin_se_23`, `rg_bin_23`, `rg_bin_se_23`

</details>

<details><summary>T50: 212 columns</summary>

`pop`, `method`, `arch`, `h2_cfg`, `rg_cfg`, `nbins`, `i`, `h2_1`, `h2_se_1`, `h2_2`, `h2_se_2`, `intercept_c`, `intercept_se`, `gamma_g`, `gamma_se`, `rg`, `rg_se`, `n_bins_parsed`, `parse_ok`, `parse_error`, `h2_1_bin_0`, `h2_1_bin_se_0`, `h2_2_bin_0`, `h2_2_bin_se_0`, `gamma_bin_0`, `gamma_bin_se_0`, `rg_bin_0`, `rg_bin_se_0`, `h2_1_bin_1`, `h2_1_bin_se_1`, `h2_2_bin_1`, `h2_2_bin_se_1`, `gamma_bin_1`, `gamma_bin_se_1`, `rg_bin_1`, `rg_bin_se_1`, `h2_1_bin_2`, `h2_1_bin_se_2`, `h2_2_bin_2`, `h2_2_bin_se_2`, `gamma_bin_2`, `gamma_bin_se_2`, `rg_bin_2`, `rg_bin_se_2`, `h2_1_bin_3`, `h2_1_bin_se_3`, `h2_2_bin_3`, `h2_2_bin_se_3`, `gamma_bin_3`, `gamma_bin_se_3`, `rg_bin_3`, `rg_bin_se_3`, `h2_1_bin_4`, `h2_1_bin_se_4`, `h2_2_bin_4`, `h2_2_bin_se_4`, `gamma_bin_4`, `gamma_bin_se_4`, `rg_bin_4`, `rg_bin_se_4`, `h2_1_bin_5`, `h2_1_bin_se_5`, `h2_2_bin_5`, `h2_2_bin_se_5`, `gamma_bin_5`, `gamma_bin_se_5`, `rg_bin_5`, `rg_bin_se_5`, `h2_1_bin_6`, `h2_1_bin_se_6`, `h2_2_bin_6`, `h2_2_bin_se_6`, `gamma_bin_6`, `gamma_bin_se_6`, `rg_bin_6`, `rg_bin_se_6`, `h2_1_bin_7`, `h2_1_bin_se_7`, `h2_2_bin_7`, `h2_2_bin_se_7`, `gamma_bin_7`, `gamma_bin_se_7`, `rg_bin_7`, `rg_bin_se_7`, `h2_1_bin_8`, `h2_1_bin_se_8`, `h2_2_bin_8`, `h2_2_bin_se_8`, `gamma_bin_8`, `gamma_bin_se_8`, `rg_bin_8`, `rg_bin_se_8`, `h2_1_bin_9`, `h2_1_bin_se_9`, `h2_2_bin_9`, `h2_2_bin_se_9`, `gamma_bin_9`, `gamma_bin_se_9`, `rg_bin_9`, `rg_bin_se_9`, `h2_1_bin_10`, `h2_1_bin_se_10`, `h2_2_bin_10`, `h2_2_bin_se_10`, `gamma_bin_10`, `gamma_bin_se_10`, `rg_bin_10`, `rg_bin_se_10`, `h2_1_bin_11`, `h2_1_bin_se_11`, `h2_2_bin_11`, `h2_2_bin_se_11`, `gamma_bin_11`, `gamma_bin_se_11`, `rg_bin_11`, `rg_bin_se_11`, `h2_1_bin_12`, `h2_1_bin_se_12`, `h2_2_bin_12`, `h2_2_bin_se_12`, `gamma_bin_12`, `gamma_bin_se_12`, `rg_bin_12`, `rg_bin_se_12`, `h2_1_bin_13`, `h2_1_bin_se_13`, `h2_2_bin_13`, `h2_2_bin_se_13`, `gamma_bin_13`, `gamma_bin_se_13`, `rg_bin_13`, `rg_bin_se_13`, `h2_1_bin_14`, `h2_1_bin_se_14`, `h2_2_bin_14`, `h2_2_bin_se_14`, `gamma_bin_14`, `gamma_bin_se_14`, `rg_bin_14`, `rg_bin_se_14`, `h2_1_bin_15`, `h2_1_bin_se_15`, `h2_2_bin_15`, `h2_2_bin_se_15`, `gamma_bin_15`, `gamma_bin_se_15`, `rg_bin_15`, `rg_bin_se_15`, `h2_1_bin_16`, `h2_1_bin_se_16`, `h2_2_bin_16`, `h2_2_bin_se_16`, `gamma_bin_16`, `gamma_bin_se_16`, `rg_bin_16`, `rg_bin_se_16`, `h2_1_bin_17`, `h2_1_bin_se_17`, `h2_2_bin_17`, `h2_2_bin_se_17`, `gamma_bin_17`, `gamma_bin_se_17`, `rg_bin_17`, `rg_bin_se_17`, `h2_1_bin_18`, `h2_1_bin_se_18`, `h2_2_bin_18`, `h2_2_bin_se_18`, `gamma_bin_18`, `gamma_bin_se_18`, `rg_bin_18`, `rg_bin_se_18`, `h2_1_bin_19`, `h2_1_bin_se_19`, `h2_2_bin_19`, `h2_2_bin_se_19`, `gamma_bin_19`, `gamma_bin_se_19`, `rg_bin_19`, `rg_bin_se_19`, `h2_1_bin_20`, `h2_1_bin_se_20`, `h2_2_bin_20`, `h2_2_bin_se_20`, `gamma_bin_20`, `gamma_bin_se_20`, `rg_bin_20`, `rg_bin_se_20`, `h2_1_bin_21`, `h2_1_bin_se_21`, `h2_2_bin_21`, `h2_2_bin_se_21`, `gamma_bin_21`, `gamma_bin_se_21`, `rg_bin_21`, `rg_bin_se_21`, `h2_1_bin_22`, `h2_1_bin_se_22`, `h2_2_bin_22`, `h2_2_bin_se_22`, `gamma_bin_22`, `gamma_bin_se_22`, `rg_bin_22`, `rg_bin_se_22`, `h2_1_bin_23`, `h2_1_bin_se_23`, `h2_2_bin_23`, `h2_2_bin_se_23`, `gamma_bin_23`, `gamma_bin_se_23`, `rg_bin_23`, `rg_bin_se_23`

</details>

<details><summary>T51: 10 columns</summary>

`pop`, `annot`, `out_prefix`, `sig1`, `sig2`, `rho_g`, `gamma_e`, `num_reps`, `seed`, `max_mem`

</details>

<details><summary>T52: 16 columns</summary>

`pop`, `method`, `h2`, `pol`, `rho_g`, `i`, `h2_1`, `h2_se_1`, `h2_2`, `h2_se_2`, `intercept_c`, `intercept_se`, `gamma_g`, `gamma_se`, `rg`, `rg_se`

</details>

<details><summary>T53: 6 columns</summary>

`Phenotype`, `Acronym`, `h2_ped`, `h2_ped_se`, `h2_wgs`, `h2_wgs_se`

</details>

<details><summary>T54: 10 columns</summary>

`Phenotype`, `Acr.`, `ICD10`, `Field`, `Dist.`, `Pop`, `N`, `Cases`, `Ctrls`, `Prev`

</details>

<details><summary>T55: 9 columns</summary>

`cohort`, `genotype_n`, `female_field31_n`, `male_field31_n`, `missing_field31_n`, `age_2023_mean`, `age_2023_sd`, `age_2023_min`, `age_2023_max`

</details>

<details><summary>T56: 10 columns</summary>

`cohort`, `source`, `release_group`, `trait`, `in_selected_enrichment_37`, `download_url`, `snps`, `n_min`, `n_median`, `n_max`

</details>

<details><summary>T57: 3 columns</summary>

`Sheet`, `Column`, `Definition`

</details>
