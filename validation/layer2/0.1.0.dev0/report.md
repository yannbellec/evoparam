# Layer-2 validation, evoparam 0.1.0.dev0

Design `weekly`, 50 replicates per cell, 204 cells, 27.7 min.

## REQ tests on simulations

| test | REQs | status | s |
|---|---|---|---|
| evoparam.tests.models.test_engine_sim::test_req105_recall_and_precision | REQ-104 to REQ-112 | passed | 6.4 |
| evoparam.tests.models.test_engine_sim::test_req106_accuracy_at_high_snr | REQ-104 to REQ-112 | passed | 1.6 |
| evoparam.tests.models.test_engine_sim::test_req104_detection_threshold_changes_no_decision | REQ-104 to REQ-112 | passed | 5.2 |
| evoparam.tests.models.test_engine_sim::test_req107_req112_no_component_in_pure_noise[0.0] | REQ-104 to REQ-112 | passed | 0.1 |
| evoparam.tests.models.test_engine_sim::test_req107_req112_no_component_in_pure_noise[0.5] | REQ-104 to REQ-112 | passed | 0.1 |
| evoparam.tests.models.test_engine_sim::test_req107_req112_no_component_in_pure_noise[0.9] | REQ-104 to REQ-112 | passed | 0.8 |
| evoparam.tests.models.test_engine_sim::test_req107_whitening_is_what_controls_false_components | REQ-104 to REQ-112 | passed | 2.7 |
| evoparam.tests.models.test_engine_sim::test_req110_edge_components | REQ-104 to REQ-112 | passed | 0.3 |
| evoparam.tests.models.test_engine_sim::test_req111_resolution_and_req121_conditioning | REQ-104 to REQ-112 | passed | 0.6 |
| evoparam.tests.models.test_engine_sim::test_req116_coverage_of_jacobian_intervals | REQ-104 to REQ-112 | passed | 1.0 |
| evoparam.tests.models.test_engine_sim::test_req122_fixed_and_cross_validated_selection | REQ-104 to REQ-112 | passed | 3.0 |
| evoparam.tests.existence.test_existence_null::test_req130_type_one_error[max_t-0.0] | REQ-130, REQ-131 | passed | 5.2 |
| evoparam.tests.existence.test_existence_null::test_req130_type_one_error[max_t-0.9] | REQ-130, REQ-131 | passed | 4.2 |
| evoparam.tests.existence.test_existence_null::test_req130_type_one_error[cluster-0.0] | REQ-130, REQ-131 | passed | 13.7 |
| evoparam.tests.existence.test_existence_null::test_req130_type_one_error[cluster-0.9] | REQ-130, REQ-131 | passed | 13.8 |
| evoparam.tests.existence.test_existence_null::test_req136_corrections | REQ-130, REQ-131 | passed | 0.2 |
| evoparam.tests.existence.test_existence_power::test_req131_power_transient_and_sustained | REQ-132, REQ-133 | passed | 1.6 |
| evoparam.tests.existence.test_existence_power::test_req132_duration_and_onset | REQ-132, REQ-133 | passed | 0.3 |
| evoparam.tests.existence.test_existence_power::test_projection_duration_and_significance | REQ-132, REQ-133 | passed | 0.0 |
| evoparam.tests.existence.test_existence_power::test_req139_train_discordance | REQ-132, REQ-133 | passed | 0.1 |
| evoparam.tests.trials.test_trials_sim::test_req150_latency_amplitude_and_jitter[0.002] | REQ-150 to REQ-155 | passed | 2.7 |
| evoparam.tests.trials.test_trials_sim::test_req150_latency_amplitude_and_jitter[0.005] | REQ-150 to REQ-155 | passed | 2.7 |
| evoparam.tests.trials.test_trials_sim::test_req151_no_invented_jitter_and_realign_refused | REQ-150 to REQ-155 | passed | 8.8 |
| evoparam.tests.trials.test_trials_sim::test_req152_jitter_corrected_amplitude | REQ-150 to REQ-155 | passed | 0.8 |
| evoparam.tests.trials.test_trials_sim::test_req153_regularized_beats_free_at_low_snr | REQ-150 to REQ-155 | passed | 12.7 |
| evoparam.tests.trials.test_trials_sim::test_req154_dose_response_hill | REQ-150 to REQ-155 | passed | 0.5 |
| evoparam.tests.trials.test_trials_sim::test_req155_prestimulus_phase | REQ-150 to REQ-155 | passed | 45.2 |
| evoparam.tests.hierarchical.test_hier_eb::test_req170_p_response_discriminates_and_is_calibrated | REQ-170 to REQ-172, REQ-175 | passed | 36.9 |
| evoparam.tests.hierarchical.test_hier_eb::test_level_tables_and_summary | REQ-170 to REQ-172, REQ-175 | passed | 0.0 |
| evoparam.tests.hierarchical.test_hier_eb::test_req171_shrunken_latencies_beat_the_channel_fits | REQ-170 to REQ-172, REQ-175 | passed | 202.9 |
| evoparam.tests.hierarchical.test_hier_eb::test_req172_variance_components_by_eb | REQ-170 to REQ-172, REQ-175 | passed | 275.2 |
| evoparam.tests.hierarchical.test_hier_eb::test_req175_formula_effect_is_covered | REQ-170 to REQ-172, REQ-175 | passed | 284.3 |
| evoparam.tests.hierarchical.test_hier_eb::test_formula_errors | REQ-170 to REQ-172, REQ-175 | passed | 0.0 |
| evoparam.tests.hierarchical.test_hier_eb::test_req176_eb_model_time_per_subject | REQ-170 to REQ-172, REQ-175 | passed | 0.0 |
| evoparam.tests.hierarchical.test_hier_eb::test_req176_eb_time_per_subject | REQ-170 to REQ-172, REQ-175 | skipped | 0.0 |
| evoparam.tests.hierarchical.test_hier_eb::test_req177_no_new_dependency_for_eb | REQ-170 to REQ-172, REQ-175 | passed | 0.0 |
| evoparam.tests.hierarchical.test_hier_eb::test_one_subject_drops_the_subject_level | REQ-170 to REQ-172, REQ-175 | passed | 4.7 |
| evoparam.tests.hierarchical.test_hier_eb::test_channel_with_few_trusted_trials_is_prior_dominated | REQ-170 to REQ-172, REQ-175 | passed | 7.4 |
| evoparam.tests.hierarchical.test_hier_eb::test_single_trialset_with_subject_column | REQ-170 to REQ-172, REQ-175 | passed | 3.7 |
| evoparam.tests.hierarchical.test_hier_eb::test_round_trip | REQ-170 to REQ-172, REQ-175 | passed | 6.0 |
| evoparam.tests.tf.test_tf_sim::test_req212_footprint_removes_the_sharp_n1_confound | REQ-212 to REQ-215 | passed | 17.0 |
| evoparam.tests.tf.test_tf_sim::test_req213_zone_recovery_of_induced_gamma_bursts | REQ-212 to REQ-215 | passed | 122.1 |
| evoparam.tests.tf.test_tf_sim::test_narrow_band_twenty_percent_is_below_pixel_resolution | REQ-212 to REQ-215 | passed | 1.4 |
| evoparam.tests.tf.test_tf_sim::test_req214_fdr_type_one_error | REQ-212 to REQ-215 | passed | 13.5 |
| evoparam.tests.tf.test_tf_sim::test_req214_cluster_family_wise_error | REQ-212 to REQ-215 | passed | 27.3 |
| evoparam.tests.tf.test_tf_sim::test_req214_tfce_family_wise_error | REQ-212 to REQ-215 | passed | 79.3 |
| evoparam.tests.tf.test_tf_sim::test_req215_single_trial_normalisation_resists_a_noisy_trial | REQ-212 to REQ-215 | passed | 2.9 |
| evoparam.tests.presets.test_presets_sim::test_req276_erna_frequency_and_first_peak | REQ-276, REQ-277 | passed | 1.4 |
| evoparam.tests.presets.test_presets_sim::test_req277_sep_n20_latency | REQ-276, REQ-277 | passed | 2.1 |
| evoparam.tests.presets.test_presets_sim::test_sep_with_coloured_noise_is_a_documented_limit | REQ-276, REQ-277 | passed | 2.3 |

## Grid

| modality | snr | K | sep | width | trials | jitter | noise | F1 | latency MAE (samples) | amp. bias | false/waveform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| erp | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.03 | 2.64 | 0.95 | 0.00 |
| erp | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.66 | 1.21 | 0.19 | 0.00 |
| erp | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.98 | 0.85 | 0.05 | 0.02 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.45 | 0.02 | 0.00 |
| erp | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.20 | 0.01 | 0.00 |
| erp | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.10 | -0.00 | 0.00 |
| erp | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.00 |
| erp | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 0.99 | 0.54 | -0.00 | 0.02 |
| erp | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.97 | 0.50 | 0.29 | 0.06 |
| erp | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.54 | 3.35 | -0.40 | 0.04 |
| erp | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.99 | 2.25 | -0.02 | 0.00 |
| erp | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.94 | 0.99 | 0.36 | 0.12 |
| erp | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 0.97 | 0.37 | 0.15 | 0.06 |
| erp | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 1.00 | 0.32 | 0.01 | 0.00 |
| erp | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 1.00 | 0.70 | 0.00 | 0.00 |
| erp | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.98 | 0.62 | 0.02 | 0.06 |
| erp | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 1.00 | 0.54 | 0.01 | 0.00 |
| erp | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 1.00 | 0.37 | 0.00 | 0.00 |
| erp | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 1.00 | 0.41 | 0.00 | 0.00 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 1.00 | 0.48 | -0.01 | 0.00 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 0.97 | 1.01 | -0.05 | 0.06 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.90 | 1.72 | -0.16 | 0.20 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 0.99 | 1.01 | 0.07 | 0.02 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.80 | 0.94 | 0.13 | 0.24 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.81 | 1.03 | 0.19 | 0.48 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.75 | 0.66 | 0.18 | 0.68 |
| ccep | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.01 | 0.24 | 1.83 | 0.00 |
| ccep | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.57 | 1.50 | 0.24 | 0.04 |
| ccep | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.99 | 1.04 | 0.03 | 0.02 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.91 | -0.01 | 0.00 |
| ccep | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.80 | -0.00 | 0.02 |
| ccep | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.84 | 1.20 | 0.09 | 0.82 |
| ccep | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.00 |
| ccep | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.75 | 0.01 | 0.00 |
| ccep | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.70 | 1.11 | -0.20 | 0.54 |
| ccep | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.38 | 3.42 | -0.47 | 0.08 |
| ccep | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.99 | 2.00 | -0.12 | 0.02 |
| ccep | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.88 | 1.45 | 0.67 | 0.24 |
| ccep | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 0.99 | 0.56 | 0.01 | 0.02 |
| ccep | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 1.00 | 0.54 | -0.02 | 0.00 |
| ccep | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 1.00 | 1.62 | -0.01 | 0.00 |
| ccep | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.99 | 1.07 | 0.06 | 0.04 |
| ccep | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 1.00 | 0.87 | -0.00 | 0.00 |
| ccep | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 1.00 | 0.82 | -0.00 | 0.00 |
| ccep | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 1.00 | 0.83 | -0.00 | 0.00 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 0.98 | 0.84 | -0.03 | 0.04 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 1.00 | 1.05 | -0.11 | 0.00 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.82 | 1.89 | -0.13 | 0.36 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 0.98 | 1.25 | 0.12 | 0.04 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.78 | 0.86 | 0.21 | 0.14 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.88 | 0.75 | 0.07 | 0.20 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.90 | 0.36 | -0.05 | 0.56 |
| tep | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| tep | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.32 | 1.64 | 0.35 | 0.02 |
| tep | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.97 | 1.23 | 0.06 | 0.02 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.56 | -0.00 | 0.00 |
| tep | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.29 | -0.00 | 0.00 |
| tep | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.17 | -0.00 | 0.00 |
| tep | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.00 |
| tep | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.63 | -0.02 | 0.00 |
| tep | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.90 | 2.11 | 0.39 | 0.44 |
| tep | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.23 | 4.84 | 0.00 | 0.04 |
| tep | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.84 | 4.09 | 0.09 | 0.30 |
| tep | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.84 | 2.02 | 0.75 | 0.32 |
| tep | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 1.00 | 0.55 | 0.00 | 0.00 |
| tep | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 1.00 | 0.67 | -0.01 | 0.00 |
| tep | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 1.00 | 0.57 | -0.00 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.98 | 1.19 | 0.09 | 0.06 |
| tep | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 1.00 | 0.65 | 0.01 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 1.00 | 0.59 | 0.00 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 1.00 | 0.56 | 0.00 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 1.00 | 0.80 | -0.03 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 0.96 | 1.27 | -0.12 | 0.08 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.84 | 4.10 | -0.01 | 0.30 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 1.00 | 1.20 | -0.01 | 0.00 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.91 | 2.29 | 0.06 | 0.08 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.93 | 1.57 | 0.05 | 0.06 |
| tep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.97 | 0.49 | 0.00 | 0.12 |
| erna | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| erna | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.08 | 2.36 | 0.08 | 0.00 |
| erna | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.40 | 1.92 | -0.10 | 0.00 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.84 | 1.81 | -0.05 | 0.00 |
| erna | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.92 | 1.79 | 0.36 | 0.18 |
| erna | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.85 | 1.78 | 0.70 | 0.44 |
| erna | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.00 |
| erna | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 0.69 | 1.83 | 0.06 | 0.02 |
| erna | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.80 | 1.87 | -0.04 | 0.00 |
| erna | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.65 | 1.18 | 0.36 | 0.00 |
| erna | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.68 | 1.91 | 0.42 | 0.00 |
| erna | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.15 | 2.36 | 0.08 | 0.00 |
| erna | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 0.21 | 0.96 | 0.12 | 0.18 |
| erna | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 0.03 | 1.56 | -0.08 | 0.02 |
| erna | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 0.81 | 1.99 | -0.02 | 0.04 |
| erna | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.79 | 2.13 | -0.04 | 0.02 |
| erna | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 0.69 | 1.93 | -0.09 | 0.02 |
| erna | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 0.82 | 2.02 | -0.30 | 0.00 |
| erna | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 0.82 | 1.65 | -0.06 | 0.00 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 0.70 | 2.43 | -0.52 | 0.00 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 0.22 | 3.35 | -1.28 | 0.02 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.04 | 1.97 | -0.51 | 0.02 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 0.87 | 1.88 | -0.03 | 0.04 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.79 | 1.91 | -0.14 | 0.06 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.85 | 2.22 | -0.18 | 0.04 |
| erna | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.85 | 2.10 | -0.44 | 0.16 |
| ecap | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.03 | 0.65 | 1.91 | 0.00 |
| ecap | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.35 | 2.08 | 0.87 | 0.02 |
| ecap | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.71 | 1.50 | 0.49 | 0.04 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.92 | 1.34 | 0.22 | 0.10 |
| ecap | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.89 | 0.68 | -0.02 | 0.54 |
| ecap | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.88 | 0.26 | -0.01 | 0.58 |
| ecap | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.00 |
| ecap | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 0.84 | 0.70 | 0.05 | 0.14 |
| ecap | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.56 | 3.30 | 0.53 | 0.76 |
| ecap | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.62 | 1.38 | 0.17 | 0.02 |
| ecap | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.70 | 0.67 | 0.31 | 0.06 |
| ecap | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.84 | 1.41 | 0.31 | 0.22 |
| ecap | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 0.72 | 0.54 | 0.05 | 0.06 |
| ecap | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 0.95 | 0.79 | -0.00 | 0.02 |
| ecap | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 0.86 | 4.93 | -0.49 | 0.28 |
| ecap | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.90 | 1.28 | 0.21 | 0.12 |
| ecap | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 0.91 | 1.28 | 0.04 | 0.16 |
| ecap | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 0.90 | 1.09 | 0.08 | 0.22 |
| ecap | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 0.92 | 1.26 | 0.05 | 0.16 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 0.91 | 1.50 | 0.14 | 0.08 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 0.79 | 1.91 | 0.39 | 0.14 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.61 | 2.59 | 0.43 | 0.24 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 0.86 | 1.21 | 0.08 | 0.52 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.63 | 2.07 | 0.30 | 1.28 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.77 | 1.65 | 0.11 | 0.76 |
| ecap | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.60 | 1.83 | 0.22 | 1.32 |
| animal | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.01 | 2.59 | 2.38 | 0.02 |
| animal | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.51 | 1.55 | 0.16 | 0.06 |
| animal | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.98 | 1.28 | 0.05 | 0.00 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.85 | 0.01 | 0.00 |
| animal | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.45 | 0.00 | 0.00 |
| animal | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.25 | 0.00 | 0.00 |
| animal | 5 | 0 | 2.0 | 12 | 20 | 0.0 | white |  |  |  | 0.02 |
| animal | 5 | 1 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.84 | 0.03 | 0.00 |
| animal | 5 | 3 | 2.0 | 12 | 20 | 0.0 | white | 0.83 | 1.15 | 0.11 | 0.46 |
| animal | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.56 | 2.42 | -0.08 | 0.08 |
| animal | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.82 | 1.41 | 0.01 | 0.12 |
| animal | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.90 | 1.41 | 0.48 | 0.20 |
| animal | 5 | 2 | 2.0 | 3 | 20 | 0.0 | white | 0.87 | 0.39 | 0.08 | 0.20 |
| animal | 5 | 2 | 2.0 | 6 | 20 | 0.0 | white | 0.97 | 0.49 | 0.04 | 0.04 |
| animal | 5 | 2 | 2.0 | 25 | 20 | 0.0 | white | 1.00 | 1.16 | 0.01 | 0.00 |
| animal | 5 | 2 | 2.0 | 12 | 5 | 0.0 | white | 0.97 | 0.92 | 0.03 | 0.08 |
| animal | 5 | 2 | 2.0 | 12 | 10 | 0.0 | white | 1.00 | 0.84 | 0.01 | 0.00 |
| animal | 5 | 2 | 2.0 | 12 | 40 | 0.0 | white | 1.00 | 0.78 | 0.02 | 0.02 |
| animal | 5 | 2 | 2.0 | 12 | 100 | 0.0 | white | 1.00 | 0.81 | 0.03 | 0.00 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.1 | white | 1.00 | 0.72 | -0.01 | 0.00 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.25 | white | 0.96 | 1.48 | -0.10 | 0.08 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.5 | white | 0.83 | 2.31 | -0.06 | 0.34 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.5 | 0.97 | 0.99 | 0.06 | 0.04 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.0 | ar1:0.9 | 0.74 | 0.87 | 0.13 | 0.12 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:1 | 0.81 | 1.05 | 0.15 | 0.18 |
| animal | 5 | 2 | 2.0 | 12 | 20 | 0.0 | pink:2 | 0.85 | 0.58 | 0.09 | 0.52 |
| erp | 1 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| erp | 1 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| erp | 1 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.02 |
| erp | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.04 | 0.76 | 1.14 | 0.00 |
| erp | 2 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| erp | 2 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.04 | 2.96 | 0.37 | 0.02 |
| erp | 2 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.43 | 1.52 | 0.32 | 0.00 |
| erp | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.68 | 0.94 | 0.18 | 0.00 |
| erp | 3 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.05 | 3.68 | -0.01 | 0.00 |
| erp | 3 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.42 | 2.98 | -0.02 | 0.00 |
| erp | 3 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.91 | 1.24 | 0.29 | 0.18 |
| erp | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.99 | 0.87 | 0.04 | 0.00 |
| erp | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.43 | 3.48 | -0.42 | 0.08 |
| erp | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.97 | 2.07 | -0.24 | 0.06 |
| erp | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.93 | 1.22 | 0.40 | 0.14 |
| erp | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.48 | -0.01 | 0.00 |
| erp | 10 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.99 | 2.94 | -0.37 | 0.02 |
| erp | 10 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.97 | 1.81 | -0.14 | 0.06 |
| erp | 10 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.91 | 0.97 | 0.65 | 0.18 |
| erp | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.22 | 0.00 | 0.00 |
| erp | 20 | 2 | 0.3 | 12 | 20 | 0.0 | white | 1.00 | 2.41 | -0.42 | 0.00 |
| erp | 20 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.97 | 1.38 | -0.27 | 0.06 |
| erp | 20 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.99 | 0.55 | 0.15 | 0.02 |
| erp | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.10 | 0.00 | 0.00 |
| ccep | 1 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| ccep | 1 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| ccep | 1 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| ccep | 1 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.01 | 0.30 | 1.72 | 0.00 |
| ccep | 2 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| ccep | 2 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.03 | 2.25 | 0.32 | 0.00 |
| ccep | 2 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.36 | 1.40 | 0.40 | 0.00 |
| ccep | 2 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.71 | 1.19 | 0.24 | 0.04 |
| ccep | 3 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.00 |  |  | 0.00 |
| ccep | 3 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.40 | 2.65 | 0.03 | 0.04 |
| ccep | 3 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.89 | 1.67 | 0.51 | 0.20 |
| ccep | 3 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.97 | 1.23 | 0.11 | 0.06 |
| ccep | 5 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.38 | 3.51 | -0.38 | 0.02 |
| ccep | 5 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.96 | 2.07 | -0.22 | 0.02 |
| ccep | 5 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.91 | 1.60 | 0.44 | 0.18 |
| ccep | 5 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.90 | -0.00 | 0.00 |
| ccep | 10 | 2 | 0.3 | 12 | 20 | 0.0 | white | 0.98 | 2.59 | -0.44 | 0.04 |
| ccep | 10 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.99 | 1.77 | -0.10 | 0.02 |
| ccep | 10 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.83 | 1.32 | 0.74 | 0.34 |
| ccep | 10 | 2 | 2.0 | 12 | 20 | 0.0 | white | 1.00 | 0.81 | -0.00 | 0.00 |
| ccep | 20 | 2 | 0.3 | 12 | 20 | 0.0 | white | 1.00 | 2.17 | -0.47 | 0.00 |
| ccep | 20 | 2 | 0.5 | 12 | 20 | 0.0 | white | 0.99 | 1.35 | -0.09 | 0.02 |
| ccep | 20 | 2 | 1.0 | 12 | 20 | 0.0 | white | 0.74 | 2.01 | 1.11 | 0.52 |
| ccep | 20 | 2 | 2.0 | 12 | 20 | 0.0 | white | 0.84 | 1.23 | 0.13 | 0.78 |
