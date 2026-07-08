# Artifact 迁移清单

本文档由 `configs/artifact_layout.yaml` 和 `configs/profile_sources.yaml` 生成，用于核对本地源文件如何进入 Hugging Face artifact bundle。

机器可读版本是 `configs/artifact_manifest.yaml`。公开仓库不包含 `docs/` 下的维护文档。

## Bundle 汇总

| Bundle | Hugging Face 文件 | 状态 | 文件数 | 大小 | 解压后的目标内容 |
| --- | --- | --- | --- | --- | --- |
| profile_sources | bundles/profile_sources_v1.tar.zst | missing_required_sources | 1 | 73.96 MiB | data/profile_sources/chronos/<br>data/profile_sources/moirai_timesfm/ |
| tsfm_results_stage20 | bundles/stage20_tsfm_results_v1.tar.zst | ready | 69 | 2.05 GiB | artifacts/tsfm_results/results_csv/TSFM/cl_512/ |
| tsrouter_core_stage20 | bundles/stage20_tsrouter_core_v1.tar.zst | missing_required_sources | 3 | 451.69 KiB | artifacts/tsrouter_core/results_artifacts/TSRouter/<br>artifacts/tsrouter_core/results_artifacts/caches/Sampled_repr_pool/<br>artifacts/tsrouter_core/results_csv/TSRouter/ |
| baselines_stage20 | bundles/stage20_baselines_v1.tar.zst | missing_required_sources | 2 | 397.50 KiB | artifacts/baseline_results/results_artifacts/baselines/selectors/<br>artifacts/baseline_results/results_csv/baselines/ |
| task_cache_stage20 | bundles/stage20_task_cache_v1.tar.zst | missing_required_sources | 0 | 0 B | artifacts/task_sample_cache/results_artifacts/caches/GE_test_sample/ |
| tables_figures_stage20 | bundles/stage20_tables_figures_v1.tar.zst | ready | 43 | 4.97 MiB | artifacts/tables_figures/Dataset/<br>artifacts/tables_figures/results_csv/TSRouter/vldb/<br>artifacts/tables_figures/figs/vldb_results/stage20/ |

## 源文件汇总

### profile_sources

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| chronos | Dataset/Repr_data_sourse/c62.tsf | true | present | 1 | 73.96 MiB | data/profile_sources/chronos/chronos_profile_source.tsf |
| domain_energy | Dataset/Repr_data_sourse/energy_num1w_len992_sd2029_std.npy | true | missing | 0 | 0 B | data/profile_sources/moirai_timesfm/domain_energy.npy |
| domain_nature | Dataset/Repr_data_sourse/nature_num1w_len992_sd2029_std.npy | true | missing | 0 | 0 B | data/profile_sources/moirai_timesfm/domain_nature.npy |
| domain_healthcare | Dataset/Repr_data_sourse/healthcare_num1w_len992_sd2029_std.npy | true | missing | 0 | 0 B | data/profile_sources/moirai_timesfm/domain_healthcare.npy |
| domain_web_cloudops | Dataset/Repr_data_sourse/web_cloudops_num1w_len992_sd2029_std.npy | true | missing | 0 | 0 B | data/profile_sources/moirai_timesfm/domain_web_cloudops.npy |
| domain_sales | Dataset/Repr_data_sourse/sales_num1w_len992_sd2029_std.npy | true | missing | 0 | 0 B | data/profile_sources/moirai_timesfm/domain_sales.npy |

### tsfm_results_stage20

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| tsfm_results_stage20:1 | results_csv/TSFM/cl_512/ | true | present | 69 | 2.05 GiB | artifacts/tsfm_results/results_csv/TSFM |

### tsrouter_core_stage20

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| tsrouter_core_stage20:1 | results_artifacts/TSRouter/Sampled_repr_anchor/StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:2 | results_artifacts/TSRouter/Sampled_repr_anchor/StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_meta.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:3 | results_artifacts/caches/Sampled_repr_pool/c-e-n-h-w-s_x3000_in512_pl480_std_sd2029_awsfirst_pool.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/caches |
| tsrouter_core_stage20:4 | results_artifacts/caches/Sampled_repr_pool/c-e-n-h-w-s_x3000_in512_pl480_std_sd2029_awsfirst_pool_meta.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/caches |
| tsrouter_core_stage20:5 | results_artifacts/TSRouter/Model_zoo_repr/stage20/weight_zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:6 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:7 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_subset_assign.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:8 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_model_manifest.json | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:9 | results_artifacts/TSRouter/Model_zoo_repr/stage20/weight_zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_rfast.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:10 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_rfast.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:11 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_rfast_subset_assign.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:12 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4C_repr-all_sub0_1.0_rfast_model_manifest.json | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:13 | results_artifacts/TSRouter/Model_zoo_repr/stage20/weight_zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4M_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:14 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4M_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:15 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4M_repr-all_sub0_1.0_subset_assign.pkl | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:16 | results_artifacts/TSRouter/Model_zoo_repr/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v4M_repr-all_sub0_1.0_model_manifest.json | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_artifacts/TSRouter |
| tsrouter_core_stage20:17 | results_csv/TSRouter/Repr_forward/StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025_*results.csv | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_csv/TSRouter |
| tsrouter_core_stage20:18 | results_csv/TSRouter/Repr_forward/c-e-n-h-w-s_x3000_in512_pl480_std_sd2029_awsfirst_pool_sf2025_*results.csv | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_csv/TSRouter |
| tsrouter_core_stage20:19 | results_csv/TSRouter/Model_zoo_repr/ | false | present | 3 | 451.69 KiB | artifacts/tsrouter_core/results_csv/TSRouter |
| tsrouter_core_stage20:20 | results_csv/TSRouter/Selector_results/stage*/ | true | missing | 0 | 0 B | artifacts/tsrouter_core/results_csv/TSRouter |

### baselines_stage20

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| autoforecast_step3_weight | results_artifacts/baselines/selectors/AutoForecast_Select/stage20/weight_zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoForecast_Select/stage20/weight_zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt.pkl |
| autoforecast_step3_model_repr | results_artifacts/baselines/selectors/AutoForecast_Select/stage20/zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoForecast_Select/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt.pkl |
| autoforecast_step3_manifest | results_artifacts/baselines/selectors/AutoForecast_Select/stage20/zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt_model_manifest.json | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoForecast_Select/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_afgbdt_model_manifest.json |
| autoxpcr_step3_weight | results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/weight_zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/weight_zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt.pkl |
| autoxpcr_step3_model_repr | results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt.pkl |
| autoxpcr_step3_manifest | results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/zoo20-23_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt_model_manifest.json | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/AutoXPCR_Select/stage20/zoo20-20_StatsRandomFourier_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v7C_repr-all_sub0_1.0_rfast_afgbdt_model_manifest.json |
| simplets_step3_weight | results_artifacts/baselines/selectors/SimpleTS_Select/stage20/weight_zoo20-23_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/SimpleTS_Select/stage20/weight_zoo20-20_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0.pkl |
| simplets_step3_model_repr | results_artifacts/baselines/selectors/SimpleTS_Select/stage20/zoo20-23_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0.pkl | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/SimpleTS_Select/stage20/zoo20-20_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0.pkl |
| simplets_step3_manifest | results_artifacts/baselines/selectors/SimpleTS_Select/stage20/zoo20-23_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0_model_manifest.json | true | missing | 0 | 0 B | artifacts/baseline_results/results_artifacts/baselines/selectors/SimpleTS_Select/stage20/zoo20-20_TS2Vec_512to256_pl480_std_sd2029_se2025_c-e-n-h-w-s_x3000_awsfirst_kmeans-n_sf2025-v6C_repr-all_sub0_1.0_model_manifest.json |
| baselines_stage20:10 | results_csv/baselines/vldb/ | false | missing | 0 | 0 B | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:11 | results_csv/baselines/selectors/Task_probe_Select/ | true | present | 1 | 133.35 KiB | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:12 | results_csv/baselines/selectors/AutoForecast_Select/step3_insert_timing.csv | true | present | 1 | 264.15 KiB | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:13 | results_csv/baselines/selectors/AutoXPCR_Select/step3_insert_timing.csv | true | missing | 0 | 0 B | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:14 | results_csv/baselines/selectors/SimpleTS_Select/step3_insert_timing.csv | true | missing | 0 | 0 B | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:15 | results_csv/baselines/selectors/Rank_Truth_Select/ | false | missing | 0 | 0 B | artifacts/baseline_results/results_csv/baselines |
| baselines_stage20:16 | results_csv/TSRouter/vldb/logs/ | false | missing | 0 | 0 B | artifacts/baseline_results/results_csv/TSRouter/vldb |

### task_cache_stage20

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| task_cache_stage20:1 | results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl | true | missing | 0 | 0 B | artifacts/task_sample_cache/results_artifacts/caches |
| task_cache_stage20:2 | results_artifacts/caches/GE_test_sample/cl512_n20_std_wseven_ss2025.pkl.meta.json | true | missing | 0 | 0 B | artifacts/task_sample_cache/results_artifacts/caches |

### tables_figures_stage20

| Source | 源模式 | 必须 | 状态 | 文件数 | 大小 | 发布路径 |
| --- | --- | --- | --- | --- | --- | --- |
| tables_figures_stage20:1 | Dataset/channel_meta.csv | true | present | 1 | 36.31 KiB |  |
| tables_figures_stage20:2 | Dataset/channel_meta_with_real_rank.csv | true | present | 1 | 942.96 KiB |  |
| tables_figures_stage20:3 | Dataset/dataset_properties.json | true | present | 1 | 2.23 KiB |  |
| tables_figures_stage20:4 | results_csv/TSRouter/vldb/tables/ | true | present | 13 | 1.17 MiB | artifacts/tables_figures/results_csv/TSRouter/vldb |
| tables_figures_stage20:5 | figs/vldb_results/stage20/ | true | present | 27 | 2.84 MiB | artifacts/tables_figures/figs/vldb_results |

## 迁移流程

1. 在服务器上核对每个 bundle 的源模式只命中论文采用参数。
2. 使用 `scripts/stage_local_artifacts.py` 本地暂存，或按 `configs/artifact_layout.yaml` 的 `release_path` 手工迁移到同样结构。
3. 使用 `scripts/pack_hf_artifacts.py` 生成压缩包，并上传 `bundles/*.tar.zst`、`manifest.json`、`checksums.sha256` 与 Hugging Face Dataset Card。
4. 下载或解压后运行 artifact check、backend prepare 和 workflow 测试。

## 校验策略

本地迁移清单可以只对小文件计算 checksum。公开发布时必须为每个压缩包和 `manifest.json` 提供 checksum。
