# results/ · 逐样本结果

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

Per-sample and summary results.

| File | Content |
|---|---|
| `table4_per_sample.csv` | 7 methods × 150 test samples: CD₁, V_pred, V_field, V_ref, both error columns |
| `id_baseline_per_sample.csv` | ID-1024 baseline on the 150 validation samples: V_field, V_id1024, N_slice, error, CD₁ (original and metric frame) |
| `test_id1024_masked_common_per_sample.csv` | masked configuration vs ID-1024 on common support (test) |
| `nine_config_summary.csv` / `nine_config_per_sample.csv` | nine grid–operator combinations |
| `val_main_per_sample.csv`, `val_order1_per_sample.csv`, `val_order2_per_sample.csv` | masked configuration across the three validation runs |
| `repeats_per_sample.csv` | both configurations × two data-order repeats (600 rows) |
| `three_run_endpoint.csv` | cavity-level endpoint of the three runs |
| `replication_check.csv` | reproduction of the test-side endpoint against the archived table |
| `negative_control_summary.csv` / `negative_control_init_per_sample.csv` | three negative controls |
| `init_summary.csv` / `init_repeats_per_sample.csv` | initialization stratum (900 rows) |
| `density_strata_test.csv` / `density_strata_val_repeats.csv` | endpoint stratified by the ID-1024 valid azimuth-cell count |
| `slice_azimuth_counts.csv` | valid azimuth cells per slice (150 000 rows) |
| `multivalue_per_sample.csv` | radial multi-value geometry per sample (1350 rows) |
| `sample_geometry_test.csv` | centroid, z range and observed z span per test sample |

Column headers are kept in Chinese to match the released working files. All geometric quantities are
in the recorded frame (`z_m = 0.3048 z` applied to obtain metric units).

`sample_geometry_test.csv` provides the per-sample closure inputs; together with `predictions/` the
volumes can be recomputed without the complete dataset.

---

## 中文

逐样本与汇总结果。

| 文件 | 内容 |
|---|---|
| `table4_per_sample.csv` | 7 种方法 × 150 个测试样本：CD₁、V_pred、V_field、V_ref 与两个误差列 |
| `id_baseline_per_sample.csv` | ID-1024 基线在 150 个验证样本上的体积、切片数、误差与 CD₁（原坐标与米制） |
| `test_id1024_masked_common_per_sample.csv` | 掩码配置与 ID-1024 在共同支撑上的对比（测试集） |
| `nine_config_summary.csv` / `nine_config_per_sample.csv` | 九种网格—算子组合 |
| `val_main_per_sample.csv`、`val_order1_per_sample.csv`、`val_order2_per_sample.csv` | 掩码配置在三次验证集运行中的结果 |
| `repeats_per_sample.csv` | 两个配置 × 两次数据顺序重复（600 行） |
| `three_run_endpoint.csv` | 三次运行的腔级终点 |
| `replication_check.csv` | 测试集终点与归档表的逐项复现核对 |
| `negative_control_summary.csv` / `negative_control_init_per_sample.csv` | 三项阴性对照 |
| `init_summary.csv` / `init_repeats_per_sample.csv` | 初始化维度（900 行） |
| `density_strata_test.csv` / `density_strata_val_repeats.csv` | 按 ID-1024 有效方位格数分层的终点 |
| `slice_azimuth_counts.csv` | 逐切片有效方位格数（150 000 行） |
| `multivalue_per_sample.csv` | 逐样本径向多值几何（1350 行） |
| `sample_geometry_test.csv` | 每个测试样本的质心、z 范围与观测 z 跨度 |

列名保留中文，以便与发布的工作文件一致。全部几何量处于记录坐标（乘 `z_m = 0.3048 z` 后即为
米制）。

`sample_geometry_test.csv` 提供逐样本的闭格架输入；配合 `predictions/` 即可在不访问完整数据集的
情况下重算体积。
