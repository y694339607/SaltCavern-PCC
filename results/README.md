# results/ · 逐样本结果

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

Per-sample and summary results backing every table in the manuscript.

| File | Content | Manuscript |
|---|---|---|
| `table4_per_sample.csv` | 7 arms × 150 test samples: CD₁, V_pred, V_field, V_ref, both error columns | Table 4 (test) |
| `test_ID1024_BM_common_per_sample.csv` | masked configuration vs ID-1024 on common support (test) | Table 5 |
| `nine_config_summary.csv` / `nine_config_per_sample.csv` | nine grid–operator combinations | Table 6 / Figure 7 |
| `val_main_per_sample.csv`, `val_order1_per_sample.csv`, `val_order2_per_sample.csv` | masked arm across the three validation runs | Table 8 |
| `repeats_per_sample.csv` | both arms × two data-order repeats (600 rows) | Table 7 |
| `three_run_endpoint.csv` | cavity-level endpoint of the three runs | Table 8 |
| `replication_check.csv` | reproduction of the test-side endpoint against the archived table | Table 5 |
| `placebo_summary.csv` / `placebo_init_per_sample.csv` | three placebo controls | Section 4.5 |
| `init_summary.csv` / `init_repeats_per_sample.csv` | initialization stratum (900 rows) | Section 4.5 |
| `density_strata_test.csv` / `density_strata_val_repeats.csv` | endpoint stratified by the ID-1024 valid azimuth-cell count | Section 4.4 |
| `slice_azimuth_counts.csv` | valid azimuth cells per slice (150 000 rows) | Section 4.4 |
| `multivalue_per_sample.csv` | radial multi-value geometry per sample (1350 rows) | Section 4.6 / Figure 9 |
| `sample_geometry_test.csv` | centroid, z range and observed z span per test sample | closure inputs |

Column headers are kept in Chinese to match the released working files. All geometric quantities are
in the recorded frame (`z_m = 0.3048 z` applied to obtain metric units).

`sample_geometry_test.csv` provides what the closure needs besides the clouds; together with
`predictions/` it allows the volumes in Table 4 to be recomputed without the complete dataset.

---

## 中文

论文各表所依据的逐样本与汇总结果。

| 文件 | 内容 | 对应论文 |
|---|---|---|
| `table4_per_sample.csv` | 7 个臂 × 150 个测试样本：CD₁、V_pred、V_field、V_ref 与两个误差列 | 表 4（测试） |
| `test_ID1024_BM_common_per_sample.csv` | 掩码配置与 ID-1024 在共同支撑上的对比（测试集） | 表 5 |
| `nine_config_summary.csv` / `nine_config_per_sample.csv` | 九种网格—算子组合 | 表 6 / 图 7 |
| `val_main_per_sample.csv`、`val_order1_per_sample.csv`、`val_order2_per_sample.csv` | 掩码臂在三次验证集运行中的结果 | 表 8 |
| `repeats_per_sample.csv` | 两个臂 × 两次数据顺序重复（600 行） | 表 7 |
| `three_run_endpoint.csv` | 三次运行的腔级主终点 | 表 8 |
| `replication_check.csv` | 测试集主终点与归档表的逐项复现核对 | 表 5 |
| `placebo_summary.csv` / `placebo_init_per_sample.csv` | 三项安慰剂对照 | §4.5 |
| `init_summary.csv` / `init_repeats_per_sample.csv` | 初始化维度（900 行） | §4.5 |
| `density_strata_test.csv` / `density_strata_val_repeats.csv` | 按 ID-1024 有效方位格数分层的主终点 | §4.4 |
| `slice_azimuth_counts.csv` | 逐切片有效方位格数（150 000 行） | §4.4 |
| `multivalue_per_sample.csv` | 逐样本径向多值几何（1350 行） | §4.6 / 图 9 |
| `sample_geometry_test.csv` | 每个测试样本的质心、z 范围与观测 z 跨度 | 闭格架输入 |

列名保留中文，以便与发布的工作文件一致。全部几何量处于记录坐标（乘 `z_m = 0.3048 z` 后即为
米制）。

`sample_geometry_test.csv` 提供闭格架除点云之外所需的几何量；配合 `predictions/` 即可在不访问
完整数据集的情况下重算表 4 的体积。
