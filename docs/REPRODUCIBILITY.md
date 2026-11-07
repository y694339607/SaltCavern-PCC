# Reproducibility map (manuscript ↔ repository scripts) · 复现对照表（论文 ↔ 仓库脚本）

> Bilingual document. Tables carry bilingual headers and are given once.
> 中英双语；表格采用双语表头，只列一次。
>
> Conventions · 通用约定: (1) scripts resolve inputs through `SCC_ROOT` (default `/data`), see
> `docs/DATASET.md` (脚本通过 `SCC_ROOT` 定位输入，见 `docs/DATASET.md`); (2) geometric metrics are
> reported in the post-map metric frame, volume-class metrics are invariant to the map (几何指标在映射后
> 米制口径报告，体积类与该映射无关); (3) seeds: initialization 20260921, data/order 20260921 and
> 20261001, data-order repeats 20261101 and 20261102, statistics 20261103 (随机源：初始化 20260921，
> 数据/顺序 20260921 与 20261001，顺序重复 20261101 与 20261102，统计 20261103).

## 一、Data build chain · 数据构建链 (manuscript §3.1–§3.4, Figures 1–4)

| Step · 环节 | Script · 脚本 | Note · 说明 |
|---|---|---|
| Leaching configuration · 造腔配置修订 | `code/datagen/configure_leaching.py` | injection depth 25–35 m, blank radius 0.41–0.57 ft (注水点 25–35 m、blank 0.41–0.57 ft) |
| Batch simulation · Sansmic 批量仿真 | `code/datagen/run_simulations.py` | batching and stage management (批量运行与阶段管理) |
| Profile checks · 剖面健康检查 | `code/datagen/sim_profile_check.py`, `code/datagen/gt_slender_check.py` | bottom-undissolved statistics, slender-band check (底部未溶段统计、细长带核查) |
| Cavity replacement · 腔体替换/复核 | `code/datagen/replace_unstable_cavities.py`, `bottom_check.py`, `verify_build.py` | substitution record and numerical validity (替换记录与数值有效性复核) |
| Source-profile volume · 源剖面体积复算 | `code/evaluation/cavity_source_volume.py` | 500-cavity source volumes (§3.2; 500 腔源剖面体积) |
| Missingness synthesiser · 缺失合成器 | `code/datagen/missing_synth.py` | four mechanisms + angular limit, per-cavity calibration (§3.4, Table 2, Figure 4) |
| Dataset assembly · 数据集组装/校验 | `code/datagen/build_dataset.py`, `code/evaluation/dataset_precheck.py` | pairing, counts, naming, coordinate range, CD₁ regression, SHA manifest |
| Coordinate map · 坐标缩放（发布口径） | `code/evaluation/rescale_coordinates.py` | z × 0.3048, float64 per-point check (§3.1) |
| Asset integrity · 完整性校验 | `code/datagen/verify_assets.py` | SHA-256 verification of archives |

LHS sampling is performed during the build; the final 500 TOML configurations and the radius table are
released with the data (see `docs/DATASET.md`). · LHS 采样在构建阶段执行；最终 500 份 TOML 配置与
半径表随数据包发布。

## 二、Training · 训练 (manuscript §3.5, Figure 5 left)

| Arm · 臂 | Configuration · 配置 | Entry point · 入口 |
|---|---|---|
| Masked supervision (`adapointr_masked`, ep179) · 掩码配置 | `configs/pc_v2_pilot_bm.yaml` | `code/models/train_pilot.py` |
| Full-region supervision (`adapointr_full`, ep169) · 全区域配置 | `configs/ms_e2_fixedinit.yaml` (`band_mask: false`); see `configs/cfg_e2_from_ckpt.yaml` for warm-start | `code/models/train_pilot.py` |
| PoinTr arm · PoinTr 臂 | `configs/pc_v2_pilot_pointr.yaml` | `code/models/train_pilot.py` (includes the PoinTr compat unpack) |
| PCN | argparse, no YAML | `code/baselines/pcn/train.py` (self-contained re-implementation, Chamfer/EMD CUDA extensions) |
| Data-order repeats · 顺序重复 | `configs/ms_bm_fixedinit.yaml`, `configs/ms_e2_fixedinit.yaml` | `code/models/train_pilot.py` |
| Blind-window ablations w30/w100 · 窗口臂 | `configs/pc_v2_pilot_bm_w30.yaml`, `configs/pc_v2_pilot_bm_w100.yaml` | same entry; not reported in the manuscript (论文未报告) |

Own components · 自有组件: `code/models/paired_data/` (paired dataset and transforms),
`code/models/losses/adapt_modules.py` (masked two-sided reconstruction term),
`code/models/completion_validator.py`. The backbone is AdaPoinTr and is not vendored (see
`third_party/`). · 骨干为 AdaPoinTr，不入库。

## 三、Baselines and export · 基线与导出 (manuscript §3.5, §3.7)

| Object · 对象 | Script · 脚本 |
|---|---|
| Rule-based baselines (`cylnn` / `spline` / `idw`) · 规则化基线 | `code/baselines/baseline_recon.py` (completion export), `code/baselines/baseline_volume.py` (volume) |
| Deep-model export · 深度模型导出 | `code/evaluation/export_adapointr_test.py` (both arms), `export_pointr_test.py`, `export_pcn_test.py`, `export_pointr_validation.py`, `export_adapointr_validation.py` |
| No-completion baselines ID-1024 / ID-8192 · 不补全基线 | `code/baselines/identity_sampling.py` (Philox spawn `[cavity, level, 2]`, sampling without replacement), `identity_common_support.py` (common-support integration), `identity_pair_stats.py` (paired statistics) |

## 四、Evaluation metrics · 评价指标 (manuscript §3.6–§3.7)

| Metric · 指标 | Script · 脚本 (validation / test · 验证集 / 测试集) |
|---|---|
| CD₁ (per sample, summary, coordinate sensitivity, re-mapped pairing) | `code/evaluation/val_cd1.py` / `test_cd1.py` |
| Volume / N_slice (fixed closure pipeline) · 体积 / N_slice | `val_volume.py` (built on `volume_pipeline.py`) / `test_volume.py` |
| Common support (zero-flip slice pairing) · 共同支撑 | `val_common_support.py` / `test_common_support.py` |
| Local surface (k = 17 PCA normal / curvature / spread) · 局部表面 | `val_surface.py` / `test_surface.py` |
| Per-cell CD (10×12×3 atlas) · 逐格 CD | `cell_cd.py`, `code/evaluation/cell_atlas.py` |
| Sparsity / density · 稀疏/密度 | `sparse_density.py` |
| Band occupancy statistics (§3.5, median in-band share 0.0027) · 盲区带观测统计 | `code/evaluation/band_occupancy_stats.py` |
| Per-level and fidelity (training-monitor frame) · 逐层/保真 | `per_level_eval.py`, `fidelity_eval.py`, `eval_masked.py` |

## 五、Statistical inference and the main endpoint · 统计推断与主要终点 (§3.8, Tables 5–8)

Per-sample and summary results are in `results/`; the file-to-table mapping is in
`results/README.md`. · 逐样本与汇总结果见 `results/`，文件名与图表的对应见 `results/README.md`。

| Analysis · 分析 | Script · 脚本 |
|---|---|
| Main endpoint, test split, common support (Tables 5–6 default row) · 主终点 | `code/stats/common_support_test.py`, `common_support_val_main.py`; validation repeats · 验证集重复 `common_support_val_order1.py` / `common_support_val_order2.py` |
| Nine grid–operator combinations (Table 6, Figure 7) · 九配置 | parameterised loop in `identity_common_support.py`; inputs in the data release |
| Masked vs full-region pairing (Table 7 repeats) · 监督配对 | `code/stats/repeats_eval.py`, `repeats_stats.py` |
| Initialization stratum (§4.5) · 初始化维度 | `code/stats/init_eval.py`, `init_stats.py`, `init_fingerprint.py`; results in `results/init_summary.csv` |
| Negative controls (§4.5) · 阴性对照 | `code/stats/negative_control_eval.py`; clouds on request |
| Masked vs full-region CD₁ pairing (§4.3) | `code/stats/supervision_arms_cd1.py` |
| Three-run endpoint (Table 8) · 三次运行主终点 | from the three `common_support_val_*` scripts; results in `results/three_run_endpoint.csv` |
| Observational-density stratification (§4.4) · 观测密度分层 | results `results/density_strata_test.csv`, `density_strata_val_repeats.csv`; input `results/slice_azimuth_counts.csv` |

Cavity-level cluster bootstrap (10 000 resamples, seed 20261103), exact Wilcoxon signed-rank and sign
tests are implemented in the stats scripts above; with n = 50 cavities the minimum attainable
two-sided sign-test p-value is ≈ 1.8×10⁻¹⁵, so it certifies sign consistency only.
· 腔级聚类 bootstrap（10000 次，种子 20261103）、精确 Wilcoxon 符号秩与符号检验实现在上述脚本内；
n = 50 腔时符号检验最小可达 p ≈ 1.8×10⁻¹⁵，仅表示符号一致性。

## 六、Figures · 图件 (manuscript Figures 1–9 · 论文图 1–图 9)

| Figure · 图 | Script / data source · 脚本 / 数据源 |
|---|---|
| Fig. 1 benchmark build and evaluation flow · 基准构建与评价流程 | schematic, typeset in the final layout · 示意图，终稿排版完成 |
| Fig. 2 simulation check and geometry statistics (a)(b)(c) · 仿真核查与几何统计 | `code/figures/fig_volume_distribution.py` (BH-104 overlay, volume distribution), `fig_perturbation_profiles.py` (representative profiles) |
| Fig. 3 observation synthesis and mask example (a)(b)(c) · 观测合成与掩码示例 | `code/figures/fig_perturbation_types.py` ((r, z) profiles and the B band) |
| Fig. 4 four mechanisms and the angular-resolution limit · 四类机制与角分辨率上限 | `fig_missingness_mechanisms.py`; superseded version in `code/figures/_archive/fig_mechanisms_legacy.py` |
| Fig. 5 masked supervision and dual reference · 掩码监督与双参考 | schematic plus data chain; typeset in the final layout |
| Fig. 6 the three coverage tiers (a)(b) · 三档保留率 | panels of `fig_volume_distribution.py`; source data `figsrc_*` in the data release |
| Fig. 7 main endpoint over nine configurations · 九配置主终点 | `fig_nine_config_sensitivity_legacy.py` (archived in `code/figures/_archive/`); values in `results/nine_config_summary.csv` |
| Fig. 8 spatial error and local surface (a)(b) · 空间误差与局部表面 | per-sample npy in the data release (`figsrc_*`); redraw scripts in `code/figures/` |
| Fig. 9 radial multi-value geometry (a)(b) · 径向多值 | `fig_radial_multivalue.py`, `fig_conditional_differences.py` |

Figure source data (CSV/NPY) ship with the data release, indexed by `figsrc_manifest.csv`.
· 各图源数据随数据包发布，由 `figsrc_manifest.csv` 索引。
