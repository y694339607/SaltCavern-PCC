# Dataset card (`pc_v2`) · 数据集卡片

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

### Composition

- 500 salt-cavern geometries generated with Sansmic (Latin-hypercube sampling of cavity-bottom
  depth 1200–2200 m, cavity height 300–600 m, insoluble content 0.03–0.08, 3–6 ordinary leaching
  stages, injection rate 1000–30 000 bbl/d). 400 cavities carry statistical perturbations (Matérn
  radial random field, local interbed necking, low-frequency axis offset, irregular sump profile;
  all 15 non-empty combinations are covered); 100 are axisymmetric controls.
- Three observation-retention tiers L0/L1/L2 (nominal 70%/50%/30%, calibrated per cavity and per
  tier to within ±0.5%), giving 1500 paired (incomplete, complete) samples.
- Missingness mechanisms: M1 inter-layer loss, M3 self-occlusion (composite geometric criterion,
  parameter window ≤ 10 rows), M4 observation-side radial noise (σ = 0.3 m, truth unchanged), M5
  whole-layer loss (nominal 5%, calibrated 2%–23% per cavity); M2 is the angular-resolution
  representation limit (0.36° acquisition step mapped onto a 1° source grid, no extra deletion).
- Layer spacing in physical metric units: L0 1.83–3.66 m, L1 2.44–4.88 m, L2 4.88–9.75 m
  (medians 3.05 / 3.66 / 7.32 m).
- Cloud sizes: incomplete clouds stored at 8192 points; model input 1024 points (common input
  protocol); truth and deep-model output 16384 points.
- Splits: cavity-isolated train/val/test = 400/50/50 cavities (1200/150/150 pairs); the three tiers
  of a cavity always fall in the same subset. The split list is `data/splits/samples.csv` in the
  repository; the full data is a Release asset.

### Directory convention and environment variables

Analysis scripts locate inputs through `SCC_ROOT` (default `/data`), with optional `SCC_ARCHIVE`
(figure source archives) and `SCC_SUPP` (supplementary directory name, default `supplementary`).
The unpacked tree is shown in "Where the scripts read data from" in the root `README.md`. The split
list in the repository (`data/splits/samples.csv`) is identical to
`$SCC_ROOT/dataset/pc_v2/samples.csv`.

### Coordinate frame (important)

- Released coordinates: radial coordinates are converted to metres with 0.3048; vertical
  coordinates retain foot values. The affine map `z_m = 0.3048 z` is supplied with the data.
- Volumes, integrable-slice counts and the common-support set are invariant under this map
  (verified: no slice flips, relative errors identical digit for digit, per-point scale deviation
  ≤ 1.7×10⁻⁶).
- CD₁ and the local surface metrics are computed in the post-map frame, which applies a weight of
  0.3048 to vertical distances; the weight is the same for all methods, so the evaluation definition
  is consistent within one frame. Training, however, was carried out in the unmapped frame; the
  effect of the scale difference on training behaviour is not isolated.
- After mapping, the vertical span of the 150 evaluation cavities is about 299–590 m, consistent in
  endpoints and magnitude with the design range 300–600 m (the LHS design is in physical units, the
  TOML file stores feet). This comparison is a magnitude check only.
- The z coordinate of a cloud is local and its origin does not correspond to the ground surface;
  absolute depth cannot be read from z.

### Source-profile verification statistics (§3.2; not the V_field used in evaluation)

- Unperturbed axisymmetric source-profile volume over 500 cavities: median 359 094 m³, 10th
  percentile ≈ 9.5×10⁴ m³, maximum ≈ 1.1×10⁶ m³.
- Injection-to-dissolution ratio ΔV/V_inj: 0.127–0.163, median 0.149.
- BH-104 reproduction check: observed well-bottom depth agrees with the SMRI 2024 value of
  2721/2720 ft (≈ 829.4/829.1 m); volume cross-check deviation 0.3%. This verifies pipeline
  consistency only and is not extrapolated to field validation of all cavities.
- Post-perturbation volume ratio V′/V: 0.9155–1.0735, mean 1.0012.

### Files in the release

- `samples.csv` — sample list (cavity, tier, split, retention).
- `splits/{train,val,test}.txt` — fixed split lists.
- `partial/`, `complete/` — incomplete and complete clouds in the released frame.
- `gtfield/` — independent simulated radius field (the `V_field` reference).
- `figsrc_*` — figure source data (npy/csv), indexed by `figsrc_manifest.csv`.
- `predictions/` — normalised prediction tree (`test/`, `val/`, `repeats/`; 17 groups × 150 files,
  478 MB). Binaries are not tracked in git and ship as the Release asset
  `predictions_v1.0.tar.gz`; see `predictions/README.md` for the layout, the table mapping and the
  list of deliberately excluded items.
- SHA-256 manifest: `SHA256SUMS`; verify with `code/datagen/verify_assets.py`.

### Licence and citation

Dataset: CC BY 4.0. Citation information in `CITATION.cff` and the root README.

---

## 中文

### 组成

- 500 个盐穴腔体：Sansmic 仿真（LHS 参数采样，腔底深度 1200–2200 m、腔高 300–600 m、不溶物含量
  0.03–0.08、3–6 个 ordinary leaching 阶段、注水流量 1000–30000 bbl/d）；其中 400 腔施加统计扰动
  （Matérn 径向随机场、局部夹层缩颈、低频轴线偏心、沉渣轮廓不规则化，15 种非空组合均有覆盖），
  100 腔为轴对称对照。
- 三种观测保留率 L0/L1/L2（名义 70%/50%/30%，逐腔逐档标定，偏差 ≤ ±0.5%），共 1500 对样本
  （残缺/完整）。
- 缺失机制：M1 层间缺失、M3 自遮挡（复合几何判据，参数窗口 ≤ 10 行）、M4 观测侧径向噪声
  （σ = 0.3 m，不修改真值）、M5 整层丢失（名义 5%，逐腔标定 2%–23%）；M2 为角分辨率表示上限
  （0.36° 作业步长映射至 1° 源网格，不产生额外删除）。
- 物理米制下层间距：L0 1.83–3.66 m、L1 2.44–4.88 m、L2 4.88–9.75 m（中位 3.05/3.66/7.32 m）。
- 点云规模：残缺存储 8192 点；模型实际输入 1024 点（统一输入协议）；真值与深度模型输出 16384 点。
- 划分：按腔体隔离 train/val/test = 400/50/50 腔（1200/150/150 对）；同一腔体三档进入同一子集；
  划分清单见 `data/splits/samples.csv`（仓库内），完整数据见 Releases。

### 目录约定与环境变量

分析脚本通过 `SCC_ROOT` 定位输入（默认 `/data`），可选 `SCC_ARCHIVE`（图件源数据归档）与 `SCC_SUPP`
（补充目录名，默认 `supplementary`）。解包后的目录树见仓库根 `README.md` 的 "Where the scripts read
data from"。划分清单在仓库内为 `data/splits/samples.csv`，在数据树中对应
`$SCC_ROOT/dataset/pc_v2/samples.csv`（内容相同）。

### 坐标口径（重要）

- 发布坐标：径向坐标已按 0.3048 换算为米；垂向坐标保留英尺数值；附沿垂向仿射映射 z_m = 0.3048 z。
- 体积、可积分切片数与共同支撑集合不受该映射影响（已核验：可积分切片数无翻转、相对误差逐位一致、
  逐点尺度偏差 ≤ 1.7×10⁻⁶）。
- CD₁ 与局部表面指标在映射后口径计算，等价于对垂向误差施加 0.3048 权重；该权重对各方法一致，
  因此同口径内的评价定义一致。但训练在未换算坐标下完成，坐标尺度差异对训练行为的影响未单独隔离。
- 换算后 150 腔的 z 跨度约 299–590 m，与造腔设计范围（300–600 m）在端点与量级上一致（LHS 设计为
  物理量纲，TOML 中存英尺）；该比较仅作量级核对。
- 点云 z 为局部坐标，零点不与地表对应，绝对深度不由 z 值读出。

### 源剖面核查统计（§3.2，不作为评价中的 V_field）

- 500 腔未扰动轴对称源剖面体积：中位数 359 094 m³，第 10 百分位约 9.5×10⁴ m³，最大约 1.1×10⁶ m³。
- 注入—溶蚀比 ΔV/V_inj：0.127–0.163，中位 0.149。
- BH-104 复现核查：观测井底深度与 SMRI 2024 报告值 2721/2720 ft（约 829.4/829.1 m）一致，体积交叉
  核查偏差 0.3%。该核查支持流程自洽，不外推为全部腔体已通过现场验证。
- 扰动后体积比 V′/V：0.9155–1.0735，均值 1.0012。

### 随包文件

- `samples.csv`：样本清单（腔号、档位、划分、观测保留率）。
- `splits/{train,val,test}.txt`：固定划分清单。
- `partial/`、`complete/`：残缺与完整点云（发布坐标）。
- `gtfield/`：独立仿真半径场（V_field 参考）。
- `figsrc_*`：论文图件源数据（npy/csv），`figsrc_manifest.csv` 索引。
- `predictions/`：各方法预测点云的规范树（`test/`、`val/`、`repeats/`，17 组 × 150 件，478 MB），
  二进制不进 git，随 Release 资产 `predictions_v1.0.tar.gz` 发布；目录约定、各组对应的图表与
  未收录项说明见 `predictions/README.md`。
- SHA-256 清单：`SHA256SUMS`；核验用 `code/datagen/verify_assets.py`。

### 许可与引用

数据集：CC BY 4.0。引用信息见根目录 `CITATION.cff` 与 README。
