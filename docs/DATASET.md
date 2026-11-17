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

```
pc_v2/
├── samples.csv                  1500 rows, utf-8-sig
├── splits/{train,val,test}.txt   400 / 50 / 50 cavity names
├── train/{partial,complete,gt_field}/
├── val/{partial,complete,gt_field}/
└── test/{partial,complete,gt_field}/
```

| Path | Format | Shape | Columns / layout |
|---|---|---|---|
| `*/partial/cav_NNN_Lk.npy` | float32 | (8192, 3) | `x, y, z`; x and y in metres, z in recorded vertical units (×0.3048 for metres) |
| `*/complete/cav_NNN_Lk.npy` | float32 | (16384, 3) | same frame as `partial/` |
| `*/gt_field/cav_NNN.npy` | float32 | (201, 360) | radius field in metres; rows from the cavity floor upward (201 layers), 360 columns at 1° azimuth |
| `samples.csv` | CSV | 1500 × 11 | `name, num, group, split, level, combo, coverage, spacing_m, p_loss, shadow_limited, seed`; `spacing_m` is a layer spacing in recorded units (×0.3048 for metres), `coverage` the source-grid retention |
| `splits/{train,val,test}.txt` | text | 400 / 50 / 50 | one cavity name per line |

`train/`, `val/` and `test/` hold 1200, 150 and 150 samples; the three tiers of one cavity always
sit in the same split. Sansmic TOML configurations and source radius tables are build-time inputs on
the generating side and are not part of this release.

Prediction clouds are a separate Release asset, `predictions_v1.0.tar.gz`; layout and naming are in
`predictions/README.md`. Checksums: `SHA256SUMS_data-v1.0.txt` for the release assets and
`predictions/SHA256SUMS.txt` for the cloud tree.

### Licence and citation

Data, exported predictions and the four released checkpoints: CC BY 4.0. Code: MIT. Citation
information in `CITATION.cff` and the root README; see also `LICENSES.md`.

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

```
pc_v2/
├── samples.csv                  1500 行，utf-8-sig
├── splits/{train,val,test}.txt   400 / 50 / 50 个腔名
├── train/{partial,complete,gt_field}/
├── val/{partial,complete,gt_field}/
└── test/{partial,complete,gt_field}/
```

| 路径 | 格式 | 形状 | 列 / 布局 |
|---|---|---|---|
| `*/partial/cav_NNN_Lk.npy` | float32 | (8192, 3) | `x, y, z`；x、y 为米，z 为记录垂向单位（×0.3048 得米） |
| `*/complete/cav_NNN_Lk.npy` | float32 | (16384, 3) | 与 `partial/` 同一坐标系 |
| `*/gt_field/cav_NNN.npy` | float32 | (201, 360) | 半径场（米）；行自腔底向上共 201 层，360 列为 1° 方位 |
| `samples.csv` | CSV | 1500 × 11 | `name, num, group, split, level, combo, coverage, spacing_m, p_loss, shadow_limited, seed`；`spacing_m` 为记录单位下的层间距（×0.3048 得米），`coverage` 为源网格保留率 |
| `splits/{train,val,test}.txt` | 文本 | 400 / 50 / 50 | 每行一个腔名 |

train/val/test 分别包含 1200/150/150 个样本；同一腔体的三个档位始终落在同一划分内。Sansmic 的 TOML
配置与源半径表是生成侧的构建输入，不属于本次发布。

预测点云为另一项 Release 资产 `predictions_v1.0.tar.gz`，目录与命名见 `predictions/README.md`。
校验值：发布资产用 `SHA256SUMS_data-v1.0.txt`，预测树用 `predictions/SHA256SUMS.txt`。

### 许可与引用

数据、导出预测与四个发布检查点：CC BY 4.0。代码：MIT。引用信息见根目录 `CITATION.cff` 与 README，
另见 `LICENSES.md`。
