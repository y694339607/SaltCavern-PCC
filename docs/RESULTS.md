# Main results of the manuscript · 论文主要结果

> Bilingual document. Tables carry bilingual headers and are given once. · 中英双语；表格用双语表头，只列一次。
> All values are in the physical metric frame. Differences are "masked configuration − comparison";
> intervals are cavity-level cluster bootstrap (10 000 resamples, seed 20261103).
> 数值均为物理米制；差值为"掩码配置 − 对照"；区间为腔级聚类 bootstrap（10000 次，种子 20261103）。

## Table 4 · 表 4 — One protocol run, overall results (mean) · 一次协议运行的整体结果（均值）

| Method · 方法 | Val CD₁ (m) · 验证 | Val V_field (%) · 验证 | Val V_ref (%) · 验证 | Test CD₁ (m) · 测试 | Test V_field (%) · 测试 | Test V_ref (%) · 测试 | N_slice (val/test) · 可积分切片 |
|---|---|---|---|---|---|---|---|
| Cyl-NN | 0.90±0.20 | 1.660 | 0.754 | 0.92±0.17 | 1.643 | 0.635 | 493.2/494.6 |
| AdaPoinTr masked (`adapointr_masked`, ep179) | 0.90±0.21 | 1.459 | 1.628 | 0.97±0.20 | 1.587 | 1.889 | 406.6/402.2 |
| AdaPoinTr full-region (`adapointr_full`, ep169) | 0.95±0.22 | 1.468 | 2.123 | 1.00±0.20 | 1.765 | 2.322 | 405.3/402.6 |
| PCN | 1.52±0.37 | 2.538 | 3.210 | 1.62±0.44 | 2.292 | 2.844 | 453.7/459.3 |
| IDW | 1.19±0.32 | 4.260 | 5.618 | 1.27±0.32 | 4.695 | 6.081 | 493.2/494.6 |
| PoinTr | 1.50±0.41 | 8.853 | 9.852 | 1.61±0.42 | 7.821 | 8.814 | 294.3/300.2 |
| Periodic spline · 周期样条 | 1.77±2.92 | 198.399* | 202.430* | 1.60±0.59 | 8.668 | 9.836 | 493.2/494.6 |
| ID-1024 (no completion) | 3.28±4.17 | 2.296 | — | 1.97±0.37 | 1.757 | — | 147.6/157.1 |
| ID-8192 (no completion) | 2.56±4.28 | 1.503 | — | 1.12±0.23 | 1.197 | — | 247.0/255.5 |

\* The validation mean of the periodic spline is dominated by extreme failures; medians are
4.723%/5.969%. · 周期样条验证集均值受极端失效样本主导，中位数 4.723%/5.969%。

Supplementary · 补充：masked vs full-region CD₁ paired difference, val −0.051 m [−0.059, −0.042],
test −0.038 m [−0.049, −0.027]（掩码相对全区域 CD₁ 配对差）; masked vs Cyl-NN CD₁ interval crosses
zero on validation (p = 0.315)（验证集掩码与 Cyl-NN 的配对区间跨零）.

## Table 5 · 表 5 — Same method pair under two frames (test, default 500×72/max)
## 同一方法对两种口径比较（测试集，默认 500×72/max）

| Frame · 口径 | ID-1024 (%) | AdaPoinTr masked (%) · 掩码配置 | Difference [95% CI] · 差值 [95% 区间] | Same-direction cavities · 同向腔 |
|---|---|---|---|---|
| Full support, own slices · 全支撑（各自 N_slice） | 1.757 (157.1 slices) | 1.587 (402.2 slices) | −0.170 [−0.526, +0.185] | — |
| Common support, 155.6 slices · 共同支撑（155.6 片） | 0.170 | 1.285 | **+1.115 [+0.882, +1.371]** | 50/50 |

By tier (common support, descriptive) · 分档（共同支撑，描述性）：L0 +0.984, L1 +1.081, L2 +1.281 pp.
Stratified by the baseline's valid azimuth-cell count · 按基线有效方位格数分层：≤6 cells +1.135
(49/50 cavities positive · 腔为正); ≥30 cells +0.780 (40 cavities positive).

## Table 6 · 表 6 — Main endpoint over nine grid–operator combinations (test, common support)
## 九种网格—算子组合下的主终点（测试集，共同支撑）

| Grid · 网格 | Operator · 算子 | N_common | ID-1024 (%) | Masked (%) · 掩码 | Difference [95% CI] · 差值 | Same-direction · 同向腔 |
|---|---|---|---|---|---|---|
| 500×72 | max | 155.6 | 0.170 | 1.285 | +1.115 [+0.882, +1.371] | 50/50 |
| 500×72 | mean | 155.6 | 0.158 | 1.165 | +1.007 [+0.829, +1.194] | 50/50 |
| 250×36 | max | 105.8 | 0.453 | 2.412 | +1.960 [+1.521, +2.431] | 48/50 |
| 250×36 | mean | 105.8 | 0.234 | 1.132 | +0.899 [+0.723, +1.082] | 48/50 |
| 1000×144 | max | 203.7 | 0.192 | 1.343 | +1.150 [+0.918, +1.430] | 50/50 |
| 1000×144 | mean | 203.7 | 0.191 | 1.355 | +1.164 [+0.930, +1.441] | 50/50 |

Note · 注：the three p95 configurations coincide with the max configuration of the same grid
(difference < 10⁻³ pp) and are merged here; all nine combinations enter the "direction unchanged"
check（三种 p95 配置与同网格 max 配置数值一致，合并列示；九种组合全部参与"方向不变"判断）.
Full-support range over the same configurations · 全支撑口径同批配置：masked 1.587–2.542 %,
ID-1024 1.757–2.799 %.

## Table 7 · 表 7 — Fixed-initialization data-order repeats (validation, masked − full)
## 固定初始化顺序重复（验证集，掩码 − 全区域）

| Run · 运行 | Metric · 指标 | Masked · 掩码 | Full · 全区域 | Difference [95% CI] · 差值 |
|---|---|---|---|---|
| Order repeat 1 · 顺序重复 1 | V_field (%) | 1.679 | 3.329 | −1.649 [−2.292, −1.078] |
| Order repeat 1 · 顺序重复 1 | V_ref (%) | 2.183 | 4.590 | −2.406 [−3.022, −1.882] |
| Order repeat 1 · 顺序重复 1 | CD₁ (m) | 1.069 | 1.124 | −0.055 [−0.066, −0.044] |
| Order repeat 2 · 顺序重复 2 | V_field (%) | 1.917 | 1.571 | +0.347 [+0.008, +0.744] |
| Order repeat 2 · 顺序重复 2 | V_ref (%) | 1.823 | 1.672 | +0.151 [−0.135, +0.464] |
| Order repeat 2 · 顺序重复 2 | CD₁ (m) | 0.998 | 0.982 | +0.015 [+0.005, +0.025] |

## Table 8 · 表 8 — Main endpoint across three runs (validation, masked − ID-1024)
## 三次运行共同支撑主终点（验证集，掩码 − ID-1024）

| Run · 运行 | Masked V_field (%) · 掩码 | Endpoint (pp) · 主终点 | Cavity-level 95% CI · 腔级区间 | Same-direction · 同向腔 | N_common |
|---|---|---|---|---|---|
| Main run · 主运行 | 1.459 | +0.952 | [+0.738, +1.193] | 50/50 | 151.0 |
| Order repeat 1 · 顺序重复 1 | 1.679 | +1.282 | [+1.040, +1.544] | 50/50 | 151.0 |
| Order repeat 2 · 顺序重复 2 | 1.917 | +1.345 | [+1.010, +1.727] | 50/50 | 150.9 |

The corresponding main-run value on the test split is +1.115 pp [+0.882, +1.371].
· 主运行测试集对应值 +1.115 pp [+0.882, +1.371]。

## Negative controls (validation, control − masked, V_field) · 阴性对照

| Control · 对照 | Semantics · 语义 | Control V_field (%) | Paired difference [95% CI] · 配对差 | Same-direction · 同向腔 |
|---|---|---|---|---|
| Control 1 · 对照 1 | mask band shifted · 掩码带错位 | 2.543 | +1.084 [+0.607, +1.577] | 35/15 |
| Control 2 · 对照 2 | same number of points removed at random · 随机同量剔除 | 2.359 | +0.900 [+0.141, +1.879] | 26/24 |
| Control 3 · 对照 3 | prediction side only · 仅预测侧剔除 | 2.372 | +0.913 [+0.365, +1.527] | 32/18 |

Masked configuration 1.459 %; all three means are positive. · 掩码配置 1.459%；三项均值均为正。

## Initialization stratum (validation) · 初始化维度

| Arm · 臂 | V_field (%) | V_ref (%) | CD₁ (m) |
|---|---|---|---|
| Masked · 掩码 | 6.745 | 7.808 | 2.164 |
| Full-region · 全区域 | 21.020 | 22.690 | 2.238 |

Four of the five arbitrary-initialization repeats degrade both configurations geometrically (CD₁
13–19 m); only the run above converged, and the internal normalized CD₂ did not warn. Its masked
V_field is 4.6× worse than the main run (1.459 %).
· 五组任意初始化重复中四组两种配置同时几何退化（CD₁ 13–19 m），仅上表一组收敛；内部归一化 CD₂ 未预警。
该组掩码 V_field 较主运行（1.459%）劣化 4.6 倍。

## Exploratory results (§4.6, validation) · 探索性结果速览

- Masked V_field by tier · 掩码 V_field 分档：L0 1.303 % / L1 1.514 % / L2 1.945 %.
- Local surface · 局部表面：masked vs Cyl-NN normal-deviation median 7.78° vs 21.23°, p90 13.46° vs
  64.44°; spread p95 1.51 vs 1.35 m; sparse-region coverage 3.9 pp lower.
- Multi-valued cells · 多值几何子集：masked vs Cyl-NN per-cell CD −0.041 m [−0.095, +0.011] (crosses
  zero · 跨零); continuous-gap band +0.191 m [0.074, 0.245]; observed cells +0.154 m [0.125, 0.180]
  (radial quantity, invariant under the vertical map · 径向量，不随垂向缩放变化).
- Over-estimation share (L2, test) · 高估比例：masked signed error mean +0.231 % [−0.444, +0.936],
  46 % of samples over-estimate; 28 pp higher than ID-8192 (p = 0.00233) and 16 pp higher than
  ID-1024 (p = 0.108).
