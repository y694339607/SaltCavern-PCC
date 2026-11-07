# examples/ · 示例数据

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

One cavity (`cav_021`) at all three coverage tiers, plus its independent simulated radius field.
Small enough to track in git; the full 150-sample set is a Release asset.

```
examples/
├── partial/     cav_021_L{0,1,2}.npy   observed clouds, 8192 points (float32)
├── complete/    cav_021_L{0,1,2}.npy   complete ground truth, 16384 points (float32)
├── gt_field/    cav_021.npy            independent simulated radius field, (201, 360)
├── demo_dual_reference.py
└── expected_output.txt
```

### Run

From the repository root:

```bash
python examples/demo_dual_reference.py --root examples
```

Expected output (see `expected_output.txt`):

```
tier  V_field(m3) V_ref(m3)   N_ref  N_id   E_8192(%)   E_1024(%)   E_1024_vs_ref(%)
L0    1231988     1218266     479    200    0.8652      0.4559      0.6653
L1    1212699     1218192     485    191    0.7513      0.6720      0.2181
L2    1228357     1219276     479    158    0.1157      0.3810      0.3609
```

### Why this is a real check, not just a smoke test

Two values in row L0 are anchored to the released per-sample table
(`results/nine_config_per_sample.csv`, row `cav=cav_021, level=0, cfg=G1_500x72_max`):

| Quantity | Demo | Released table |
|---|---:|---:|
| `N_slice` of the ID-1024 baseline | 200 | 200 |
| `N_slice` of the complete cloud | 479 | 479 |
| ID-1024 error vs `V_field` (%) | 0.4559 | 0.45587418886863484 |
| ID-1024 error vs `V_ref` (%) | 0.6653 | 0.6652896277084518 |

Agreement at this level means the closure, the in-cell maximum aggregation, the 3-azimuth-cell
validity rule, the 10-slice minimum, the trapezoid over slice centres, and the fixed 1024-point
subsampling all reproduce outside the original environment.

### Conventions used

- Wellbore closure: radius 0.61 m, height range `zmin + 2` to `zmin + 60` (recorded frame), 36 azimuths.
- Grid: 500 vertical slices × 72 azimuths; cell value = maximum radius in that cell.
- A slice is integrable with at least 3 valid azimuth cells; a cloud yields a volume with at least
  10 integrable slices.
- Slice cross-section area = `0.5 * ∫ r² dθ` with periodic interpolation; volume = trapezoid over
  slice centres.
- ID-1024: `Philox(SeedSequence(20260921, spawn_key=[cavity, level, 2])).permutation(8192)[:1024]`.
- `V_field = π · Σ_z mean_θ(r²) · H / 200`, with `H` taken from the complete cloud of that tier.

All quantities are in the recorded frame; the vertical affine map `z_m = 0.3048 z` leaves
volume-class quantities unchanged (manuscript Section 3.1).

---

## 中文

一个腔体（`cav_021`）在三个观测保留率档位下的完整数据，以及它对应的独立仿真半径场。体量足够小，
可以直接纳入 git；完整的 150 个样本随 Release 发布。

目录结构与上节相同：`partial/` 为 8192 点观测、`complete/` 为 16384 点完整真值、
`gt_field/` 为 (201, 360) 的独立半径场。

### 运行

在仓库根目录执行：

```bash
python examples/demo_dual_reference.py --root examples
```

预期输出见 `expected_output.txt`（与上节代码块一致）。

### 这不只是"跑得动"，而是可核对的复现

L0 行的两个值锚定在已发布的逐样本表中（`results/nine_config_per_sample.csv` 的
`cav=cav_021, level=0, cfg=G1_500x72_max` 行）：ID-1024 的可积分切片数 200、完整云的切片数 479、
ID-1024 对 `V_field` 的误差 0.4559%、对 `V_ref` 的误差 0.6653%。

达到这一精度的一致，意味着井筒闭格架、格内最大半径聚合、≥3 个有效方位格的判定、≥10 个可积分
切片的下限、层中心梯形积分，以及固定的 1024 点抽样，全部都能在原环境之外复现。

### 使用的口径

- 井筒闭合：半径 0.61 m，高度范围 `zmin + 2` 至 `zmin + 60`（记录坐标），36 个方位。
- 网格：500 个垂向切片 × 72 个方位；格内取最大半径。
- 一个切片至少需 3 个有效方位格才可积分；一个点云至少需 10 个可积分切片才有体积。
- 切片横截面积 = `0.5 * ∫ r² dθ`，周向按周期插值；体积 = 层中心梯形积分。
- ID-1024：`Philox(SeedSequence(20260921, spawn_key=[腔号, 档位, 2])).permutation(8192)[:1024]`。
- `V_field = π · Σ_z mean_θ(r²) · H / 200`，其中 `H` 取该档位完整点云的垂向跨度。

全部量处于记录坐标；垂向仿射映射 `z_m = 0.3048 z` 不改变体积类量（论文 §3.1）。
