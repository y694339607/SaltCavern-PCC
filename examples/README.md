# examples/ · 示例数据

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

One cavity (`cav_021`) at all three coverage tiers, plus its independent simulated radius field.

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
tier  V_field(m3) V_ref(m3)   N_ref  N_id   E_1024(%)   E_1024_vs_ref(%)
L0    1231988     1218266     479    200    0.4559      0.6653     
L1    1212699     1218192     485    191    0.6720      0.2181     
L2    1228357     1219276     479    158    0.3810      0.3609     
```

---

## 中文

一个腔体（`cav_021`）在三个观测保留率档位下的完整数据，以及它对应的独立仿真半径场。

目录结构与上节相同：`partial/` 为 8192 点观测、`complete/` 为 16384 点完整真值、
`gt_field/` 为 (201, 360) 的独立半径场。

### 运行

在仓库根目录执行：

```bash
python examples/demo_dual_reference.py --root examples
```

预期输出见 `expected_output.txt`。
