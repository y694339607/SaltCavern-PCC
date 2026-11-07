# Checkpoints and release assets · 检查点与发布资产

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

### Released weights (GitHub Release assets)

File names are given verbatim. **Two of them contain a space**; URL-encode it as `%20` when
downloading from a shell.

| File | Arm | Epoch of the selected checkpoint | Role in the manuscript |
|---|---|---|---|
| `BM_full_ s21_o250e250_pcV2_best.pt` | AdaPoinTr, blind-zone masked supervision | 179 | Table 4 row `adapointr_masked`; clouds in `predictions/*/pred_adapointr_masked/` |
| `E2_full_s21_o250e250_pcV2_best.pt` | AdaPoinTr, full-region supervision | 169 | Table 4 row `adapointr_full`; clouds in `predictions/*/pred_adapointr_full/` |
| `PCN_full_ s21_o250e250_pcV2_best.pt` | PCN | 243 | Table 4 row `pcn`; clouds in `predictions/*/pred_pcn/` |
| `gpointr_s21_best.pt` | PoinTr | 250 | Table 4 row `pointr`; clouds in `predictions/*/pred_pointr/` |

Weights from runs that are **not reported in the manuscript**:

| File | Arm | Note |
|---|---|---|
| `gbmw30v_s21_best.pt` | AdaPoinTr, masked, blind window 30 | width ablation; no row in any table |
| `gbmw100v_s21_best.pt` | AdaPoinTr, masked, blind window 100 | width ablation; no row in any table |
| `gseedok_s21_best.pt` | AdaPoinTr, masked, repeat | seed-reproduction run; no row in any table |

### Decoding the names

| Token | Meaning |
|---|---|
| `BM` / `E2` | historical tags of the two supervision arms; the manuscript and the released prediction tree use `adapointr_masked` / `adapointr_full` |
| `s21` | initialization seed 20260921 |
| `o250e250` | the **training budget** of that run (250 epochs). It does not mean the released file is the end-of-budget checkpoint: `best` marks the checkpoint selected by validation CD₂ during training, and the epoch at which it was selected is given in the table above. No end-of-budget checkpoint is released. |
| `pcV2` | the `pc_v2` benchmark build |

### Not released as weights

The data-order repeats (Tables 7–8), the initialization stratum and the placebo controls
(Section 4.5) were run, but only their **per-sample metrics** are released:
`results/repeats_per_sample.csv`, `results/init_repeats_per_sample.csv`,
`results/placebo_init_per_sample.csv`. Their weights and clouds are available from the authors on
request.

### Loading a checkpoint

```python
import torch
ck = torch.load("BM_full_ s21_o250e250_pcV2_best.pt", map_location="cuda", weights_only=False)
state = ck["model"] if "model" in ck else ck
```

Training recipes: `configs/pc_v2_pilot_bm.yaml` (`band_mask: true`) and
`configs/ms_e2_fixedinit.yaml` (`band_mask: false`). Entry point: `code/models/train_pilot.py`.
Re-export: `code/evaluation/export_adapointr_test.py` (test) or
`code/evaluation/export_adapointr_validation.py` (validation).

---

## 中文

### 已发布的权重（GitHub Release 资产）

文件名按原名给出。**其中两个含空格**，命令行下载时需写成 `%20`。

| 文件 | 臂 | 被选中检查点的轮次 | 论文中的位置 |
|---|---|---|---|
| `BM_full_ s21_o250e250_pcV2_best.pt` | AdaPoinTr，盲区掩码监督 | 179 | 表 4 `adapointr_masked` 行；点云在 `predictions/*/pred_adapointr_masked/` |
| `E2_full_s21_o250e250_pcV2_best.pt` | AdaPoinTr，全区域监督 | 169 | 表 4 `adapointr_full` 行；点云在 `predictions/*/pred_adapointr_full/` |
| `PCN_full_ s21_o250e250_pcV2_best.pt` | PCN | 243 | 表 4 `pcn` 行；点云在 `predictions/*/pred_pcn/` |
| `gpointr_s21_best.pt` | PoinTr | 250 | 表 4 `pointr` 行；点云在 `predictions/*/pred_pointr/` |

论文**未报告**的运行所对应的权重：

| 文件 | 臂 | 说明 |
|---|---|---|
| `gbmw30v_s21_best.pt` | AdaPoinTr，掩码，盲窗 30 | 宽度消融；任何表中均无对应行 |
| `gbmw100v_s21_best.pt` | AdaPoinTr，掩码，盲窗 100 | 宽度消融；任何表中均无对应行 |
| `gseedok_s21_best.pt` | AdaPoinTr，掩码，重复运行 | 种子复现运行；任何表中均无对应行 |

### 名称解读

| 记号 | 含义 |
|---|---|
| `BM` / `E2` | 两个监督臂的历史标签；论文与发布的预测树使用 `adapointr_masked` / `adapointr_full` |
| `s21` | 初始化种子 20260921 |
| `o250e250` | 该次运行的**训练预算**（250 轮）。**不代表**发布的是跑满预算时的权重：`best` 表示训练过程中按验证 CD₂ 选出的检查点，其被选中的轮次见上表；训练结束时的检查点未发布 |
| `pcV2` | 基准数据集 `pc_v2` |

### 未以权重形式发布

顺序重复（表 7–表 8）、初始化维度与安慰剂对照（§4.5）均已运行，但只发布其**逐样本指标**：
`results/repeats_per_sample.csv`、`results/init_repeats_per_sample.csv`、
`results/placebo_init_per_sample.csv`。其权重与点云可向作者索取。

### 加载检查点

```python
import torch
ck = torch.load("BM_full_ s21_o250e250_pcV2_best.pt", map_location="cuda", weights_only=False)
state = ck["model"] if "model" in ck else ck
```

训练配置：`configs/pc_v2_pilot_bm.yaml`（`band_mask: true`）与 `configs/ms_e2_fixedinit.yaml`
（`band_mask: false`）。入口：`code/models/train_pilot.py`。重新导出：
`code/evaluation/export_adapointr_test.py`（测试集）或
`code/evaluation/export_adapointr_validation.py`（验证集）。
