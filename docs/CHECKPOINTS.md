# Checkpoints and release assets · 检查点与发布资产

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

### Released weights (GitHub Release `data-v1.0`)

| File | Arm | Selected epoch | Role in the manuscript |
|---|---|---|---|
| `masked_main_e179.pt` | AdaPoinTr, blind-zone masked supervision | 179 | Table 4 row `adapointr_masked`; clouds in `predictions/*/pred_adapointr_masked/` |
| `full_region_main_e169.pt` | AdaPoinTr, full-region supervision | 169 | Table 4 row `adapointr_full`; clouds in `predictions/*/pred_adapointr_full/` |
| `pcn_main_e250.pt` | PCN | 243 | Table 4 row `pcn`; clouds in `predictions/*/pred_pcn/` |
| `pointr_main_e250.pt` | PoinTr | 250 | Table 4 row `pointr`; clouds in `predictions/*/pred_pointr/` |

A SHA-256 list accompanies the four assets; verify a download with
`sha256sum -c SHA256SUMS_checkpoints.txt`.

### Reading the file names

`<arm>_main_e<epoch>.pt`, where `<epoch>` is the epoch at which the released checkpoint was
selected by validation CD₂ during a 250-epoch training budget. The file is therefore **not** the
end-of-budget checkpoint; no end-of-budget weights are released.

### Not released as weights

- Data-order repeats (Tables 7–8), the initialization stratum and the placebo controls
  (Section 4.5) were run, but only their per-sample metrics are released
  (`results/repeats_per_sample.csv`, `results/init_repeats_per_sample.csv`,
  `results/placebo_init_per_sample.csv`).
- The blind-window width ablations (window 30 m and 100 m) and one seed-reproduction run were also
  carried out; no row of the manuscript reports them and their weights are not published.

Weights and clouds of all of the above are available from the authors on request.

### Loading a checkpoint

```python
import torch
ck = torch.load("masked_main_e179.pt", map_location="cuda", weights_only=False)
state = ck["model"] if "model" in ck else ck
```

Training recipes: `configs/pc_v2_pilot_bm.yaml` (`band_mask: true`) and
`configs/ms_e2_fixedinit.yaml` (`band_mask: false`). Entry point: `code/models/train_pilot.py`.
Re-export: `code/evaluation/export_adapointr_test.py` (test) or
`code/evaluation/export_adapointr_validation.py` (validation).

---

## 中文

### 已发布的权重（GitHub Release `data-v1.0`）

| 文件 | 臂 | 被选中检查点的轮次 | 论文中的位置 |
|---|---|---|---|
| `masked_main_e179.pt` | AdaPoinTr，盲区掩码监督 | 179 | 表 4 `adapointr_masked` 行；点云在 `predictions/*/pred_adapointr_masked/` |
| `full_region_main_e169.pt` | AdaPoinTr，全区域监督 | 169 | 表 4 `adapointr_full` 行；点云在 `predictions/*/pred_adapointr_full/` |
| `pcn_main_e250.pt` | PCN | 243 | 表 4 `pcn` 行；点云在 `predictions/*/pred_pcn/` |
| `pointr_main_e250.pt` | PoinTr | 250 | 表 4 `pointr` 行；点云在 `predictions/*/pred_pointr/` |

四份资产附一份 SHA-256 清单，下载后用 `sha256sum -c SHA256SUMS_checkpoints.txt` 核验。

### 文件名的读法

`<臂>_main_e<轮次>.pt`。其中 `<轮次>` 是在 250 轮训练预算内、按验证 CD₂ 选出该检查点时所在的轮次，
因此发布的**不是**训练结束时的权重；训练结束权重未发布。

### 未以权重形式发布

- 顺序重复（表 7–表 8）、初始化维度与安慰剂对照（§4.5）均已运行，但只发布**逐样本指标**：
  `results/repeats_per_sample.csv`、`results/init_repeats_per_sample.csv`、
  `results/placebo_init_per_sample.csv`。
- 盲窗宽度消融（窗宽 30 m 与 100 m）与一次种子复现运行也已执行；论文没有任何一行报告它们，
  其权重不发布。

以上各项的权重与点云均可向作者索取。

### 加载检查点

```python
import torch
ck = torch.load("masked_main_e179.pt", map_location="cuda", weights_only=False)
state = ck["model"] if "model" in ck else ck
```

训练配置：`configs/pc_v2_pilot_bm.yaml`（`band_mask: true`）与 `configs/ms_e2_fixedinit.yaml`
（`band_mask: false`）。入口：`code/models/train_pilot.py`。重新导出：
`code/evaluation/export_adapointr_test.py`（测试集）或
`code/evaluation/export_adapointr_validation.py`（验证集）。
