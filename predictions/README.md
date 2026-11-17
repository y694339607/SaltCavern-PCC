# predictions/ · 预测点云

> Bilingual document: English first, Chinese below. · 中英双语：上为英文，下为中文。

## English

Layout expected by the evaluation scripts:

```
predictions/
├── MANIFEST.csv          index -> cavity / level / source file names
├── SHA256SUMS.txt        one line per .npy (sha256, relative path)
├── test/
│   ├── gt/               150 complete clouds, 16384 x 3 float32
│   └── pred_{cylnn,spline,idw,adapointr_masked,adapointr_full,pcn,pointr}/   150 each
├── val/
│   ├── gt/               150 complete clouds
│   └── pred_{adapointr_masked,adapointr_full,pointr,pcn}/                    150 each
└── repeats/
    ├── order1/pred_{adapointr_masked,adapointr_full}/   first data-order repeat, 150 each
    └── order2/pred_{adapointr_masked,adapointr_full}/   second data-order repeat, 150 each
```

Arm names: `cylnn` cylindrical nearest neighbour, `spline` periodic spline, `idw` inverse-distance
weighting, `adapointr_masked` AdaPoinTr with blind-zone masked supervision, `adapointr_full`
AdaPoinTr with full-region supervision, `pcn` PCN, `pointr` PoinTr.

### Fields

All arrays are `(16384, 3)` float32 in the recorded coordinate frame (radial coordinates in metres,
vertical coordinates retaining foot values; the affine map `z_m = 0.3048 z` is documented in
`docs/DATASET.md`). File index `i` maps to a cavity and coverage tier through `MANIFEST.csv`, which
is identical to the row order of `data/splits/samples.csv` for the same split.

### Using these files

The evaluation scripts take a prediction root and look for `pred_<method>/<i:03d>.npy`:

```bash
export SCC_ROOT=/path/to/benchmark_root
python code/evaluation/volume_pipeline.py \
    --split test \
    --pred_root predictions/test \
    --gt_dir predictions/test/gt \
    --field_dir $SCC_ROOT/dataset/pc_v2/test/gt_field \
    --methods adapointr_masked,adapointr_full,pcn,pointr,cylnn,spline,idw \
    --outdir /tmp/vol
```

---

## 中文

`test/` 含 7 种方法的预测与 150 件完整真值；`val/` 含 4 种方法与真值；
`repeats/order1`、`repeats/order2` 为两次数据顺序重复的掩码与全区域两种配置。
`MANIFEST.csv` 给出「序号 → 腔号 / 档位 / 源文件名」的映射，`SHA256SUMS.txt` 每行一个文件的
SHA-256。

### 字段

全部为 `(16384, 3)` float32，处于记录坐标（径向为米，垂向保留英尺数值；仿射映射
`z_m = 0.3048 z` 见 `docs/DATASET.md`）。文件序号 `i` 通过 `MANIFEST.csv` 对应到腔号与观测档位，
其行序与 `data/splits/samples.csv` 中同一划分的行序一致。

### 使用方式

评价脚本接收一个预测根目录，并按 `pred_<方法>/<i:03d>.npy` 查找文件，命令示例见上节。
