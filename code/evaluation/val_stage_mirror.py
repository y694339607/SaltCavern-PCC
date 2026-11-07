# Build a symlink mirror of the validation split for the legacy analysis layout.

from pathlib import Path
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
ZVAL = WORK / "raw/val_from_zip"
MIR = WORK / "val_mirror"
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}

VAL_PRED = {"BM": ROOT/"validate/volume/pred_adapointr_masked","E2": ROOT/"validate/volume/pred_adapointr_full",
            "PoinTr": ROOT/"validate/volume/pred_GPoinTr","B1": ROOT/"validate/baseline/B1",
            "B2": ROOT/"validate/baseline/B2","B3": ROOT/"validate/baseline/B3",
            "PCN": ROOT/"validate/baseline_dl/pcn/pred"}

samples = pd.read_csv(ROOT/"dataset/pc_v2/samples.csv", encoding="utf-8-sig")
val = samples[samples["split"]=="val"].reset_index(drop=True)
assert len(val) == 150
stems = [f"{r['name']}_L{int(r['level'])}" for _, r in val.iterrows()]

(MIR/"gt").mkdir(parents=True, exist_ok=True)
for i, st in enumerate(stems):
    link = MIR/"gt"/f"{i:03d}.npy"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(ZVAL/"complete"/f"{st}.npy")

for m, d in VAL_PRED.items():
    m = ALIAS.get(m, m)
    link = MIR/f"pred_{m}"
    if link.exists() or link.is_symlink():
        if link.is_symlink(): link.unlink()
        else: raise SystemExit(f"{link} 存在且非符号链接，停止")
    link.symlink_to(d)

for name, src in [("gt_field", ZVAL/"gt_field"), ("obs", ZVAL/"partial")]:
    link = MIR/name
    if link.exists() or link.is_symlink():
        if link.is_symlink(): link.unlink()
        else: raise SystemExit(f"{link} 存在且非符号链接，停止")
    link.symlink_to(src)

# 校验：通过符号链接读取与直接读取一致（抽 3 件）
import numpy as np
for i in [0, 1, 2]:
    a = np.load(MIR/"gt"/f"{i:03d}.npy")
    b = np.load(ZVAL/"complete"/f"{stems[i]}.npy")
    assert np.array_equal(a, b), f"gt 链接不一致 i={i}"
a = np.load(MIR/"pred_adapointr_masked"/"000.npy")
b = np.load(VAL_PRED["BM"]/"000.npy")
assert np.array_equal(a, b)
print("VAL_MIRROR_OK", MIR)
