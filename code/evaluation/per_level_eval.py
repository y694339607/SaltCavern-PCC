# -*- coding: utf-8 -*-
"""Per-level CD-L2 eval of pilot best ckpt on val split (normalised space, same口径 as training)."""
import sys, time
from pathlib import Path
import numpy as np
import torch
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
import argparse as _ap
_ap = _ap.ArgumentParser(); _ap.add_argument("--ckpt", default=f"{_ROOT}/checkpoints/bm_ep179.pt"); _ap.add_argument("--csv", default=f"{_ROOT}/per_level_val_eval.csv"); _a = _ap.parse_args()
CKPT = Path(_a.ckpt)
OUTCSV = _a.csv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
sys.path.insert(0, str(ROOT / "train" / "vendor_src" / "PoinTr-master"))

from train_pilot import load_config, build_model, make_cd_l2
from paired_data.dataset import PairedDataset

cfg = load_config(str(str(Path(__file__).resolve().parents[2] / 'configs' / 'pc_v2_pilot_bm.yaml')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = build_model(cfg["model"], "vendor").to(device)
try:
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
except TypeError:
    ck = torch.load(str(CKPT), map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval()
cd_l2 = make_cd_l2("chamfer")

val_ds = PairedDataset(
    root=str(ROOT / "dataset" / "pc_v2"),
    split=str(cfg["data"]["val_split"]),
    input_points=int(cfg["data"]["input_points"]),
    gt_points=int(cfg["data"]["gt_points"]),
    downsample_seed=int(cfg["data"]["downsample_seed"]),
)
n = len(val_ds)
df = pd.read_csv(ROOT / "dataset" / "pc_v2" / "samples.csv", encoding="utf-8-sig")
dfv = df[df["split"] == str(cfg["data"]["val_split"])].reset_index(drop=True)
assert len(dfv) == n, (len(dfv), n)
print("val samples:", n, "| meta keys:", list(val_ds.get_meta(0).keys()))

rows, t0 = [], time.time()
with torch.no_grad():
    for i in range(n):
        item = val_ds[i]
        partial, gt = (item["partial"], item["gt"]) if isinstance(item, dict) else item[2]
        partial = torch.as_tensor(partial, dtype=torch.float32).unsqueeze(0).to(device)
        gt = torch.as_tensor(gt, dtype=torch.float32).unsqueeze(0).to(device)
        v = float(cd_l2(model(partial)[-1], gt))
        rows.append((dfv.loc[i, "name"], int(dfv.loc[i, "level"]), (dfv.loc[i, "group"] == "perturb"), v))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}  elapsed {time.time()-t0:.0f}s")

r = pd.DataFrame(rows, columns=["name", "level", "is_perturb", "cd_l2"])
print("\n== per level ==")
print(r.groupby("level")["cd_l2"].agg(["mean", "std", "count"]).to_string())
print("\n== level x is_perturb (mean cd_l2) ==")
print(r.groupby(["level", "is_perturb"])["cd_l2"].mean().unstack().to_string())
out = Path(OUTCSV)
r.to_csv(out, index=False)
print("\nsaved ->", out)