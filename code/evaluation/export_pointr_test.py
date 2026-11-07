# export_pointr_test.py — test split export: PoinTr baseline
# Predictions are written to $SCC_ROOT/validate/test_final/pred_pointr/ with per-file checks and a SHA256 list.
# 用法：cd /data && python validate/test/export_pointr_test.py
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))

from paired_data.dataset import PairedDataset  # noqa: E402
from paired_data.transforms import denormalize  # noqa: E402
from train_pilot import load_config, build_model  # noqa: E402

OUT = ROOT / "validate/test_final"
ds = PairedDataset(root=f"{_ROOT}/dataset/pc_v2", split="test", input_points=1024,
                     gt_points=16384, downsample_seed=20260921)
dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == len(ds) == 150

cfg = load_config(str(ROOT / "train/cfg/pc_v2_pilot_pointr.yaml"))["model"]
model = build_model(cfg, "vendor").cuda()
ck = torch.load(str(ROOT / "validate/pilot/ckpt/gpointr_s21_best.pt"),
                map_location="cuda", weights_only=False)
sd = ck.get("model", ck.get("model_state", ck)) if isinstance(ck, dict) else ck
model.load_state_dict(sd)
model.eval()
print(f"ckpt=gpointr_s21_best.pt best_epoch={int(ck.get('epoch', -1))}", flush=True)

pdir = OUT / "pred_pointr"
pdir.mkdir(parents=True, exist_ok=True)
hashes, t0 = [], time.time()
with torch.no_grad():
    for i in range(len(ds)):
        item = ds[i]
        cav_item = str(item[1])
        partial, _gt = item[2]
        assert cav_item == str(dfv.loc[i, "name"]), f"i={i} 映射不一致"
        tp = ds.get_meta(i, include_transform=True)["transform_params"]
        pred = model(partial.unsqueeze(0).cuda())[-1]
        pred = denormalize(pred[0].cpu().numpy(), tp).astype(np.float32)
        assert pred.shape == (16384, 3) and np.isfinite(pred).all(), f"i={i} 形状/有限性异常"
        fp = pdir / f"{i:03d}.npy"
        np.save(fp, pred)
        hashes.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.name}")
        if (i + 1) % 30 == 0:
            print(f"PoinTr {i+1}/150 {time.time()-t0:.0f}s", flush=True)

(pdir / "pred_pointr_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
print(f"EXPORT_DONE PoinTr n=150 best_epoch={int(ck.get('epoch', -1))}", flush=True)
