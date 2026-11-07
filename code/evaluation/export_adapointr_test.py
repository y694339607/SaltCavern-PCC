# Export AdaPoinTr predictions on the test split with per-file checks and a SHA256 manifest.

import argparse
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

from train_pilot import load_config, build_model  # noqa: E402
from paired_data.dataset import PairedDataset  # noqa: E402
from paired_data.transforms import denormalize  # noqa: E402

OUT = ROOT / "validate/test_final"
ARMS = {
    "BM": ROOT / "validate/pilot/ckpt/BM_full_s21_o250e250_pcV2_best.pt",
    "E2": ROOT / "validate/pilot/ckpt/E2_full_s21_o250e250_pcV2_best.pt",
}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["BM", "E2"])
a = ap.parse_args()
tag = a.tag
ckpt = ARMS[tag]

cfg = load_config(str(ROOT / "train/cfg/ms_bm_fixedinit.yaml"))
device = torch.device("cuda")
model = build_model(cfg["model"], "vendor").to(device)
ck = torch.load(str(ckpt), map_location=device, weights_only=False)
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval()
best_epoch = int(ck.get("epoch", -1))
print(f"ckpt={ckpt.name} best_epoch={best_epoch} val_cd_l2={ck.get('val_cd_l2')}", flush=True)

ds = PairedDataset(root=str(ROOT / "dataset/pc_v2"), split="test", input_points=1024,
                     gt_points=16384, downsample_seed=20260921)
dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == len(ds) == 150

pdir = OUT / f"pred_{tag}"
pdir.mkdir(parents=True, exist_ok=True)
hashes, t0 = [], time.time()
with torch.no_grad():
    for i in range(len(ds)):
        item = ds[i]
        cav_item = str(item[1])
        partial, _gt = item[2]
        assert cav_item == str(dfv.loc[i, "name"]), f"i={i} 映射不一致"
        tp = ds.get_meta(i, include_transform=True)["transform_params"]
        ret = model(partial.unsqueeze(0).to(device))
        pred = denormalize(ret[-1][0].float().cpu().numpy(), tp).astype(np.float32)
        assert pred.shape == (16384, 3) and np.isfinite(pred).all(), f"i={i} 形状/有限性异常"
        fp = pdir / f"{i:03d}.npy"
        np.save(fp, pred)
        hashes.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.name}")
        if (i + 1) % 30 == 0:
            print(f"{tag} {i+1}/150 {time.time()-t0:.0f}s", flush=True)

(pdir / f"pred_{tag}_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
print(f"EXPORT_DONE {tag} n=150 best_epoch={best_epoch}", flush=True)
