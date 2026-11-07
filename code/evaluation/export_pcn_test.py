# export_pcn_test.py —  test 导出：PCN 基线（PCN_full_s21_o250e250_pcV2_best）
# 装载与归一化代码段复用 归档 pcn_export.py（val 版）同式链路；
# Predictions are written to $SCC_ROOT/validate/test_final/pred_pcn/ with per-file checks and a SHA256 list.
# 用法：cd /data && python validate/test/export_pcn_test.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))            # 先显式注入训练链路径，不依赖任何 import 副作用
sys.path.insert(0, f"{_ROOT}/coding/pcn_baseline")
from models import PCN  # noqa: E402
from dataset_pc import PCv2  # noqa: E402
from paired_data.transforms import denormalize  # noqa: E402
from paired_data.dataset import PairedDataset  # noqa: E402

CKPT = ROOT / "validate/pilot/ckpt/PCN_full_s21_o250e250_pcV2_best.pt"
OUT = ROOT / "validate/test_final"
DEVICE = "cuda"

base = PairedDataset(root=str(ROOT / "dataset/pc_v2"), split="test", input_points=1024,
                       gt_points=16384, downsample_seed=20260921)
ds = PCv2("test")
assert len(ds) == 150, len(ds)
dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == 150

model = PCN(16384, 1024, 4).to(DEVICE)
ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval()
print(f"ckpt=PCN_full_s21_o250e250_pcV2_best.pt best_epoch={int(ck.get('epoch', -1))}",
      flush=True)

pdir = OUT / "pred_pcn"
pdir.mkdir(parents=True, exist_ok=True)
hashes, t0 = [], time.time()
with torch.no_grad():
    for i in range(len(ds)):
        item = base[i]
        cav_item = str(item[1])
        assert cav_item == str(dfv.loc[i, "name"]), f"i={i} 映射不一致: {cav_item}"
        obs, _gt = ds[i]
        tp = base.get_meta(i, include_transform=True)["transform_params"]
        _, fine = model(obs.unsqueeze(0).to(DEVICE))
        pred = denormalize(fine[0].cpu().numpy(), tp).astype(np.float32)
        assert pred.shape == (16384, 3) and np.isfinite(pred).all(), f"i={i} 形状/有限性异常"
        fp = pdir / ("%03d.npy" % i)
        np.save(fp, pred)
        hashes.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.name}")
        if (i + 1) % 30 == 0:
            print(f"PCN {i+1}/150 {time.time()-t0:.0f}s", flush=True)

(pdir / "pred_pcn_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
print(f"EXPORT_DONE PCN n=150 best_epoch={int(ck.get('epoch', -1))}", flush=True)
