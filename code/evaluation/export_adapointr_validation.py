import sys, time
from pathlib import Path
import numpy as np, torch, pandas as pd
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
sys.path.insert(0, str(ROOT / "train" / "vendor_src" / "PoinTr-master"))
from train_pilot import load_config, build_model
from paired_data.dataset import PairedDataset
from paired_data.transforms import denormalize

cfg = load_config(str(str(Path(__file__).resolve().parents[2] / 'configs' / 'pc_v2_pilot_bm.yaml')))
device = torch.device("cuda")
model = build_model(cfg["model"], "vendor").to(device)
import argparse as _ap
_ap = _ap.ArgumentParser(); _ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "bm_ep179.pt")); _ap.add_argument("--outdir", default=str(ROOT / "validate/pilot/val150_dump")); _a = _ap.parse_args()
ck = torch.load(_a.ckpt,
                map_location=device, weights_only=False)
model.load_state_dict(ck["model"] if "model" in ck else ck); model.eval()

val_ds = PairedDataset(root=str(ROOT / "dataset" / "pc_v2"), split=str(cfg["data"]["val_split"]),
    input_points=int(cfg["data"]["input_points"]), gt_points=int(cfg["data"]["gt_points"]),
    downsample_seed=int(cfg["data"]["downsample_seed"]))
dfv = pd.read_csv(ROOT / "dataset" / "pc_v2" / "samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == str(cfg["data"]["val_split"])].reset_index(drop=True)
assert len(dfv) == len(val_ds)

outdir = Path(_a.outdir); outdir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with torch.no_grad():
    for i in range(len(val_ds)):
        _, model_id, (partial_n, gt_n) = val_ds[i]
        meta = val_ds.get_meta(i, include_transform=True)
        ret = model(partial_n.unsqueeze(0).to(device))
        fine_m = denormalize(ret[-1][0].float().cpu().numpy(), meta["transform_params"])
        gt_m = denormalize(gt_n.numpy(), meta["transform_params"])
        np.savez(outdir / f"{dfv.loc[i,'name']}_L{int(dfv.loc[i,'level'])}.npz",
                 fine=fine_m.astype(np.float32), gt=gt_m.astype(np.float32))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(val_ds)} elapsed {time.time()-t0:.0f}s")
n = len(list(outdir.glob("*.npz")))
assert n == len(val_ds), n
print("DUMP_DONE", n)
