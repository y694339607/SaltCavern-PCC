# PCN baseline: export 16384-point predictions on the validation split.

import sys
import numpy as np, pandas as pd, torch
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(_ROOT)
sys.path.insert(0, f'{_ROOT}/coding/pcn_baseline')
from models import PCN
from dataset_pc import PCv2
from pcn_loss import cd_l1
from paired_data.transforms import denormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models'))
from paired_data.dataset import PairedDataset

CKPT = ROOT/'validate/pilot/ckpt/pcn_main_e250.pt'
OUT = ROOT/'validate/baseline_dl/pcn'
DEVICE = 'cuda'


@torch.no_grad()
def main():
    base = PairedDataset(root=str(ROOT/'dataset/pc_v2'), split='val',
                           input_points=1024, gt_points=16384, downsample_seed=20260921)
    ds = PCv2('val')
    model = PCN(16384, 1024, 4).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model']); model.eval()
    (OUT/'pred').mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(ds)):
        obs, gt = ds[i]
        meta = base.get_meta(i, include_transform=True)
        tp = meta['transform_params']
        _, fine = model(obs.unsqueeze(0).to(DEVICE))
        pred = fine[0].cpu().numpy()
        pred_phys = denormalize(pred, tp).astype(np.float32)
        np.save(OUT/'pred'/('%03d.npy' % i), pred_phys)
        cd = float(cd_l1(fine.cpu(), gt.unsqueeze(0)))
        rows.append(dict(i=i, cav=meta.get('taxonomy', ''), l1_cd=cd))
        if i % 25 == 0: print('progress', i, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'pcn_val_cd.csv', index=False)
    print(df.describe().to_string(), flush=True)
    print('saved', OUT)


if __name__ == '__main__':
    main()
