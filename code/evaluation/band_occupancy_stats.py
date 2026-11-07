# -*- coding: utf-8 -*-
"""WP1 观测侧统计：val 150 样本带内观测/gt/面积占比（带界 zmin+2..zmin+60，辅列 2..40）。
列：name,level,n_obs1024,n_obs1024_40,o1024,o1024_40,n_obs8192,o8192,n_gt,g,a_area(近似),H,zmin
o=带内观测/总观测; g=带内gt/16384; a_area=带表面积占比(gt_field逐行中位半径×dz加权,线性z映射近似)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from paired_data.dataset import PairedDataset
from paired_data.transforms import denormalize

Z_LO, Z_HI = 2.0, 60.0
ds = PairedDataset(root=str(ROOT/"dataset/pc_v2"), split='val', input_points=1024,
                     gt_points=16384, downsample_seed=20260921)
dfv = pd.read_csv(ROOT/"dataset/pc_v2/samples.csv", encoding='utf-8-sig')
dfv = dfv[dfv['split']=='val'].reset_index(drop=True)
rows = []
for i in range(len(ds)):
    name = dfv.loc[i,'name']; level = int(dfv.loc[i,'level'])
    stem = f"{name}_L{level}"
    _, _, (partial_n, gt_n) = ds[i]
    meta = ds.get_meta(i, include_transform=True)
    obs1024 = denormalize(partial_n.numpy(), meta['transform_params'])
    obs8192 = np.load(ROOT/f"dataset/pc_v2/val/partial/{stem}.npy")
    gt = denormalize(gt_n.numpy(), meta['transform_params'])
    gf_path = ROOT/f"dataset/pc_v2/val/gt_field/{name}.npy"
    zmin = float(gt[:,2].min()); zmax = float(gt[:,2].max()); H = zmax - zmin
    m60_o1 = (obs1024[:,2]>zmin+Z_LO)&(obs1024[:,2]<zmin+Z_HI)
    m40_o1 = (obs1024[:,2]>zmin+Z_LO)&(obs1024[:,2]<zmin+40.0)
    m60_o8 = (obs8192[:,2]>zmin+Z_LO)&(obs8192[:,2]<zmin+Z_HI)
    m60_g  = (gt[:,2]>zmin+Z_LO)&(gt[:,2]<zmin+Z_HI)
    a_area = np.nan
    if gf_path.exists() and H > 0:
        gf = np.load(gf_path)              # (201,360) 半径网格
        r_row = np.median(gf, axis=1)
        n_r = len(r_row)
        z_row = zmin + np.arange(n_r) * H / (n_r - 1)
        w = r_row
        a_area = float(w[(z_row>zmin+Z_LO)&(z_row<zmin+Z_HI)].sum() / w.sum())
    rows.append(dict(name=name, level=level,
        n_obs1024=int(m60_o1.sum()), n_obs1024_40=int(m40_o1.sum()),
        o1024=round(float(m60_o1.mean()),5), o1024_40=round(float(m40_o1.mean()),5),
        n_obs8192=int(m60_o8.sum()), o8192=round(float(m60_o8.mean()),5),
        n_gt=int(m60_g.sum()), g=round(float(m60_g.mean()),5),
        a_area=round(a_area,5) if a_area==a_area else np.nan,
        H=round(H,1), zmin=round(zmin,1)))
out = pd.DataFrame(rows)
out.to_csv(ROOT/"validate/pilot/band_occupancy_stats.csv", index=False)
print(out.head(8).to_string(index=False))
print('rows:', len(out), '| NaN count:', int(out.isna().sum().sum() - out['a_area'].isna().sum()))
print('o1024 describe:'); print(out['o1024'].describe().round(5).to_string())

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives
