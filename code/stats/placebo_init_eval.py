# placebo_init_eval.py — R4 安慰剂对照 / R5 初始化维度 全支撑评估
# 管线复刻 source_archive/mu/init_eval.py（常量与函数逐行一致，不依赖 ckpt）
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES, N_THETA = 500, 72
DTH = 2 * np.pi / N_THETA

ROOT = Path(_ROOT)
VOL = ROOT / 'validate/volume'
GT_FIELD = ROOT / 'dataset/pc_v2/val/gt_field'
WORK = ROOT / 'v44_补算'

def wellbore_points(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZ_STEP)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([xc + RW * np.cos(T).ravel(), yc + RW * np.sin(T).ravel(),
                            Z.ravel()]).astype(np.float64)

def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H / N_SLICES
    k = np.clip(((pts[:, 2] - zmin) / dz).astype(int), 0, N_SLICES - 1)
    th = np.mod(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, N_THETA - 1)
    A = np.zeros((N_SLICES, N_THETA))
    C = np.zeros((N_SLICES, N_THETA), dtype=int)
    np.maximum.at(A, (k, tb), rr)
    np.add.at(C, (k, tb), 1)
    idx = np.arange(N_THETA)
    areas = np.full(N_SLICES, np.nan)
    for s in range(N_SLICES):
        m = C[s] > 0
        if m.sum() < 3:
            continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi - N_THETA, oi, oi + N_THETA])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        filled = np.interp(idx, ex_i[o], ex_v[o])
        areas[s] = 0.5 * float(np.trapz(filled ** 2, dx=DTH))
    return areas

def volume_of(areas, zmin, H):
    dz = H / N_SLICES
    zc = zmin + (np.arange(N_SLICES) + 0.5) * dz
    m = ~np.isnan(areas)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.trapz(areas[m], zc[m])), int(m.sum())

def cd_l1_m(P, G):
    """与 init_eval 同公式：0.5*(mean_min(P,G)+mean_min(G,P))，精确最近邻"""
    tP, tG = cKDTree(P), cKDTree(G)
    return 0.5 * (tP.query(G, k=1)[0].mean() + tG.query(P, k=1)[0].mean())

# ---- meta（每样本一次） ----
dfv = pd.read_csv(_ROOT / 'dataset/pc_v2/samples.csv', encoding='utf-8-sig')
dfv = dfv[dfv['split'] == 'val'].reset_index(drop=True)
assert len(dfv) == 150

meta = {}
for i in range(150):
    gt = np.load(VOL / 'gt' / f'{i:03d}.npy').astype(np.float64)
    zmin = float(gt[:, 2].min()); zmax = float(gt[:, 2].max()); H = zmax - zmin
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)
    V_ref, nsl = volume_of(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)
    gf = np.load(GT_FIELD / f"{dfv.loc[i, 'name']}.npy")
    gf2 = np.where(gf > 0, gf, 0.0)
    V_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
    meta[i] = dict(gt=gt, xc=xc, yc=yc, zmin=zmin, H=H, wb=wb,
                   V_ref=V_ref, V_field=V_field)
print('meta ok', flush=True)

# ---- 臂定义 ----
def find_pred_dir(tag):
    for cand in [WORK / tag, WORK / tag / tag]:
        if (cand / '000.npy').exists():
            return cand
    raise FileNotFoundError(tag)

ARMS = [
    dict(pair='HIST',  arm='BM', tag='掩码配置(主运行)',  path=VOL / 'pred_adapointr_masked'),
    dict(pair='MV',    arm='PL1', tag='对照1-掩码带错位', path=find_pred_dir('pred_MV_PL1')),
    dict(pair='MV',    arm='PL2', tag='对照2-随机同量剔除', path=find_pred_dir('pred_MV_PL2')),
    dict(pair='MV',    arm='PL3', tag='对照3-仅预测侧剔除', path=find_pred_dir('pred_MV_PL3')),
    dict(pair='MU2',   arm='BM', tag='随机初始化-掩码',  path=find_pred_dir('pred_MU2-BM')),
    dict(pair='MU2',   arm='E2', tag='随机初始化-全区域', path=find_pred_dir('pred_MU2-E2')),
]

rows = []
for spec in ARMS:
    for i in range(150):
        m = meta[i]
        pred = np.load(spec['path'] / f'{i:03d}.npy').astype(np.float64)
        V_pred, n_pred = volume_of(radial_areas(np.vstack([pred, m['wb']]), m['xc'], m['yc'], m['zmin'], m['H']), m['zmin'], m['H'])
        rows.append({
            'pair': spec['pair'], 'arm': spec['arm'], 'tag': spec['tag'],
            'i': i, 'cav': dfv.loc[i, 'name'], 'level': int(dfv.loc[i, 'level']),
            'cd_l1_m': cd_l1_m(pred, m['gt']),
            'V_pred': V_pred, 'V_field': m['V_field'], 'V_ref': m['V_ref'],
            'err_field_abs': abs(V_pred - m['V_field']) / m['V_field'],
            'err_ref_abs': abs(V_pred - m['V_ref']) / m['V_ref'],
            'N_slice_pred': n_pred,
        })
    print(f"{spec['tag']} done", flush=True)

out = pd.DataFrame(rows)
out.to_csv(WORK / 'R4_R5_eval_per_sample.csv', index=False, encoding='utf-8-sig')
print('saved', WORK / 'R4_R5_eval_per_sample.csv')
print(out.groupby(['pair', 'arm'])[['err_field_abs', 'err_ref_abs', 'cd_l1_m']].mean().round(5))
