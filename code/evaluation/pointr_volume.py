# PoinTr baseline: volume through the shared closure pipeline.

import numpy as np, pandas as pd
from pathlib import Path

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT); VOL = ROOT/'validate/volume'
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2*np.pi/N_THETA

dfv = pd.read_csv(ROOT/'dataset/pc_v2/samples.csv', encoding='utf-8-sig')
dfv = dfv[dfv['split']=='val'].reset_index(drop=True)
assert len(dfv) == 150

def wellbore_points(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZ_STEP)
    th = np.linspace(0, 2*np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([xc+RW*np.cos(T).ravel(), yc+RW*np.sin(T).ravel(), Z.ravel()]).astype(np.float64)

def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H/N_SLICES
    z = pts[:,2]
    k = np.clip(((z-zmin)/dz).astype(int), 0, N_SLICES-1)
    th = np.mod(np.arctan2(pts[:,1]-yc, pts[:,0]-xc), 2*np.pi)
    rr = np.hypot(pts[:,0]-xc, pts[:,1]-yc)
    tb = np.clip((th/DTH).astype(int), 0, N_THETA-1)
    A = np.zeros((N_SLICES, N_THETA)); C = np.zeros((N_SLICES, N_THETA), dtype=int)
    np.maximum.at(A, (k, tb), rr); np.add.at(C, (k, tb), 1)
    idx = np.arange(N_THETA)
    areas = np.full(N_SLICES, np.nan)
    for s in range(N_SLICES):
        m = C[s] > 0
        if m.sum() < 3:
            continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi-N_THETA, oi, oi+N_THETA])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        filled = np.interp(idx, ex_i[o], ex_v[o])
        areas[s] = 0.5*float(np.trapz(filled**2, dx=DTH))
    return areas

def volume_of(areas, zmin, H):
    dz = H/N_SLICES
    zc = zmin + (np.arange(N_SLICES)+0.5)*dz
    m = ~np.isnan(areas)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.trapz(areas[m], zc[m])), int(m.sum())

rows = []
for i in range(150):
    gt = np.load(VOL/'gt'/f'{i:03d}.npy').astype(np.float64)
    zmin = float(gt[:,2].min()); zmax = float(gt[:,2].max()); H = zmax-zmin
    xc, yc = float(gt[:,0].mean()), float(gt[:,1].mean())
    name = dfv.loc[i,'name']
    gf = np.load(ROOT/f'dataset/pc_v2/val/gt_field/{name}.npy')
    gf2 = np.where(gf > 0, gf, 0.0)
    V_field = float(np.pi*np.mean(gf2**2, axis=1).sum()*H/200)
    wb = wellbore_points(xc, yc, zmin+Z_LO, zmin+Z_HI)
    clouds = {'ref': np.vstack([gt, wb]),
              'GPoinTr': np.vstack([np.load(VOL/'pred_GPoinTr'/f'{i:03d}.npy').astype(np.float64), wb])}
    V = {}
    for tag, c in clouds.items():
        a = radial_areas(c, xc, yc, zmin, H)
        V[tag], _nsl = volume_of(a, zmin, H)
    calib = V['ref']/V_field
    rows.append(dict(i=i, cav=name, level=int(dfv.loc[i,'level']),
        V_ref=round(V['ref'],1), V_field=round(V_field,1),
        V_GPoinTr=round(V['GPoinTr'],1),
        err_field_pct=round((V['GPoinTr']/V_field-1)*100, 3),
        err_ref_pct=round((V['GPoinTr']/V['ref']-1)*100, 3),
        calib=round(calib, 6)))
    if i % 25 == 0:
        print('progress', i, flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT/'validate/pilot/volume_error_pointr.csv', index=False)
print('mean err_field_pct', round(df['err_field_pct'].mean(),3), 'median', round(df['err_field_pct'].median(),3))
print('mean err_ref_pct', round(df['err_ref_pct'].mean(),3), 'median', round(df['err_ref_pct'].median(),3))
print('pointr_volume done', flush=True)
