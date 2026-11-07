# Volume evaluation of the rule-based baselines against both reference volumes.

import numpy as np, pandas as pd
from pathlib import Path

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2*np.pi/N_THETA

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

meta = pd.read_csv(ROOT/'validate/baseline/baseline_meta.csv')
ve = pd.read_csv(ROOT/'validate/pilot/volume_error.csv')
rows = []
for _, m in meta.iterrows():
    i = int(m['i']); xc = float(m['xc']); yc = float(m['yc'])
    zmin = float(m['zmin']); H = float(m['H'])
    wb = wellbore_points(xc, yc, zmin+Z_LO, zmin+Z_HI)
    r = dict(i=i, cav=str(m['cav']), level=int(m['level']))
    for b in ['B1','B2','B3']:
        c = np.vstack([np.load(ROOT/f'validate/baseline/{b}/{i:03d}.npy').astype(np.float64), wb])
        a = radial_areas(c, xc, yc, zmin, H)
        V, _nsl = volume_of(a, zmin, H)
        Vref = float(ve.loc[ve['i']==i, 'V_ref'].iloc[0])
        r['err_'+b] = round(abs(V-Vref)/Vref, 5) if np.isfinite(V) else np.nan
    e = ve.loc[ve['i']==i].iloc[0]
    r['err_BM'] = float(e['err_BM']); r['err_E2'] = float(e['err_E2'])
    rows.append(r)
    if i % 25 == 0:
        print('progress', i, flush=True)

out = pd.DataFrame(rows)[['i','cav','level','err_B1','err_B2','err_B3','err_BM','err_E2']]
out.to_csv(ROOT/'validate/baseline/volume_error_baseline.csv', index=False)
print(out[['err_B1','err_B2','err_B3','err_BM','err_E2']].describe().to_string())
print('means:', out[['err_B1','err_B2','err_B3','err_BM','err_E2']].mean().round(5).to_dict())
print('volume baseline done', len(out), flush=True)
