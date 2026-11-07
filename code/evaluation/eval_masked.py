# Unified CD/HD and point-density diagnostics for one exported run.

import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
V = ROOT/'validate/volume'
B = ROOT/'validate/baseline'
RW = 0.61; Z_LO, Z_HI = 2.0, 60.0; DZS = 0.5
NS, NT = 500, 72
DTH = 2*np.pi/NT

def wb_pts(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZS)
    th = np.linspace(0, 2*np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.c_[xc+RW*np.cos(T).ravel(), yc+RW*np.sin(T).ravel(), Z.ravel()]

def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H/NS
    k = np.clip(((pts[:,2]-zmin)/dz).astype(int), 0, NS-1)
    th = np.mod(np.arctan2(pts[:,1]-yc, pts[:,0]-xc), 2*np.pi)
    rr = np.hypot(pts[:,0]-xc, pts[:,1]-yc)
    tb = np.clip((th/DTH).astype(int), 0, NT-1)
    A = np.zeros((NS, NT)); C = np.zeros((NS, NT), dtype=int)
    np.maximum.at(A, (k, tb), rr); np.add.at(C, (k, tb), 1)
    idx = np.arange(NT)
    areas = np.full(NS, np.nan)
    for s in range(NS):
        m = C[s] > 0
        if m.sum() < 3: continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi-NT, oi, oi+NT])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        filled = np.interp(idx, ex_i[o], ex_v[o])
        areas[s] = 0.5*float(np.trapz(filled**2, dx=DTH))
    return areas

def volume_of(areas, zmin, H):
    dz = H/NS
    zc = zmin + (np.arange(NS)+0.5)*dz
    m = ~np.isnan(areas)
    if m.sum() < 10: return np.nan, int(m.sum())
    return float(np.trapz(areas[m], zc[m])), int(m.sum())

def cd_hd(a, b):
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    return float(d_ab.mean()+d_ba.mean()), float(max(d_ab.max(), d_ba.max()))

meta = pd.read_csv(B/'baseline_meta.csv').set_index('i')
ve = pd.read_csv(ROOT/'validate/pilot/volume_error.csv')
arms = {'BM': V/'pred_adapointr_masked', 'E2': V/'pred_adapointr_full', 'GPtrV': V/'pred_GPtrV',
        'GBMw30': V/'pred_GBMw30', 'GBMw100': V/'pred_GBMw100',
        'PCN': ROOT/'validate/baseline_dl/pcn/pred'}
rows = []
for tag, pth in arms.items():
    cds, hds, errs = [], [], []
    for i in range(150):
        gt = np.load(V/f'gt/{i:03d}.npy').astype(np.float64)
        p = np.load(pth/f'{i:03d}.npy').astype(np.float64)
        c, h = cd_hd(p, gt); cds.append(c); hds.append(h)
        m = meta.loc[i]
        xc, yc = float(m['xc']), float(m['yc'])
        zmin, H = float(m['zmin']), float(m['H'])
        cl = np.vstack([p, wb_pts(xc, yc, zmin+Z_LO, zmin+Z_HI)])
        a = radial_areas(cl, xc, yc, zmin, H)
        Vol, _ = volume_of(a, zmin, H)
        Vref = float(ve.loc[ve['i']==i, 'V_ref'].iloc[0])
        errs.append(abs(Vol-Vref)/Vref if np.isfinite(Vol) else np.nan)
    rows.append(dict(arm=tag, cd_mean=float(np.mean(cds)), cd_median=float(np.median(cds)),
                     hd_mean=float(np.mean(hds)), hd_median=float(np.median(hds)),
                     vol_mean=float(np.nanmean(errs)), vol_median=float(np.nanmedian(errs)),
                     vol_max=float(np.nanmax(errs))))
    print(tag, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(B/'g_positioning.csv', index=False)
print(df.round(4).to_string())
print('g_eval done', flush=True)
