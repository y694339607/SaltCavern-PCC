# Rule-based completion baselines (Cyl-NN / periodic spline / IDW), validation split.

import sys, numpy as np, pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from paired_data.dataset import PairedDataset
from paired_data.transforms import denormalize
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

OUT = ROOT/'validate/baseline'
for b in ['B1','B2','B3']:
    (OUT/b).mkdir(parents=True, exist_ok=True)

N_SLICES, N_THETA = 500, 72
DTH = 2*np.pi/N_THETA
SEED, N_OUT = 20261015, 16384
B2_MIN_OBS, B3_POWER, B3_K = 3, 2, 8

ds = PairedDataset(root=f'{_ROOT}/dataset/pc_v2', split='val', input_points=1024,
                     gt_points=16384, downsample_seed=20260921)
dfv = pd.read_csv(ROOT/'dataset/pc_v2/samples.csv', encoding='utf-8-sig')
dfv = dfv[dfv['split']=='val'].reset_index(drop=True)
assert len(dfv)==150, len(dfv)

def bin_obs(obs, xc, yc, zmin, H):
    dz = H/N_SLICES
    k = np.clip(((obs[:,2]-zmin)/dz).astype(int), 0, N_SLICES-1)
    th = np.mod(np.arctan2(obs[:,1]-yc, obs[:,0]-xc), 2*np.pi)
    rr = np.hypot(obs[:,0]-xc, obs[:,1]-yc)
    tb = np.clip((th/DTH).astype(int), 0, N_THETA-1)
    A = np.zeros((N_SLICES, N_THETA)); C = np.zeros((N_SLICES, N_THETA), dtype=int)
    np.maximum.at(A, (k, tb), rr); np.add.at(C, (k, tb), 1)
    return A, C

def circ_fill(A, C):
    idx = np.arange(N_THETA)
    Af = np.where(C>0, A, np.nan)
    for s in range(N_SLICES):
        m = C[s]>0
        if m.sum()==0 or m.sum()==N_THETA: continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi-N_THETA, oi, oi+N_THETA])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        Af[s] = np.interp(idx, ex_i[o], ex_v[o])
    return Af

def z_fill(Af):
    occ = np.where(np.all(np.isfinite(Af), axis=1))[0]
    if len(occ)<2: return Af
    out = Af.copy()
    for t in range(N_THETA):
        out[:,t] = np.interp(np.arange(N_SLICES), occ, Af[occ,t])
    return out

def field_b1(obs, xc, yc, zmin, H):
    A, C = bin_obs(obs, xc, yc, zmin, H)
    return z_fill(circ_fill(A, C))

def field_b2(obs, xc, yc, zmin, H):
    A, C = bin_obs(obs, xc, yc, zmin, H)
    Rz = np.full((N_SLICES, N_THETA), np.nan)
    tb_ctr = (np.arange(N_THETA)+0.5)*DTH
    for s in range(N_SLICES):
        if C[s].sum() < B2_MIN_OBS: continue
        m = C[s]>0
        th = np.mod(np.arctan2(obs[:,1]-yc, obs[:,0]-xc), 2*np.pi)
        # 按切片取观测角
        k = np.clip(((obs[:,2]-zmin)/(H/N_SLICES)).astype(int), 0, N_SLICES-1)
        sel = (k==s)
        th_s = np.mod(np.arctan2(obs[sel,1]-yc, obs[sel,0]-xc), 2*np.pi)
        rr_s = np.hypot(obs[sel,0]-xc, obs[sel,1]-yc)
        o = np.argsort(th_s)
        th_s, rr_s = th_s[o], rr_s[o]
        th2 = np.concatenate([th_s, th_s[:1]+2*np.pi])
        rr2 = np.concatenate([rr_s, rr_s[:1]])
        sp = CubicSpline(th2, rr2, bc_type='periodic')
        Rz[s] = sp(tb_ctr)
    return z_fill(Rz)

def idw_b3(obs, xc, yc, z, th):
    z0, z1 = obs[:,2].min(), obs[:,2].max()
    if z1-z0 < 1e-6: z1 = z0+1.0
    th_o = np.mod(np.arctan2(obs[:,1]-yc, obs[:,0]-xc), 2*np.pi)
    rr_o = np.hypot(obs[:,0]-xc, obs[:,1]-yc)
    u, v = th_o/(2*np.pi), (obs[:,2]-z0)/(z1-z0)
    P3 = np.concatenate([np.c_[u-1,v], np.c_[u,v], np.c_[u+1,v]])
    R3 = np.tile(rr_o, 3)
    tree = cKDTree(P3)
    k = min(B3_K, len(obs))
    vq = np.clip((z-z0)/(z1-z0), 0, 1)
    uq = th/(2*np.pi)
    d, idx = tree.query(np.c_[uq,vq], k=k)
    d = np.maximum(d, 1e-12)
    w = 1.0/d**B3_POWER
    return (w*R3[idx]).sum(axis=1)/w.sum(axis=1)

rng = np.random.default_rng(SEED)
meta_rows = []
for i in range(150):
    _, cav, (obs, gt) = ds[i]
    meta = ds.get_meta(i, include_transform=True)
    tp = meta['transform_params']
    obs = denormalize(obs.numpy(), tp).astype(np.float64)
    gt = denormalize(gt.numpy(), tp).astype(np.float64)
    zmin = float(gt[:,2].min()); H = float(gt[:,2].max())-zmin
    xc = float(gt[:,0].mean()); yc = float(gt[:,1].mean())
    zo0, zo1 = float(obs[:,2].min()), float(obs[:,2].max())
    z = rng.uniform(zo0, zo1, N_OUT)
    th = rng.uniform(0, 2*np.pi, N_OUT)
    dz = H/N_SLICES
    s_f = np.clip((z-zmin)/dz, 0, N_SLICES-1)
    t_f = th/DTH
    F1 = field_b1(obs, xc, yc, zmin, H)
    F2 = field_b2(obs, xc, yc, zmin, H)
    F1e = np.concatenate([F1, F1[:,:1]], axis=1)
    F2e = np.concatenate([F2, F2[:,:1]], axis=1)
    r1 = map_coordinates(F1e, [s_f, t_f], order=1, mode='nearest')
    r2 = map_coordinates(F2e, [s_f, t_f], order=1, mode='nearest')
    r3 = idw_b3(obs, xc, yc, z, th)
    for tag, r in [('B1',r1),('B2',r2),('B3',r3)]:
        pts = np.column_stack([xc+r*np.cos(th), yc+r*np.sin(th), z]).astype(np.float32)
        np.save(OUT/tag/f'{i:03d}.npy', pts)
    meta_rows.append(dict(i=i, cav=str(dfv.loc[i,'name']), level=int(dfv.loc[i,'level']),
                          xc=xc, yc=yc, zmin=zmin, H=H, zobs0=zo0, zobs1=zo1))
    if i==0:
        print('sample0 obs z range %.1f..%.1f  gt H %.1f  r1 nan%%=%.1f' %
              (zo0, zo1, H, 100*np.isnan(r1).mean()), flush=True)
    if i%25==0:
        print('progress', i, flush=True)
pd.DataFrame(meta_rows).to_csv(OUT/'baseline_meta.csv', index=False)
print('recon done', len(meta_rows), flush=True)
