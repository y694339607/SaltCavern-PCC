import os, time
import numpy as np
import pandas as pd
import matplotlib
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

GT_DIR = f'{_ROOT}/validate/volume/gt'
PRED_DIRS = {'masked': f'{_ROOT}/validate/volume/pred_adapointr_masked',
             'full': f'{_ROOT}/validate/volume/pred_adapointr_full',
             'pcn': f'{_ROOT}/validate/baseline_dl/pcn/pred',
             'cylnn': f'{_ROOT}/validate/baseline/B1'}
OBS_DIR = f'{_ROOT}/dataset/pc_v2/val/partial'
META = f'{_ROOT}/dataset/pc_v2/samples.csv'
N = 150
OUT = f'{_ROOT}/validate/pilot/atlas_out'
os.makedirs(OUT, exist_ok=True)

meta = pd.read_csv(META)
val = meta[meta['split'] == 'val'].reset_index(drop=True)
assert len(val) == N, f'val rows {len(val)}'

rows = []
t0 = time.time()
for i in range(N):
    gt = np.load(f'{GT_DIR}/{i:03d}.npy').astype(np.float64)
    obs_f = f"{OBS_DIR}/{val.loc[i, 'name']}_L{int(val.loc[i, 'level'])}.npy"
    obs = np.load(obs_f).astype(np.float64)
    base_xy = gt[:, :2].mean(axis=0)
    zmin, zmax = gt[:, 2].min(), gt[:, 2].max()
    def cyl(p):
        r = np.hypot(p[:, 0] - base_xy[0], p[:, 1] - base_xy[1])
        th = np.mod(np.arctan2(p[:, 1] - base_xy[1], p[:, 0] - base_xy[0]), 2 * np.pi)
        zn = (p[:, 2] - zmin) / (zmax - zmin)
        return r, th, zn
    r_gt, th_gt, zn_gt = cyl(gt)
    r_obs, th_obs, zn_obs = cyl(obs)
    q1, q2 = np.quantile(r_gt, [1.0/3.0, 2.0/3.0])
    def bands(r, th, zn):
        z10 = np.clip((zn * 10).astype(int), 0, 9)
        th12 = np.clip((th / (np.pi / 6.0)).astype(int), 0, 11)
        rl = np.digitize(r, [q1, q2])
        return z10, th12, rl
    z10_g, th12_g, rl_g = bands(r_gt, th_gt, zn_gt)
    z10_o, th12_o, rl_o = bands(r_obs, th_obs, zn_obs)
    gdf = pd.DataFrame({'z10': z10_g, 'th12': th12_g, 'rl': rl_g, 'r': r_gt})
    ginfo = gdf.groupby(['z10', 'th12', 'rl']).agg(r_std=('r', 'std'), max_r_gt=('r', 'max')).reset_index()
    odf = pd.DataFrame({'z10': z10_o, 'th12': th12_o, 'rl': rl_o})
    ocnt = odf.groupby(['z10', 'th12', 'rl']).size().rename('obs_cnt').reset_index()
    for m, d in PRED_DIRS.items():
        pred = np.load(f'{d}/{i:03d}.npy').astype(np.float64)
        d_p2g = cKDTree(gt).query(pred)[0]
        d_g2p = cKDTree(pred).query(gt)[0]
        r_pr, th_pr, zn_pr = cyl(pred)
        z10_p, th12_p, rl_p = bands(r_pr, th_pr, zn_pr)
        pdf = pd.DataFrame({'z10': z10_p, 'th12': th12_p, 'rl': rl_p, 'r': r_pr, 'd': d_p2g})
        pinf = pdf.groupby(['z10', 'th12', 'rl']).agg(m_p2g=('d', 'mean'), max_r_pred=('r', 'max')).reset_index()
        gside = pd.DataFrame({'z10': z10_g, 'th12': th12_g, 'rl': rl_g, 'd': d_g2p}).groupby(['z10', 'th12', 'rl']).agg(m_g2p=('d', 'mean')).reset_index()
        cell = pinf.merge(gside, on=['z10', 'th12', 'rl'], how='outer').merge(ginfo, on=['z10', 'th12', 'rl'], how='left').merge(ocnt, on=['z10', 'th12', 'rl'], how='left')
        cell['obs_cnt'] = cell['obs_cnt'].fillna(0)
        cell['sym'] = (cell['m_p2g'] + cell['m_g2p']) / 2.0
        cell['inflate'] = cell['max_r_pred'] - cell['max_r_gt']
        cell['idx'] = i; cell['cav'] = val.loc[i, 'name']; cell['method'] = m
        cell['level'] = int(val.loc[i, 'level']); cell['shadow'] = int(val.loc[i, 'shadow_limited'])
        cell['coverage'] = float(val.loc[i, 'coverage'])
        cell['z_center'] = (cell['z10'] + 0.5) / 10.0
        rows.append(cell[['idx', 'cav', 'method', 'level', 'shadow', 'coverage', 'z10', 'th12', 'rl', 'z_center', 'm_p2g', 'm_g2p', 'sym', 'inflate', 'obs_cnt', 'r_std']])
    if i % 25 == 0:
        print(f'progress {i} elapsed {time.time()-t0:.0f}s', flush=True)

df = pd.concat(rows, ignore_index=True)
df.rename(columns={'r_std': 'rough'}, inplace=True)
df.to_csv(f'{OUT}/cell_errors.csv', index=False)

methods = ['BM', 'E2', 'PCN', 'B1']
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for j, m in enumerate(methods):
    sub = df[df.method == m]
    piv = sub.pivot_table(index='z10', columns='th12', values='sym', aggfunc='mean')
    im = axes[0, j].imshow(piv.values, origin='lower', aspect='auto', cmap='viridis')
    axes[0, j].set_title(f'{m} (z,theta) sym CD (m)')
    axes[0, j].set_xlabel('theta sector'); axes[0, j].set_ylabel('z band (0=bottom)')
    fig.colorbar(im, ax=axes[0, j])
    prof = sub.groupby('z10')['sym'].mean().reindex(range(10))
    axes[1, j].plot(np.arange(10), prof.values, marker='o')
    axes[1, j].set_title(f'{m} z profile')
    axes[1, j].set_xlabel('z band (0=bottom)'); axes[1, j].set_ylabel('sym CD (m)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_error_atlas.png', dpi=150); plt.close()

def attrib(sub, name):
    d = sub.dropna(subset=['sym', 'rough', 'obs_cnt']).reset_index(drop=True)
    feats_all = ['obs_cnt', 'rough', 'z_center', 'rl', 'shadow', 'coverage', 'level']
    X = d[feats_all].values.astype(float)
    sd = X.std(0)
    keep = [j for j in range(len(feats_all)) if np.isfinite(sd[j]) and sd[j] > 1e-12]
    dropped = [feats_all[j] for j in range(len(feats_all)) if j not in keep]
    if dropped:
        print(f'attrib {name}: constant columns dropped: {dropped}', flush=True)
    feats = [feats_all[j] for j in keep]
    X = X[:, keep]
    y = d['sym'].values
    Xs = (X - X.mean(0)) / X.std(0); ys = (y - y.mean()) / y.std()
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
    r2 = 1.0 - ((ys - Xd @ beta) ** 2).sum() / (ys ** 2).sum()
    sp = d[feats + ['sym']].corr(method='spearman')['sym'].drop('sym')
    out = pd.DataFrame({'factor': feats, 'ols_beta': beta[1:], 'spearman': sp.values, 'r2': round(r2, 4)})
    out.to_csv(f'{OUT}/attribution_{name}.csv', index=False)
    print(f'--- attribution {name} R2={r2:.3f}', flush=True)
    print(out.reindex(out['ols_beta'].abs().sort_values(ascending=False).index).to_string(index=False), flush=True)
attrib(df, 'all')
attrib(df[df.method == 'BM'], 'bm')

bm = df[df.method == 'BM'].dropna(subset=['sym'])
thr = bm['sym'].quantile(0.95)
w = bm[bm['sym'] >= thr]
summary = pd.DataFrame({'worst_median': w[['obs_cnt', 'rough', 'z_center', 'rl']].median(),
                        'global_median': bm[['obs_cnt', 'rough', 'z_center', 'rl']].median()})
summary.to_csv(f'{OUT}/worst_cell_summary.csv')
print(f'--- worst-cell top5% (BM, threshold {thr:.3f} m)', flush=True)
print(summary.to_string(), flush=True)
print('worst shadow ratio %.3f global %.3f' % (w['shadow'].mean(), bm['shadow'].mean()), flush=True)
print('worst level counts:'); print(w['level'].value_counts().sort_index().to_string(), flush=True)

print('--- band means bottom(0-1) vs top(8-9)', flush=True)
for m in methods:
    s = df[df.method == m]
    b = s[s.z10 <= 1]['sym'].mean(); t = s[s.z10 >= 8]['sym'].mean()
    print(f'{m} bottom={b:.3f} top={t:.3f} ratio={b/t:.2f}', flush=True)

cand = {'obs_cnt': '观测密度主导 -> 稀疏区加权监督 / 逐点残差自校准重启',
        'rough': '粗糙度主导 -> 解码容量/细粒度损失',
        'z_center': '端部带主导 -> 端部范围正则（论文展望，决议41路线②）',
        'rl': '径向层梯度 -> 分区加权监督',
        'shadow': '阴影标志相关 -> shadow 工况分层干预',
        'coverage': '覆盖度主导 -> 按覆盖分层部署/低覆盖专项',
        'level': '遮挡等级相关 -> 按 level 分层干预'}
att = pd.read_csv(f'{OUT}/attribution_all.csv')
tt = att.reindex(att['ols_beta'].abs().sort_values(ascending=False).index).copy()
tt['candidate'] = tt['factor'].map(cand)
tt.to_csv(f'{OUT}/target_table.csv', index=False)
print('--- target table', flush=True); print(tt.to_string(index=False), flush=True)

ve = pd.read_csv(f'{_ROOT}/validate/pilot/volume_error.csv')
print('volume_error.csv columns:', ve.columns.tolist(), flush=True)
cd_cols = [c for c in ve.columns if 'cd' in c.lower()]
if cd_cols:
    bm_cav = bm.groupby('idx')['sym'].mean().sort_index().values
    ref = ve[cd_cols[0]].values[:N]
    k = min(len(bm_cav), len(ref))
    print('mapping self-check corr(BM sym, %s) = %.3f' % (cd_cols[0], np.corrcoef(bm_cav[:k], ref[:k])[0, 1]), flush=True)
else:
    print('no cd column found, self-check skipped', flush=True)
print(f'ATLAS DONE elapsed {time.time()-t0:.0f}s', flush=True)
