# -*- coding: utf-8 -*-
"""重绘 fig_perturbation_profile_compare.png（v2）
变更：v1 (b) 竖直轮廓对比中扰动件因径向尺度差显得过小。v2 (b) 改为形状对比：
两条剖面各除以自身最大半径后叠绘，直接展示形状保持一致（行级 Pearson r ≈ 1.0），
并标注归一化偏差的最大绝对值；原始尺度差在 (a) 中按原值呈现（不做缩放）。
数据源同 fig_perturbation_types.py。无平滑。
"""
import io
import tarfile
import numpy as np
import pandas as pd
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = f'{_ARCHIVE}'
OUT = f'{BASE}/figures/fig_perturbation_profile_compare.png'

with tarfile.open(f'{BASE}/source_archive/source_data/radius500_v2.tar.gz') as t:
    ctrl = pd.read_csv(t.extractfile('cav_000.radius.csv'))
with tarfile.open(f'{BASE}/source_archive/data/independent_radius_field.tar.gz') as t:
    fld = np.load(io.BytesIO(t.extractfile('train/gt_field/cav_000.npy').read())).astype(np.float64)

prof_r = ctrl['1353.9395833333335'].values
depth = ctrl['z'].values
row_med = np.median(fld, axis=1)
vrow = prof_r > 0.5
r_corr = np.corrcoef(row_med[vrow], prof_r[vrow])[0, 1]

n_ctl = prof_r / prof_r.max()
n_fld = row_med / row_med.max()
dn = np.abs(n_ctl - n_fld)[vrow]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200)
fig.suptitle('Control vs perturbed cavern geometry, cav_000 (perturb group): identical outline shape '
             f'(row-wise Pearson r ≈ {r_corr:.2f}) at a reduced, near-uniform radial scale',
             fontsize=12)

ax = axes[0]
ax.plot(prof_r, depth, color='#1f77b4', lw=1.7, label='Control design profile')
ax.plot(row_med, depth, color='#c8321e', lw=1.7, label='Perturbed field (median over θ)')
ax.fill_betweenx(depth, fld.min(axis=1), fld.max(axis=1), color='#c8321e', alpha=0.15,
                 label='Field min–max over θ')
ax.set_ylim(depth.max(), depth.min())
ax.set_xlabel('Radius (m)'); ax.set_ylabel('Depth (m)')
ax.set_title('(a) Profile comparison (as-is, no rescaling)', fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower right', frameon=False)

ax = axes[1]
ax.plot(n_ctl, depth, color='#1f77b4', lw=1.7, label='Control (÷ own max)')
ax.plot(n_fld, depth, color='#c8321e', lw=1.7, ls='--', label='Perturbed (÷ own max)')
ax.set_ylim(depth.max(), depth.min())
ax.set_xlabel('Normalised radius'); ax.set_ylabel('Depth (m)')
ax.set_title('(b) Shape comparison (each curve ÷ its own maximum)', fontsize=10.5)
ax.text(0.03, 0.96, f'max |Δ normalised| = {dn.max():.3f} (valid rows, r > 0.5 m)',
        transform=ax.transAxes, fontsize=8.5, va='top', color='0.25')
ax.legend(fontsize=8.5, loc='center right', frameon=False)

for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig.text(0.5, -0.03,
         'Row medians and min–max computed cell-wise on valid cells (r > 0.5 m). Depths from the Sansmic profile rows (201 rows, floor → top). '
         'The perturbed field is azimuthally uniform\nwithin each grid row, so the min–max band collapses onto the median. '
         'The radial-scale difference between the design profile and the calibrated field is part of the  calibration and is presented as-is in panel (a).',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)
