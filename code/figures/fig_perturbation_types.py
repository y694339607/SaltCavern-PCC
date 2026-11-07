# -*- coding: utf-8 -*-
"""重绘 fig_perturbation_quad.png（v2）
变更：v1 的残差面板（场 − 控制剖面，mean −29.27 m）由近均匀径向比例因子（≈0.311，
 标定的一部分）主导，易被误读为扰动误差。v2 将残差分解呈现：
  (c) 逐行径向比 field/control（有效行）；
  (d) 去除最小二乘比例后的非仿射残差直方图（真实几何扰动量级），
      并注明原始单元格偏移统计（mean −29.27 m, σ 27.47 m，按 v1 有效格 r>0.5 口径）。
新增披露：扰动场在每一网格行内沿 θ 恒定（cav_000 仅个别行有微小起伏），变化全部沿 z 向。
数据源：radius500_v2.tar.gz（cav_000，interface 4）；
independent_radius_field.tar.gz（train/gt_field/cav_000.npy）。无平滑。
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
OUT = f'{BASE}/figures/fig_perturbation_quad.png'

with tarfile.open(f'{BASE}/source_archive/source_data/radius500_v2.tar.gz') as t:
    ctrl = pd.read_csv(t.extractfile('cav_000.radius.csv'))
with tarfile.open(f'{BASE}/source_archive/data/independent_radius_field.tar.gz') as t:
    fld = np.load(io.BytesIO(t.extractfile('train/gt_field/cav_000.npy').read())).astype(np.float64)

prof_r = ctrl['1353.9395833333335'].values          # interface 4
depth = ctrl['z'].values                             # 行序 floor → top（depth 递减）
row_med = np.median(fld, axis=1)

vrow = prof_r > 0.5
vcell = fld > 0.5
raw = (fld - prof_r[:, None])[vcell]
s = (row_med[vrow] * prof_r[vrow]).sum() / (prof_r[vrow] ** 2).sum()
aff = row_med[vrow] - s * prof_r[vrow]
ratio = row_med[vrow] / prof_r[vrow]
r_corr = np.corrcoef(row_med[vrow], prof_r[vrow])[0, 1]

fig = plt.figure(figsize=(15.5, 4.2), dpi=200)
fig.suptitle('Perturbation synthesis for a perturb-group training cavern (cav_000): '
             'control geometry, perturbed radius field, and perturbation decomposition', fontsize=12.5)
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.25, 1.0, 1.0], wspace=0.32,
                      left=0.045, right=0.985, top=0.80, bottom=0.16)

ax = fig.add_subplot(gs[0])
ax.plot(prof_r, depth, color='#1f77b4', lw=1.6)
ax.set_ylim(depth.max(), depth.min())
ax.set_xlabel('Radius (m)', fontsize=9.5); ax.set_ylabel('Depth (m)', fontsize=9.5)
ax.set_title('(a) Control design profile\n(cav_000, interface 4)', fontsize=10)

ax = fig.add_subplot(gs[1])
im = ax.imshow(fld, cmap='viridis', aspect='auto', origin='lower',
               extent=[0, 360, 0, 201], interpolation='nearest')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('r (m)', fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
ax.set_xlabel('θ (°)', fontsize=9.5); ax.set_ylabel('grid row (floor → top)', fontsize=9)
ax.set_title('(b) Perturbed radius field\nr(z, θ), 201 × 360 grid', fontsize=10)

ax = fig.add_subplot(gs[2])
ax.plot(ratio, depth[vrow], color='#5e3c99', lw=1.6)
ax.axvline(ratio.mean(), color='0.4', ls='--', lw=0.9)
ax.text(0.05, 0.04, f'mean {ratio.mean():.3f} ± {ratio.std():.3f}',
        transform=ax.transAxes, fontsize=8.5, color='0.3')
ax.set_ylim(depth.max(), depth.min())
ax.set_xlim(0.29, 0.34)
ax.set_xlabel('Radial ratio (field / control)', fontsize=9.5)
ax.set_yticklabels([])
ax.set_title('(c) Per-row radial ratio\n(near-uniform scale,  calibration)', fontsize=10)

ax = fig.add_subplot(gs[3])
ax.hist(aff, bins=40, color='#7b5ea7', alpha=0.85)
ax.set_xlabel('Non-affine residual (m)', fontsize=9.5); ax.set_ylabel('Rows', fontsize=9.5)
ax.set_title(f'(d) Residual after scale removal\nmean {aff.mean():.2f} m, σ {aff.std():.2f} m', fontsize=10)
ax.text(0.03, 0.96, f'raw cell-wise offset vs control (valid cells, r > 0.5 m):\n'
                    f'mean {raw.mean():.2f} m, σ {raw.std():.2f} m — dominated by the scale factor',
        transform=ax.transAxes, fontsize=8, va='top', color='0.25')

for ax in fig.axes:
    if ax not in fig.axes[-5:]:
        continue
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig.text(0.5, 0.005,
         'Sources: control profile from the Sansmic batch archive (radius500_v2.tar.gz); perturbed field from the archived  perturbation fields '
         '(independent_radius_field.tar.gz, train split).\nRow-wise correspondence between profile and grid (both 201 rows, floor → top; row-wise Pearson r ≈ 1.0). '
         'The perturbed field is azimuthally uniform within each grid row (all variation is along z). The radial-scale factor is part of the  calibration and is presented as-is. No smoothing.',
         ha='center', fontsize=7.8)
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT, f'| r={r_corr:.4f} s={s:.4f} raw mean {raw.mean():.2f} σ {raw.std():.2f}')