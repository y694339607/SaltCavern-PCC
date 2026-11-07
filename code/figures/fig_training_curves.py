# -*- coding: utf-8 -*-
"""重绘 fig_training_curves.png（v2）
变更：图例去除内部代号 BM/E2，改为论文正式名（Ours (masked) / Baseline (unmasked)）。
数据源：BM_full csv（source_archive/docs/logs/）、
E2_full csv（figures/logs/）、training_log_masked.log（source_archive/train/）。
口径：无平滑；best 点 ep179/169/93；确认臂预置停臂准则（best 停滞 ≥30 epoch）。
"""
import re
import numpy as np
import pandas as pd
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = f'{_ARCHIVE}'
OUT = f'{BASE}/figures/fig_training_curves.png'

bm = pd.read_csv(f'{BASE}/source_archive/docs/logs/BM_full_s21_o250e250_pcV2.csv')
e2 = pd.read_csv(f'{BASE}/figures/logs/E2_full_s21_o250e250_pcV2.csv')
with open(f'{BASE}/source_archive/train/training_log_masked.log') as f:
    lines = f.readlines()
pat = re.compile(r'\[epoch (\d+)\] train_loss=([\d.]+) val_cd_l2=([\d.]+)')
gs = pd.DataFrame([(int(m.group(1)), float(m.group(2)), float(m.group(3)))
                   for l in lines for m in [pat.search(l)] if m],
                  columns=['epoch', 'train_loss', 'val_cd_l2'])

C = dict(bm='#1f77b4', e2='#d95319', gs='#1b9e77')
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=200)
fig.suptitle('Training curves of the three paper-track runs (seed 20260921; '
             '250-epoch budget unless stopped early)', fontsize=13, y=0.99)

series = [(bm, C['bm'], 'Ours (masked)'), (e2, C['e2'], 'Baseline (unmasked)'),
          (gs, C['gs'], 'Confirmatory arm (seed 20260921)')]
best_eps = [179, 169, 93]

ax = axes[0]
for (df, c, lab), be in zip(series, best_eps):
    ax.plot(df.epoch, df.val_cd_l2, color=c, lw=1.1, label=lab)
    row = df[df.epoch == be].iloc[0]
    ax.plot(be, row.val_cd_l2, marker='*', ms=14, color=c, mec='k', mew=0.7, zorder=5)
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_title('Validation CD (L2, normalised)', fontsize=11)
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.annotate('confirmatory arm stopped at ep 107\n(best plateau ≥ 30 epochs; criterion pre-set)',
            xy=(107, gs.loc[gs.epoch == 107, 'val_cd_l2'].iloc[0]),
            xytext=(130, 8e-5), fontsize=8.5, color=C['gs'],
            arrowprops=dict(arrowstyle='-', color='k', lw=0.8))

ax = axes[1]
for (df, c, lab), be in zip(series, best_eps):
    ax.plot(df.epoch, df.train_loss, color=c, lw=1.1, label=lab)
    row = df[df.epoch == be].iloc[0]
    ax.plot(be, row.train_loss, marker='*', ms=14, color=c, mec='k', mew=0.7, zorder=5)
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_title('Training loss', fontsize=11)

for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

fig.text(0.5, -0.04,
         'Note: stars mark best-checkpoint epochs (179 / 169 / 93). Validation CD is the L2 Chamfer distance in normalised units\n'
         '(identical training protocol and data order; confirmatory arm differs only in weight initialisation). No smoothing applied.',
         ha='center', fontsize=8.5)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)
