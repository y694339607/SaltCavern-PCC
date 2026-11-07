# -*- coding: utf-8 -*-
"""重绘 fig_five_modes_mechanism.png（v2）
变更：
1) 修复 M1/M2 图注文字碰撞——统一换行宽度、统一图注纵坐标；
2) M4 面板不再留空：径向噪声不改变掩膜，改为以发散色标展示作用于观测格的
   高斯噪声场（σ = 0.3 m，示意实现，种子 20260921），图注声明掩膜不变。
示意掩码为机制图示，非实测数据；参数口径见 table_five_modes_parameters.md
（missing_synth.py v1.3，sha256 6b20e6b5…15ac）。
"""
import textwrap
import numpy as np
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = f'{_ARCHIVE}/figures/fig_five_modes_mechanism.png'
ROWS, COLS = 201, 360
rng = np.random.default_rng(20260921)

fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.3), dpi=200)
fig.suptitle('Missingness synthesis on the (z, θ) grid — five compositional modes '
             '(grid 201 × 360, floor → top; master seed 20260921)', fontsize=12.5)

def style(ax, title):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['0', '90', '180', '270', '360'], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel('θ (°)', fontsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def caption(ax, text):
    ax.text(0.5, -0.24, textwrap.fill(text, 30), transform=ax.transAxes,
            ha='center', va='top', fontsize=8)

KW = dict(cmap='gray', vmin=0, vmax=1, aspect='auto', origin='lower',
          extent=[0, 360, 0, 201], interpolation='nearest')

# M1 层间空洞：按高度定位的测层间隙（示意：周期黑带）
m1 = np.ones((ROWS, COLS))
rows_gap = np.arange(0, ROWS, 7)
m1[rows_gap[:, None] + np.arange(2)[None, :].clip(max=ROWS - 1)] = 0
axes[0].imshow(m1, **KW)
style(axes[0], 'M1 Inter-survey gaps')
caption(axes[0], 'located by height value; spacing parameterised (target 2 m, quantised on the 201-row grid)')

# M2 角分辨率：0.36° 步长映射到 1° 源网格 → 可观测列集合（每列观测概率 0.36）
m2 = np.ones((ROWS, COLS))
obs_cols = rng.random(COLS) < 0.36
m2[:, ~obs_cols] = 0
axes[1].imshow(m2, **KW)
style(axes[1], 'M2 Angular resolution')
caption(axes[1], '0.36° survey step mapped onto the 1° source grid (observable column set)')

# M3 沉渣 + 凸壁遮挡：底部沉渣埋置带 + 凸壁背声呐侧阴影（≤10 行）
m3 = np.ones((ROWS, COLS))
m3[:14, :] = 0                       # 沉渣面之下埋置格（示意）
m3[150:160, 180:260] = 0             # 凸壁阴影，向下有限延伸 10 行（示意）
axes[2].imshow(m3, **KW)
style(axes[2], 'M3 Sediment & protrusion shadow')
caption(axes[2], 'buried cells below sediment surface + shadow ≤ 10 rows below protrusion (compound criterion)')

# M4 径向噪声：掩膜不变，仅观测点半径加性高斯扰动——展示噪声场
noise = rng.normal(0, 0.3, (ROWS, COLS))
im = axes[3].imshow(noise, cmap='RdBu_r', vmin=-0.9, vmax=0.9, aspect='auto',
                    origin='lower', extent=[0, 360, 0, 201], interpolation='nearest')
cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
cb.set_label('Δr (m)', fontsize=8)
cb.ax.tick_params(labelsize=7)
style(axes[3], 'M4 Radial noise')
caption(axes[3], 'mask unchanged; additive Gaussian, σ = 0.3 m, applied to observed points only (noise field shown)')

# M5 整层丢失：无置换随机整行丢弃（名义 5%）
m5 = np.ones((ROWS, COLS))
drop = rng.choice(ROWS, size=10, replace=False)
m5[drop, :] = 0
axes[4].imshow(m5, **KW)
style(axes[4], 'M5 Whole-layer loss')
caption(axes[4], 'random whole-row drop without replacement, p parameterised (nominal 5%)')

fig.text(0.5, 0.005,
         'Schematic masks (white = observed, black = unobserved). Modes compose into the observation mask; coverage is calibrated per cavern to the level targets\n'
         'L0/L1/L2 = 70%/50%/30% with a dedicated calibration random stream (isolated from generation). Randomness derives only from make_rng(20260921, [cavity, level, use]);\n'
         'area-weighted sampling (weight ∝ r) follows mask synthesis.',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.10, 1, 0.93])
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)
