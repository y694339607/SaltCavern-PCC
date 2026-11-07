# -*- coding: utf-8 -*-
"""重绘 fig_cav405_triptych.png（v2）
变更：
1) 主面板裁掉井筒长尾（z ≤ 6050 m，约占 gt 点数 10.2%），消除大片留白，图注声明裁剪；
2) 点标记放大、统一视角；
3) 第 5 面板改为 θ∈[345°,350°) 楔形径向剖面放大，直接展示 z≈6205–6245 m 的
   多值台面结构（同一高度两个径向表面）：Cylindrical NN interpolation 的
   max-radius-per-cell 只保留外侧表面，Ours (masked) 保留两个表面。
数据源：source_archive/cav405/ 四件 npy（指纹在档）。
"""
import numpy as np
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAV = f'{_ARCHIVE}/source_archive/cav405'
OUT = f'{_ARCHIVE}/figures/fig_cav405_triptych.png'

gt = np.load(f'{CAV}/gt_cav_405_L0.npy').astype(np.float64)
obs = np.load(f'{CAV}/obs_cav_405_L0.npy').astype(np.float64)
b1 = np.load(f'{CAV}/pred_cylnn_cav_405_L0.npy').astype(np.float64)
bm = np.load(f'{CAV}/pred_adapointr_masked_cav_405_L0.npy').astype(np.float64)

ZMIN = 6050.0
panels = [('Ground truth (16,384 pts)', gt), ('Observed (8,192 pts)', obs),
          ('Cyl. NN interpolation', b1), ('Ours (masked)', bm)]

fig = plt.figure(figsize=(15.5, 4.6), dpi=200)
fig.suptitle('Worked example cavern cav_405, level L0 (mild, 70% coverage target): '
             'ground truth, observed points, and the two reconstructions', fontsize=12.5)

# 前四个面板：3/4 视角投影（手工投影到 2D，避免 3D 坐标轴开销）
az = np.deg2rad(-60.0)
el = np.deg2rad(14.0)
def proj(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    xr = x * np.cos(az) + y * np.sin(az)
    yr = (-x * np.sin(az) + y * np.cos(az)) * np.sin(el) + (z - z.mean()) * np.cos(el)
    return xr, yr

rng = np.random.default_rng(20261017)
for k, (title, p) in enumerate(panels):
    ax = fig.add_subplot(1, 5, k + 1)
    pb = p[p[:, 2] > ZMIN]
    if len(pb) > 8192:  # 16k 面板显示降采样 ×2（沿用 v1 口径）
        pb = pb[rng.choice(len(pb), 8192, replace=False)]
    xr, yr = proj(pb)
    ax.scatter(xr, yr, s=2.2, c=pb[:, 2], cmap='viridis', alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')

# 第五面板：楔形径向剖面放大（多值台面区）
ax = fig.add_subplot(1, 5, 5)
lo, hi = np.deg2rad(345 - 180), np.deg2rad(350 - 180)  # arctan2 输出 [-pi, pi)：345° ≡ -15°
def wedge(p):
    th = np.arctan2(p[:, 1], p[:, 0])
    m = (th >= lo) & (th < hi) & (p[:, 2] > ZMIN)
    q = p[m]
    return np.hypot(q[:, 0], q[:, 1]), q[:, 2]
rg, zg = wedge(gt); rb, zb = wedge(b1); rm, zm = wedge(bm)
ax.scatter(rg, zg, s=14, c='0.45', label='Ground truth', alpha=0.8, linewidths=0)
ax.scatter(rb, zb, s=14, c='#c8321e', label='Cyl. NN interp.', alpha=0.75, linewidths=0)
ax.scatter(rm, zm, s=14, c='#1b77b4', label='Ours (masked)', alpha=0.75, linewidths=0)
ax.axhspan(6205, 6245, color='#e8a33d', alpha=0.18, zorder=0)
ax.annotate('multi-valued zone:\ntwo radial surfaces\nat the same height',
            xy=(38, 6225), xytext=(30, 6340), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='0.25', lw=0.9))
ax.set_title('Cross-section θ∈[345°, 350°)\n(multi-valued zoom)', fontsize=10)
ax.set_xlabel('Radius from well axis (m)', fontsize=9)
ax.set_ylabel('z (m)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, loc='lower right', frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.text(0.5, 0.012,
         'Cylindrical NN interpolation reproduced locally with the archived pipeline including its output jitter (seed 20261015); '
         'Ours (masked) is the eval-only single-sample inference from the fingerprint-verified best checkpoint '
         '(BM_full_s21_o250e250_pcV2_best.pt, ep 179).\nDisplay subsampling ×2 for 16k panels; '
         'the wellbore section below z = 6050 m is cropped for display (10.2% of ground-truth points); no other processing.',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)
