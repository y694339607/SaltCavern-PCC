# Forest plot of the conditional win/loss differences.

"""Redraw fig_conditional_winloss_forest.png.

Axis labels use the paper-formal method names; "Multi-valued" and "Observed-cell"
follow the terminology used in the manuscript. Values are the released per-sample
results; the Q3 threshold (44.63575 m) is fixed a priori.
"""
import matplotlib
import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms

OUT = f'{_ARCHIVE}/figures/fig_conditional_winloss_forest.png'

GREEN, RED, GREY = '#1b9e57', '#c8321e', '#7f7f7f'

# (label, est, lo, hi, note, color, marker)
panel_a = [
    ('Multi-valued cell-wise CD (all)',        -0.229, -0.285, -0.196,
     '-0.229 [-0.285, -0.196]  135/150 samples with hits; p ≈ 0', GREEN, 'o'),
    ('per-level L1',                           -0.205, None, None, '-0.205', GREEN, 'D'),
    ('per-level L2',                           -0.328, None, None, '-0.328', GREEN, 'D'),
    ('per-level L0',                           -0.182, None, None, '-0.182', GREEN, 'D'),
    ('Gap-band cell-wise CD (all)',            +0.292, +0.197, +0.341,
     '+0.292 [+0.197, +0.341]  n = 146; p < 1e-6', RED, 'o'),
    ('large-gap subset S1',                    +0.193, +0.007, +0.338,
     '+0.193 [+0.007, +0.338]  n = 37', RED, 'o'),
    ('large-gap subset S2 (L2)',               +0.129, -0.047, +0.293,
     '+0.129 [-0.047, +0.293]  n = 27; n.s.', GREY, 'o'),
    ('Observed-cell radial residual (pred − gt, debiased)', +0.141, +0.129, +0.159,
     '+0.141 [+0.129, +0.159]  n = 150; p = 2.3e-26', RED, 'o'),
]
panel_b = [
    ('Volume error (all)',                     +0.710, +0.530, +0.940,
     '+0.710 [+0.530, +0.940]  n = 150; p < 1e-6', RED, 'o'),
    ('Volume error S2 (large gap × L2)',       +0.620, -0.180, +1.100,
     '+0.620 [-0.180, +1.100]  n = 27; n.s.', GREY, 'o'),
]

fig = plt.figure(figsize=(12.6, 7.6), dpi=200)
fig.suptitle('Conditional win–loss picture, Ours (masked) vs Cylindrical NN interpolation '
             '(validation set, n = 150; pre-registered input-condition filtering)', fontsize=12.5)

def draw(ax, rows, xlabel, title, xlim):
    ax.set_title(title, fontsize=11, loc='center', pad=10)
    ys = list(range(len(rows), 0, -1))
    for y, (lab, est, lo, hi, note, c, mk) in zip(ys, rows):
        ax.axhspan(y - 0.42, y + 0.42, color='0.95', zorder=0)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle='round', zorder=2)
        ax.plot(est, y, marker=mk, ms=11 if mk == 'o' else 10, color=c, zorder=3)
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.015, y, note, transform=tr, va='center', fontsize=9, color=c)
    ax.axvline(0, color='k', ls='--', lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

gs = fig.add_gridspec(2, 1, height_ratios=[8, 2.6], hspace=0.50,
                      left=0.26, right=0.60, top=0.92, bottom=0.11)
axa = fig.add_subplot(gs[0])
draw(axa, panel_a,
     'Paired difference, Ours (masked) − Cylindrical NN interpolation (m); negative favours Ours (masked)',
     'A. Geometric and surface criteria (per-cell / per-sample median, 95% CI)', (-0.4, 0.5))
axb = fig.add_subplot(gs[1])
draw(axb, panel_b,
     'Paired difference, Ours (masked) − Cylindrical NN interpolation (percentage points); '
     'negative favours Ours (masked)',
     'B. Volumetric criteria', (-0.3, 1.2))

fig.text(0.5, 0.005,
         'Note: multi-valued cells = grid cells with multiple radial ray hits (92.8% of samples contain multi-valued cells; lower bound); '
         'gap-band threshold Q3 = 44.64 m pre-registered;\nobserved-cell analysis is post hoc (path b). '
         'Paired differences are medians with 95% bootstrap CI; Wilcoxon signed-rank for multi-valued subset (pre-registered rule n ≥ 20).',
         ha='center', fontsize=8)
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)
