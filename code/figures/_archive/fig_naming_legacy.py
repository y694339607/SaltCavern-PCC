# -*- coding: utf-8 -*-
"""仅规范图 10、图 11 的展示标签，不改变任何数值或统计口径。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / _SUPP / "figures" / "内部代号规范"
OUT.mkdir(parents=True, exist_ok=True)

# 图 10：条件化分区森林图。S1/S2 改为直接描述样本范围。
GREEN, RED, GREY = "#1b9e57", "#c8321e", "#7f7f7f"
panel_a = [
    ("Multi-valued cell-wise CD (all)", -0.229, -0.285, -0.196,
     "-0.229 [-0.285, -0.196]  135/150 samples with hits; p < 1e-6", GREEN, "o"),
    ("per-level L1", -0.205, None, None, "-0.205", GREEN, "D"),
    ("per-level L2", -0.328, None, None, "-0.328", GREEN, "D"),
    ("per-level L0", -0.182, None, None, "-0.182", GREEN, "D"),
    ("Gap-band cell-wise CD (all)", +0.292, +0.197, +0.341,
     "+0.292 [+0.197, +0.341]  n = 146; p < 1e-6", RED, "o"),
    ("Large-gap subset (all coverages)", +0.193, +0.007, +0.338,
     "+0.193 [+0.007, +0.338]  n = 37", RED, "o"),
    ("Large-gap subset (L2)", +0.129, -0.047, +0.293,
     "+0.129 [-0.047, +0.293]  n = 27; n.s.", GREY, "o"),
    ("Observed-cell radial residual (pred − gt, debiased)", +0.141, +0.129, +0.159,
     "+0.141 [+0.129, +0.159]  n = 150; p = 2.3e-26", RED, "o"),
]
panel_b = [
    ("Volume error (all)", +0.710, +0.530, +0.940,
     "+0.710 [+0.530, +0.940]  n = 150; p < 1e-6", RED, "o"),
    ("Volume error (large-gap subset, L2)", +0.620, -0.180, +1.100,
     "+0.620 [-0.180, +1.100]  n = 27; n.s.", GREY, "o"),
]

fig = plt.figure(figsize=(12.6, 7.6), dpi=200)
fig.suptitle(
    "Conditional win–loss picture, Ours (masked) vs Cylindrical NN interpolation "
    "(validation set, n = 150; pre-registered input-condition filtering)",
    fontsize=12.5,
)

def draw(ax, rows, xlabel, title, xlim):
    ax.set_title(title, fontsize=11, loc="center", pad=10)
    ys = list(range(len(rows), 0, -1))
    for y, (lab, est, lo, hi, note, color, marker) in zip(ys, rows):
        ax.axhspan(y - 0.42, y + 0.42, color="0.95", zorder=0)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=2)
        ax.plot(est, y, marker=marker, ms=11 if marker == "o" else 10, color=color, zorder=3)
        transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.015, y, note, transform=transform, va="center", fontsize=9, color=color)
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)

gs = fig.add_gridspec(2, 1, height_ratios=[8, 2.6], hspace=0.50,
                      left=0.29, right=0.60, top=0.92, bottom=0.11)
axa = fig.add_subplot(gs[0])
draw(axa, panel_a,
     "Paired difference, Ours (masked) − Cylindrical NN interpolation (m); negative favours Ours (masked)",
     "A. Geometric and surface criteria (per-cell / per-sample median, 95% CI)", (-0.4, 0.5))
axb = fig.add_subplot(gs[1])
draw(axb, panel_b,
     "Paired difference, Ours (masked) − Cylindrical NN interpolation (percentage points); "
     "negative favours Ours (masked)",
     "B. Volumetric criteria", (-0.3, 1.2))
fig.text(
    0.5, 0.005,
    "Note: multi-valued cells = grid cells with multiple radial ray hits (92.8% of samples contain multi-valued cells; lower bound); "
    "gap-band threshold Q3 = 44.64 m pre-registered;\nobserved-cell analysis is post hoc. "
    "Paired differences are medians with 95% bootstrap CI; Wilcoxon signed-rank for multi-valued subset (pre-registered rule n ≥ 20).",
    ha="center", fontsize=8,
)
forest_path = OUT / "图10_条件化分区配对差异.png"
fig.savefig(forest_path, bbox_inches="tight")
plt.close(fig)

# 按原合成方式合并径向多值个案和条件化分区森林图。
source_dir = ROOT / _SUPP / "figures"
parts = [
    Image.open(source_dir / "fig_cav405_triptych.png").convert("RGB"),
    Image.open(forest_path).convert("RGB"),
]
labels = [
    "(a) Worked example of radial multivalue geometry",
    "(b) Paired differences in conditional subsets",
]
width = 2400
resized = []
for image in parts:
    height = round(image.height * width / image.width)
    resized.append(image.resize((width, height)))
canvas = Image.new("RGB", (width, sum(image.height for image in resized) + 120), "white")
draw_canvas = ImageDraw.Draw(canvas)
y = 0
for image, label in zip(resized, labels):
    draw_canvas.text((25, y + 15), label, fill="black")
    canvas.paste(image, (0, y + 60))
    y += image.height + 60
fig10_path = OUT / "图10_径向多值几何与条件化分区结果.png"
canvas.save(fig10_path)

# 图 11：数据读取仍使用归档运行编号，展示标签改为实验协议描述。
MS = ROOT / _SUPP / "figures" / "paired_repeat_table.csv"
MU = ROOT / "source_archive" / "mu" / "mu_long_table_all.csv"
configs = [
    "250×36\nmax", "250×36\nmean", "250×36\np95",
    "500×72\nmax", "500×72\nmean", "500×72\np95",
    "1000×144\nmax", "1000×144\nmean", "1000×144\np95",
]
bm_f = [1.751, 2.542, 1.751, 1.587, 1.788, 1.587, 1.616, 1.625, 1.616]
bm_r = [2.977, 1.565, 2.977, 1.889, 1.590, 1.889, 1.756, 1.693, 1.756]
e_f = [2.354, 2.427, 2.354, 1.765, 1.689, 1.765, 1.794, 1.744, 1.794]
e_r = [3.931, 1.755, 3.930, 2.322, 1.831, 2.322, 2.077, 1.972, 2.077]
ms = pd.read_csv(MS)
mu = pd.read_csv(MU)
mu = mu[mu["pair"].isin([f"MU{i}" for i in range(1, 6)])]
mu_summary = mu.groupby(["pair", "arm"], as_index=False).agg(
    cd_l1_m=("cd_l1_m", "mean"),
    err_field_abs=("err_field_abs", "mean"),
    n=("cd_l1_m", "size"),
)
rows = []
for pair in ["P1", "P2"]:
    for arm in ["BM", "E2"]:
        cd = ms[(ms["pair"] == pair) & (ms["metric"] == "cd_l1_m") & (ms["arm"] == arm)]["mean"].iloc[0]
        vf = ms[(ms["pair"] == pair) & (ms["metric"] == "err_field_abs") & (ms["arm"] == arm)]["mean"].iloc[0]
        rows.append({"pair": pair, "arm": arm, "cd_l1_m": cd, "err_field_abs": vf})
for pair in [f"MU{i}" for i in range(1, 6)]:
    for arm in ["BM", "E2"]:
        row = mu_summary[(mu_summary["pair"] == pair) & (mu_summary["arm"] == arm)].iloc[0]
        rows.append({"pair": pair, "arm": arm, "cd_l1_m": row["cd_l1_m"], "err_field_abs": row["err_field_abs"]})
repeats = pd.DataFrame(rows)

teal, rust, gray = "#367C85", "#B26E5D", "#68767C"
pairs = ["P1", "P2", "MU1", "MU2", "MU3", "MU4", "MU5"]
display_labels = [
    "Fixed\nrepeat 1", "Fixed\nrepeat 2",
    "Arbitrary\nrepeat 1", "Arbitrary\nrepeat 2",
    "Arbitrary\nrepeat 3", "Arbitrary\nrepeat 4",
    "Arbitrary\nrepeat 5",
]
x = np.arange(len(pairs))
fig = plt.figure(figsize=(13, 8.7))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.15, 1])
for row, (masked, standard, title) in enumerate([
    (bm_f, e_f, "(a) $V_{field}$ mean absolute relative error (%)"),
    (bm_r, e_r, "(b) $V_{ref}$ mean absolute relative error (%)"),
]):
    ax = fig.add_subplot(gs[row, 0])
    matrix = np.array([masked, standard])
    ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=1.5, vmax=4.0)
    ax.set_yticks([0, 1], ["Masked", "Standard"])
    ax.set_xticks(range(9), configs, fontsize=8)
    ax.set_title(title, weight="bold")
    for i in range(2):
        for j, value in enumerate(matrix[i]):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 2.8 else "#17313D")

collapsed = {"MU1", "MU3", "MU4", "MU5"}
for row, metric, title, ylabel in [
    (0, "err_field_abs", "(c) $V_{field}$ error across repeated runs", "$V_{field}$ error (%)"),
    (1, "cd_l1_m", "(d) Physical CD-L1 across repeated runs", "CD-L1 (m)"),
]:
    ax = fig.add_subplot(gs[row, 1])
    for i, pair in enumerate(pairs):
        if pair in collapsed:
            ax.axvspan(i - 0.36, i + 0.36, color="#F2F2F2", zorder=0)
    for arm, label, color, marker in [
        ("BM", "Masked configuration", teal, "o"),
        ("E2", "Standard supervision", rust, "s"),
    ]:
        values = [
            repeats[(repeats["pair"] == pair) & (repeats["arm"] == arm)][metric].iloc[0]
            * (100 if metric == "err_field_abs" else 1)
            for pair in pairs
        ]
        ax.plot(x, values, marker=marker, lw=1.8, ms=6, label=label, color=color, zorder=3)
    ax.set_xticks(x, display_labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", alpha=0.18, zorder=1)
    if metric == "err_field_abs":
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        ax.set_ylim(0, 21)

fig.text(
    0.5, 0.015,
    "Sensitivity configurations share the same test samples. Fixed-initialization repeats vary data order only; "
    "arbitrary-initialization repeats vary initialization and data order. Shaded arbitrary-initialization repeats show geometric collapse in both arms.",
    ha="center", fontsize=8.5, color=gray,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig11_path = OUT / "图11_数值敏感性与重复实验.png"
fig.savefig(fig11_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(fig10_path)
print(fig11_path)
