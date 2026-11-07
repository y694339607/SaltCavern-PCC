# -*- coding: utf-8 -*-
"""按 v22 口径重绘图 1 与图 4：四类缺失机制与一项角分辨率上限。"""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / _SUPP / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"navy": "#294C60", "teal": "#3E7C7B", "gold": "#C28E3D", "rust": "#A85D4C", "gray": "#65727A"}

# 图 1：四阶段流程。
fig, ax = plt.subplots(figsize=(12, 6.4))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(0.5, 0.965, "Salt-cavern sonar point-cloud completion: benchmark and evaluation",
        ha="center", va="top", fontsize=16, weight="bold", color=C["navy"])

def band(y, label, lines, color):
    x, w, label_w, h = 0.03, 0.94, 0.29, 0.155
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=color, lw=1.25))
    ax.add_patch(Rectangle((x, y), label_w, h, fc=color, ec="none"))
    ax.text(x + label_w / 2, y + h / 2, label, ha="center", va="center",
            color="white", weight="bold", fontsize=10.5, linespacing=1.2)
    ax.text(x + label_w + (w - label_w) / 2, y + h / 2, "\n".join(lines),
            ha="center", va="center", color="#25333A", fontsize=10, linespacing=1.3)

band(0.735, "01  Simulation\nground truth",
     ["Sansmic: 500 caverns  |  BH-104 workflow check",
      "4 perturbation types: 400 caverns  |  100 axisymmetric controls"], C["navy"])
band(0.540, "02  Structured\nmissingness",
     ["4 missing-data mechanisms + angular-resolution limit",
      "Coverage: 70 / 50 / 30%  |  1,500 residual–complete sample pairs"], C["teal"])
band(0.345, "03  Completion\nmethods",
     ["NN / IDW / spline  |  PCN / PoinTr / AdaPoinTr",
      "Standard and masked supervision  |  1,024 → 16,384 points"], C["gold"])
band(0.150, "04  Geometric and\nvolumetric evaluation",
     ["CD-L1  |  V_field: independent radius field  |  V_ref: complete-cloud pipeline",
      "Error strata  |  Sensitivity analysis  |  Repeated training checks"], C["rust"])
fig.savefig(OUT / "图1_基准与评价流程.png", bbox_inches="tight", facecolor="white", dpi=300)
plt.close(fig)

# 图 4：五项观测处理中，M2 为角分辨率上限，其余四项为缺失或观测扰动机制。
ROWS, COLS = 201, 360
rng = np.random.default_rng(20260921)
fig, axes = plt.subplots(1, 5, figsize=(15.5, 5.2), dpi=200)
fig.suptitle("Missingness synthesis on the (z, θ) grid — four mechanisms and one angular-resolution limit\n"
             "(grid 201 × 360, floor → top; master seed 20260921)", fontsize=12.5)

def style(ax, title):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(["0", "90", "180", "270", "360"], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("θ (°)", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

def caption(ax, text):
    ax.text(0.5, -0.52, textwrap.fill(text, 30), transform=ax.transAxes,
            ha="center", va="top", fontsize=8)

KW = dict(cmap="gray", vmin=0, vmax=1, aspect="auto", origin="lower",
          extent=[0, 360, 0, 201], interpolation="nearest")

m1 = np.ones((ROWS, COLS))
rows_gap = np.arange(0, ROWS, 7)
m1[rows_gap[:, None] + np.arange(2)[None, :].clip(max=ROWS - 1)] = 0
axes[0].imshow(m1, **KW)
style(axes[0], "M1 Inter-survey gaps")
caption(axes[0], "located by height value; spacing parameterised (target 2 m, quantised on the 201-row grid)")

m2 = np.ones((ROWS, COLS))
obs_cols = rng.random(COLS) < 0.36
m2[:, ~obs_cols] = 0
axes[1].imshow(m2, **KW)
style(axes[1], "M2 Angular-resolution limit")
caption(axes[1], "0.36° survey step represented on a 1° source grid; no additional deletion after mapping")

m3 = np.ones((ROWS, COLS))
m3[:14, :] = 0
m3[150:160, 180:260] = 0
axes[2].imshow(m3, **KW)
style(axes[2], "M3 Sediment & protrusion shadow")
caption(axes[2], "buried cells below sediment surface + shadow ≤ 10 rows below protrusion (compound criterion)")

noise = rng.normal(0, 0.3, (ROWS, COLS))
im = axes[3].imshow(noise, cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="auto",
                    origin="lower", extent=[0, 360, 0, 201], interpolation="nearest")
cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
cb.set_label("Δr (m)", fontsize=8)
cb.ax.tick_params(labelsize=7)
style(axes[3], "M4 Radial noise")
caption(axes[3], "mask unchanged; additive Gaussian, σ = 0.3 m, applied to observed points only (noise field shown)")

m5 = np.ones((ROWS, COLS))
drop = rng.choice(ROWS, size=10, replace=False)
m5[drop, :] = 0
axes[4].imshow(m5, **KW)
style(axes[4], "M5 Whole-layer loss")
caption(axes[4], "random whole-row drop without replacement, p parameterised (nominal 5%)")

fig.text(0.5, 0.025,
         "Schematic masks and observation noise (white = observed, black = unobserved). M1/M3/M5 compose into the observation mask; M4 perturbs observed radii; M2 defines the angular representation.\n"
         "Coverage is calibrated per cavern to L0/L1/L2 = 70%/50%/30% with a dedicated calibration random stream. Randomness derives from master seed 20260921 and per-cavity, per-level and per-use keys;\n"
         "area-weighted sampling (weight ∝ r) follows mask synthesis.",
         ha="center", va="bottom", fontsize=8)
fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.43, wspace=0.28)
fig.savefig(OUT / "图4_结构化观测机制.png", facecolor="white", dpi=200)
plt.close(fig)
print(OUT / "图1_基准与评价流程.png")
print(OUT / "图4_结构化观测机制.png")
