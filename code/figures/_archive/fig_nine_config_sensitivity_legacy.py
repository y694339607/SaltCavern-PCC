from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / _SUPP / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MS = ROOT / _SUPP / "figures" / "paired_repeat_table.csv"
MU = ROOT / "source_archive" / "mu" / "mu_long_table_all.csv"

# 九种敏感性配置沿用 v21.1 图件制作脚本中的归档数值。
configs = ["250×36\nmax", "250×36\nmean", "250×36\np95", "500×72\nmax", "500×72\nmean", "500×72\np95", "1000×144\nmax", "1000×144\nmean", "1000×144\np95"]
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
        rows.append({"pair": pair, "arm": arm, "cd_l1_m": cd, "err_field_abs": vf, "n": 150, "repeat_type": "fixed_initialization_data_order", "status": "normal"})
for pair in [f"MU{i}" for i in range(1, 6)]:
    for arm in ["BM", "E2"]:
        row = mu_summary[(mu_summary["pair"] == pair) & (mu_summary["arm"] == arm)].iloc[0]
        rows.append({"pair": pair, "arm": arm, "cd_l1_m": row["cd_l1_m"], "err_field_abs": row["err_field_abs"], "n": int(row["n"]), "repeat_type": "varied_initialization_and_data_order", "status": "normal" if pair == "MU2" else "both_arms_geometric_collapse"})
repeats = pd.DataFrame(rows)
repeats.to_csv(OUT / "图11_重复实验臂级汇总.csv", index=False, float_format="%.12g")

teal, rust, gray = "#367C85", "#B26E5D", "#68767C"
pairs = ["P1", "P2", "MU1", "MU2", "MU3", "MU4", "MU5"]
x = np.arange(len(pairs))
fig = plt.figure(figsize=(13, 8.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.15, 1])
for row, (b, e, title) in enumerate([(bm_f, e_f, "(a) $V_{field}$ mean absolute relative error (%)"), (bm_r, e_r, "(b) $V_{ref}$ mean absolute relative error (%)")]):
    ax = fig.add_subplot(gs[row, 0])
    mat = np.array([b, e])
    ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=1.5, vmax=4.0)
    ax.set_yticks([0, 1], ["Masked", "Standard"])
    ax.set_xticks(range(9), configs, fontsize=8)
    ax.set_title(title, weight="bold")
    for i in range(2):
        for j, value in enumerate(mat[i]):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white" if value > 2.8 else "#17313D")

collapsed = {"MU1", "MU3", "MU4", "MU5"}
for row, metric, title, ylabel in [
    (0, "err_field_abs", "(c) $V_{field}$ error across repeated runs", "$V_{field}$ error (%)"),
    (1, "cd_l1_m", "(d) Physical CD-L1 across repeated runs", "CD-L1 (m)"),
]:
    ax = fig.add_subplot(gs[row, 1])
    for i, pair in enumerate(pairs):
        if pair in collapsed:
            ax.axvspan(i - 0.36, i + 0.36, color="#F2F2F2", zorder=0)
    for arm, label, color, marker in [("BM", "Masked configuration", teal, "o"), ("E2", "Standard supervision", rust, "s")]:
        values = [repeats[(repeats["pair"] == pair) & (repeats["arm"] == arm)][metric].iloc[0] * (100 if metric == "err_field_abs" else 1) for pair in pairs]
        ax.plot(x, values, marker=marker, lw=1.8, ms=6, label=label, color=color, zorder=3)
    ax.set_xticks(x, pairs)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", alpha=0.18, zorder=1)
    if metric == "err_field_abs":
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        ax.set_ylim(0, 21)

fig.text(0.5, 0.015, "Sensitivity configurations share the same test samples. P1/P2 vary data order only; MU1–MU5 vary initialization and data order. Shaded MU pairs show geometric collapse in both arms.", ha="center", fontsize=8.7, color=gray)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT / "图11_数值敏感性与重复实验.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT / "图11_数值敏感性与重复实验.png")
