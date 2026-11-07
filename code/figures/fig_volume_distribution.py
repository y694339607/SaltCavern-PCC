# -*- coding: utf-8 -*-
"""重绘 v23 的图 2 体积分布与图 7 覆盖度分层比较。"""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / _SUPP / "figures"
MEDIA = ROOT / "70_paper_draft/work_v23/unpacked/word/media"
VOLUMES = FIG / "最终配置500腔_体积复算.csv"
TEST_LONG = ROOT / "source_archive/test_long_table.csv"
IDENTITY = FIG / "identity_reanalysis/identity_统一口径逐样本复算.csv"

# 图 2：保留原组合图的（a）与（c），仅重绘右上方（b）。
source2 = MEDIA / "cc58688bc5d41f59a04dd72262fbf95b7eaa5523.png"
combo = Image.open(source2).convert("RGB")
vol = pd.read_csv(VOLUMES)["V_cell"].to_numpy() / 1e4
fig, ax = plt.subplots(figsize=(11.8, 6.6), dpi=100)
ax.hist(vol, bins=30, color="#1f77b4", edgecolor="none")
ax.axvline(np.median(vol), color="#b44c43", ls="--", lw=1.8, label=f"median = {np.median(vol):.1f}")
ax.set_title("500 caverns", fontsize=17)
ax.set_xlabel(r"$V_{cav}$ ($\times 10^4$ m$^3$)", fontsize=17)
ax.set_ylabel("count", fontsize=17)
ax.tick_params(labelsize=14)
ax.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout(pad=1.2)
panel_path = FIG / "图2b_最终配置500腔体积分布.png"
fig.savefig(panel_path, dpi=100, facecolor="white")
plt.close(fig)
panel = Image.open(panel_path).convert("RGB")
combo.paste(panel, (1200, 100))
out2 = FIG / "图2_仿真核查与最终配置基准统计.png"
combo.save(out2)
shutil.copyfile(out2, source2)

# 图 7：测试集原七种方法加 identity；误差列统一换算为百分数。
test = pd.read_csv(TEST_LONG)
identity = pd.read_csv(IDENTITY)
identity = identity[identity["split"] == "test"].copy()
method_map = {
    "B1": "Cylindrical NN",
    "B2": "IDW",
    "B3": "Periodic spline",
    "PCN": "PCN",
    "PoinTr": "PoinTr",
    "E2": "AdaPoinTr standard",
    "BM": "AdaPoinTr masked",
}
order = list(method_map) + ["identity"]
labels = [method_map[m] for m in method_map] + ["identity"]
records = []
for method in order:
    for level in [0, 1, 2]:
        if method == "identity":
            rows = identity[identity["level"] == level]
            cd = rows["cd_l1_m"].mean()
            vf = rows["err_field_abs"].mean() * 100.0
        else:
            rows = test[(test["method"] == method) & (test["level"] == level)]
            cd = rows["cd_l1_m"].mean()
            vf = rows["err_field_abs"].mean() * 100.0
        if len(rows) != 50:
            raise RuntimeError(f"分层样本数异常：{method}, L{level}, n={len(rows)}")
        records.append({"method": method, "level": level, "cd": cd, "vfield": vf})
plot_data = pd.DataFrame(records)
plot_data.to_csv(FIG / "图7_测试集覆盖度分层含identity.csv", index=False)

colors = ["#294f68", "#4b908c", "#d2a043"]
x = np.arange(len(order), dtype=float)
width = 0.24
fig, axes = plt.subplots(1, 2, figsize=(18, 7.1), dpi=200)
for j, level in enumerate([0, 1, 2]):
    sub = plot_data[plot_data["level"] == level].set_index("method").loc[order]
    offset = (j - 1) * width
    axes[0].bar(x + offset, sub["cd"], width=width, color=colors[j], label=f"L{level}: {[70, 50, 30][level]}%")
    axes[1].bar(x + offset, sub["vfield"], width=width, color=colors[j])
axes[0].set_title("(a) Global geometric error", fontsize=17, fontweight="bold")
axes[1].set_title("(b) Volume error against the independent field", fontsize=17, fontweight="bold")
axes[0].set_ylabel("CD-L1 (m)", fontsize=15)
axes[1].set_ylabel(r"$V_{field}$ mean absolute relative error (%)", fontsize=14)
axes[1].set_yscale("log")
axes[1].set_ylim(0.8, 16)
axes[0].set_ylim(0, 3.45)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#dddddd", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.01))
fig.text(
    0.5, 0.012,
    "Held-out test set, n = 150; 50 samples per coverage level. Panel (b) uses a logarithmic scale. "
    "Identity uses 8192 stored observations; other methods use 16384-point outputs.",
    ha="center", fontsize=10.5, color="#666666",
)
fig.tight_layout(rect=[0.02, 0.065, 0.99, 0.94], w_pad=3.0)
out7 = FIG / "图7_覆盖度分层结果_含identity.png"
fig.savefig(out7, dpi=200, facecolor="white")
plt.close(fig)
shutil.copyfile(out7, MEDIA / "0f352de20995d8f63ba2293e6373a2be35cecdc2.png")
print(out2)
print(out7)
print("FIGURE_2_7_REDRAW_PASS")
