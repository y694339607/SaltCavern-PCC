# -*- coding: utf-8 -*-
"""重绘 v24 图 7，加入模型输入同源 identity@1024。"""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / _SUPP / "figures"
MEDIA = ROOT / "70_paper_draft/work_v24/unpacked/word/media"
TEST_LONG = ROOT / "source_archive/test_long_table.csv"
IDENTITY = FIG / "identity_input_matched/identity8192_identity1024_掩码配置_逐样本.csv"

# 图 7：七种方法加两种 identity 对照；误差列统一换算为百分数。
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
order = list(method_map) + ["identity1024", "identity8192"]
labels = [method_map[m] for m in method_map] + ["Identity (1024)", "Identity (8192)"]
records = []
for method in order:
    for level in [0, 1, 2]:
        if method == "identity1024":
            rows = identity[identity["level"] == level]
            cd = rows["cd1024_l1_m"].mean()
            vf = rows["err1024_abs"].mean() * 100.0
        elif method == "identity8192":
            rows = identity[identity["level"] == level]
            cd = rows["cd8192_l1_m"].mean()
            vf = rows["err8192_abs"].mean() * 100.0
        else:
            rows = test[(test["method"] == method) & (test["level"] == level)]
            cd = rows["cd_l1_m"].mean()
            vf = rows["err_field_abs"].mean() * 100.0
        if len(rows) != 50:
            raise RuntimeError(f"分层样本数异常：{method}, L{level}, n={len(rows)}")
        records.append({"method": method, "level": level, "cd": cd, "vfield": vf})
plot_data = pd.DataFrame(records)
plot_data.to_csv(FIG / "图7_测试集覆盖度分层含两种identity.csv", index=False)

colors = ["#294f68", "#4b908c", "#d2a043"]
x = np.arange(len(order), dtype=float)
width = 0.24
fig, axes = plt.subplots(1, 2, figsize=(19.5, 7.4), dpi=200)
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
axes[0].set_ylim(0, 4.75)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=10.5)
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
    "Identity (1024) uses the model-matched subset; Identity (8192) uses all stored observations.",
    ha="center", fontsize=10.5, color="#666666",
)
fig.tight_layout(rect=[0.02, 0.065, 0.99, 0.94], w_pad=3.0)
out7 = FIG / "图7_覆盖度分层结果_含两种identity.png"
fig.savefig(out7, dpi=200, facecolor="white")
plt.close(fig)
shutil.copyfile(out7, MEDIA / "0f352de20995d8f63ba2293e6373a2be35cecdc2.png")
print(out7)
print("FIGURE_7_INPUT_MATCHED_REDRAW_PASS")
