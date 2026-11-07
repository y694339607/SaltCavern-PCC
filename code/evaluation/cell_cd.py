# cell_cd.py — 逐格对称 CD 物理米制重算（逐式adapted from cell_atlas.py）
# 格架：z 按真值 z 范围归一 ×10 带；θ 12 扇区（π/6）；径向 3 层按真值半径三分位（q1,q2）。
# 格内 sym CD = (mean 预测→真值 + mean 真值→预测)/2；双向都有点才计。
# 附带：合并样本第 99 百分位截断值与底/顶带均值比（论文 [84] orig 锚：BM 1.28 / E2 1.83 / PCN 0.98）。
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
TEST_FIN = ROOT / "validate/test_final"
SCALE = 0.3048
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}   # accept legacy tokens
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "B1"]]

def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q

def cell_sym_cd(P, G):
    """返回 (10,12,3) 对称 CD 与有效格掩码；G 为真值。"""
    tG = cKDTree(G); tP = cKDTree(P)
    dg, _ = tG.query(P, k=1)   # 预测→真值
    dp, _ = tP.query(G, k=1)   # 真值→预测
    zmin, zmax = float(G[:, 2].min()), float(G[:, 2].max())
    H = zmax - zmin
    xc, yc = float(G[:, 0].mean()), float(G[:, 1].mean())
    def bands(Q):
        zn = (Q[:, 2] - zmin) / H
        z10 = np.clip((zn * 10).astype(int), 0, 9)
        th = np.mod(np.arctan2(Q[:, 1] - yc, Q[:, 0] - xc), 2 * np.pi)
        th12 = np.clip((th / (np.pi / 6)).astype(int), 0, 11)
        r = np.hypot(Q[:, 0] - xc, Q[:, 1] - yc)
        rg = np.hypot(G[:, 0] - xc, G[:, 1] - yc)
        q1, q2 = np.quantile(rg, [1/3, 2/3])
        rl = np.clip(np.searchsorted([q1, q2], r), 0, 2)
        return z10, th12, rl
    z10p, thp, rlp = bands(P)
    z10g, thg, rlg = bands(G)
    S = np.full((10, 12, 3), np.nan); V = np.zeros((10, 12, 3), dtype=int)
    for name, (z, t, r, d) in {"p2g": (z10p, thp, rlp, dg), "g2p": (z10g, thg, rlg, dp)}.items():
        acc = {}
        cnt = {}
        for a, b, c, dist in zip(z, t, r, d):
            acc[(a, b, c)] = acc.get((a, b, c), 0.0) + float(dist)
            cnt[(a, b, c)] = cnt.get((a, b, c), 0) + 1
        for key in acc:
            if name == "p2g":
                S[key] = acc[key] / cnt[key]
            else:
                if np.isfinite(S[key]):
                    S[key] = 0.5 * (S[key] + acc[key] / cnt[key])
                    V[key] = 1
    return S, V.astype(bool)

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

out_rows = []
prof_rows = []
p99 = {}
ratio = {}
for split, src in [("val", None), ("test", TEST_FIN)]:
    samples = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
    dfx = samples[samples["split"] == split].reset_index(drop=True)
    MIR = WORK / "val_mirror"
    all_cells = []
    for tag in ["orig", "new"]:
        cells_tag = []
        for i in range(150):
            if split == "val":
                gt = np.load(MIR / "gt" / f"{i:03d}.npy").astype(np.float64)
            else:
                gt = np.load(src / "gt" / f"{i:03d}.npy").astype(np.float64)
            if tag == "new":
                gt = scale_z(gt)
            for m in METHODS:
                if split == "val":
                    pred = np.load(MIR / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
                else:
                    pred = np.load(src / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
                if tag == "new":
                    pred = scale_z(pred)
                S, V = cell_sym_cd(pred, gt)
                cells_tag.append(S[V])
                for z in range(10):
                    row = S[z][V[z]]
                    if row.size:
                        prof_rows.append(dict(split=split, state=tag, method=m, i=i,
                                              band=z, sym=float(row.mean()), n=int(row.size)))
            if (i + 1) % 50 == 0:
                print("CELLCD_PROGRESS", split, tag, i + 1, flush=True)
        merged = np.concatenate(cells_tag)
        p99[(split, tag)] = float(np.percentile(merged, 99))
    prof = pd.DataFrame(prof_rows)
    pr = prof[(prof["split"] == split) & (prof["state"] == "orig")]
    for m in METHODS:
        pm = pr[pr["method"] == m]
        bt = pm[pm["band"] == 0]["sym"].mean() / pm[pm["band"] == 9]["sym"].mean()
        ratio[(split, m)] = float(bt)

prof_df = pd.DataFrame(prof_rows)
prof_df.to_csv(WORK / "metrics" / "cellCD_高度剖面.csv", index=False, encoding="utf-8-sig")
summ = prof_df.groupby(["split", "state", "method", "band"])["sym"].mean().reset_index()
summ.to_csv(WORK / "metrics" / "cellCD_汇总.csv", index=False, encoding="utf-8-sig")

print("[p99]", {f"{k[0]}/{k[1]}": round(v, 3) for k, v in p99.items()}, flush=True)
print("[底/顶带比 orig]", {f"{k[0]}/{k[1]}": round(v, 3) for k, v in ratio.items()}, flush=True)
check(abs(p99[("val", "orig")] - 6.61) <= 0.10, f"val orig 合并 99 分位 {p99[('val','orig')]:.3f} vs 论文 6.61（判定图示划分）")
check(abs(p99[("test", "orig")] - 6.61) <= 0.10, f"test orig 合并 99 分位 {p99[('test','orig')]:.3f} vs 论文 6.61（判定图示划分）")
for m, ref in [("BM", 1.28), ("E2", 1.83), ("PCN", 0.98)]:
    for split in ["val", "test"]:
        v = ratio[(split, m)]
        if abs(v - ref) <= 0.05:
            print(f"[info] 底/顶带比 {split} {m} = {v:.3f} ≈ 论文 {ref} → 图7a 图示划分为 {split}", flush=True)

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
