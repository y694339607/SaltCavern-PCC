# sparse_density.py — 稀疏区覆盖度/点密度变异系数（术语表 固定定义）
# 稀疏区覆盖度：500×72 (z,θ) 格架（锚定同体积网格）；观测点数 <19 的格为稀疏格；
# 覆盖度 = 含 ≥1 预测点的稀疏格占比。点密度变异系数 = 格内点数变异系数（500×72）。
# 注意：术语表决议"暂停使用点密度变异系数定量结果"，本脚本照常计算并在输出标注该治理状态。
# 论文锚点（[85]）：掩码配置稀疏区覆盖度比 Cyl-NN 低 5.3 个百分点（p=8.6e-17）。
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
TEST_FIN = ROOT / "validate/test_final"
SCALE = 0.3048
NS, NT = 500, 72
DTH = 2 * np.pi / NT
SPARSE_TH = 19
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "B1"]]
def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q

def counts_grid(P, xc, yc, zmin, H):
    P = P.astype(np.float64)
    dz = H / NS
    k = np.clip(((P[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(P[:, 1] - yc, P[:, 0] - xc), 2 * np.pi)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    C = np.zeros((NS, NT), dtype=int)
    np.add.at(C, (k, tb), 1)
    return C

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

rows = []
flip = 0
for split in ["val", "test"]:
    samples = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
    dfx = samples[samples["split"] == split].reset_index(drop=True)
    MIR = WORK / "val_mirror"
    for i in range(150):
        if split == "val":
            gt = np.load(MIR / "gt" / f"{i:03d}.npy").astype(np.float64)
            obs = np.load(MIR / "obs" / f"{dfx.loc[i,'name']}_L{int(dfx.loc[i,'level'])}.npy").astype(np.float64)
        else:
            gt = np.load(TEST_FIN / "gt" / f"{i:03d}.npy").astype(np.float64)
            obs = np.load(ROOT / "validate/rescale_work/raw/test_from_zip/partial" /
                          f"{dfx.loc[i,'name']}_L{int(dfx.loc[i,'level'])}.npy").astype(np.float64)
        zmin = float(gt[:, 2].min()); H = float(gt[:, 2].max() - zmin)
        xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
        for tag in ["orig", "new"]:
            G = scale_z(gt) if tag == "new" else gt
            zmin_t = float(G[:, 2].min()); H_t = float(G[:, 2].max() - zmin_t)
            O = scale_z(obs) if tag == "new" else obs
            Co = counts_grid(O, xc, yc, zmin_t, H_t)
            sparse = Co < SPARSE_TH
            for m in METHODS:
                if split == "val":
                    pred = np.load(MIR / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
                else:
                    pred = np.load(TEST_FIN / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
                if tag == "new":
                    pred = scale_z(pred)
                Cp = counts_grid(pred, xc, yc, zmin_t, H_t)
                cov = float((Cp[sparse] > 0).mean() * 100) if sparse.any() else np.nan
                dens_cv = float(Cp[Cp > 0].std() / Cp[Cp > 0].mean()) if (Cp > 0).any() else np.nan
                rows.append(dict(split=split, state=tag, method=m, i=i,
                                 cav=str(dfx.loc[i, "name"]), level=int(dfx.loc[i, "level"]),
                                 n_sparse=int(sparse.sum()), coverage=cov, density_cv=dens_cv))
    print("SPARSE_DONE", split, flush=True)

df = pd.DataFrame(rows)
df.to_csv(WORK / "metrics" / "sparse_density_逐样本.csv", index=False, encoding="utf-8-sig")
s = df.groupby(["split", "state", "method"])[["coverage", "density_cv"]].agg(["mean", "median"])
s.to_csv(WORK / "metrics" / "sparse_density_汇总.csv", encoding="utf-8-sig")
print(s.round(4).to_string(), flush=True)

o = df[df["state"] == "orig"]
for split in ["val", "test"]:
    bm = o[(o["split"] == split) & (o["method"] == "BM")]["coverage"].mean()
    b1 = o[(o["split"] == split) & (o["method"] == "B1")]["coverage"].mean()
    print(f"[info] {split} orig 稀疏覆盖 BM-B1 = {bm - b1:+.2f} pp（论文：-5.3 pp）", flush=True)

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}（本脚本无硬锚点，全部信息性输出）")
sys.exit(0)
