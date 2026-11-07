# val_surface.py — P1 验证集局部表面物理米制重算（逐式适配 test_surface.py）
# 论文锚点（v34 §3.3 [85]，验证集中位数，orig 态）：BM 法向均值 6.54 / p90 10.22 / 散布 p95 2.41；
# B1 22.99 / 68.80 / 2.35（单位：度、m）
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
REC = WORK / "val_mirror"
MET = WORK / "metrics"
MET.mkdir(parents=True, exist_ok=True)
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "PoinTr", "B1", "B2", "B3"]]
SCALE = 0.3048
NS, NT = 500, 72
DTH = 2 * np.pi / NT

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q

def pca_normals(P, k=17, chunk=4096):
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)
    N = np.empty((P.shape[0], 3), dtype=np.float64)
    for s in range(0, P.shape[0], chunk):
        X = P[idx[s:s + chunk]]
        Xc = X - X.mean(axis=1, keepdims=True)
        C = np.einsum("bki,bkj->bij", Xc, Xc) / k
        _, V = np.linalg.eigh(C)
        n = V[:, :, 0]
        nrm = np.linalg.norm(n, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        N[s:s + chunk] = n / nrm
    return N

def normal_angles(P, tG, Ng):
    Np = pca_normals(P)
    _, j = tG.query(P, k=1)
    dot = np.abs(np.einsum("ij,ij->i", Np, Ng[j]))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))

def mean_radius_grid(P, xc, yc, zmin, H):
    P = P.astype(np.float64)
    dz = H / NS
    k = np.clip(((P[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(P[:, 1] - yc, P[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(P[:, 0] - xc, P[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    S = np.zeros((NS, NT)); C = np.zeros((NS, NT), dtype=int)
    np.add.at(S, (k, tb), rr); np.add.at(C, (k, tb), 1)
    R = np.full((NS, NT), np.nan)
    np.divide(S, C, out=R, where=C > 0)
    return R

def curvature_err(Rp, Rg):
    vp = np.isfinite(Rp); vg = np.isfinite(Rg)
    valid = vp[:-2] & vp[1:-1] & vp[2:] & vg[:-2] & vg[1:-1] & vg[2:]
    Dp = Rp[2:] - 2.0 * Rp[1:-1] + Rp[:-2]
    Dg = Rg[2:] - 2.0 * Rg[1:-1] + Rg[:-2]
    return np.abs(Dp - Dg)[valid]

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "val"].reset_index(drop=True)
assert len(dfv) == 150

rows = []
anom = []
for tag in ["orig", "new"]:
    for i in range(150):
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        if tag == "new":
            gt = scale_z(gt)
        zmin = float(gt[:, 2].min()); H = float(gt[:, 2].max() - zmin)
        xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
        tG = cKDTree(gt); Ng = pca_normals(gt)
        Rg = mean_radius_grid(gt, xc, yc, zmin, H)
        for m in METHODS:
            pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
            if tag == "new":
                pred = scale_z(pred)
            assert pred.shape == (16384, 3) and np.isfinite(pred).all()
            ang = normal_angles(pred, tG, Ng)
            d, _ = tG.query(pred, k=1)
            Rp = mean_radius_grid(pred, xc, yc, zmin, H)
            cerr = curvature_err(Rp, Rg)
            rec = dict(method=m, i=i, cav=str(dfv.loc[i, "name"]), level=int(dfv.loc[i, "level"]),
                       state=tag, normal_mean=float(ang.mean()),
                       normal_p90=float(np.percentile(ang, 90)),
                       scatter_p95=float(np.percentile(d, 95)), scatter_max=float(d.max()))
            if cerr.size:
                rec["curv_mean"] = float(cerr.mean()); rec["curv_p90"] = float(np.percentile(cerr, 90))
            else:
                rec["curv_mean"] = np.nan; rec["curv_p90"] = np.nan
                anom.append(dict(method=m, i=i, state=tag, note="curvature valid-cell empty"))
            assert np.isfinite(ang).all() and np.isfinite(d).all()
            rows.append(rec)
        if (i + 1) % 25 == 0:
            print("SURFACE_PROGRESS", tag, i + 1, flush=True)

df = pd.DataFrame(rows)
assert len(df) == 2100
df.to_csv(MET / "val_局部表面_逐样本坐标敏感性_长表.csv", index=False, encoding="utf-8-sig")
piv = {}
for metric in ["normal_mean", "normal_p90", "curv_mean", "curv_p90", "scatter_p95", "scatter_max"]:
    t = df.pivot_table(index=["method", "i", "cav", "level"], columns="state", values=metric)
    t = t.rename(columns={"orig": f"{metric}_orig", "new": f"{metric}_new"}).reset_index()
    piv[metric] = t
wide = piv["normal_mean"]
for metric in ["normal_p90", "curv_mean", "curv_p90", "scatter_p95", "scatter_max"]:
    wide = wide.merge(piv[metric], on=["method", "i", "cav", "level"], how="inner")
assert len(wide) == 1050
wide.to_csv(MET / "val_局部表面_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")

med = wide.groupby("method")[["normal_mean_orig", "normal_mean_new",
                              "normal_p90_orig", "normal_p90_new",
                              "scatter_p95_orig", "scatter_p95_new"]].median()
med.to_csv(MET / "val_局部表面_中位数汇总.csv", encoding="utf-8-sig")
print(med.round(4).to_string(), flush=True)

# 论文锚点（v34 §3.3[85]，验证集中位数，orig 态）
check(abs(med.loc["BM", "normal_mean_orig"] - 6.54) <= 0.05,
      f"val BM normal_mean 中位 {med.loc['BM','normal_mean_orig']:.3f} vs 论文 6.54")
check(abs(med.loc["BM", "normal_p90_orig"] - 10.22) <= 0.05,
      f"val BM normal_p90 中位 {med.loc['BM','normal_p90_orig']:.3f} vs 论文 10.22")
check(abs(med.loc["BM", "scatter_p95_orig"] - 2.41) <= 0.05,
      f"val BM scatter_p95 中位 {med.loc['BM','scatter_p95_orig']:.3f} vs 论文 2.41")
check(abs(med.loc["B1", "normal_mean_orig"] - 22.99) <= 0.05,
      f"val B1 normal_mean 中位 {med.loc['B1','normal_mean_orig']:.3f} vs 论文 22.99")
check(abs(med.loc["B1", "normal_p90_orig"] - 68.80) <= 0.10,
      f"val B1 normal_p90 中位 {med.loc['B1','normal_p90_orig']:.3f} vs 论文 68.80")
check(abs(med.loc["B1", "scatter_p95_orig"] - 2.35) <= 0.05,
      f"val B1 scatter_p95 中位 {med.loc['B1','scatter_p95_orig']:.3f} vs 论文 2.35")

if anom:
    pd.DataFrame(anom).to_csv(MET / "val_局部表面_异常清单.csv", index=False, encoding="utf-8-sig")
print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
