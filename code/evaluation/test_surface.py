# Local surface metrics under the re-scaled coordinate frame (test split).

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/test_predictions"
REC = WORK / "recovered"
MET = WORK / "metrics"
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "PoinTr", "B1", "B2", "B3"]]
NS, NT = 500, 72
DTH = 2 * np.pi / NT
K_NN = 17            # 自身 + 16 邻点
SCALE = 0.3048

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == 150


def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q


def pca_normals(P, k=K_NN, chunk=4096):
    """局部 PCA 法向：每个点的邻域为自身+k-1 个最近邻；返回单位法向（最小特征值对应特征向量）。"""
    P = np.asarray(P, dtype=np.float64)
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)                     # (n,k)，第 0 列为自身
    N = np.empty((P.shape[0], 3), dtype=np.float64)
    for s in range(0, P.shape[0], chunk):
        X = P[idx[s:s + chunk]]                     # (b,k,3)
        Xc = X - X.mean(axis=1, keepdims=True)
        C = np.einsum("bki,bkj->bij", Xc, Xc) / k   # (b,3,3) 协方差
        _, V = np.linalg.eigh(C)                    # 升序特征值
        n = V[:, :, 0]
        nrm = np.linalg.norm(n, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        N[s:s + chunk] = n / nrm
    return N


def normal_angles(P, G, tG, Ng):
    """预测法向与最近真值点法向的夹角（度）。tG/Ng 为真值侧已建好的树与法向。"""
    Np = pca_normals(P)
    _, j = tG.query(P, k=1)
    dot = np.abs(np.einsum("ij,ij->i", Np, Ng[j]))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def mean_radius_grid(P, xc, yc, zmin, H):
    """500x72 (z,theta) 网格，格内平均半径；空格为 NaN；不填充。"""
    P = P.astype(np.float64)
    dz = H / NS
    k = np.clip(((P[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(P[:, 1] - yc, P[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(P[:, 0] - xc, P[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    S = np.zeros((NS, NT))
    C = np.zeros((NS, NT), dtype=int)
    np.add.at(S, (k, tb), rr)
    np.add.at(C, (k, tb), 1)
    R = np.full((NS, NT), np.nan)
    np.divide(S, C, out=R, where=C > 0)
    return R


def curvature_err(Rp, Rg):
    """半径场沿 z 二阶差分的绝对误差，仅在两端三格均有效处计算。返回误差向量（可能为空）。"""
    vp = np.isfinite(Rp)
    vg = np.isfinite(Rg)
    valid = vp[:-2] & vp[1:-1] & vp[2:] & vg[:-2] & vg[1:-1] & vg[2:]
    Dp = Rp[2:] - 2.0 * Rp[1:-1] + Rp[:-2]
    Dg = Rg[2:] - 2.0 * Rg[1:-1] + Rg[:-2]
    return np.abs(Dp - Dg)[valid]


rows = []
anom = []
for tag in ["orig", "new"]:
    for i in range(150):
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        if tag == "new":
            gt = scale_z(gt)
        zmin = float(gt[:, 2].min())
        H = float(gt[:, 2].max() - zmin)
        xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
        tG = cKDTree(gt)
        Ng = pca_normals(gt)
        Rg = mean_radius_grid(gt, xc, yc, zmin, H)
        for m in METHODS:
            pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
            if tag == "new":
                pred = scale_z(pred)
            assert pred.shape == (16384, 3) and np.isfinite(pred).all()
            ang = normal_angles(pred, gt, tG, Ng)
            d, _ = tG.query(pred, k=1)              # 逐点预测至真值距离
            Rp = mean_radius_grid(pred, xc, yc, zmin, H)
            cerr = curvature_err(Rp, Rg)
            rec = dict(method=m, i=i, cav=str(dfv.loc[i, "name"]),
                       level=int(dfv.loc[i, "level"]), state=tag,
                       normal_mean=float(ang.mean()), normal_p90=float(np.percentile(ang, 90)),
                       scatter_p95=float(np.percentile(d, 95)), scatter_max=float(d.max()))
            if cerr.size:
                rec["curv_mean"] = float(cerr.mean())
                rec["curv_p90"] = float(np.percentile(cerr, 90))
            else:
                rec["curv_mean"] = np.nan
                rec["curv_p90"] = np.nan
                anom.append(dict(method=m, i=i, cav=str(dfv.loc[i, "name"]), state=tag,
                                 note="curvature valid-cell empty"))
            assert np.isfinite(ang).all() and np.isfinite(d).all()
            rows.append(rec)
        if (i + 1) % 25 == 0:
            print("SURFACE_PROGRESS", tag, i + 1, flush=True)

df = pd.DataFrame(rows)
assert len(df) == 2100
df.to_csv(MET / "test_局部表面_逐样本坐标敏感性_长表.csv", index=False, encoding="utf-8-sig")

piv = {}
for metric in ["normal_mean", "normal_p90", "curv_mean", "curv_p90", "scatter_p95", "scatter_max"]:
    t = df.pivot_table(index=["method", "i", "cav", "level"], columns="state", values=metric)
    t = t.rename(columns={"orig": f"{metric}_orig", "new": f"{metric}_new"}).reset_index()
    piv[metric] = t
wide = piv["normal_mean"]
for metric in ["normal_p90", "curv_mean", "curv_p90", "scatter_p95", "scatter_max"]:
    wide = wide.merge(piv[metric], on=["method", "i", "cav", "level"], validate="one_to_one")
assert len(wide) == 1050
wide.to_csv(MET / "test_局部表面_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")
print("saved test_局部表面_逐样本坐标敏感性.csv", flush=True)

metrics = ["normal_mean", "normal_p90", "curv_mean", "curv_p90", "scatter_p95", "scatter_max"]
summ = df.groupby(["method", "state"])[metrics].mean().unstack("state")
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ[[f"{m}_{s}" for m in metrics for s in ["orig", "new"]]]
summ.to_csv(MET / "test_局部表面_均值汇总.csv", encoding="utf-8-sig")
print(summ.round(4).to_string(), flush=True)
if anom:
    pd.DataFrame(anom).to_csv(MET / "test_局部表面_异常清单.csv", index=False, encoding="utf-8-sig")
print("SURFACE_SENSITIVITY_DONE anomalies=", len(anom), flush=True)
