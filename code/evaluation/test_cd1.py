# test_cd1.py — MT 测试集 CD1 坐标敏感性复算
# 原坐标逐样本值取自 mt_long_table_recalc_orig.csv（已与原主表逐行零差异复现，不重算）；
# 重标定坐标 z_new = 0.3048*z_old 同步作用于预测与真值，float64；
# 最近邻实现与 mt_eval.py 固定实现一致（torch float64 分块精确最近邻）。
# 统计：腔体聚类 bootstrap 10000 次（种子 20261017）；配对 Wilcoxon 报效应量、区间与 p 值；
# 预注册配对仅 BM-B1、BM-E2；不得因结果跨零删除比较。
# 用法：cd /data && python validate/test_predictions/code/test_cd1.py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats

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
PAIRS = [("BM", "B1"), ("BM", "E2")]
SCALE = 0.3048
N_BOOT = 10000
SEED = 20261017
DEVICE = "cuda"

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == 150
orig = pd.read_csv(MET / "mt_long_table_recalc_orig.csv")
assert len(orig) == 1050 and orig["method"].nunique() == 7
assert set(orig["method"].unique()) == set(METHODS)


def cd_l1_m(P, G):
    P = torch.as_tensor(P, dtype=torch.float64, device=DEVICE)
    G = torch.as_tensor(G, dtype=torch.float64, device=DEVICE)

    def mean_min(A, B):
        s = 0.0
        for c in range(0, A.shape[0], 2048):
            d = torch.cdist(A[c:c + 2048], B)
            s += float(d.min(dim=1).values.sum().item())
        return s / A.shape[0]

    return 0.5 * (mean_min(P, G) + mean_min(G, P))


rows = []
for m in METHODS:
    for i in range(150):
        pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        assert pred.shape == (16384, 3) and gt.shape == (16384, 3)
        assert np.isfinite(pred).all() and np.isfinite(gt).all()
        pn = pred.copy()
        pn[:, 2] *= SCALE
        gn = gt.copy()
        gn[:, 2] *= SCALE
        assert np.isfinite(pn).all() and np.isfinite(gn).all()
        cdn = cd_l1_m(pn, gn)
        assert np.isfinite(cdn) and cdn > 0
        rows.append(dict(method=m, i=i, cd_new=round(cdn, 5)))
    print("CD_NEW_DONE", m, flush=True)

new = pd.DataFrame(rows)
out = orig.merge(new, on=["method", "i"], how="inner", validate="one_to_one")
assert len(out) == 1050
out = out.rename(columns={"cd_l1_m": "cd_orig"})
out = out[["method", "i", "cav", "level", "cd_orig", "cd_new"]]
out.to_csv(MET / "test_CD1_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")
print("saved test_CD1_逐样本坐标敏感性.csv rows=", len(out), flush=True)


def summ(col, tag):
    s = out.groupby("method")[col].agg(["mean", "median", "std", "count"])
    s["rank_" + tag] = s["mean"].rank().astype(int)
    return s


s_o = summ("cd_orig", "orig").rename(columns={"mean": "mean_orig", "median": "median_orig",
                                             "std": "std_orig", "count": "n",
                                             "rank_orig": "rank_orig"})
s_n = summ("cd_new", "new").rename(columns={"mean": "mean_new", "median": "median_new",
                                            "std": "std_new", "count": "count_new",
                                            "rank_new": "rank_new"})
summary = s_o.join(s_n.drop(columns=["count_new"]))
summary = summary[["n", "mean_orig", "median_orig", "std_orig", "rank_orig",
                   "mean_new", "median_new", "std_new", "rank_new"]]
summary.to_csv(MET / "test_CD1_汇总.csv", encoding="utf-8-sig")
print(summary.round(5).to_string(), flush=True)

lvl = out.groupby(["method", "level"])[["cd_orig", "cd_new"]].mean().reset_index()
lvl.to_csv(MET / "test_CD1_分覆盖度.csv", index=False, encoding="utf-8-sig")
print("saved test_CD1_分覆盖度.csv", flush=True)


def pair_stats(a, b, col):
    da = out[out["method"] == a].set_index("i")[col]
    db = out[out["method"] == b].set_index("i")[col]
    d = (da - db).loc[np.arange(150)].values  # 逐样本配对差 A - B
    cav = dfv["name"].values
    w = stats.wilcoxon(d)
    rng = np.random.default_rng(SEED)
    uniq = np.unique(cav)
    vals = np.array([d[cav == c].mean() for c in uniq])
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sel = rng.integers(0, len(uniq), len(uniq))
        boot[k] = vals[sel].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(A=a, B=b, metric=col, n=150, n_cav=50,
                mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                wilcoxon_stat=float(w.statistic), p_value=float(w.pvalue),
                boot_ci_lo=float(lo), boot_ci_hi=float(hi), n_boot=N_BOOT, seed=SEED)


prows = [pair_stats(a, b, c) for (a, b) in PAIRS for c in ["cd_orig", "cd_new"]]
pdf = pd.DataFrame(prows)
pdf.to_csv(MET / "test_CD1_重标定配对核查.csv", index=False, encoding="utf-8-sig")
print(pdf.round(6).to_string(index=False), flush=True)
print("CD1_SENSITIVITY_DONE", flush=True)
