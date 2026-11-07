# val_cd1.py — P1 验证集 CD1 物理米制重算（适配自 test_cd1.py，固定实现）
# 固定口径：torch float64 分块精确最近邻，半和；z_new = 0.3048*z_old 同步作用于预测/真值/观测。
# 锚点：val BM 前三样本 main_cd（recompute_BM_validation.csv）；
#       七方法 orig 均值 = 表 3 验证列；ID-1024/ID-8192 orig 均值 = 表 3（5.042 / 3.771）。
# ID-1024 按固定链 Philox(SeedSequence(20260921,[num,level,2])).permutation(8192)[:1024] 抽取。
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

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
PAIRS = [("BM", "B1"), ("BM", "E2")]
SCALE = 0.3048
N_BOOT = 10000
SEED = 20261017
DEVICE = "cuda"
T3_VAL_MEAN = {"B1": 1.509, "BM": 1.399, "E2": 1.459, "PCN": 1.932,
               "B3": 1.858, "PoinTr": 2.134, "B2": 2.411}
# 验证侧重标定 CD1 均值锚点（v34 表 5 重标定口径；用户核定 ，tol 0.01）
T3_VAL_NEW_MEAN = {"B1": 0.898, "BM": 0.896, "E2": 0.946, "PCN": 1.518,
                   "B3": 1.185, "PoinTr": 1.502, "B2": 1.769}
T3_VAL_ID = {"ID-1024": 5.042, "ID-8192": 3.771}
VAL_BM_MAINCD = {0: 1.76163, 1: 1.63431, 2: 1.62878}
MASTER_SEED = 20260921

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q

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

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "val"].reset_index(drop=True)
assert len(dfv) == 150

rows = []
for m in METHODS:
    for i in range(150):
        pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        assert pred.shape == (16384, 3) and gt.shape == (16384, 3)
        cdo = cd_l1_m(pred, gt)
        cdn = cd_l1_m(scale_z(pred), scale_z(gt))
        rows.append(dict(method=m, i=i, cav=dfv.loc[i, "name"], level=int(dfv.loc[i, "level"]),
                         cd_orig=round(cdo, 5), cd_new=round(cdn, 5)))
    print("METHOD_DONE", m, flush=True)

# ID 基线（不补全：观测 vs 真值）
for tag in ["ID-8192", "ID-1024"]:
    for i in range(150):
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        obs = np.load(REC / "obs" / f"{dfv.loc[i,'name']}_L{int(dfv.loc[i,'level'])}.npy")
        if tag == "ID-1024":
            num = int(str(dfv.loc[i, "name"]).split("_")[1])
            rng = np.random.Generator(np.random.Philox(
                np.random.SeedSequence(MASTER_SEED, spawn_key=[num, int(dfv.loc[i, "level"]), 2])))
            obs = obs[rng.permutation(obs.shape[0])[:1024]]
        obs = obs.astype(np.float64)
        cdo = cd_l1_m(obs, gt)
        cdn = cd_l1_m(scale_z(obs), scale_z(gt))
        rows.append(dict(method=tag, i=i, cav=dfv.loc[i, "name"], level=int(dfv.loc[i, "level"]),
                         cd_orig=round(cdo, 5), cd_new=round(cdn, 5)))
    print("ID_DONE", tag, flush=True)

out = pd.DataFrame(rows)
out.to_csv(MET / "val_CD1_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")
print("saved val_CD1_逐样本坐标敏感性.csv rows=", len(out), flush=True)

methods7 = out[out["method"].isin(METHODS)]
idrows = out[out["method"].isin(["ID-1024", "ID-8192"])]

# 锚点核查
for i in range(3):
    c = float(methods7[(methods7.method == "BM") & (methods7.i == i)]["cd_orig"].iloc[0])
    check(abs(c - VAL_BM_MAINCD[i]) <= 1e-3, f"val BM i={i} cd_orig {c:.5f} vs main_cd {VAL_BM_MAINCD[i]:.5f}")
for m in METHODS:
    mean_o = float(methods7[methods7.method == m]["cd_orig"].mean())
    tol = 3e-3 if m == "PCN" else 1e-3
    check(abs(mean_o - T3_VAL_MEAN[m]) <= tol, f"val {m} cd_orig 均值 {mean_o:.4f} vs 表3 {T3_VAL_MEAN[m]:.3f}")
for tag in ["ID-1024", "ID-8192"]:
    mean_o = float(idrows[idrows.method == tag]["cd_orig"].mean())
    check(abs(mean_o - T3_VAL_ID[tag]) <= 1e-3, f"val {tag} cd_orig 均值 {mean_o:.4f} vs 表3 {T3_VAL_ID[tag]:.3f}")

# 重标定均值锚点 + 比值恒定性自检（判据：复现归档值；比值不得恒等 0.3048，隐含系数应逐方法不同）
for m in METHODS:
    sub = methods7[methods7.method == m]
    mean_n = float(sub["cd_new"].mean())
    check(abs(mean_n - T3_VAL_NEW_MEAN[m]) <= 0.01,
          f"val {m} cd_new 均值 {mean_n:.4f} vs 归档重标定 {T3_VAL_NEW_MEAN[m]:.3f}（tol 0.01）")
    r = sub["cd_new"] / sub["cd_orig"]
    implied = mean_n / float(sub["cd_orig"].mean())
    check(not np.allclose(r, SCALE, atol=1e-6),
          f"val {m} 比值非恒等 0.3048（range [{r.min():.4f},{r.max():.4f}]，隐含系数 {implied:.3f}）→ 重算而非换算")

# 汇总与分档
def summ(col, tag):
    s = methods7.groupby("method")[col].agg(["mean", "median", "std", "count"])
    s["rank_" + tag] = s["mean"].rank().astype(int)
    return s
s_o = summ("cd_orig", "orig").rename(columns={"mean": "mean_orig", "median": "median_orig",
                                             "std": "std_orig", "count": "n", "rank_orig": "rank_orig"})
s_n = summ("cd_new", "new").rename(columns={"mean": "mean_new", "median": "median_new",
                                            "std": "std_new", "count": "count_new", "rank_new": "rank_new"})
summary = s_o.join(s_n.drop(columns=["count_new"]))
summary = summary[["n", "mean_orig", "median_orig", "std_orig", "rank_orig",
                   "mean_new", "median_new", "std_new", "rank_new"]]
summary.to_csv(MET / "val_CD1_汇总.csv", encoding="utf-8-sig")
print(summary.round(5).to_string(), flush=True)

ids = idrows.groupby("method")[["cd_orig", "cd_new"]].agg(["mean", "median", "std"])
ids.to_csv(MET / "val_CD1_ID汇总.csv", encoding="utf-8-sig")
print(ids.round(5).to_string(), flush=True)

lvl = out.groupby(["method", "level"])[["cd_orig", "cd_new"]].mean().reset_index()
lvl.to_csv(MET / "val_CD1_分覆盖度.csv", index=False, encoding="utf-8-sig")

# 配对核查（预注册 BM-B1、BM-E2）
def pair_stats(a, b, col):
    da = out[out["method"] == a].set_index("i")[col]
    db = out[out["method"] == b].set_index("i")[col]
    d = (da - db).loc[np.arange(150)].values
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

pdf = pd.DataFrame([pair_stats(a, b, c) for (a, b) in PAIRS for c in ["cd_orig", "cd_new"]])
pdf.to_csv(MET / "val_CD1_重标定配对核查.csv", index=False, encoding="utf-8-sig")
print(pdf.round(6).to_string(index=False), flush=True)

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
