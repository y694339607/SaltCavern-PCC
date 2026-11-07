# val_common_support.py — P1 验证集共同高度物理米制重算（逐式适配 test_common_support.py）
# 固定口径：500×72 max 半径格；有效方位格 ≥3 插值；共同支撑 isfinite(A)&isfinite(B)&isfinite(REF)，
# 无连续性要求；共同有限层 <10 → NaN；尺度不变性容差 1e-3。
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

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
PAIRS = sorted(combinations(METHODS, 2))
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

def areas(P, xc, yc, zmin, H):
    P = P.astype(np.float64)
    dz = H / NS
    k = np.clip(((P[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(P[:, 1] - yc, P[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(P[:, 0] - xc, P[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    A = np.zeros((NS, NT)); C = np.zeros((NS, NT), dtype=int)
    np.maximum.at(A, (k, tb), rr); np.add.at(C, (k, tb), 1)
    idx = np.arange(NT)
    out = np.full(NS, np.nan)
    for s in range(NS):
        m = C[s] > 0
        if m.sum() < 3:
            continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi - NT, oi, oi + NT])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        out[s] = 0.5 * float(np.trapz(np.interp(idx, ex_i[o], ex_v[o]) ** 2, dx=DTH))
    return out

def vol_common(A, B, REF, zmin, H):
    dz = H / NS
    zc = zmin + (np.arange(NS) + 0.5) * dz
    common = np.isfinite(A) & np.isfinite(B) & np.isfinite(REF)
    n_common = int(common.sum())
    if n_common < 10:
        return np.nan, np.nan, n_common
    z_cc, v_cc = zc[common], REF[common]
    A_cc, B_cc = A[common], B[common]
    V_ref = float(np.trapz(np.interp(zc, z_cc, v_cc), zc))
    V_A = float(np.trapz(np.interp(zc, z_cc, A_cc), zc))
    V_B = float(np.trapz(np.interp(zc, z_cc, B_cc), zc))
    return V_A, V_B, n_common

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "val"].reset_index(drop=True)
assert len(dfv) == 150

rows = []
n_flip = 0
for i in range(150):
    name = str(dfv.loc[i, "name"]); level = int(dfv.loc[i, "level"])
    gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
    zmin = float(gt[:, 2].min()); H = float(gt[:, 2].max() - zmin)
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    states = {"orig": (gt, {}), "new": (scale_z(gt), {})}
    clouds = {}
    for tag in ["orig", "new"]:
        G = states[tag][0]
        zmin_t = float(G[:, 2].min()); H_t = float(G[:, 2].max() - zmin_t)
        clouds[tag] = {"REF": areas(G, xc, yc, zmin_t, H_t)}
        for m in METHODS:
            pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
            if tag == "new":
                pred = scale_z(pred)
            clouds[tag][m] = areas(pred, xc, yc, zmin_t, H_t)
    for (a, b) in PAIRS:
        rec = dict(A=a, B=b, i=i, cav=name, level=level)
        V = {}
        for tag in ["orig", "new"]:
            G = states[tag][0]
            zmin_t = float(G[:, 2].min()); H_t = float(G[:, 2].max() - zmin_t)
            V_A, V_B, n_c = vol_common(clouds[tag][a], clouds[tag][b], clouds[tag]["REF"],
                                       zmin_t, H_t)
            V[tag] = (V_A, V_B, n_c)
        (Vao, Vbo, nco), (Van, Vbn, ncn) = V["orig"], V["new"]
        rec.update(V_A_orig=Vao, V_B_orig=Vbo, N_common_orig=nco,
                   V_A_new=Van, V_B_new=Vbn, N_common_new=ncn,
                   diff_signed_orig=(Vao - Vbo) if (np.isfinite(Vao) and np.isfinite(Vbo)) else np.nan,
                   diff_signed_new=(Van - Vbn) if (np.isfinite(Van) and np.isfinite(Vbn)) else np.nan)
        if nco != ncn:
            n_flip += 1
        if np.isfinite(Vao) and np.isfinite(Van):
            dev = abs(Van - Vao * SCALE) / abs(Vao)
            if dev > 1e-3:
                check(False, f"{a}-{b} i={i} A 尺度偏差 {dev:.2e}")
        if np.isfinite(Vbo) and np.isfinite(Vbn):
            dev = abs(Vbn - Vbo * SCALE) / abs(Vbo)
            if dev > 1e-3:
                check(False, f"{a}-{b} i={i} B 尺度偏差 {dev:.2e}")
        if nco >= 10 and np.isfinite(Vao) and np.isfinite(Vbo):
            rec["rel_diff_orig"] = float((Vao - Vbo) / Vbo)
            rec["abs_diff_m3_orig"] = float(Vao - Vbo)
        else:
            rec["rel_diff_orig"] = np.nan; rec["abs_diff_m3_orig"] = np.nan
        if ncn >= 10 and np.isfinite(Van) and np.isfinite(Vbn):
            rec["rel_diff_new"] = float((Van - Vbn) / Vbn)
            rec["abs_diff_m3_new"] = float(Van - Vbn)
        else:
            rec["rel_diff_new"] = np.nan; rec["abs_diff_m3_new"] = np.nan
        rows.append(rec)
    if (i + 1) % 25 == 0:
        print("COMMON_PROGRESS", i + 1, flush=True)

df = pd.DataFrame(rows)
assert len(df) == 3150
df.to_csv(MET / "val_共同高度_逐样本.csv", index=False, encoding="utf-8-sig")
s = df.groupby(["A", "B"])[["N_common_orig", "N_common_new"]].mean()
s.to_csv(MET / "val_共同高度_汇总.csv", encoding="utf-8-sig")
print(s.round(2).to_string(), flush=True)
check(n_flip == 0, f"N_common 尺度不变（翻格 {n_flip}/3150）")
print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
