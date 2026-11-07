# test_common_support.py — MT 测试集共同支撑体积坐标敏感性复算
# Fixed definition of the cross-section area used throughout this file:
#   500x72 (z,theta) 网格，格内径向聚合为 max；每层有效方位格 >=3 周向插值并计算截面积，
#   否则该层面积为 NaN；areas/vol/wb 三个函数与原实现逐式一致（np.trapz 代 np.trapezoid，
#   数学相同，numpy 1.26.4 无 trapezoid）。
#   共同支撑掩码 common = isfinite(A) & isfinite(B) & isfinite(REF)，为共同有效切片集合，
#   【不要求空间连续】；"共同高度区间"为历史名称，不得改写为连续起止区间。
#   N_common < 10：该组合共同体积与误差记 NaN 并列入异常清单，不得插值补足或删样本。
#   误差 E_X_pct = 100*abs(V_X_common - V_REF_common)/V_REF_common。
# 坐标纪律：点云、真值、原井筒闭合点（原坐标 zmin+2..zmin+60 生成）及积分坐标同步 z*0.3048；
# 不得在新坐标下重新生成井筒闭合点；mean/p95 仅可作另行标注的机制诊断，主口径保持 max。
# 用法：cd /data && python validate/test_predictions/code/test_common_support.py
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

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
PAIRS = list(combinations(METHODS, 2))          # 21 组
SCALE = 0.3048
NS, NT = 500, 72
DTH = 2 * np.pi / NT

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == 150


def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q


def wb(xc, yc, z0, z1):
    zs = np.arange(z0, z1, .5)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([xc + .61 * np.cos(T).ravel(),
                            yc + .61 * np.sin(T).ravel(), Z.ravel()])


def areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H / NS
    k = np.clip(((pts[:, 2] - zmin) / dz).astype(int), 0, NS - 1)
    th = np.mod(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, NT - 1)
    A = np.zeros((NS, NT))
    C = np.zeros((NS, NT), int)
    np.maximum.at(A, (k, tb), rr)
    np.add.at(C, (k, tb), 1)
    idx = np.arange(NT)
    out = np.full(NS, np.nan)
    for s in range(NS):
        m = C[s] > 0
        if m.sum() < 3:
            continue
        oi = np.where(m)[0]
        ei = np.concatenate([oi - NT, oi, oi + NT])
        ev = np.tile(A[s][oi], 3)
        o = np.argsort(ei)
        filled = np.interp(idx, ei[o], ev[o])
        out[s] = .5 * float(np.trapz(filled ** 2, dx=DTH))
    return out


def vol(a, zmin, H):
    zc = zmin + (np.arange(NS) + .5) * H / NS
    m = np.isfinite(a)
    return (float(np.trapz(a[m], zc[m])) if m.sum() >= 10 else np.nan), int(m.sum())


# ---- 逐样本面积场：两坐标态一次算齐 ----
store = {tag: {m: np.full((150, NS), np.nan) for m in METHODS + ["REF"]} for tag in ["orig", "new"]}
meta = []
for i in range(150):
    gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
    assert gt.shape == (16384, 3) and np.isfinite(gt).all()
    zmin = float(gt[:, 2].min())
    H = float(gt[:, 2].max() - zmin)
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    w = wb(xc, yc, zmin + 2, zmin + 60)             # 原坐标闭合点
    clouds = {"REF": gt}
    for m in METHODS:
        p = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
        assert p.shape == (16384, 3) and np.isfinite(p).all()
        clouds[m] = p
    for tag in ["orig", "new"]:
        zmin_t, H_t = (zmin, H) if tag == "orig" else (zmin * SCALE, H * SCALE)
        for m, c in clouds.items():
            c_t = c if tag == "orig" else scale_z(c)
            w_t = w if tag == "orig" else scale_z(w)
            store[tag][m][i] = areas(np.vstack([c_t, w_t]), xc, yc, zmin_t, H_t)
    meta.append(dict(i=i, cav=str(dfv.loc[i, "name"]), level=int(dfv.loc[i, "level"]),
                     zmin=zmin, H=H))
    if (i + 1) % 25 == 0:
        print("COMMON_AREAS_PROGRESS", i + 1, flush=True)

# ---- 交叉核对锚点：原坐标 REF 体积与原实现运行结果比对（打印供回贴核对）----
print("XCHECK_REF_ORIG i=0..4", flush=True)
for i in range(5):
    v, n = vol(store["orig"]["REF"][i], meta[i]["zmin"], meta[i]["H"])
    print(f"  i={i} cav={meta[i]['cav']} L{meta[i]['level']} V_REF={v:.4f} N_REF={n}", flush=True)

# ---- 共同支撑体积：21 组 x 150 样本 x 2 态 ----
rows = []
anom = []
for tag in ["orig", "new"]:
    for A, B in PAIRS:
        for i in range(150):
            zmin = meta[i]["zmin"]
            H = meta[i]["H"]
            if tag == "new":
                zmin, H = zmin * SCALE, H * SCALE
            zc = zmin + (np.arange(NS) + .5) * H / NS
            common = np.isfinite(store[tag][A][i]) & np.isfinite(store[tag][B][i]) \
                & np.isfinite(store[tag]["REF"][i])
            n_common = int(common.sum())
            if n_common < 10:
                va = vb = vr = np.nan
                anom.append(dict(state=tag, A=A, B=B, i=i, cav=meta[i]["cav"],
                                 level=meta[i]["level"], N_common=n_common,
                                 note="N_common<10 -> NaN"))
            else:
                va = float(np.trapz(store[tag][A][i][common], zc[common]))
                vb = float(np.trapz(store[tag][B][i][common], zc[common]))
                vr = float(np.trapz(store[tag]["REF"][i][common], zc[common]))
                if not (np.isfinite(va) and np.isfinite(vb) and np.isfinite(vr)):
                    anom.append(dict(state=tag, A=A, B=B, i=i, cav=meta[i]["cav"],
                                     level=meta[i]["level"], N_common=n_common,
                                     note="non-finite volume"))
            rows.append(dict(state=tag, A=A, B=B, i=i, cav=meta[i]["cav"],
                             level=meta[i]["level"], N_common=n_common,
                             common_idx=";".join(map(str, np.where(common)[0])),
                             V_A_common=va, V_B_common=vb, V_REF_common=vr,
                             E_A_pct=100 * abs(va - vr) / vr if np.isfinite(va) else np.nan,
                             E_B_pct=100 * abs(vb - vr) / vr if np.isfinite(vb) else np.nan))

df = pd.DataFrame(rows)
assert len(df) == 2 * 21 * 150
df.to_csv(MET / "test_共同高度_逐样本.csv", index=False, encoding="utf-8-sig")
print("saved test_共同高度_逐样本.csv rows=", len(df), flush=True)

s = df.groupby(["state", "A", "B"]).agg(
    N_common_mean=("N_common", "mean"),
    E_A_mean=("E_A_pct", "mean"), E_B_mean=("E_B_pct", "mean"),
    n=("i", "count"), n_nan=("E_A_pct", lambda x: int(x.isna().sum()))).reset_index()
s["A_minus_B_pp"] = s.E_A_mean - s.E_B_mean
s.to_csv(MET / "test_共同高度_汇总.csv", index=False, encoding="utf-8-sig")
print(s.round(4).to_string(index=False), flush=True)

if anom:
    pd.DataFrame(anom).to_csv(MET / "test_共同高度_异常清单.csv", index=False, encoding="utf-8-sig")
print("COMMON_SUPPORT_DONE anomalies=", len(anom), flush=True)
