# val_volume.py — P1 验证集体积物理米制重算（适配自 test_volume.py，固定实现）
# 固定口径（逐式沿用）：井筒闭合点原坐标 zmin+2..zmin+60 生成（RW=0.61, dz=0.5, 36 方位）后随云缩放；
# 500×72 网格 max 半径，有效方位格 ≥3 周向插值否则该层 NaN；有限层 ≥10 梯形积分。
# 锚点：v27《可积分切片_逐样本复算.csv》（REF/ID/B1/B2/B3/PoinTr 逐样本 V 与 N）与
#       《recompute_BM_validation.csv》（BM 逐样本 V_pred/V_field/V_ref/n_slices）；E2/PCN 无逐样本锚点仅记录。
# N_slice 双重不变性：N_orig == N_new（尺度不变性）且 N_orig == N_归档（验收清单 #2），翻格逐条留档不静默。
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
REC = WORK / "val_mirror"
ANCH = WORK / "anchors"
MET = WORK / "metrics"
MET.mkdir(parents=True, exist_ok=True)
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "PoinTr", "B1", "B2", "B3"]]
IDTAGS = ["ID-1024", "ID-8192"]
SCALE = 0.3048
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2 * np.pi / N_THETA
MASTER_SEED = 20260921

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "val"].reset_index(drop=True)
assert len(dfv) == 150
kq = pd.read_csv(ANCH / "可积分切片_逐样本复算.csv")
bm = pd.read_csv(ANCH / "recompute_BM_validation.csv")
kq = kq[kq["split"].isin(["val", "validation"])].reset_index(drop=True)   # 源表含 val+test 共 300 行
bm = bm[bm["split"].isin(["val", "validation"])].reset_index(drop=True)
assert len(kq) == 150 and len(bm) == 150

def scale_z(P):
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 2] *= SCALE
    return Q

def wellbore_points(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZ_STEP)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([xc + RW * np.cos(T).ravel(), yc + RW * np.sin(T).ravel(),
                            Z.ravel()]).astype(np.float64)

def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H / N_SLICES
    z = pts[:, 2]
    k = np.clip(((z - zmin) / dz).astype(int), 0, N_SLICES - 1)
    th = np.mod(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc), 2 * np.pi)
    rr = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    tb = np.clip((th / DTH).astype(int), 0, N_THETA - 1)
    A = np.zeros((N_SLICES, N_THETA))
    C = np.zeros((N_SLICES, N_THETA), dtype=int)
    np.maximum.at(A, (k, tb), rr)
    np.add.at(C, (k, tb), 1)
    idx = np.arange(N_THETA)
    areas = np.full(N_SLICES, np.nan)
    for s in range(N_SLICES):
        m = C[s] > 0
        if m.sum() < 3:
            continue
        oi = np.where(m)[0]
        ex_i = np.concatenate([oi - N_THETA, oi, oi + N_THETA])
        ex_v = np.tile(A[s][oi], 3)
        o = np.argsort(ex_i)
        filled = np.interp(idx, ex_i[o], ex_v[o])
        areas[s] = 0.5 * float(np.trapz(filled ** 2, dx=DTH))
    return areas

def volume_of(areas, zmin, H):
    dz = H / N_SLICES
    zc = zmin + (np.arange(N_SLICES) + 0.5) * dz
    m = ~np.isnan(areas)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.trapz(areas[m], zc[m])), int(m.sum())

def obs_of(i, tag):
    name = str(dfv.loc[i, "name"]); level = int(dfv.loc[i, "level"])
    obs = np.load(REC / "obs" / f"{name}_L{level}.npy")
    if tag == "ID-1024":
        num = int(name.split("_")[1])
        rng = np.random.Generator(np.random.Philox(
            np.random.SeedSequence(MASTER_SEED, spawn_key=[num, level, 2])))
        obs = obs[rng.permutation(obs.shape[0])[:1024]]
    return obs.astype(np.float64)

def anchor_val(method, i, field):
    """field: 'V' 或 'N'；返回 v27 锚点（BM 走 bm 表，其余走 kq 表）。"""
    if method == "BM":
        col = {"V": "V_pred", "N": "n_slices"}[field]
        return float(bm.loc[i, col])
    if method == "REF":
        col = {"V": "V_REF", "N": "N_REF"}[field]
    else:
        col = {"V": f"V_{method}", "N": f"N_{method}"}[field]
    return float(kq.loc[i, col])

rows = []
anom = []
dev_rows = []
n_anchor_v_fail = 0
n_n_mismatch_arch = 0
n_n_flip = 0
for i in range(150):
    name = str(dfv.loc[i, "name"])
    level = int(dfv.loc[i, "level"])
    gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
    assert gt.shape == (16384, 3) and np.isfinite(gt).all()
    zmin = float(gt[:, 2].min()); zmax = float(gt[:, 2].max()); H = zmax - zmin
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)
    gt_s = scale_z(gt); wb_s = scale_z(wb)
    zmin_s, H_s = zmin * SCALE, H * SCALE

    gf = np.load(REC / "gt_field" / f"{name}.npy")
    gf2 = np.where(gf > 0, gf, 0.0)
    V_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
    V_field_s = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H_s / 200)
    assert np.isfinite(V_field) and V_field > 0
    assert abs(V_field_s - V_field * SCALE) <= abs(V_field) * 1e-12

    V_ref_o, n_ref_o = volume_of(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)
    V_ref_n, n_ref_n = volume_of(radial_areas(np.vstack([gt_s, wb_s]), xc, yc, zmin_s, H_s), zmin_s, H_s)
    rec = dict(i=i, cav=name, level=level, zmin=zmin, H=H,
               V_field_orig=V_field, V_field_new=V_field_s,
               V_ref_orig=V_ref_o, N_ref_orig=n_ref_o, V_ref_new=V_ref_n, N_ref_new=n_ref_n)

    clouds = [(m, np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)) for m in METHODS]
    clouds += [(t, obs_of(i, t)) for t in IDTAGS]
    for tag, pts in clouds:
        assert np.isfinite(pts).all()
        pts_s = scale_z(pts)
        V_o, n_o = volume_of(radial_areas(np.vstack([pts, wb]), xc, yc, zmin, H), zmin, H)
        V_n, n_n = volume_of(radial_areas(np.vstack([pts_s, wb_s]), xc, yc, zmin_s, H_s), zmin_s, H_s)
        rec[f"V_pred_orig_{tag}"] = V_o; rec[f"N_pred_orig_{tag}"] = n_o
        rec[f"V_pred_new_{tag}"] = V_n; rec[f"N_pred_new_{tag}"] = n_n
        if np.isfinite(V_o) and np.isfinite(V_n):
            dev = abs(V_n - V_o * SCALE) / abs(V_o)
            dev_rows.append(dict(method=tag, i=i, cav=name, level=level, field="V_pred",
                                 rel_dev=float(dev), V_orig=V_o, V_new=V_n,
                                 N_finite_orig=n_o, N_finite_new=n_n))
            if dev > 1e-3:
                check(False, f"{tag} i={i} 尺度偏差 {dev:.2e} > 1e-3")
            if n_o != n_n:
                n_n_flip += 1
                anom.append(dict(i=i, cav=name, field=f"V_pred_{tag}", V_o=V_o, V_n=V_n,
                                 note=f"N_finite changed {n_o}->{n_n} (bin-boundary flip)"))
        else:
            anom.append(dict(i=i, cav=name, field=f"V_pred_{tag}", V_o=V_o, V_n=V_n,
                             note="volume non-finite"))
        # v27 锚点核对（E2/PCN 无锚点跳过）
        if tag in ("BM", "REF", "B1", "B2", "B3", "PoinTr", "ID-1024", "ID-8192"):
            aV = anchor_val(tag, i, "V"); aN = int(anchor_val(tag, i, "N"))
            if np.isfinite(V_o) and abs(V_o - aV) > max(1e-4 * abs(aV), 1.0):
                n_anchor_v_fail += 1
                if n_anchor_v_fail <= 5:
                    check(False, f"{tag} i={i} V_orig {V_o:.1f} vs v27 锚点 {aV:.1f}")
            if n_o != aN:
                n_n_mismatch_arch += 1
                if n_n_mismatch_arch <= 5:
                    check(False, f"{tag} i={i} N_orig {n_o} vs v27 归档 N {aN}")
    rows.append(rec)
    if (i + 1) % 25 == 0:
        print("VOLUME_PROGRESS", i + 1, flush=True)

df = pd.DataFrame(rows)
long_rows = []
for m in METHODS:
    for _, r in df.iterrows():
        V_o, V_n = r[f"V_pred_orig_{m}"], r[f"V_pred_new_{m}"]
        ef_o = abs(V_o - r["V_field_orig"]) / r["V_field_orig"] if np.isfinite(V_o) else np.nan
        ef_n = abs(V_n - r["V_field_new"]) / r["V_field_new"] if np.isfinite(V_n) else np.nan
        er_o = abs(V_o - r["V_ref_orig"]) / r["V_ref_orig"] if np.isfinite(V_o) and np.isfinite(r["V_ref_orig"]) else np.nan
        er_n = abs(V_n - r["V_ref_new"]) / r["V_ref_new"] if np.isfinite(V_n) and np.isfinite(r["V_ref_new"]) else np.nan
        if np.isfinite(ef_o) and np.isfinite(ef_n) and abs(ef_n - ef_o) >= 1e-3:
            check(False, f"{m} i={int(r['i'])} 相对误差漂移 {abs(ef_n-ef_o):.2e}")
        long_rows.append(dict(method=m, i=int(r["i"]), cav=r["cav"], level=int(r["level"]),
                              V_pred_orig=V_o, V_field_orig=r["V_field_orig"], V_ref_orig=r["V_ref_orig"],
                              err_field_orig=ef_o, err_ref_orig=er_o,
                              V_pred_new=V_n, V_field_new=r["V_field_new"], V_ref_new=r["V_ref_new"],
                              err_field_new=ef_n, err_ref_new=er_n,
                              N_finite_orig=int(r[f"N_pred_orig_{m}"]), N_finite_new=int(r[f"N_pred_new_{m}"])))
long_df = pd.DataFrame(long_rows)
assert len(long_df) == 1050
long_df.to_csv(MET / "val_体积_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")

id_rows = []
for t in IDTAGS:
    for _, r in df.iterrows():
        V_o, V_n = r[f"V_pred_orig_{t}"], r[f"V_pred_new_{t}"]
        ef_o = abs(V_o - r["V_field_orig"]) / r["V_field_orig"] if np.isfinite(V_o) else np.nan
        ef_n = abs(V_n - r["V_field_new"]) / r["V_field_new"] if np.isfinite(V_n) else np.nan
        id_rows.append(dict(method=t, i=int(r["i"]), cav=r["cav"], level=int(r["level"]),
                            V_pred_orig=V_o, V_pred_new=V_n, err_field_orig=ef_o, err_field_new=ef_n,
                            N_finite_orig=int(r[f"N_pred_orig_{t}"]), N_finite_new=int(r[f"N_pred_new_{t}"])))
id_df = pd.DataFrame(id_rows)
id_df.to_csv(MET / "val_体积_ID逐样本.csv", index=False, encoding="utf-8-sig")
print("saved val_体积_逐样本坐标敏感性.csv / val_体积_ID逐样本.csv", flush=True)

dev_df = pd.DataFrame(dev_rows)
dev_df.to_csv(MET / "val_体积_尺度偏差.csv", index=False, encoding="utf-8-sig")
print("scale dev: n=%d max_rel=%.3e mean_rel=%.3e" %
      (len(dev_df), dev_df["rel_dev"].max(), dev_df["rel_dev"].mean()), flush=True)

summ = long_df.groupby("method")[["err_field_orig", "err_field_new",
                                  "err_ref_orig", "err_ref_new"]].agg(["mean", "median", "std"])
summ.to_csv(MET / "val_体积_汇总.csv", encoding="utf-8-sig")
print(summ.round(6).to_string(), flush=True)
ids = id_df.groupby("method")[["err_field_orig", "err_field_new"]].agg(["mean", "median", "std"])
ids.to_csv(MET / "val_体积_ID汇总.csv", encoding="utf-8-sig")
print(ids.round(6).to_string(), flush=True)

if anom:
    pd.DataFrame(anom).to_csv(MET / "val_体积_异常清单.csv", index=False, encoding="utf-8-sig")
print(f"[stats] N_flip={n_n_flip}  N_vs_归档不符={n_n_mismatch_arch}  V锚点不符={n_anchor_v_fail}", flush=True)
check(n_anchor_v_fail == 0, f"v27 V 锚点核对（不符 {n_anchor_v_fail}）")
check(n_n_mismatch_arch == 0, f"N_slice vs v27 归档逐样本一致（不符 {n_n_mismatch_arch}）")

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
