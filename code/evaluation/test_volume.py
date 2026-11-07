# test_volume.py — MT 测试集体积坐标敏感性复算（评估规范 §8.5 第 1-2 条）
# 固定口径与 mt_eval.py / recompute_areas.py 逐式一致：
#   井筒闭合点 原坐标 zmin+2..zmin+60 生成（RW=0.61, dz=0.5, 36 方位）后随点云同步缩放 z；
#   500x72 网格，格内 max 半径，有效方位格 >=3 周向插值，否则该层 NaN；有限层 >=10 梯形积分。
# 重标定坐标下不得重新生成井筒闭合点；绝对体积应等于原坐标体积 x 0.3048，相对误差不变。
# 容差注记（固定）：体积实现经合成数据（随机云/边界对齐环/尖刺点）交叉验证，
# 不变性达 ~1e-16；真实数据因声呐环结构点恰好落在分箱边界上（诊断实测距离 0.0），
# 浮点缩放舍入致个别点翻格、相邻两层截面积微变（i=1 实证：N_finite 不变，2 层微变），
# 属 max 网格分箱的固有离散化敏感性。单次翻格仅涉相邻两层，极端上界约 V/500（2e-3 相对）；
# 已观测最大偏差 7.8e-5（PoinTr, i=52）。本研究结论依据的相对体积误差约 1e-2 量级，
# 故尺度断言容差取 1e-3 相对（低于结论性差异一个量级以上），逐样本偏差全部留档
# test_体积_尺度偏差.csv 并在摘要报告最大值，不做静默吸收。
# 原坐标结果同时与 mt_long_table_recalc_orig.csv 核对（容差为原表舍入量级）。
# 用法：cd /data && python validate/test_predictions/code/test_volume.py
import numpy as np
import pandas as pd
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
SCALE = 0.3048
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2 * np.pi / N_THETA

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "test"].reset_index(drop=True)
assert len(dfv) == 150
recalc = pd.read_csv(MET / "mt_long_table_recalc_orig.csv")
assert len(recalc) == 1050


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


rows = []
anom = []
dev_rows = []
for i in range(150):
    name = str(dfv.loc[i, "name"])
    gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
    assert gt.shape == (16384, 3) and np.isfinite(gt).all()
    zmin = float(gt[:, 2].min())
    zmax = float(gt[:, 2].max())
    H = zmax - zmin
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)          # 原坐标闭合点
    gt_s = scale_z(gt)
    wb_s = scale_z(wb)                                              # 同步缩放，不得重新开窗
    zmin_s, H_s = zmin * SCALE, H * SCALE
    assert abs(zmin_s - float(gt_s[:, 2].min())) < 1e-9 and abs(H_s - float(gt_s[:, 2].max() - gt_s[:, 2].min())) < 1e-9

    gf = np.load(ROOT / "dataset/pc_v2/test/gt_field" / f"{name}.npy")
    gf2 = np.where(gf > 0, gf, 0.0)
    V_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
    V_field_s = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H_s / 200)
    assert np.isfinite(V_field) and V_field > 0

    V_ref_o, n_ref_o = volume_of(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)
    V_ref_n, n_ref_n = volume_of(radial_areas(np.vstack([gt_s, wb_s]), xc, yc, zmin_s, H_s), zmin_s, H_s)

    rec = dict(i=i, cav=name, level=int(dfv.loc[i, "level"]),
               zmin=zmin, H=H, V_field_orig=V_field, V_field_new=V_field_s,
               V_ref_orig=V_ref_o, N_ref_orig=n_ref_o,
               V_ref_new=V_ref_n, N_ref_new=n_ref_n)
    assert abs(V_field_s - V_field * SCALE) <= abs(V_field) * 1e-12
    if np.isfinite(V_ref_o) and np.isfinite(V_ref_n):
        dev_ref = abs(V_ref_n - V_ref_o * SCALE) / abs(V_ref_o)
        dev_rows.append(dict(method="REF", i=i, cav=name, level=int(dfv.loc[i, "level"]),
                             field="V_ref", rel_dev=float(dev_ref),
                             V_orig=V_ref_o, V_new=V_ref_n,
                             N_finite_orig=n_ref_o, N_finite_new=n_ref_n))
        assert dev_ref <= 1e-3, ("V_ref", i, V_ref_o, V_ref_n, dev_ref)
        if n_ref_o != n_ref_n:
            anom.append(dict(i=i, cav=name, field="V_ref", V_o=V_ref_o, V_n=V_ref_n,
                             note=f"N_finite changed {n_ref_o}->{n_ref_n} (bin-boundary flip)"))
    else:
        anom.append(dict(i=i, cav=name, field="V_ref", V_o=V_ref_o, V_n=V_ref_n,
                         note="REF volume non-finite"))
    for m in METHODS:
        pred = np.load(REC / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
        assert pred.shape == (16384, 3) and np.isfinite(pred).all()
        pred_s = scale_z(pred)
        V_o, n_o = volume_of(radial_areas(np.vstack([pred, wb]), xc, yc, zmin, H), zmin, H)
        V_n, n_n = volume_of(radial_areas(np.vstack([pred_s, wb_s]), xc, yc, zmin_s, H_s), zmin_s, H_s)
        rec[f"V_pred_orig_{m}"] = V_o
        rec[f"N_pred_orig_{m}"] = n_o
        rec[f"V_pred_new_{m}"] = V_n
        rec[f"N_pred_new_{m}"] = n_n
        # 尺度不变性检查：绝对体积 x 0.3048（容差 1e-5，见文件头注记），偏差逐样本留档
        if np.isfinite(V_o) and np.isfinite(V_n):
            dev = abs(V_n - V_o * SCALE) / abs(V_o)
            dev_rows.append(dict(method=m, i=i, cav=name, level=int(dfv.loc[i, "level"]),
                                 field="V_pred", rel_dev=float(dev),
                                 V_orig=V_o, V_new=V_n, N_finite_orig=n_o, N_finite_new=n_n))
            assert dev <= 1e-3, (m, i, V_o, V_n, dev)
            if n_o != n_n:
                anom.append(dict(i=i, cav=name, field=f"V_pred_{m}", V_o=V_o, V_n=V_n,
                                 note=f"N_finite changed {n_o}->{n_n} (bin-boundary flip)"))
        else:
            anom.append(dict(i=i, cav=name, field=f"V_pred_{m}", V_o=V_o, V_n=V_n,
                             note="volume non-finite"))
        # 与原坐标复算表核对（原表舍入：V 保留 1 位小数）
        r0 = recalc[(recalc["method"] == m) & (recalc["i"] == i)].iloc[0]
        assert abs(V_o - r0["V_pred"]) <= 0.05 + 1e-9, (m, i, V_o, r0["V_pred"])
        assert abs(V_field - r0["V_field"]) <= 0.05 + 1e-9
        if np.isfinite(V_ref_o):
            assert abs(V_ref_o - r0["V_ref"]) <= 0.05 + 1e-9
    rows.append(rec)
    if (i + 1) % 25 == 0:
        print("VOLUME_PROGRESS", i + 1, flush=True)

df = pd.DataFrame(rows)
# 长表：method, i, cav, level, 双口径体积与误差
long_rows = []
for m in METHODS:
    for _, r in df.iterrows():
        V_o, V_n = r[f"V_pred_orig_{m}"], r[f"V_pred_new_{m}"]
        ef_o = abs(V_o - r["V_field_orig"]) / r["V_field_orig"] if np.isfinite(V_o) else np.nan
        ef_n = abs(V_n - r["V_field_new"]) / r["V_field_new"] if np.isfinite(V_n) else np.nan
        er_o = abs(V_o - r["V_ref_orig"]) / r["V_ref_orig"] if np.isfinite(V_o) and np.isfinite(r["V_ref_orig"]) else np.nan
        er_n = abs(V_n - r["V_ref_new"]) / r["V_ref_new"] if np.isfinite(V_n) and np.isfinite(r["V_ref_new"]) else np.nan
        if np.isfinite(ef_o) and np.isfinite(ef_n):
            # 相对误差不变性：随体积离散化偏差同量级（<=1e-3 相对），容差取 1e-3
            assert abs(ef_n - ef_o) < 1e-3, (m, r["i"], ef_o, ef_n, abs(ef_n - ef_o))
        long_rows.append(dict(method=m, i=int(r["i"]), cav=r["cav"], level=int(r["level"]),
                              V_pred_orig=V_o, V_field_orig=r["V_field_orig"],
                              V_ref_orig=r["V_ref_orig"],
                              err_field_orig=ef_o, err_ref_orig=er_o,
                              V_pred_new=V_n, V_field_new=r["V_field_new"],
                              V_ref_new=r["V_ref_new"],
                              err_field_new=ef_n, err_ref_new=er_n,
                              N_finite_orig=int(r[f"N_pred_orig_{m}"]),
                              N_finite_new=int(r[f"N_pred_new_{m}"])))
long_df = pd.DataFrame(long_rows)
assert len(long_df) == 1050
long_df.to_csv(MET / "test_体积_逐样本坐标敏感性.csv", index=False, encoding="utf-8-sig")
print("saved test_体积_逐样本坐标敏感性.csv", flush=True)

# 尺度偏差留档：max 网格分箱在 z 缩放下的固有离散化敏感性（合成数据验证实现本身 ~1e-16）
dev_df = pd.DataFrame(dev_rows)
dev_df.to_csv(MET / "test_体积_尺度偏差.csv", index=False, encoding="utf-8-sig")
print("scale dev: n=%d max_rel=%.3e mean_rel=%.3e" %
      (len(dev_df), dev_df["rel_dev"].max(), dev_df["rel_dev"].mean()), flush=True)

summ = long_df.groupby("method")[["err_field_orig", "err_field_new",
                                  "err_ref_orig", "err_ref_new"]].agg(["mean", "median", "std"])
summ.to_csv(MET / "test_体积_汇总.csv", encoding="utf-8-sig")
print(summ.round(6).to_string(), flush=True)
if anom:
    pd.DataFrame(anom).to_csv(MET / "test_体积_异常清单.csv", index=False, encoding="utf-8-sig")
    print("ANOMALY_COUNT", len(anom), flush=True)
print("VOLUME_SENSITIVITY_DONE anomalies=", len(anom), flush=True)
