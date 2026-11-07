# volume_pipeline.py — N_slice 固定口径补算
# 计算依据：the frozen reference implementation（wellbore_points/
# radial_areas/volume_of 逐式一致）；仅保留 volume_of 第二返回值并导出。
# 不得改动：点云读取与样本筛选、井筒闭合、切片与方位网格、格内 max、周期插值、
# 面积与体积积分、坐标。原基准坐标，不执行 z*0.3048。
# N_slice 定义：预测点云（含井筒闭合点）在 500 层网格上具有有限面积的层数。
# zmin/H/xc/yc 一律取自对应完整点云真值；可积分层 <10 时 V 为 NaN 但 N_slice 仍输出。
# 用法：
#   python volume_pipeline.py --split test --pred_root $SCC_ROOT/validate/test_predictions/recovered \
#       --gt_dir $SCC_ROOT/validate/test_predictions/recovered/gt \
#       --field_dir $SCC_ROOT/dataset/pc_v2/test/gt_field \
#       --methods masked,full,pcn,pointr,cylnn,spline,idw --outdir $SCC_ROOT/validate/test_predictions/metrics \
#       --check_table $SCC_ROOT/validate/test_predictions/metrics/mt_long_table_recalc_orig.csv
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
from pathlib import Path

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2 * np.pi / N_THETA


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test", "val"])
    ap.add_argument("--pred_root", required=True, help="内含 pred_<M>/<i:03d>.npy")
    ap.add_argument("--gt_dir", required=True, help="真值 npy 目录（dataset 链）")
    ap.add_argument("--field_dir", required=True, help="gt_field npy 目录")
    ap.add_argument("--methods", required=True, help="逗号分隔，不得含 ID-1024/ID-8192")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--check_table", default=None, help="可选：原坐标复算表路径，逐样本核对 V_pred")
    a = ap.parse_args()

    # accept either the academic arm names used by the released prediction tree
    # or the legacy short tokens
    ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
             "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
    METHODS = [ALIAS.get(m, m) for m in a.methods.split(",")]
    assert all(m in ("adapointr_masked", "adapointr_full", "pcn", "pointr",
                     "cylnn", "spline", "idw")
               for m in METHODS), METHODS
    assert not any(m.startswith("ID-") for m in METHODS), "ID-1024/ID-8192 为不补全基线，禁止混入"

    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("ENV python=%s numpy=%s pandas=%s" % (sys.version.split()[0], np.__version__, pd.__version__), flush=True)
    print("CODE_SHA256", code_sha, flush=True)
    print("ARGS", vars(a), flush=True)

    dfv = pd.read_csv(f"{_ROOT}/dataset/pc_v2/samples.csv", encoding="utf-8-sig")
    dfv = dfv[dfv["split"].eq(a.split)].reset_index(drop=True)
    assert len(dfv) == 150, f"{a.split} 划分样本数 {len(dfv)} != 150"
    gt_dir = Path(a.gt_dir)
    for i in range(150):
        p = gt_dir / f"{i:03d}.npy"
        assert p.exists(), f"gt 缺失 {p}"

    check = None
    if a.check_table:
        t = pd.read_csv(a.check_table)
        if "arm" in t.columns:
            t = t.rename(columns={"arm": "method"})
        check = {}
        for (mm, ii), g in t.groupby(["method", "i"]):
            check[(mm, ii)] = list(g["V_pred"])

    rows = []
    anom = []
    for i in range(150):
        name = str(dfv.loc[i, "name"])
        gt = np.load(gt_dir / f"{i:03d}.npy").astype(np.float64)
        assert gt.shape == (16384, 3) and np.isfinite(gt).all(), f"gt 异常 i={i}"
        zmin = float(gt[:, 2].min())
        zmax = float(gt[:, 2].max())
        H = zmax - zmin
        xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
        wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)

        gf = np.load(Path(a.field_dir) / f"{name}.npy")
        gf2 = np.where(gf > 0, gf, 0.0)
        V_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
        V_ref, N_ref = volume_of(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)

        for m in METHODS:
            p = Path(a.pred_root) / f"pred_{m}" / f"{i:03d}.npy"
            assert p.exists(), f"预测缺失 {p}"
            pred = np.load(p).astype(np.float64)
            assert pred.shape == (16384, 3) and np.isfinite(pred).all(), f"预测异常 {m} i={i}"
            V_pred, N_slice = volume_of(radial_areas(np.vstack([pred, wb]), xc, yc, zmin, H), zmin, H)
            assert isinstance(N_slice, int) and 0 <= N_slice <= N_SLICES
            if N_slice < 10:
                anom.append(dict(method=m, i=i, cav=name, level=int(dfv.loc[i, "level"]),
                                 N_slice=N_slice, note="N_slice<10, V_pred=NaN"))
            if check is not None:
                if np.isfinite(V_pred):
                    cands = check.get((m, i)) or []
                    assert (not cands) or any(abs(V_pred - c) <= 0.05 + 1e-9 for c in cands), (m, i, V_pred, cands)
            rows.append(dict(split=a.split, method=m, i=i, cav=name,
                             level=int(dfv.loc[i, "level"]), N_slice=N_slice,
                             V_pred=V_pred, V_field=V_field, V_ref=V_ref, N_ref=N_ref))
        if (i + 1) % 25 == 0:
            print("PROGRESS", a.split, i + 1, flush=True)

    df = pd.DataFrame(rows)
    assert len(df) == 150 * len(METHODS)
    assert not df.duplicated(subset=["split", "method", "i"]).any()
    assert df["N_slice"].notna().all() and (df["N_slice"] % 1 == 0).all()
    out = Path(a.outdir)
    df.to_csv(out / f"n_slice_per_sample_{a.split}.csv", index=False, encoding="utf-8-sig")
    print("saved", f"n_slice_per_sample_{a.split}.csv", "rows=", len(df), flush=True)

    s = df.groupby(["split", "method"])["N_slice"].agg(
        n="count", N_slice_mean="mean", N_slice_median="median",
        N_slice_min="min", N_slice_max="max", N_slice_lt10_count=lambda x: int((x < 10).sum()))
    s = s.reset_index()
    s["N_slice_mean_rounded1"] = s["N_slice_mean"].round(1)
    s.to_csv(out / f"n_slice_summary_{a.split}.csv", index=False, encoding="utf-8-sig")
    print(s.round(4).to_string(index=False), flush=True)
    print("V_pred NaN count:", int(df["V_pred"].isna().sum()), flush=True)
    if anom:
        pd.DataFrame(anom).to_csv(out / f"n_slice_anomaly_{a.split}.csv", index=False, encoding="utf-8-sig")
    print("N_SLICE_DONE anomalies=", len(anom), flush=True)


if __name__ == "__main__":
    main()
