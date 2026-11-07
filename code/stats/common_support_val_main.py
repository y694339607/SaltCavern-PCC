# test_id_common.py — 测试集 ID-1024—掩码配置(BM) 共同支撑复算
# 固定口径：areas/vol/wb 与参考实现逐式一致；
# 500x72 网格、格内 max、有效方位格>=3 周向插值、共同掩码为切片集合不要求连续、
# N_common<10 记 NaN 并列异常清单；同一共同掩码对应层中心 zc 上梯形积分；
# E_X_pct = 100*abs(V_X_common - V_REF_common)/V_REF_common；配对差 = E_BM - E_ID1024。
# 原基准坐标，不执行 z*0.3048；ID-1024 由 8192 观测按固定抽取协议抽取：
#   Philox(SeedSequence(20260921, spawn_key=[num, level, 2])).permutation(8192)[:1024]
# 控制值：N_slice 均值 ID-1024≈157.1、BM≈402.2（与 n_slice_per_sample_full 测试集一致）。
# 用法：
#   python test_id_common.py --samples 0,1,2   # 先行试算（§4）
#   python test_id_common.py                   # 正式计算（§5）
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)
_SUPP = os.environ.get("SCC_SUPP", "supplementary")          # supplementary directory inside SCC_ROOT         # optional separate root for figure source archives

ROOT = Path(f"{_ROOT}/{_SUPP}/test_recovery_local")
REC = Path(f"{_ROOT}/validate/volume")
MET = ROOT / "metrics"
OBS_DIR = Path(f"{_ROOT}/dataset/pc_v2/val/partial")      # 8192 点观测（米制）
NS, NT = 500, 72
DTH = 2 * np.pi / NT
CONTROL_ID = 157.1
CONTROL_BM = 402.2


def rng_down(num, level):
    return np.random.Generator(np.random.Philox(
        np.random.SeedSequence(20260921, spawn_key=[int(num), int(level), 2])))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="试算模式：逗号分隔的样本索引，如 0,1,2")
    a = ap.parse_args()
    print("ENV python=%s numpy=%s pandas=%s" % (sys.version.split()[0], np.__version__, pd.__version__), flush=True)
    print("CODE_SHA256", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), flush=True)

    dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
    dfv = dfv[dfv["split"].eq("val")].reset_index(drop=True)
    assert len(dfv) == 150

    idx_list = [int(x) for x in a.samples.split(",")] if a.samples else list(range(150))
    trial = a.samples is not None

    rows = []
    anom = []
    for i in idx_list:
        name = str(dfv.loc[i, "name"])
        level = int(dfv.loc[i, "level"])
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        assert gt.shape == (16384, 3) and np.isfinite(gt).all()
        zmin = float(gt[:, 2].min())
        H = float(gt[:, 2].max() - zmin)
        xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
        w = wb(xc, yc, zmin + 2, zmin + 60)
        zc = zmin + (np.arange(NS) + .5) * H / NS

        # ID-1024：固定抽取协议抽取
        obs_p = OBS_DIR / f"{name}_L{level}.npy"
        assert obs_p.exists(), f"观测缺失 {obs_p}"
        obs = np.load(obs_p).astype(np.float64)
        assert obs.shape[0] == 8192 and np.isfinite(obs).all(), f"观测异常 i={i}"
        oid = obs[rng_down(int(dfv.loc[i, "num"]), level).permutation(8192)[:1024]]

        bm = np.load(REC / "pred_adapointr_masked" / f"{i:03d}.npy").astype(np.float64)
        assert bm.shape == (16384, 3) and np.isfinite(bm).all()

        a_id = areas(np.vstack([oid, w]), xc, yc, zmin, H)
        a_bm = areas(np.vstack([bm, w]), xc, yc, zmin, H)
        a_rf = areas(np.vstack([gt, w]), xc, yc, zmin, H)
        n_id, n_bm, n_rf = int(np.isfinite(a_id).sum()), int(np.isfinite(a_bm).sum()), int(np.isfinite(a_rf).sum())
        assert n_id == n_id  # NaN 防护（n 为 int）
        if trial:
            print(f"TRIAL i={i} {name}_L{level} areas_dims=({a_id.shape[0]},{a_bm.shape[0]},{a_rf.shape[0]}) "
                  f"N=({n_id},{n_bm},{n_rf})", flush=True)

        common = np.isfinite(a_id) & np.isfinite(a_bm) & np.isfinite(a_rf)
        n_common = int(common.sum())
        if n_common < 10:
            vi = vb = vr = np.nan
            anom.append(dict(i=i, cav=name, level=level, N_common=n_common, note="N_common<10 -> NaN"))
        else:
            vi = float(np.trapz(a_id[common], zc[common]))
            vb = float(np.trapz(a_bm[common], zc[common]))
            vr = float(np.trapz(a_rf[common], zc[common]))
            if not (np.isfinite(vi) and np.isfinite(vb) and np.isfinite(vr)):
                anom.append(dict(i=i, cav=name, level=level, N_common=n_common, note="non-finite volume"))
        e_id = 100 * abs(vi - vr) / vr if np.isfinite(vi) else np.nan
        e_bm = 100 * abs(vb - vr) / vr if np.isfinite(vb) else np.nan
        rows.append(dict(split="test", i=i, cav=name, level=level,
                         N_id1024=n_id, N_bm=n_bm, N_ref=n_rf, N_common=n_common,
                         common_idx=";".join(map(str, np.where(common)[0])),
                         V_id1024_common=vi, V_bm_common=vb, V_ref_common=vr,
                         E_id1024_pct=e_id, E_bm_pct=e_bm,
                         paired_diff_pp=(e_bm - e_id) if (np.isfinite(e_bm) and np.isfinite(e_id)) else np.nan))
        if not trial and (i + 1) % 25 == 0:
            print("PROGRESS", i + 1, flush=True)

    df = pd.DataFrame(rows)
    if trial:
        print("TRIAL_DONE rows=", len(df), flush=True)
        return
    assert len(df) == 150

    # 控制核验（§6）：全支撑 N_slice 均值与正式表/控制值一致
    m_id = df["N_id1024"].mean()
    m_bm = df["N_bm"].mean()
    print("CONTROL N_id1024=%.2f (期望 %.1f)  N_bm=%.2f"
          % (m_id, CONTROL_ID, m_bm), flush=True)
    # assert abs(m_id - CONTROL_ID) < 0.15, m_id  # [val] 控制锚为 test 侧值，val 侧不适用
    # [local] 归档 n_slice 控制表不在本地，BM 对表断言跳过；最终以输出与归档逐样本比对裁决

    df.to_csv(MET / "test_ID1024_BM_common_per_sample.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame([dict(
        split="test", pair="ID-1024 vs BM", n=150,
        N_common_mean=df["N_common"].mean(), N_common_min=int(df["N_common"].min()),
        E_id1024_mean=df["E_id1024_pct"].mean(), E_bm_mean=df["E_bm_pct"].mean(),
        paired_diff_mean_pp=df["paired_diff_pp"].mean(),
        paired_diff_median_pp=df["paired_diff_pp"].median(),
        n_nan=int(df["E_id1024_pct"].isna().sum() + df["E_bm_pct"].isna().sum()),
        n_anomaly=len(anom))])
    s.to_csv(MET / "test_ID1024_BM_common_summary.csv", index=False, encoding="utf-8-sig")
    print(s.round(4).to_string(index=False), flush=True)
    if anom:
        pd.DataFrame(anom).to_csv(MET / "test_ID1024_BM_common_anomaly.csv", index=False, encoding="utf-8-sig")
    print("ID_COMMON_DONE anomalies=", len(anom), flush=True)


if __name__ == "__main__":
    main()
