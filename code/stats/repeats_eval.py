# Evaluate the two supervision arms of the paired-repeat experiment.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
VOL = ROOT / "validate/volume"
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2 * np.pi / N_THETA

ARMS = [
    dict(tag="MS_P1_BM", pair="P1", arm="BM", init_seed=20260921, order_seed=20261101,
         ckpt="validate/pilot/ckpt/MS_P1_BM_i20260921_o20261101_best.pt"),
    dict(tag="MS_P1_E2", pair="P1", arm="E2", init_seed=20260921, order_seed=20261101,
         ckpt="validate/pilot/ckpt/MS_P1_E2_i20260921_o20261101_best.pt"),
    dict(tag="MS_P2_BM", pair="P2", arm="BM", init_seed=20260921, order_seed=20261102,
         ckpt="validate/pilot/ckpt/MS_P2_BM_i20260921_o20261102_best.pt"),
    dict(tag="MS_P2_E2", pair="P2", arm="E2", init_seed=20260921, order_seed=20261102,
         ckpt="validate/pilot/ckpt/MS_P2_E2_i20260921_o20261102_best.pt"),
]
HIST = [  # 历史参照（可选）：须已有 pred_adapointr_full / pred_adapointr_masked 导出存档
    dict(tag="E2", pair="HIST", arm="E2", init_seed=20260921, order_seed=20260921,
         ckpt="validate/pilot/ckpt/full_region_main_e169.pt", dirname="pred_adapointr_full"),
    dict(tag="BM", pair="HIST", arm="BM", init_seed=20260921, order_seed=20260921,
         ckpt="validate/pilot/ckpt/masked_main_e179.pt", dirname="pred_adapointr_masked"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--with-historical", action="store_true")
ap.add_argument("--out", default=str(VOL / "ms_long_table.csv"))
a = ap.parse_args()

dfv = pd.read_csv(ROOT / "dataset/pc_v2/samples.csv", encoding="utf-8-sig")
dfv = dfv[dfv["split"] == "val"].reset_index(drop=True)
assert len(dfv) == 150


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cd_l1_m(P, G):
    P = torch.as_tensor(P, dtype=torch.float64, device=device)
    G = torch.as_tensor(G, dtype=torch.float64, device=device)

    def mean_min(A, B):
        s = 0.0
        for c in range(0, A.shape[0], 2048):
            d = torch.cdist(A[c:c + 2048], B)
            s += float(d.min(dim=1).values.sum().item())
        return s / A.shape[0]

    return 0.5 * (mean_min(P, G) + mean_min(G, P))


# 每样本的 gt 几何量与 V_ref、V_field（两臂共用，只算一次）
meta = {}
for i in range(150):
    gt = np.load(VOL / "gt" / f"{i:03d}.npy").astype(np.float64)
    zmin = float(gt[:, 2].min())
    zmax = float(gt[:, 2].max())
    H = zmax - zmin
    xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
    wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)
    V_ref, nsl = volume_of(radial_areas(np.vstack([gt, wb]), xc, yc, zmin, H), zmin, H)
    name = str(dfv.loc[i, "name"])
    gf = np.load(ROOT / "dataset/pc_v2/val/gt_field" / f"{name}.npy")
    gf2 = np.where(gf > 0, gf, 0.0)
    V_field = float(np.pi * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
    meta[i] = dict(gt=gt, xc=xc, yc=yc, zmin=zmin, H=H, wb=wb, V_ref=V_ref, V_field=V_field)

arms = list(ARMS)
if a.with_historical:
    for h in HIST:
        if (VOL / h["dirname"]).exists():
            h = dict(h)
            h["tag"] = "HIST_" + h["tag"]
            arms.append(h)
        else:
            print("HIST_SKIP 缺少导出目录:", h["dirname"], flush=True)

rows = []
for spec in arms:
    ck = torch.load(str(ROOT / spec["ckpt"]), map_location="cpu", weights_only=False)
    best_epoch = int(ck.get("epoch", -1))
    pdir = VOL / (spec.get("dirname") or f"pred_{spec['tag']}")
    n_files = len(list(pdir.glob("*.npy")))
    assert n_files == 150, f"{spec['tag']} 导出件数 {n_files} != 150"
    for i in range(150):
        pred = np.load(pdir / f"{i:03d}.npy").astype(np.float64)
        assert pred.shape == (16384, 3) and np.isfinite(pred).all()
        m = meta[i]
        cd = cd_l1_m(pred, m["gt"])
        cloud = np.vstack([pred, m["wb"]])
        V_pred, _ = volume_of(radial_areas(cloud, m["xc"], m["yc"], m["zmin"], m["H"]),
                              m["zmin"], m["H"])
        rows.append(dict(
            pair=spec["pair"], arm=spec["arm"], init_seed=spec["init_seed"],
            order_seed=spec["order_seed"], i=i, cav=str(dfv.loc[i, "name"]),
            level=int(dfv.loc[i, "level"]), cd_l1_m=round(cd, 5),
            V_pred=round(V_pred, 1), V_field=round(m["V_field"], 1),
            V_ref=round(m["V_ref"], 1),
            err_field_abs=round(abs(V_pred - m["V_field"]) / m["V_field"], 6),
            err_ref_abs=round(abs(V_pred - m["V_ref"]) / m["V_ref"], 6),
            best_epoch=best_epoch))
    print(f"EVAL_DONE {spec['tag']} best_epoch={best_epoch}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
calib = df[df.pair != "HIST"].drop_duplicates("i")
ratio = calib["V_ref"] / calib["V_field"]
warn = int(((ratio < 0.95) | (ratio > 1.05)).sum())
print(f"rows={len(df)} saved={a.out}", flush=True)
print(f"V_ref/V_field out-of-band samples: {warn}/150（判读：>0 须停报核查）", flush=True)
print("REPEATS_EVAL_PASS", flush=True)
