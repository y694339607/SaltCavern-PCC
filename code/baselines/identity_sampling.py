# -*- coding: utf-8 -*-
"""identity@1024 输入匹配复算、配对 bootstrap 与有符号误差分析。

1024 点子集严格按 PairedDataset 的归档实现重建：
SeedSequence(20260921, spawn_key=[腔号, 覆盖度, 2]) -> Philox -> permutation(8192)[:1024]。
"""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "source_archive/identity_reanalysis"
OUT = ROOT / "source_archive/identity_input_matched"
RAW = {"validation": BASE / "val_raw.zip", "test": BASE / "test_raw.zip"}
PREVIOUS = BASE / "identity_统一口径逐样本复算.csv"
BOOT = 10000
RNG_SEED = 20261103
MASTER_SEED = 20260921
TAG_DOWN = 2
RW = 0.61
Z_LO, Z_HI = 2.0, 60.0
DZ_STEP = 0.5
N_SLICES = 500
N_THETA = 72
DTH = 2 * np.pi / N_THETA


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_sample_rng(num, level):
    # 与 paired_data.transforms.make_sample_rng 及 missing_synth.make_rng 一致。
    sq = np.random.SeedSequence(MASTER_SEED, spawn_key=[int(num), int(level), TAG_DOWN])
    return np.random.Generator(np.random.Philox(sq))


def sample_indices(cav, level):
    num = int(re.search(r"(\d+)", str(cav)).group(1))
    return make_sample_rng(num, level).permutation(8192)[:1024].astype(np.int64)


def wellbore_points(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZ_STEP)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([
        xc + RW * np.cos(T).ravel(), yc + RW * np.sin(T).ravel(), Z.ravel()
    ]).astype(np.float64)


def grid_coordinates(pts, xc, yc, zmin, H):
    dz = H / N_SLICES
    k = np.clip(((pts[:, 2] - zmin) / dz).astype(int), 0, N_SLICES - 1)
    th = np.mod(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc), 2 * np.pi)
    tb = np.clip((th / DTH).astype(int), 0, N_THETA - 1)
    return k, tb


def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H / N_SLICES
    k, tb = grid_coordinates(pts, xc, yc, zmin, H)
    rr = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
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
        order = np.argsort(ex_i)
        filled = np.interp(idx, ex_i[order], ex_v[order])
        areas[s] = 0.5 * float(np.trapz(filled ** 2, dx=DTH))
    return areas


def volume_of(areas, zmin, H):
    dz = H / N_SLICES
    zc = zmin + (np.arange(N_SLICES) + 0.5) * dz
    m = ~np.isnan(areas)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.trapz(areas[m], zc[m])), int(m.sum())


def cd_l1_m(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pred_to_gt = cKDTree(gt).query(pred, k=1)[0].mean()
    gt_to_pred = cKDTree(pred).query(gt, k=1)[0].mean()
    return float(0.5 * (pred_to_gt + gt_to_pred))


def read_array(zf, name):
    return np.load(io.BytesIO(zf.read(name)))


def recompute_1024(split, previous, index_arrays):
    old = previous[previous["split"] == split].sort_values(["cav", "level"]).reset_index(drop=True)
    assert len(old) == 150 and old["cav"].nunique() == 50
    rows = []
    with zipfile.ZipFile(RAW[split]) as zf:
        for r in old.itertuples(index=False):
            stem = f"{r.cav}_L{int(r.level)}"
            obs = read_array(zf, f"partial/{stem}.npy").astype(np.float32)
            gt = read_array(zf, f"complete/{stem}.npy").astype(np.float64)
            idx = sample_indices(r.cav, r.level)
            obs1024 = obs[idx].astype(np.float64)
            assert obs.shape == (8192, 3) and obs1024.shape == (1024, 3)
            assert len(np.unique(idx)) == 1024 and idx.min() >= 0 and idx.max() < 8192
            # 归一化—反归一化不改变 identity 所用的物理坐标子集。
            index_arrays[f"{split}_{stem}"] = idx
            zmin = float(gt[:, 2].min())
            zmax = float(gt[:, 2].max())
            H = zmax - zmin
            xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
            wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)
            V_id1024, n_slices = volume_of(
                radial_areas(np.vstack([obs1024, wb]), xc, yc, zmin, H), zmin, H
            )
            assert np.isfinite(V_id1024) and V_id1024 > 0, (split, stem, V_id1024, n_slices)
            k, tb = grid_coordinates(obs1024, xc, yc, zmin, H)
            occupied = len(np.unique(k * N_THETA + tb))
            V_field = float(r.V_field)
            rows.append(dict(
                split=split,
                cav=r.cav,
                level=int(r.level),
                V_field=V_field,
                V_id8192=float(r.V_id),
                err8192_abs=float(r.err_field_abs),
                err8192_signed=float(r.err_field_signed),
                n8192_slices=int(r.n_slices),
                cd8192_l1_m=float(r.cd_l1_m),
                V_id1024=V_id1024,
                err1024_abs=abs(V_id1024 - V_field) / V_field,
                err1024_signed=(V_id1024 - V_field) / V_field,
                n1024_slices=n_slices,
                occupied1024_cells=occupied,
                occupied1024_pct=100.0 * occupied / (N_SLICES * N_THETA),
                cd1024_l1_m=cd_l1_m(obs1024, gt),
                BM_V_pred=float(r.BM_V_pred),
                BM_err_abs=float(r.BM_err_field_abs),
                BM_err_signed=float(r.BM_err_field_signed),
                BM_cd_l1_m=float(r.BM_cd_l1_m),
            ))
    out = pd.DataFrame(rows)
    assert len(out) == 150 and out["cav"].nunique() == 50
    assert np.isfinite(out.select_dtypes(include=[np.number])).all().all()
    return out


def cluster_values(s, value):
    # 总体先在腔体内对三个覆盖度取均值；分层时每腔一个样本。
    return s.groupby("cav")[value].mean().to_numpy(dtype=np.float64)


def boot_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(BOOT, len(values)))
    boots = values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prob_negative": float((boots < 0).mean()),
        "interval_crosses_zero": bool(lo <= 0 <= hi),
    }


def paired_bootstrap(data):
    specs = [
        ("identity1024_minus_BM", "err1024_abs", "BM_err_abs"),
        ("identity8192_minus_BM", "err8192_abs", "BM_err_abs"),
        ("identity8192_minus_identity1024", "err8192_abs", "err1024_abs"),
    ]
    rows = []
    for split in ["validation", "test", "pooled_descriptive"]:
        d = data if split == "pooled_descriptive" else data[data["split"] == split]
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            n_cav = s["cav"].nunique()
            assert n_cav == (100 if split == "pooled_descriptive" else 50)
            for comparison, a, b in specs:
                work = s[["cav", a, b]].copy()
                work["diff_pp"] = 100 * (work[a] - work[b])
                rng = np.random.default_rng(RNG_SEED)
                ci = boot_ci(cluster_values(work, "diff_pp"), rng)
                rows.append(dict(
                    split=split,
                    level=level,
                    comparison=comparison,
                    n_cavities=n_cav,
                    n_samples=len(s),
                    left_mean_pct=float(100 * s[a].mean()),
                    right_mean_pct=float(100 * s[b].mean()),
                    diff_pp=float(work["diff_pp"].mean()),
                    bootstrap=BOOT,
                    seed=RNG_SEED,
                    **ci,
                ))
    return pd.DataFrame(rows)


def signed_summary(data):
    methods = [
        ("identity@8192", "err8192_signed"),
        ("identity@1024", "err1024_signed"),
        ("AdaPoinTr masked", "BM_err_signed"),
    ]
    rows = []
    for split in ["validation", "test", "pooled_descriptive"]:
        d = data if split == "pooled_descriptive" else data[data["split"] == split]
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            for method, col in methods:
                signed_pct = 100 * s[col]
                rng = np.random.default_rng(RNG_SEED)
                work = pd.DataFrame({"cav": s["cav"].to_numpy(), "signed_pct": signed_pct.to_numpy()})
                ci = boot_ci(cluster_values(work, "signed_pct"), rng)
                rows.append(dict(
                    split=split,
                    level=level,
                    method=method,
                    n_cavities=s["cav"].nunique(),
                    n_samples=len(s),
                    signed_mean_pct=float(signed_pct.mean()),
                    signed_median_pct=float(signed_pct.median()),
                    overestimated_pct=float(100 * (signed_pct > 0).mean()),
                    ci_low_pct=ci["ci_low"],
                    ci_high_pct=ci["ci_high"],
                    interval_crosses_zero=ci["interval_crosses_zero"],
                ))
    return pd.DataFrame(rows)


def geometry_summary(data):
    methods = [
        ("identity@8192", "cd8192_l1_m", "8192 stored observations versus 16384 complete points"),
        ("identity@1024", "cd1024_l1_m", "model-matched 1024-point subset versus 16384 complete points"),
        ("AdaPoinTr masked", "BM_cd_l1_m", "16384 predicted points versus 16384 complete points"),
    ]
    rows = []
    for split in ["validation", "test"]:
        d = data[data["split"] == split]
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            for method, col, protocol in methods:
                rows.append(dict(
                    split=split,
                    level=level,
                    method=method,
                    n=len(s),
                    cd_mean_m=float(s[col].mean()),
                    cd_median_m=float(s[col].median()),
                    point_protocol=protocol,
                ))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(PREVIOUS)
    index_arrays = {}
    parts = [recompute_1024(split, previous, index_arrays) for split in ["validation", "test"]]
    data = pd.concat(parts, ignore_index=True)
    paired = paired_bootstrap(data)
    signed = signed_summary(data)
    geometry = geometry_summary(data)
    data.to_csv(OUT / "identity8192_identity1024_掩码配置_逐样本.csv", index=False)
    paired.to_csv(OUT / "输入匹配与点数影响_腔体聚类配对bootstrap.csv", index=False)
    signed.to_csv(OUT / "三种配置_有符号误差分层汇总.csv", index=False)
    geometry.to_csv(OUT / "三种配置_几何基线分层汇总.csv", index=False)
    index_path = OUT / "identity@1024_模型输入同源索引.npz"
    np.savez_compressed(index_path, **index_arrays)
    checks = {
        "sampling": {
            "master_seed": MASTER_SEED,
            "spawn_key": ["cavity_number", "level", TAG_DOWN],
            "bit_generator": "Philox",
            "selection": "permutation(8192)[:1024], without replacement",
            "index_count": len(index_arrays),
            "all_indices_unique_per_sample": True,
            "source": "paired-dataset implementation in this repository",
        },
        "bootstrap": {
            "replicates": BOOT,
            "seed": RNG_SEED,
            "unit": "cavity",
            "interval": "95% percentile",
            "pooled_status": "descriptive sensitivity analysis; validation was used for model selection",
        },
        "rows_per_split": 150,
        "cavities_per_split": 50,
        "inputs_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in list(RAW.values()) + [PREVIOUS]
        },
        "outputs_sha256": {},
    }
    for name in [
        "identity8192_identity1024_掩码配置_逐样本.csv",
        "输入匹配与点数影响_腔体聚类配对bootstrap.csv",
        "三种配置_有符号误差分层汇总.csv",
        "三种配置_几何基线分层汇总.csv",
        "identity@1024_模型输入同源索引.npz",
    ]:
        checks["outputs_sha256"][name] = sha256(OUT / name)
    (OUT / "复算摘要.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Overall paired comparisons ===")
    print(paired[paired["level"] == "overall"].to_string(index=False))
    print("\n=== L2 signed errors ===")
    print(signed[(signed["level"] == 2) & (signed["split"] != "pooled_descriptive")].to_string(index=False))
    print("\n=== Overall geometry ===")
    print(geometry[geometry["level"] == "overall"].to_string(index=False))
    print("IDENTITY_1024_INPUT_MATCHED_PASS")


if __name__ == "__main__":
    main()
