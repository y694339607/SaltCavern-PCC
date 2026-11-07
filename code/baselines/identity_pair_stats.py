# identity 与 AdaPoinTr 掩码配置的统一口径复算
# 参考体积、封闭积分和物理米制 CD-L1 均按主评估管线计算。
import os
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_ROOT = os.environ.get("SCC_ROOT", "/data")
_SUPP = os.environ.get("SCC_SUPP", "supplementary")
ROOT = Path(_ROOT)
OUT = ROOT / _SUPP / "identity_reanalysis"
RAW = {"validation": OUT / "val_raw.zip", "test": OUT / "test_raw.zip"}
MAIN = {
    "validation": ROOT / _SUPP / "long_table_validation.csv",
    "test": ROOT / _SUPP / "long_table_test.csv",
}
SOURCE_IDENTITY = {
    "validation": ROOT / _SUPP / "identity_validation_150.csv",
    "test": ROOT / _SUPP / "identity_test_150.csv",
}
BOOT = 10000
RNG_SEED = 20261103
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


def wellbore_points(xc, yc, z0, z1):
    zs = np.arange(z0, z1, DZ_STEP)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    T, Z = np.meshgrid(th, zs)
    return np.column_stack([
        xc + RW * np.cos(T).ravel(),
        yc + RW * np.sin(T).ravel(),
        Z.ravel(),
    ]).astype(np.float64)


def radial_areas(pts, xc, yc, zmin, H):
    pts = pts.astype(np.float64)
    dz = H / N_SLICES
    k = np.clip(((pts[:, 2] - zmin) / dz).astype(int), 0, N_SLICES - 1)
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


def load_main(split):
    d = pd.read_csv(MAIN[split])
    if split == "validation":
        d = d[(d["pair"] == "HIST") & (d["arm"] == "BM")].copy()
    else:
        d = d[d["method"] == "BM"].copy()
    assert len(d) == 150 and d["cav"].nunique() == 50
    return d.sort_values(["cav", "level"]).reset_index(drop=True)


def read_array(zf, name):
    return np.load(io.BytesIO(zf.read(name))).astype(np.float64)


def recompute_split(split):
    main = load_main(split)
    source = pd.read_csv(SOURCE_IDENTITY[split]).sort_values(["cav", "level"]).reset_index(drop=True)
    assert main[["cav", "level"]].equals(source[["cav", "level"]])
    rows = []
    with zipfile.ZipFile(RAW[split]) as zf:
        for r in main.itertuples(index=False):
            stem = f"{r.cav}_L{int(r.level)}"
            obs = read_array(zf, f"partial/{stem}.npy")
            gt = read_array(zf, f"complete/{stem}.npy")
            # 主评估脚本对半径场保留 float32 运算；此处保持相同精度以复现其参考体积。
            gf = np.load(io.BytesIO(zf.read(f"gt_field/{r.cav}.npy")))
            assert obs.shape == (8192, 3) and gt.shape == (16384, 3)
            assert gf.shape == (201, 360) and gf.dtype == np.float32
            zmin = float(gt[:, 2].min())
            zmax = float(gt[:, 2].max())
            H = zmax - zmin
            xc, yc = float(gt[:, 0].mean()), float(gt[:, 1].mean())
            wb = wellbore_points(xc, yc, zmin + Z_LO, zmin + Z_HI)
            V_id, n_slices = volume_of(
                radial_areas(np.vstack([obs, wb]), xc, yc, zmin, H), zmin, H
            )
            gf2 = np.where(gf > 0, gf, 0.0)
            # 显式提升 pi 为 float64，复现主评估原环境的 NumPy 标量提升行为。
            V_field = float(np.float64(np.pi) * np.mean(gf2 ** 2, axis=1).sum() * H / 200)
            src = source[(source["cav"] == r.cav) & (source["level"] == r.level)].iloc[0]
            rows.append(dict(
                split=split,
                cav=r.cav,
                level=int(r.level),
                V_id=V_id,
                V_field=V_field,
                err_field_abs=abs(V_id - V_field) / V_field,
                err_field_signed=(V_id - V_field) / V_field,
                n_slices=n_slices,
                cd_l1_m=cd_l1_m(obs, gt),
                n_obs=len(obs),
                n_gt=len(gt),
                source_V_id=float(src["V_id"]),
                source_V_field=float(src["V_field"]),
                source_n_slices=int(src["n_slices"]),
                main_V_field=float(r.V_field),
                BM_V_pred=float(r.V_pred),
                BM_err_field_abs=float(r.err_field_abs),
                BM_err_field_signed=(float(r.V_pred) - V_field) / V_field,
                BM_cd_l1_m=float(r.cd_l1_m),
            ))
    out = pd.DataFrame(rows)
    assert len(out) == 150 and out["cav"].nunique() == 50
    assert np.max(np.abs(out["V_id"] - out["source_V_id"])) < 0.01
    field_delta = np.abs(out["V_field"] - out["main_V_field"])
    assert field_delta.max() < 0.051, (split, field_delta.max(), out.loc[field_delta.idxmax(), ["cav", "level", "V_field", "main_V_field"]].to_dict())
    assert (out["n_slices"] == out["source_n_slices"]).all()
    assert np.max(np.abs(out["err_field_abs"] - abs(out["V_id"] - out["V_field"]) / out["V_field"])) < 1e-12
    return out


def boot_ci(diff, rng):
    diff = np.asarray(diff, dtype=np.float64)
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(
        ci_low_pp=float(lo),
        ci_high_pp=float(hi),
        prob_diff_negative=float((boots < 0).mean()),
        interval_crosses_zero=bool(lo <= 0 <= hi),
    )


def paired_bootstrap(all_data):
    rows = []
    for split in ["validation", "test"]:
        d = all_data[all_data["split"] == split].copy()
        d["diff_pp"] = 100 * (d["err_field_abs"] - d["BM_err_field_abs"])
        rng = np.random.default_rng(RNG_SEED)
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            if level == "overall":
                cav = s.groupby("cav")[["err_field_abs", "BM_err_field_abs", "diff_pp"]].mean()
                diff = cav["diff_pp"].to_numpy()
                n_samples = len(s)
            else:
                cav = s.set_index("cav")
                diff = cav["diff_pp"].to_numpy()
                n_samples = len(s)
            assert len(diff) == 50
            row = dict(
                split=split,
                level=level,
                n_cavities=50,
                n_samples=n_samples,
                identity_mean_pct=float(100 * s["err_field_abs"].mean()),
                BM_mean_pct=float(100 * s["BM_err_field_abs"].mean()),
                diff_identity_minus_BM_pp=float(s["diff_pp"].mean()),
                bootstrap=BOOT,
                seed=RNG_SEED,
            )
            row.update(boot_ci(diff, rng))
            rows.append(row)
    return pd.DataFrame(rows)


def signed_summary(all_data):
    rows = []
    for split in ["validation", "test"]:
        d = all_data[all_data["split"] == split]
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            for method, prefix in [("identity", ""), ("AdaPoinTr masked", "BM_")]:
                a = s[f"{prefix}err_field_abs"]
                g = s[f"{prefix}err_field_signed"]
                rows.append(dict(
                    split=split,
                    level=level,
                    method=method,
                    n=len(s),
                    abs_mean_pct=float(100 * a.mean()),
                    signed_mean_pct=float(100 * g.mean()),
                    signed_median_pct=float(100 * g.median()),
                    overestimated_pct=float(100 * (g > 0).mean()),
                ))
    return pd.DataFrame(rows)


def geometry_summary(all_data):
    rows = []
    for split in ["validation", "test"]:
        d = all_data[all_data["split"] == split]
        for level in ["overall", 0, 1, 2]:
            s = d if level == "overall" else d[d["level"] == level]
            for method, col, note in [
                ("identity stored observation", "cd_l1_m", "8192 stored observation points versus 16384 complete points"),
                ("AdaPoinTr masked", "BM_cd_l1_m", "16384 predicted points versus 16384 complete points"),
            ]:
                rows.append(dict(
                    split=split,
                    level=level,
                    method=method,
                    n=len(s),
                    cd_mean_m=float(s[col].mean()),
                    cd_median_m=float(s[col].median()),
                    point_protocol=note,
                ))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for split in ["validation", "test"]:
        d = recompute_split(split)
        parts.append(d)
        print(split, "identity aligned V_field mean %", 100 * d["err_field_abs"].mean(),
              "BM mean %", 100 * d["BM_err_field_abs"].mean(),
              "identity CD mean m", d["cd_l1_m"].mean(), flush=True)
    all_data = pd.concat(parts, ignore_index=True)
    paired = all_data[[
        "split", "cav", "level", "V_id", "V_field", "err_field_abs", "err_field_signed",
        "BM_V_pred", "BM_err_field_abs", "BM_err_field_signed", "cd_l1_m", "BM_cd_l1_m",
    ]].copy()
    paired["diff_identity_minus_BM_pp"] = 100 * (paired["err_field_abs"] - paired["BM_err_field_abs"])
    boot = paired_bootstrap(all_data)
    signed = signed_summary(all_data)
    geometry = geometry_summary(all_data)
    all_data.to_csv(OUT / "identity_统一口径逐样本复算.csv", index=False)
    paired.to_csv(OUT / "identity与掩码配置_配对逐样本.csv", index=False)
    boot.to_csv(OUT / "identity与掩码配置_腔体聚类配对bootstrap.csv", index=False)
    signed.to_csv(OUT / "identity与掩码配置_有符号库容误差汇总.csv", index=False)
    geometry.to_csv(OUT / "identity几何基线_汇总.csv", index=False)
    summary = {
        "protocol": {
            "volume_reference": "per-sample independent simulated radius-field volume used by the main evaluation",
            "identity_prediction": "8192 stored observation points plus the same wellbore closure points",
            "geometry": "symmetric physical-metre CD-L1; 8192 stored observations versus 16384 complete points",
            "bootstrap": "50-cavity cluster resampling; all three levels retained; 10000 replicates; percentile 95% interval",
            "seed": RNG_SEED,
            "difference_direction": "identity minus AdaPoinTr masked configuration, in percentage points",
        },
        "validation_checks": {
            "max_abs_V_id_minus_source_m3": float(np.max(np.abs(all_data["V_id"] - all_data["source_V_id"]))),
            "max_abs_V_field_minus_main_m3": float(np.max(np.abs(all_data["V_field"] - all_data["main_V_field"]))),
            "all_n_slices_match_source": bool((all_data["n_slices"] == all_data["source_n_slices"]).all()),
            "rows_per_split": 150,
            "cavities_per_split": 50,
        },
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in list(RAW.values()) + list(MAIN.values()) + list(SOURCE_IDENTITY.values())},
        "paired_bootstrap": boot.to_dict(orient="records"),
    }
    (OUT / "复算摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(boot.to_string(index=False), flush=True)
    print("IDENTITY_REANALYSIS_PASS", flush=True)


if __name__ == "__main__":
    main()
