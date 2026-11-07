# dataset_precheck.py — P1 前置校验（执行方案_P1 §3.2；只读输入，不改写任何原始数据）
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

SCALE = 0.3048
ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
ZVAL = WORK / "raw/val_from_zip"
ZTEST = WORK / "raw/test_from_zip"
SAMPLES = ROOT / "dataset/pc_v2/samples.csv"
TEST_FIN = ROOT / "validate/test_final"
VAL_PRED = {
    "BM": ROOT / "validate/volume/pred_adapointr_masked",
    "E2": ROOT / "validate/volume/pred_adapointr_full",
    "PoinTr": ROOT / "validate/volume/pred_GPoinTr",
    "B1": ROOT / "validate/baseline/B1",
    "B2": ROOT / "validate/baseline/B2",
    "B3": ROOT / "validate/baseline/B3",
    "PCN": ROOT / "validate/baseline_dl/pcn/pred",
}
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM", "E2", "PCN", "PoinTr", "B1", "B2", "B3"]]
OUT = WORK / "precheck"
(OUT / "sha").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T3_VAL_MEAN = {"B1": 1.509, "BM": 1.399, "E2": 1.459, "PCN": 1.932,
               "B3": 1.858, "PoinTr": 2.134, "B2": 2.411}
VAL_BM_MAINCD = {0: 1.76163, 1: 1.63431, 2: 1.62878}

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        fails.append(msg)

def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

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

samples = pd.read_csv(SAMPLES, encoding="utf-8-sig")
val_rows = samples[samples["split"] == "val"].reset_index(drop=True)
test_rows = samples[samples["split"] == "test"].reset_index(drop=True)
check(len(val_rows) == 150 and len(test_rows) == 150, "samples.csv 两划分各 150 行")
val_stems = [f"{r['name']}_L{int(r['level'])}" for _, r in val_rows.iterrows()]
test_stems = [f"{r['name']}_L{int(r['level'])}" for _, r in test_rows.iterrows()]

def count_shape(d, pat):
    fs = sorted(Path(d).glob(pat))
    return len(fs), (np.load(fs[0]).shape if fs else None)

n, s = count_shape(ZVAL / "partial", "*.npy")
check(n == 150 and s == (8192, 3), f"zip val partial: {n} 件, 首件 {s}")
n, s = count_shape(ZVAL / "complete", "*.npy")
check(n == 150 and s == (16384, 3), f"zip val complete: {n} 件, 首件 {s}")
n, s = count_shape(ZVAL / "gt_field", "*.npy")
check(n == 50, f"zip val gt_field: {n} 件")
n, s = count_shape(ZTEST / "partial", "*.npy")
check(n == 150 and s == (8192, 3), f"zip test partial: {n} 件, 首件 {s}")
n, s = count_shape(ZTEST / "complete", "*.npy")
check(n == 150 and s == (16384, 3), f"zip test complete: {n} 件, 首件 {s}")
n, s = count_shape(ZTEST / "gt_field", "*.npy")
check(n == 50, f"zip test gt_field: {n} 件")
for m in METHODS:
    n, s = count_shape(VAL_PRED[m], "*.npy")
    check(n == 150 and s == (16384, 3), f"val pred {m}: {n} 件, 首件 {s}")
    n, s = count_shape(TEST_FIN / f"pred_{m}", "*.npy")
    check(n == 150 and s == (16384, 3), f"test pred {m}: {n} 件, 首件 {s}")
n, s = count_shape(TEST_FIN / "gt", "*.npy")
check(n == 150 and s == (16384, 3), f"test gt: {n} 件, 首件 {s}")

miss_val = [st for st in val_stems if not (ZVAL / "partial" / f"{st}.npy").exists()
            or not (ZVAL / "complete" / f"{st}.npy").exists()]
check(not miss_val, f"zip val 命名完整（缺失 {len(miss_val)}）")
miss_test = [st for st in test_stems if not (ZTEST / "partial" / f"{st}.npy").exists()
             or not (ZTEST / "complete" / f"{st}.npy").exists()]
check(not miss_test, f"zip test 命名完整（缺失 {len(miss_test)}）")

def span_stats(zdir, stems):
    ratios, zspans = [], []
    for st in stems:
        a = np.load(Path(zdir) / f"{st}.npy").astype(np.float64)
        xs, ys, zs = np.ptp(a, axis=0)
        ratios.append(zs / (0.5 * (xs + ys)))
        zspans.append(zs)
    return np.array(ratios), np.array(zspans)

for tag, zdir, stems in [("val", ZVAL / "complete", val_stems),
                         ("test", ZTEST / "complete", test_stems)]:
    r, z = span_stats(zdir, stems)
    zsc = z * SCALE
    print(f"[range {tag}] z/xy 比值 min/med/max = {r.min():.3f}/{np.median(r):.3f}/{r.max():.3f}; "
          f"缩放后 z 跨度 min/max = {zsc.min():.1f}/{zsc.max():.1f} m", flush=True)
    check(3.1 <= np.median(r) <= 3.5, f"{tag} z/xy 比值中位 ≈3.28（实测 {np.median(r):.3f}）")
    check(zsc.min() >= 300 and zsc.max() <= 600, f"{tag} 缩放后 z 跨度全部落入 300–600 m")

mt = pd.read_csv(ROOT / "validate/test_predictions/metrics/mt_long_table_recalc_orig.csv")
bad = []
for m in METHODS:
    for i in range(3):
        pred = np.load(TEST_FIN / f"pred_{m}" / f"{i:03d}.npy").astype(np.float64)
        gt = np.load(TEST_FIN / "gt" / f"{i:03d}.npy").astype(np.float64)
        c = cd_l1_m(pred, gt)
        ref = float(mt[(mt.method == m) & (mt.i == i)]["cd_l1_m"].iloc[0])
        if abs(c - ref) > 1e-3:
            bad.append((m, i, round(c, 5), ref))
check(not bad, f"test CD1 三样本回归（七方法×3，tol 1e-3），异常 {len(bad)} 项 {bad[:3]}")

mism = sum(1 for i, st in enumerate(test_stems)
           if sha16(TEST_FIN / "gt" / f"{i:03d}.npy") != sha16(ZTEST / "complete" / f"{st}.npy"))
check(mism == 0, f"test gt 双源（test_final vs zip）指纹一致（不符 {mism}/150）")

val_pred_cache = {}
for m in METHODS:
    cds = []
    for i, st in enumerate(val_stems):
        pred = np.load(VAL_PRED[m] / f"{i:03d}.npy").astype(np.float64)
        gt = np.load(ZVAL / "complete" / f"{st}.npy").astype(np.float64)
        cds.append(cd_l1_m(pred, gt))
    val_pred_cache[m] = np.array(cds)
    mean = float(np.mean(cds))
    ref = T3_VAL_MEAN[m]
    tol = 3e-3 if m == "PCN" else 1e-3
    ok = abs(mean - ref) <= tol
    check(ok, f"val {m} CD1 均值 {mean:.4f} vs 表3 {ref:.3f}（tol {tol:g}，差 {mean - ref:+.4f}）"
             + (" [PCN 既定基线目录，来源限制已注]" if m == "PCN" else ""))

for i in range(3):
    c = val_pred_cache["BM"][i]
    ref = VAL_BM_MAINCD[i]
    check(abs(c - ref) <= 1e-3, f"val BM i={i} CD1 {c:.5f} vs main_cd {ref:.5f}")

def write_manifest(path_list, out):
    with open(out, "w") as f:
        for p in sorted(path_list):
            f.write(f"{sha16(p)}  {p.name}\n")

write_manifest(list((ZVAL / 'partial').glob('*.npy')) + list((ZVAL / 'complete').glob('*.npy'))
               + list((ZVAL / 'gt_field').glob('*.npy')), OUT / "sha/val_zip.txt")
write_manifest(list((ZTEST / 'partial').glob('*.npy')) + list((ZTEST / 'complete').glob('*.npy'))
               + list((ZTEST / 'gt_field').glob('*.npy')), OUT / "sha/test_zip.txt")
for m in METHODS:
    write_manifest(list(VAL_PRED[m].glob('*.npy')), OUT / f"sha/val_pred_{m}.txt")
    write_manifest(list((TEST_FIN / f'pred_{m}').glob('*.npy')), OUT / f"sha/test_pred_{m}.txt")
write_manifest(list((TEST_FIN / 'gt').glob('*.npy')), OUT / "sha/test_gt.txt")

def parse_sha_file(p):
    d = {}
    for ln in Path(p).read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            d[parts[-1]] = parts[0][:16]
    return d

for m in METHODS:
    sf = TEST_FIN / f"pred_{m}" / f"pred_{m}_sha256.txt"
    if not sf.exists():
        print(f"[sha] pred_{m} 无既有清单，跳过交叉核对", flush=True)
        continue
    old = parse_sha_file(sf)
    new = parse_sha_file(OUT / f"sha/test_pred_{m}.txt")
    mm = [k for k in old if k in new and old[k] != new[k]]
    check(not mm, f"test pred_{m} 与归档 SHA 清单一致（不符 {len(mm)}）")

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails:
    print("  -", x)
sys.exit(1 if fails else 0)
