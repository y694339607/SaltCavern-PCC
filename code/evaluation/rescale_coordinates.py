import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

SCALE = 0.3048
MASTER_SEED = 20260921
TAG_DOWN = 2
ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
ZVAL = WORK / "raw/val_from_zip"
ZTEST = WORK / "raw/test_from_zip"
SAMPLES = ROOT / "dataset/pc_v2/samples.csv"
TEST_FIN = ROOT / "validate/test_final"
VAL_PRED = {"BM": ROOT/"validate/volume/pred_adapointr_masked","E2": ROOT/"validate/volume/pred_adapointr_full",
            "PoinTr": ROOT/"validate/volume/pred_GPoinTr","B1": ROOT/"validate/baseline/B1",
            "B2": ROOT/"validate/baseline/B2","B3": ROOT/"validate/baseline/B3",
            "PCN": ROOT/"validate/baseline_dl/pcn/pred"}
ALIAS = {"BM": "adapointr_masked", "E2": "adapointr_full", "PCN": "pcn",
         "PoinTr": "pointr", "B1": "cylnn", "B2": "spline", "B3": "idw"}
METHODS = [ALIAS.get(m, m) for m in ["BM","E2","PCN","PoinTr","B1","B2","B3"]]
OUT = WORK / "rescaled"

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

def sha16(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def make_rng(num, level):
    sq = np.random.SeedSequence(MASTER_SEED, spawn_key=[int(num), int(level), TAG_DOWN])
    return np.random.Generator(np.random.Philox(sq))

samples = pd.read_csv(SAMPLES, encoding="utf-8-sig")
rows = {s: samples[samples["split"]==s].reset_index(drop=True) for s in ["val","test"]}
stems = {s: [f"{r['name']}_L{int(r['level'])}" for _, r in rows[s].iterrows()] for s in rows}

def rescale_arr(a):
    b = np.array(a, dtype=np.float64, copy=True)
    b[:, 2] *= SCALE
    return b

def process(src, dst):
    a = np.load(src).astype(np.float64)
    b = rescale_arr(a)
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, b)
    return a, b

def verify_pair(a, b, tag):
    ok_xy = np.array_equal(a[:, :2], b[:, :2])
    err = np.abs(b[:, 2] - SCALE * a[:, 2]).max()
    ok_z = err < 1e-9
    check(ok_xy and ok_z, f"{tag}: x/y 不变={ok_xy}, max|z'-0.3048z|={err:.2e}")

n = 0
for m in METHODS:
    for sp, srcdir in [("val", VAL_PRED[m]), ("test", TEST_FIN/f"pred_{m}")]:
        d = OUT/sp/f"pred_{m}"; d.mkdir(parents=True, exist_ok=True)
        for i in range(150):
            a, b = process(srcdir/f"{i:03d}.npy", d/f"{i:03d}.npy"); n += 1
            if i < 3: verify_pair(a, b, f"{sp}/pred_{m}/{i:03d}")

for sp, zdir in [("val", ZVAL), ("test", ZTEST)]:
    d = OUT/sp/"complete"; d.mkdir(parents=True, exist_ok=True)
    for st in stems[sp]:
        process(zdir/"complete"/f"{st}.npy", d/f"{st}.npy"); n += 1
    for i in [0,1,2]:
        st = stems[sp][i]
        verify_pair(np.load(zdir/"complete"/f"{st}.npy").astype(np.float64),
                    np.load(d/f"{st}.npy"), f"{sp}/complete/{st}")

for sp, zdir in [("val", ZVAL), ("test", ZTEST)]:
    d = OUT/sp/"partial"; d.mkdir(parents=True, exist_ok=True)
    for st in stems[sp]:
        process(zdir/"partial"/f"{st}.npy", d/f"{st}.npy"); n += 1
    for i in [0,1,2]:
        st = stems[sp][i]
        verify_pair(np.load(zdir/"partial"/f"{st}.npy").astype(np.float64),
                    np.load(d/f"{st}.npy"), f"{sp}/partial/{st}")

for sp, zdir in [("val", ZVAL), ("test", ZTEST)]:
    src = zdir/"gt_field"; d = OUT/sp/"gt_field"; d.mkdir(parents=True, exist_ok=True)
    fs = sorted(src.glob("*.npy")); check(len(fs)==50, f"{sp} gt_field 源 50 件")
    for f in fs:
        process(f, d/f.name); n += 1

for sp, zdir in [("val", ZVAL), ("test", ZTEST)]:
    d = OUT/sp/"obs1024"; d.mkdir(parents=True, exist_ok=True)
    for i, r in rows[sp].iterrows():
        st = stems[sp][i]
        partial = np.load(zdir/"partial"/f"{st}.npy")
        num = int(str(r["name"]).split("_")[1])
        rng = make_rng(num, int(r["level"]))
        idx = rng.permutation(partial.shape[0])[:1024]
        obs = partial[idx].astype(np.float32)
        b = rescale_arr(obs)
        np.save(d/f"{i:03d}.npy", b); n += 1
        if i < 3: verify_pair(obs.astype(np.float64), b, f"{sp}/obs1024/{i:03d}")

print(f"[info] 写入文件总数: {n}", flush=True)
check(n == 3100, "输出件数 = 2100 预测 + 300 真值 + 300 观测 + 100 gt_field + 300 obs1024")

for sp in ["val","test"]:
    zs = [np.ptp(np.load(OUT/sp/"complete"/f"{st}.npy")[:,2]) for st in stems[sp]]
    zs = np.array(zs)
    print(f"[range {sp}] 缩放后 z 跨度 min/max = {zs.min():.1f}/{zs.max():.1f} m", flush=True)
    check(zs.min() >= 295 and zs.max() <= 605, f"{sp} 缩放后 z 跨度 ∈ [295,605] m")

sha_dir = OUT/"sha"; sha_dir.mkdir(parents=True, exist_ok=True)
for sp in ["val","test"]:
    for sub in ["pred_adapointr_masked","pred_adapointr_full","pred_pcn","pred_pointr","pred_cylnn","pred_spline","pred_idw",
                "complete","partial","gt_field","obs1024"]:
        with open(sha_dir/f"{sp}_{sub}.txt","w") as f:
            for p in sorted((OUT/sp/sub).glob("*.npy")):
                f.write(f"{sha16(p)}  {p.name}\n")

print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
