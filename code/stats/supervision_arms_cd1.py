import sys
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
from scipy import stats

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
WORK = ROOT / "validate/rescale_work"
REC = WORK / "val_mirror"
MET = WORK / "metrics"
RAWZIP = WORK / "raw/val_raw.zip"
SCALE = 0.3048
N_BOOT = 10000
SEED = 20261017
ARMS = {"F-1": ("pred_MS_P1_BM", "pred_MS_P1_E2"), "F-2": ("pred_MS_P2_BM", "pred_MS_P2_E2")}
PAPER_ORIG = {"F-1": (1.586, 1.645), "F-2": (1.502, 1.505)}

fails = []
def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond: fails.append(msg)

zf = zipfile.ZipFile(RAWZIP)
stems = sorted(n.split("/")[1][:-4] for n in zf.namelist()
               if n.startswith("partial/") and n.endswith(".npy"))
zf.close()
assert len(stems) == 150
cav = np.array([s.rsplit("_L", 1)[0] for s in stems])
lvl = np.array([int(s.rsplit("_L", 1)[1]) for s in stems])
check(all(len(set(cav[i:i+3])) == 1 for i in range(0, 150, 3)), "索引对齐：每三个连续样本属同一腔")

def cd(P, G):
    P = torch.as_tensor(P, dtype=torch.float64, device="cuda")
    G = torch.as_tensor(G, dtype=torch.float64, device="cuda")
    def mm(A, B):
        s = 0.0
        for c in range(0, A.shape[0], 2048):
            d = torch.cdist(A[c:c+2048], B)
            s += float(d.min(dim=1).values.sum())
        return s / A.shape[0]
    return 0.5 * (mm(P, G) + mm(G, P))

rows = []
for tag, (da, db) in ARMS.items():
    for i in range(150):
        gt = np.load(REC / "gt" / f"{i:03d}.npy").astype(np.float64)
        A = np.load(ROOT / "validate/volume" / da / f"{i:03d}.npy").astype(np.float64)
        B = np.load(ROOT / "validate/volume" / db / f"{i:03d}.npy").astype(np.float64)
        rows.append(dict(arm=tag, i=i, cav=cav[i], level=int(lvl[i]),
                         cd_A_orig=cd(A, gt), cd_B_orig=cd(B, gt),
                         cd_A_new=cd(A * [1, 1, SCALE], gt * [1, 1, SCALE]),
                         cd_B_new=cd(B * [1, 1, SCALE], gt * [1, 1, SCALE])))
df = pd.DataFrame(rows)
df.to_csv(MET / "f_arms_cd1_逐样本.csv", index=False, encoding="utf-8-sig")

out = []
for tag in ARMS:
    d = df[df.arm == tag]
    a_o, b_o = PAPER_ORIG[tag]
    check(abs(d["cd_A_orig"].mean() - a_o) <= 0.005, f"{tag} BM 原坐标均值 {d['cd_A_orig'].mean():.4f} vs 论文 {a_o}")
    check(abs(d["cd_B_orig"].mean() - b_o) <= 0.005, f"{tag} E2 原坐标均值 {d['cd_B_orig'].mean():.4f} vs 论文 {b_o}")
    for col in ["orig", "new"]:
        da = (d[f"cd_A_{col}"] - d[f"cd_B_{col}"]).values
        vals = np.array([da[cav == c].mean() for c in np.unique(cav)])
        w = stats.wilcoxon(vals)
        rng = np.random.default_rng(SEED)
        boot = np.array([vals[rng.integers(0, 50, 50)].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(arm=tag, state=col, A_mean=d[f"cd_A_{col}"].mean(), B_mean=d[f"cd_B_{col}"].mean(),
                        diff=da.mean(), ci_lo=lo, ci_hi=hi, p=w.pvalue, n_boot=N_BOOT, seed=SEED))
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv(MET / "f_arms_cd1_配对.csv", index=False, encoding="utf-8-sig")
print("\n==== 汇总 ====")
print(f"FAIL 数：{len(fails)}")
for x in fails: print("  -", x)
sys.exit(1 if fails else 0)
