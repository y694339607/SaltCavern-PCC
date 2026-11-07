"""gt_slender_check.py — v2 点云侧细长带统计（阶段四，standalone，纯标准库）

用法：python gt_slender_check.py <数据集目录> [--out 汇总csv路径]

输入：数据集根目录（递归查找 cav_*_L*.npy；按点数为 gt 取较大件，默认 gt=16384 点）
口径：z 向 40 m 带，带内点 <20 记空带；r_eq = 带内点到带 xy 质心的 RMS 距离（等效半径）；
      slender 阈值 2.0 m（物理坐标）。输出逐腔汇总 csv 并打印对照线。
"""
import argparse
import ast
import csv
import glob
import math
import os
import struct

BAND_M = 40.0
SLENDER_M = 2.0
MIN_PTS = 20
GT_PTS = 16384


def read_npy(path):
    """极简 npy v1/v2 读取：<f4 二维数组，无 numpy 依赖。"""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("非 npy 文件")
        major = f.read(1)[0]
        f.read(1)  # minor
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
        if hdr["descr"] not in ("<f4", "|f4", "f4") or hdr["fortran_order"]:
            raise ValueError(f"不支持的 dtype/order: {hdr['descr']}")
        shape = hdr["shape"]
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(f"期望 (N,3)，实际 {shape}")
        n = shape[0]
        data = struct.unpack(f"<{n*3}f", f.read(n * 12))
        return [data[i*3:(i+1)*3] for i in range(n)], n


def per_cavity(path):
    pts, n = read_npy(path)
    if n < GT_PTS:  # partial 件（8192）跳过，取同样本 gt 件
        return None
    zs = [p[2] for p in pts]
    zmin, zmax = min(zs), max(zs)
    nb = int(math.ceil((zmax - zmin) / BAND_M)) or 1
    bands = [[] for _ in range(nb)]
    for p in pts:
        b = min(int((p[2] - zmin) / BAND_M), nb - 1)
        bands[b].append(p)
    reqs, slender = [], []
    for bp in bands:
        if len(bp) < MIN_PTS:
            reqs.append(float("nan")); slender.append(False); continue
        cx = sum(p[0] for p in bp) / len(bp)
        cy = sum(p[1] for p in bp) / len(bp)
        r_eq = math.sqrt(sum((p[0]-cx)**2 + (p[1]-cy)**2 for p in bp) / len(bp))
        reqs.append(r_eq); slender.append(r_eq < SLENDER_M)
    L = 0  # 自腔底（z 最小带）向上连续 slender
    for v in slender:
        if v:
            L += 1
        else:
            break
    valid = [r for r in reqs if r == r]
    return dict(H_m=zmax - zmin, n_bands=nb, L_slen_m=L * BAND_M,
                frac=L * BAND_M / (zmax - zmin), r_eq_min=min(valid), r_eq_med=st_median(valid))


def st_median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m-1] + xs[m]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="gt_slender_summary.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "**", "cav_*_L*.npy"), recursive=True))
    recs = {}
    for p in files:
        name = os.path.basename(p).replace(".npy", "")
        try:
            s = per_cavity(p)
        except Exception as e:
            print(f"  [跳过] {name}: {e}")
            continue
        if s is None:
            continue
        recs[name] = s
    if not recs:
        raise SystemExit("未找到 gt 点云件（16384 点 npy），请检查目录")
    S = list(recs.values())
    Ls = sorted(s["L_slen_m"] for s in S)
    fr = sorted(s["frac"] for s in S)
    print(f"解析 {len(S)} 件 gt（目录含 {len(files)} 个 npy）")
    print(f"{'指标':<22}{'v2 实测':>12}{'对照（v1 剖面）':>16}")
    print(f"{'L_slen 中位 m':<22}{st_median(Ls):>12.1f}{'167':>16}")
    print(f"{'L_slen 最大 m':<22}{Ls[-1]:>12.1f}{'425':>16}")
    print(f"{'细长占比中位':<22}{st_median(fr):>12.3f}{'0.38':>16}")
    print(f"{'r_eq 最小 m':<22}{min(s['r_eq_min'] for s in S):>12.2f}{'≈0.61(blank)':>16}")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "H_m", "n_bands", "L_slen_m", "frac", "r_eq_min", "r_eq_med"])
        for k in sorted(recs):
            s = recs[k]
            w.writerow([k, round(s["H_m"], 1), s["n_bands"], round(s["L_slen_m"], 1),
                        round(s["frac"], 4), round(s["r_eq_min"], 2), round(s["r_eq_med"], 2)])
    print(f"\n逐腔明细已写 {a.out}；与剖面统计逐腔对照（相关 >0.8）用两 csv 的 L 列计算")
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
