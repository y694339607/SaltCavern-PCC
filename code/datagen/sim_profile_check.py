# Sanity checks on simulated source profiles before dataset assembly.

"""sim_profile_check.py — v2 仿真剖面健康性检查（阶段二，standalone，纯标准库）

用法：python sim_profile_check.py <radius_csv目录> [--out 汇总csv路径]

输入：cav_*.radius.csv（首列 z 深度 ft，自腔底向腔顶递减；末列为最终时刻半径 ft）
口径与判定线：同v2 数据集规范的第二阶段；v1 参照值内嵌于 REF。
"""
import argparse
import csv
import glob
import math
import os
import statistics as st

FT2M = 0.3048
SLENDER_FT = 2.5          # 未溶阈值（ft），略高于初始 blank 2.0 ft
GATES = {                  # v2 验收红线
    "L_med_max_m": 30.0,   # 底部未溶段中位
    "L_any_max_m": 80.0,   # 任一腔须单独复核
    "frac_med_max": 0.10,  # 占比中位
    "frac25_max_n": 0.05,  # 占比>25% 的腔数比例
}
REF = {"n": 500, "L_med_m": 167.0, "frac_med": 0.38,
       "rmax_med_m": 29.7, "rmax_min_m": 5.4, "rmax_max_m": 83.7}  # v1 实测


def cavity_stats(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    hdr = rows[0]
    if hdr[0].strip().lower() != "z":
        raise ValueError(f"{os.path.basename(path)}: 首列应为 z，实际 {hdr[0]}")
    z, r = [], []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        z.append(float(row[0])); r.append(float(row[-1]))  # 末列 = 最终时刻
    n = len(z)
    dz = abs(z[0] - z[1]) if n > 1 else 0.0
    H = (max(z) - min(z)) + dz
    Lb = 0
    for v in r:                      # 自腔底（首行）向上连续未溶段
        if v < SLENDER_FT:
            Lb += 1
        else:
            break
    return dict(
        H_m=H * FT2M, dz_m=dz * FT2M,
        r_max_m=max(r) * FT2M, r_med_m=st.median(r) * FT2M,
        L_bottom_m=Lb * dz * FT2M, frac_bottom=Lb * dz / H,
        r_uniq=len(set(round(v, 3) for v in r)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="sim_profile_check_summary.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.radius.csv")))
    if not files:
        raise SystemExit(f"未在 {a.dir} 找到 *.radius.csv")
    recs = {}
    for p in files:
        name = os.path.basename(p).replace(".radius.csv", "")
        try:
            recs[name] = cavity_stats(p)
        except Exception as e:
            print(f"  [跳过] {name}: {e}")
    S = list(recs.values())
    Ls = sorted(s["L_bottom_m"] for s in S)
    fr = sorted(s["frac_bottom"] for s in S)
    rmax = sorted(s["r_max_m"] for s in S)
    med = lambda xs: st.median(xs)
    frac25 = sum(1 for x in fr if x > 0.25) / len(fr)

    print(f"解析 {len(S)} 腔（目录含 {len(files)} 文件）")
    print(f"{'指标':<22}{'v2 实测':>12}{'v1 参照':>12}{'红线':>12}  判定")
    rows = [
        ("底部未溶段中位 m", med(Ls), REF["L_med_m"], f"<{GATES['L_med_max_m']:.0f}", med(Ls) < GATES["L_med_max_m"]),
        ("未溶段占比中位", med(fr), REF["frac_med"], f"<{GATES['frac_med_max']:.2f}", med(fr) < GATES["frac_med_max"]),
        ("占比>25% 腔数比例", frac25, "—", f"<{GATES['frac25_max_n']:.2f}", frac25 < GATES["frac25_max_n"]),
        ("最大半径中位 m", med(rmax), REF["rmax_med_m"], "同量级", 3.0 < med(rmax) < 60.0),
        ("最大半径范围 m", f"{rmax[0]:.1f}–{rmax[-1]:.1f}", f"{REF['rmax_min_m']}–{REF['rmax_max_m']}", "5–85 量级",
         rmax[0] > 1.0 and rmax[-1] < 150.0),
    ]
    for name, got, refv, gate, ok in rows:
        print(f"{name:<22}{got!s:>12}{refv!s:>12}{gate:>12}  {'PASS' if ok else 'FAIL'}")

    n_untouched = sum(1 for s in S if s["r_uniq"] <= 2)
    if n_untouched:
        print(f"[提示] {n_untouched} 腔最终剖面恒为单一半径（疑仿真未跑或死区全程），请复核")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "H_m", "r_max_m", "r_med_m", "L_bottom_m", "frac_bottom", "r_uniq"])
        for k in sorted(recs):
            s = recs[k]
            w.writerow([k, round(s["H_m"], 1), round(s["r_max_m"], 2), round(s["r_med_m"], 2),
                        round(s["L_bottom_m"], 1), round(s["frac_bottom"], 4), s["r_uniq"]])
    print(f"\n逐腔明细已写 {a.out}")
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
