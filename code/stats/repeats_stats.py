# repeats_stats.py — 阶段三：统计分析（指南 7.3、8.1）
# 统计种子固定 20261103；腔体为重采样单位的配对 bootstrap 10000 次，95% 百分位 CI。
# 主统计：err_field_abs（V_field 口径）配对差 BM-E2；err_ref_abs 为敏感性口径；
# 普通 150 样本 Wilcoxon 仅作补充，不作为主统计（指南 7.3）。
# 腔体级 CI 反映当前 50 个验证腔体内的不确定性，不得解释为任意初始化种子的总体显著性。
# 用法：cd /data && python validate/ms/repeats_stats.py [--csv validate/volume/ms_long_table.csv]
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

BOOT = 10000
RNG_SEED = 20261103

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=f"{_ROOT}/validate/volume/ms_long_table.csv")
ap.add_argument("--outdir", default=f"{_ROOT}/validate/volume")
a = ap.parse_args()
outdir = Path(a.outdir)

df = pd.read_csv(a.csv)
pairs = [p for p in ["P1", "P2"] if p in set(df.pair)]
assert len(pairs) == 2, f"配对缺失: {set(df.pair)}"

rng = np.random.default_rng(RNG_SEED)
report, cav_rows, summ_rows = [], [], []


def boot_ci(diff_vals):
    idx = rng.integers(0, len(diff_vals), size=(BOOT, len(diff_vals)))
    boots = diff_vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return lo, hi, float((boots < 0).mean()), boots


report.append(f"repeats_stats — 固定初始化配对重复实验统计报告")
report.append(f"统计种子: {RNG_SEED} | bootstrap: {BOOT} | 重采样单位: 腔体\n")

verdicts = {}
for pair in pairs:
    d = df[df.pair == pair]
    report.append(f"== 配对 {pair} ==")
    for metric, label in [("err_field_abs", "库容误差(V_field)"), ("err_ref_abs", "库容误差(V_ref)"),
                          ("cd_l1_m", "CD-L1(m)")]:
        for arm in ["BM", "E2"]:
            s = d[d.arm == arm][metric]
            report.append(f"  {label} {arm}: mean={s.mean():.5f} median={s.median():.5f} "
                          f"max={s.max():.5f} (n={len(s)})")
            summ_rows.append(dict(pair=pair, metric=metric, arm=arm, mean=s.mean(),
                                  median=s.median(), max=s.max(), n=len(s)))
    lvl = d.groupby(["level", "arm"])[["err_field_abs", "err_ref_abs", "cd_l1_m"]].mean()
    report.append("  分等级均值:")
    report.append(lvl.round(5).to_string())

    cav = d.groupby(["cav", "arm"])[["err_field_abs", "err_ref_abs", "cd_l1_m"]].mean()
    for metric, label in [("err_field_abs", "库容(V_field)"), ("err_ref_abs", "库容(V_ref)"),
                          ("cd_l1_m", "CD-L1")]:
        piv = cav[metric].unstack("arm")
        diff = (piv["BM"] - piv["E2"]).dropna()
        assert len(diff) == 50, f"{pair} {metric} 腔体数 {len(diff)} != 50"
        lo, hi, p_neg, _ = boot_ci(diff.values)
        for cav_name, v in diff.items():
            cav_rows.append(dict(pair=pair, metric=metric, cav=cav_name, diff_BM_minus_E2=v))
        report.append(f"  逐腔配对差 BM-E2 ({label}): mean={diff.mean():+.5f} "
                      f"median={diff.median():+.5f} min={diff.min():+.5f} max={diff.max():+.5f}")
        report.append(f"    腔体级 bootstrap 95% CI: [{lo:+.5f}, {hi:+.5f}] | "
                      f"P(diff<0)={p_neg:.4f}")
        summ_rows.append(dict(pair=pair, metric=metric, arm="BM-E2_diff", mean=diff.mean(),
                              median=diff.median(), max=diff.max(), n=len(diff),
                              ci_lo=lo, ci_hi=hi, p_neg=p_neg))
    dV = summ_rows[-3]["mean"]  # err_field_abs diff mean（主指标）
    cd_deterior = None
    cd = cav["cd_l1_m"].unstack("arm")
    cddiff = (cd["BM"] - cd["E2"]).dropna()
    lo_c, hi_c, _, _ = boot_ci(cddiff.values)
    cd_ratio = d[d.arm == "BM"]["cd_l1_m"].mean() / d[d.arm == "E2"]["cd_l1_m"].mean() - 1.0
    cd_deterior = cd_ratio > 0.05
    report.append(f"  CD-L1 均值恶化率 BM/E2-1: {cd_ratio:+.2%}（阈值 5%，{'超阈' if cd_deterior else '未超阈'}）"
                  f" | 腔体级 CI [{lo_c:+.5f}, {hi_c:+.5f}] m")
    summ_rows.append(dict(pair=pair, metric="cd_ratio", arm="BM/E2-1", mean=cd_ratio,
                          ci_lo=lo_c, ci_hi=hi_c))
    verdicts[pair] = dict(dV=dV, cd_ok=not cd_deterior)
    report.append("")

# Wilcoxon（补充统计，非主统计）
try:
    from scipy.stats import wilcoxon
    for pair in pairs:
        d = df[df.pair == pair]
        for metric in ["err_field_abs", "err_ref_abs"]:
            piv = d.pivot_table(index="i", columns="arm", values=metric)
            w, p = wilcoxon(piv["BM"], piv["E2"])
            report.append(f"[补充] {pair} {metric} Wilcoxon(150样本): stat={w:.0f} p={p:.3e}")
except ImportError:
    report.append("[补充] scipy 不可用，Wilcoxon 跳过（非主统计，不影响判定）")

# 跨配对方向一致性（按预注册判据 3.3 给出建议，最终裁定属用户）
both_dV_neg = all(v["dV"] < 0 for v in verdicts.values())
both_cd_ok = all(v["cd_ok"] for v in verdicts.values())
if both_dV_neg and both_cd_ok:
    sug = "重复性获得支持（建议）"
elif not any(v["dV"] < 0 for v in verdicts.values()):
    sug = "未获得支持（建议）"
else:
    sug = "重复性证据有限（建议）"
report.append(f"\n两配对 ΔV 均为负: {both_dV_neg} | 两配对 CD 均未恶化超 5%: {both_cd_ok}")
report.append(f"预注册判据建议结论: {sug}（最终裁定以用户裁决为准）")

txt = "\n".join(report) + "\n"
(outdir / "repeats_stats_report.txt").write_text(txt, encoding="utf-8")
pd.DataFrame(cav_rows).to_csv(outdir / "ms_per_cavity_paired_diff.csv", index=False)
pd.DataFrame(summ_rows).to_csv(outdir / "ms_pair_summary.csv", index=False)
print(txt)
print("MS_STATS_PASS", flush=True)
