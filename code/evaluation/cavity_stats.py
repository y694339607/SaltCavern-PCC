#!/usr/bin/env python3
"""按腔体重算配对差值的符号秩检验、符号检验与聚类 bootstrap。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="共同支撑逐样本误差表")
    p.add_argument("--col-cavity", default="cavity_id")
    p.add_argument("--col-level", default="level")
    p.add_argument("--col-a", default="err_id1024", help="方法 A 的绝对误差，单位：百分点")
    p.add_argument("--col-b", default="err_mask", help="方法 B 的绝对误差，单位：百分点")
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--signed-csv", help="可选：含逐样本有符号误差的输入匹配表")
    p.add_argument("--signed-split", default="test")
    p.add_argument("--signed-cavity", default="cav")
    p.add_argument("--signed-level", default="level")
    p.add_argument("--signed-a", default="err1024_signed")
    p.add_argument("--signed-b", default="BM_err_signed")
    return p.parse_args()


def exact_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    result = stats.wilcoxon(values, alternative="two-sided", method="exact")
    return float(result.statistic), float(result.pvalue)


def exact_sign_test(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0]
    n = len(nonzero)
    n_pos = int(np.sum(nonzero > 0))
    p = float(stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue) if n else np.nan
    return n_pos, n, p


def cluster_bootstrap(frame, cavity_col, diff_col, boot, seed):
    grouped = {key: group[diff_col].to_numpy(dtype=float) for key, group in frame.groupby(cavity_col, sort=True)}
    keys = np.array(list(grouped), dtype=object)
    observed = float(frame[diff_col].mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(boot, dtype=float)
    for b in range(boot):
        selected = rng.integers(0, len(keys), len(keys))
        estimates[b] = np.concatenate([grouped[keys[i]] for i in selected]).mean()
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return observed, float(lo), float(hi), estimates


def fmt(x):
    return f"{x:.12g}"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    required = [args.col_cavity, args.col_level, args.col_a, args.col_b]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}")
    work = df[required].copy()
    work[args.col_a] = pd.to_numeric(work[args.col_a], errors="raise")
    work[args.col_b] = pd.to_numeric(work[args.col_b], errors="raise")
    if work[required].isna().any().any():
        raise ValueError("主终点所需字段存在缺失值")
    work["diff"] = work[args.col_b] - work[args.col_a]

    grouped = work.groupby(args.col_cavity, sort=True)
    cavity = grouped.agg(
        n_samples=("diff", "size"),
        mean_diff=("diff", "mean"),
        median_diff=("diff", "median"),
        n_pos_levels=("diff", lambda x: int(np.sum(x > 0))),
    ).reset_index()
    levels = grouped[args.col_level].apply(lambda x: ";".join(map(str, sorted(x.unique())))).reset_index(name="levels")
    cavity = cavity.merge(levels, on=args.col_cavity, how="left")
    cavity = cavity[[args.col_cavity, "levels", "n_samples", "mean_diff", "median_diff", "n_pos_levels"]]
    cavity.to_csv(out / "cavity_level.csv", index=False, float_format="%.12g")

    d = cavity["mean_diff"].to_numpy(dtype=float)
    w_stat, w_p = exact_wilcoxon(d)
    n_pos, n_nonzero, sign_p = exact_sign_test(d)
    observed, ci_lo, ci_hi, boot_estimates = cluster_bootstrap(work, args.col_cavity, "diff", args.boot, args.seed)
    level_means = work.groupby(args.col_level, sort=True)["diff"].agg(["size", "mean", "median"])

    lines = [
        "腔体级统计结果",
        f"输入文件：{Path(args.csv).resolve()}",
        f"差值方向：B − A（{args.col_b} − {args.col_a}），单位为百分点",
        f"样本数：{len(work)}；腔体数：{len(cavity)}",
        f"样本级均值差（仅对照，不用于推断）：{fmt(work['diff'].mean())}",
        f"腔体级平均差：{fmt(d.mean())}",
        f"腔体均值的中位数：{fmt(np.median(d))}",
        f"腔体均值为正：{n_pos}/{len(d)}；非零腔体数：{n_nonzero}",
        f"Wilcoxon 符号秩统计量：{fmt(w_stat)}",
        f"Wilcoxon 双侧精确 p：{w_p:.16g}",
        f"符号检验双侧精确 p：{sign_p:.16g}",
        f"聚类 bootstrap 次数：{args.boot}；seed：{args.seed}",
        "重采样单位：腔体；每次保留被抽中腔体的全部档位；百分位法",
        f"均值差：{fmt(observed)}；95% 区间：[{fmt(ci_lo)}, {fmt(ci_hi)}]",
        "分档描述（不作独立推断）：",
    ]
    for level, row in level_means.iterrows():
        lines.append(f"  {level}：n={int(row['size'])}，均值差={fmt(row['mean'])}，中位数={fmt(row['median'])}")
    if (cavity["n_samples"] != 3).any():
        lines.append("缺档位腔体：" + ", ".join(cavity.loc[cavity["n_samples"] != 3, args.col_cavity].astype(str)))
    else:
        lines.append("完整性：每个腔体均含 3 个档位")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(out / "bootstrap_estimates.npy", boot_estimates)

    if args.signed_csv:
        signed = pd.read_csv(args.signed_csv)
        req = ["split", args.signed_cavity, args.signed_level, args.signed_a, args.signed_b]
        miss = [c for c in req if c not in signed.columns]
        if miss:
            raise ValueError(f"有符号误差表缺少列：{miss}")
        signed = signed.loc[signed["split"] == args.signed_split, req].copy()
        if signed[req].isna().any().any():
            raise ValueError("有符号误差所需字段存在缺失值")
        signed["over_a"] = signed[args.signed_a] > 0
        signed["over_b"] = signed[args.signed_b] > 0
        per_cavity = signed.groupby(args.signed_cavity, sort=True).agg(
            n_samples=("over_a", "size"), over_a_prop=("over_a", "mean"), over_b_prop=("over_b", "mean")
        ).reset_index()
        per_cavity["diff_pp"] = 100 * (per_cavity["over_b_prop"] - per_cavity["over_a_prop"])
        per_cavity.to_csv(out / "overestimate_cavity.csv", index=False, float_format="%.12g")

        o_lines = [
            "腔体级高估比例比较",
            f"输入文件：{Path(args.signed_csv).resolve()}",
            f"划分：{args.signed_split}；差值方向：B − A；比例差单位：百分点",
        ]
        for level in ["overall"] + sorted(signed[args.signed_level].unique().tolist()):
            sub = signed if level == "overall" else signed.loc[signed[args.signed_level] == level]
            cav = sub.groupby(args.signed_cavity, sort=True).agg(over_a=("over_a", "mean"), over_b=("over_b", "mean"))
            prop_diff = 100 * (cav["over_b"] - cav["over_a"])
            stat, p = exact_wilcoxon(prop_diff.to_numpy()) if np.any(prop_diff.to_numpy() != 0) else (0.0, 1.0)
            n_b_higher = int(np.sum(prop_diff > 0))
            o_lines.extend([
                f"档位：{level}",
                f"  腔体数：{len(cav)}",
                f"  A 平均高估占比：{fmt(100 * cav['over_a'].mean())}%",
                f"  B 平均高估占比：{fmt(100 * cav['over_b'].mean())}%",
                f"  平均占比差（B−A）：{fmt(prop_diff.mean())} 个百分点",
                f"  B 更高的腔体：{n_b_higher}/{len(cav)}",
                f"  Wilcoxon 统计量：{fmt(stat)}；双侧精确 p：{p:.16g}",
            ])
        (out / "overestimate.txt").write_text("\n".join(o_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
