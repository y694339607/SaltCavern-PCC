import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paired_data.dataset import PairedDataset
from paired_data.transforms import MASTER_SEED, denormalize, make_rng
from train_pilot import build_model, make_cd_l2, collate_fn


def load_run(logdir: Path, run: str):
    csv_path = logdir / f"{run}.csv"
    sum_path = logdir / f"{run}_summary.json"
    df = pd.read_csv(csv_path, encoding="utf-8")
    with open(sum_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return df, summary


def assert_consistency(df, summary):
    i = int(df["val_cd_l2"].idxmin())
    best_log = float(df.loc[i, "val_cd_l2"])
    best_sum = float(summary["best_val_cd_l2"])
    assert abs(best_log - best_sum) < 1e-6, (best_log, best_sum)
    return int(df.loc[i, "epoch"]), best_log


@torch.no_grad()
def eval_ckpt(model, loader, device, cd_l2):
    model.eval()
    vals = []
    for batch in loader:
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)
        ret = model(partial)
        vals.append(cd_l2(ret[-1], gt))
    return float(np.mean(vals))


@torch.no_grad()
def noise_floor(val_ds, n_samples, sigma_m, device, cd_l2, seed=20260921):
    """参考下限：partial 米制坐标加径向高斯噪声 sigma_m 后与 complete 的 CD。
    本地 dummy 后端仅验证流程，数值无效。"""
    rng_idx = make_rng(MASTER_SEED, [0, 3, 0])    # 用途 3：噪声地板样本选取
    rng_noise = make_rng(MASTER_SEED, [0, 3, 1])  # 用途 3：径向噪声
    idxs = rng_idx.choice(len(val_ds), size=min(n_samples, len(val_ds)), replace=False)
    vals = []
    for idx in idxs:
        _, _, (partial_n, gt_n) = val_ds[int(idx)]
        meta = val_ds.get_meta(int(idx), include_transform=True)
        partial_m = denormalize(partial_n.numpy(), meta["transform_params"])
        gt_m = denormalize(gt_n.numpy(), meta["transform_params"])
        center = meta["transform_params"]["centroid"].reshape(1, 3)
        direction = partial_m - center
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        direction = direction / norm
        noisy = partial_m + direction * rng_noise.normal(0.0, sigma_m, size=(partial_m.shape[0], 1))
        v = cd_l2(
            torch.from_numpy(noisy).float().unsqueeze(0).to(device),
            torch.from_numpy(gt_m).float().unsqueeze(0).to(device),
        )
        vals.append(v)
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def pick_representatives(val_ds):
    """高/中/低覆盖(level 2/1/0) × perturb/control，各取首个匹配样本。"""
    picks = {}
    for level in (2, 1, 0):
        for want_perturb in (True, False):
            for idx in range(len(val_ds)):
                m = val_ds.get_meta(idx)
                if int(m["level"]) == level and (m["group"] == "perturb") == want_perturb:
                    picks[(level, want_perturb)] = idx
                    break
    return picks


@torch.no_grad()
def export_overlays(model, val_ds, picks, outdir, device, save_png):
    rows = []
    for (level, is_pert), idx in picks.items():
        _, model_id, (partial_n, gt_n) = val_ds[idx]
        meta = val_ds.get_meta(idx, include_transform=True)
        partial_m = denormalize(partial_n.numpy(), meta["transform_params"])
        gt_m = denormalize(gt_n.numpy(), meta["transform_params"])
        ret = model(partial_n.unsqueeze(0).to(device))
        fine_n = ret[-1][0].float().cpu().numpy()
        fine_m = denormalize(fine_n, meta["transform_params"])
        coarse_m = denormalize(ret[0][0].float().cpu().numpy(), meta["transform_params"])

        tag = f"L{level}_{'perturb' if is_pert else 'control'}_{model_id}"
        np.savez(outdir / f"overlay_{tag}.npz",
                 partial=partial_m.astype(np.float32), coarse=coarse_m.astype(np.float32),
                 fine=fine_m.astype(np.float32), gt=gt_m.astype(np.float32),
                 meta=json.dumps({k: (v if not hasattr(v, "tolist") else v.tolist())
                                  for k, v in meta.items() if k not in ("sample_idx", "transform_params")}))

        png_path = ""
        if save_png:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(gt_m[:, 0], gt_m[:, 2], s=0.3, c="lightgray", label="gt")
                ax.scatter(partial_m[:, 0], partial_m[:, 2], s=1.0, c="tab:blue", label="partial")
                ax.scatter(fine_m[:, 0], fine_m[:, 2], s=0.3, c="tab:red", label="fine")
                ax.set_aspect("equal")
                ax.legend(markerscale=5)
                ax.set_title(tag)
                png_path = str(outdir / f"overlay_{tag}.png")
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                png_path = f"PNG_FAILED: {e}"

        rows.append({
            "level": level, "group": "perturb" if is_pert else "control",
            "model_id": model_id, "npz": f"overlay_{tag}.npz", "png": png_path,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="log file prefix, e.g. bm_ep179_stub_B2")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="validate/pilot")
    ap.add_argument("--model_source", choices=["stub", "vendor"], default="stub")
    ap.add_argument("--loss_backend", choices=["dummy", "chamfer"], default="dummy")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--noise_sigma_m", type=float, default=0.3)
    ap.add_argument("--noise_n", type=int, default=30)
    ap.add_argument("--save_png", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    logdir = outdir / "logs"
    df, summary = load_run(logdir, args.run)
    best_epoch, best_val = assert_consistency(df, summary)

    device = torch.device("cuda") if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], args.model_source).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = PairedDataset(
        root=cfg["data"]["root"], split=cfg["data"]["val_split"],
        input_points=int(cfg["data"]["input_points"]), gt_points=int(cfg["data"]["gt_points"]),
        downsample_seed=int(cfg["data"]["downsample_seed"]),
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]),
                                             shuffle=False, num_workers=0, collate_fn=collate_fn)
    cd_l2 = make_cd_l2(args.loss_backend)

    cd_mean = eval_ckpt(model, val_loader, device, cd_l2)
    nf_mean, nf_min, nf_max = noise_floor(val_ds, args.noise_n, args.noise_sigma_m, device, cd_l2)
    picks = pick_representatives(val_ds)
    overlay_rows = export_overlays(model, val_ds, picks, outdir, device, args.save_png)

    md = outdir / f"validator_week{args.week}.md"
    lines = [
        f"# CompletionValidator Week {args.week}",
        "",
        f"- run: `{args.run}`",
        f"- ckpt: `{args.ckpt}` (epoch {ckpt['epoch']}, saved val_cd_l2={ckpt['val_cd_l2']:.6f})",
        f"- backend: model_source={args.model_source}, loss_backend={args.loss_backend}",
        "",
        "## 1. 数值汇总",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| best epoch (from log) | {best_epoch} |",
        f"| best val CD-L2 (log) | {best_val:.6f} |",
        f"| ckpt 复算 val CD-L2 mean | {cd_mean:.6f} |",
        f"| 噪声地板 σ={args.noise_sigma_m} m (mean/min/max, n={args.noise_n}) | {nf_mean:.6f} / {nf_min:.6f} / {nf_max:.6f} |",
        "",
        "## 2. 代表样本叠加图",
        "",
        "| level | group | model_id | npz | png |",
        "|---|---|---|---|---|",
    ]
    for r in overlay_rows:
        lines.append(f"| {r['level']} | {r['group']} | {r['model_id']} | {r['npz']} | {r['png']} |")
    lines += [
        "",
        "## 3. 判定",
        "",
        "- 日志与 summary 一致性：通过（断言已执行）。",
        "- 叠加图需人工核对：缺失区形态是否连续、是否把观测点锁死、三档难度是否单调。",
    ]
    if args.loss_backend == "dummy":
        lines.append("- 注意：dummy 后端数值无效，本节仅证明流程可走通。")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[validator] wrote {md}")
    print(f"[validator] best_epoch={best_epoch} best_val={best_val:.6f} ckpt_reval={cd_mean:.6f}")
    print("VALIDATOR_PASS")


if __name__ == "__main__":
    main()