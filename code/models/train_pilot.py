import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


VENDOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "vendor_src", "PoinTr-master"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paired_data.dataset import PairedDataset  # noqa: E402


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config 为空或格式错误: {path}")
    return cfg


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_cfg: dict, model_source: str):
    if model_source == "stub":
        # 本地免编译路径：仅用于 dummy 闭环，不得用于正式结果
        from smoke_cpu_forward import load_adapointtr, to_namespace
        AdaPoinTr = load_adapointtr(VENDOR_ROOT)
        return AdaPoinTr(to_namespace(model_cfg))

    # 平台路径：要求已安装 chamfer_dist / pointnet2_ops / KNN_CUDA
    sys.path.insert(0, VENDOR_ROOT)
    from easydict import EasyDict  # noqa: E402
    from models import build_model_from_cfg  # noqa: E402

    def _edict(obj):
        return EasyDict({k: _edict(v) if isinstance(v, dict) else v for k, v in obj.items()})

    cfg2 = _edict(dict(model_cfg))
    cfg2["NAME"] = cfg2.pop("name")
    return build_model_from_cfg(cfg2)


def make_cd_l2(loss_backend: str):
    if loss_backend == "chamfer":
        sys.path.insert(0, VENDOR_ROOT)
        from extensions.chamfer_dist import ChamferDistanceL2
        criterion = ChamferDistanceL2()

        def cd_l2(pred, gt):
            return float(criterion(pred, gt).item())

        return cd_l2

    # dummy 近似：子采样到 2048 后逐样本 CD，仅供本地闭环，数值无效
    rng = torch.Generator().manual_seed(20260921)

    def cd_l2(pred, gt):
        vals = []
        for i in range(pred.size(0)):
            n = min(2048, pred.size(1), gt.size(1))
            ip = torch.randperm(pred.size(1), generator=rng)[:n].to(pred.device)
            ig = torch.randperm(gt.size(1), generator=rng)[:n].to(gt.device)
            d = torch.cdist(pred[i][ip].float(), gt[i][ig].float()) ** 2
            vals.append(d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean())
        return float(torch.stack(vals).mean().item())

    return cd_l2


def collate_fn(batch):
    partial = torch.stack([b[2][0] for b in batch])
    gt = torch.stack([b[2][1] for b in batch])
    ids = [b[1] for b in batch]
    out = {"partial": partial, "gt": gt, "id": ids}
    if len(batch[0]) > 3:
        out["scale"] = torch.tensor([float(b[3]) for b in batch], dtype=torch.float32)
    return out


def filter_by_pilot(ds: PairedDataset, pilot_csv: str):
    import pandas as pd
    pilot = pd.read_csv(pilot_csv, encoding="utf-8-sig")
    names = set(pilot["name"].astype(str))
    before = len(ds.df)
    ds.df = ds.df[ds.df["name"].astype(str).isin(names)].reset_index(drop=True)
    return before, len(ds.df)


@torch.no_grad()
def validate(model, loader, device, cd_l2):
    model.eval()
    vals = []
    for batch in loader:
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)
        ret = model(partial)
        fine = ret[-1]
        vals.append(cd_l2(fine, gt))
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def save_ckpt(path, model, optimizer, epoch, val_cd, cfg_dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_cd_l2": val_cd,
        "config": cfg_dict,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / 'configs' / 'pc_v2_pilot_bm.yaml'))
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--model_source", choices=["stub", "vendor"], default="stub")
    parser.add_argument("--loss_backend", choices=["dummy", "chamfer"], default="dummy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pilot_csv", default="validate/pilot/pilot_50.csv")
    parser.add_argument("--outdir", default="validate/pilot")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--loss_stage", default=None, help="B0 默认；B1/B2 经 adapt_modules")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="run 标识，贯穿 run_name/ckpt/log")
    parser.add_argument("--resume", default=None, help="微调起点 ckpt 路径（G 组）")
    parser.add_argument("--lam_known", type=float, default=None)
    parser.add_argument("--lam_missing", type=float, default=None)
    parser.add_argument("--order_seed", type=int, default=None, help="数据顺序种子，默认跟随 --seed")
    parser.add_argument("--save_every", type=int, default=0, help=">=1 时每隔 N epoch 存一份 {run_name}_ep{N}.pt")
    args = parser.parse_args()

    if args.model_source == "stub" and args.loss_backend == "chamfer":
        raise SystemExit("stub + chamfer 不允许：本地 stub 下只能用 dummy。")

    cfg = load_config(args.config)
    device = pick_device(args.device)

    adapt_losses = None
    if args.loss_stage in ("B1", "B2"):
        from losses.adapt_modules import build_adapt_losses
        adapt_losses = build_adapt_losses(cfg, args.loss_backend, device, args.loss_stage)
        print(f"[run] adapt stage={args.loss_stage} backend={args.loss_backend}")

    torch.manual_seed(int(args.seed or cfg.get("seed", 20260921)))
    np.random.seed(int(args.seed or cfg.get("seed", 20260921)) % (2 ** 31))

    if args.lam_known is not None or args.lam_missing is not None:
        cfg.setdefault("loss", {})
        if args.lam_known is not None: cfg["loss"]["lambda_known"] = args.lam_known
        if args.lam_missing is not None: cfg["loss"]["lambda_missing"] = args.lam_missing
        print(f"[run] lambda override known={cfg['loss'].get('lambda_known')} missing={cfg['loss'].get('lambda_missing')}")
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])
    _epochs_key = {"smoke": "epochs_smoke", "pilot": "epochs_pilot", "full": "epochs_full"}[args.mode]
    epochs = args.epochs or int(cfg["train"][_epochs_key])
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    outdir = Path(args.outdir)
    logdir = outdir / "logs"
    ckptdir = outdir / "ckpt"
    logdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)

    # 数据
    ds = PairedDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["train_split"],
        input_points=int(cfg["data"]["input_points"]),
        gt_points=int(cfg["data"]["gt_points"]),
        downsample_seed=int(cfg["data"]["downsample_seed"]),
    )
    if args.mode != "full":
        before, after = filter_by_pilot(ds, args.pilot_csv)
        print(f"[data] pilot filter: {before} -> {after} rows (mode={args.mode})")
    else:
        print(f"[data] full mode: skip pilot filter, train rows = {len(ds)}")

    if (cfg.get("loss", {}) or {}).get("band_mask", True):
        _scales = [float(ds.get_meta(_i, include_transform=True)["transform_params"]["scale"].reshape(-1)[0]) for _i in range(len(ds))]
        class _DSBandMask(torch.utils.data.Dataset):
            def __init__(self, base, scales):
                self.base = base
                self.scales = scales
            def __len__(self):
                return len(self.base)
            def __getitem__(self, i):
                return self.base[i] + (self.scales[i],)
        ds = _DSBandMask(ds, _scales)
        print(f"[data] band_mask: per-sample scale attached (n={len(ds)})")
    g = torch.Generator().manual_seed(int(args.order_seed or args.seed or cfg.get("seed", 20260921)))
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, generator=g, drop_last=False,
    )

    # val
    val_ds = PairedDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["val_split"],
        input_points=int(cfg["data"]["input_points"]),
        gt_points=int(cfg["data"]["gt_points"]),
        downsample_seed=int(cfg["data"]["downsample_seed"]),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # 模型与损失
    model = build_model(cfg["model"], args.model_source).to(device)
    if args.resume:
        _sd = torch.load(args.resume, map_location=device, weights_only=False)
        _sd = _sd.get("model", _sd) if isinstance(_sd, dict) else _sd
        model.load_state_dict(_sd)
        print(f"[run] resume weights from {args.resume}")
    model.train()
    cd_l2 = make_cd_l2(args.loss_backend)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))

    run_name = f"{cfg.get('experiment', 'run')}_{args.mode}_{args.loss_backend}_{args.model_source}"
    if args.tag:
        run_name = args.tag
    if args.loss_stage in ("B1", "B2"):
        run_name = f"{run_name}_{args.loss_stage}"
    log_path = logdir / f"{run_name}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_cd_l2", "lr", "epoch_time_s", "dummy_backend"])
        f.flush()

        print(f"[run] {run_name}")
        print(f"[run] device={device} amp={amp} batch={batch_size} epochs={epochs}")
        if args.loss_backend == "dummy":
            print("[run] WARNING: DUMMY_BACKEND=1, 数值仅用于闭环验证")

        best = float("inf")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            losses = []
            for batch in train_loader:
                partial = batch["partial"].to(device)
                gt = batch["gt"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    ret = model(partial)
                    if args.loss_backend == "chamfer":
                        loss_denoised, loss_recon = model.get_loss(ret, gt)
                        loss = loss_denoised + loss_recon
                        lc = cfg.get("loss", {}) or {}
                        if lc.get("band_mask", True):
                            # BM band mask v4: v3 mask + bottom-zone extra CD (weight lambda, zone outside band)
                            _bw = float(lc.get("bottom_weight", 0.0))
                            _bz = lc.get("bottom_zone", [60, 250])
                            scale_b = batch["scale"].to(device).view(-1, 1)
                            if len(ret) == 4:
                                pred_coarse_bm, _, _, pred_fine_bm = ret
                            else:
                                pred_coarse_bm, pred_fine_bm = ret
                            zmin_b = gt[:, :, 2].min(dim=1, keepdim=True).values
                            w_lo = zmin_b + 2.0 / scale_b
                            w_hi = zmin_b + 60.0 / scale_b
                            m_keep_gt = ~((gt[:, :, 2] > w_lo) & (gt[:, :, 2] < w_hi))
                            b_lo = zmin_b + float(_bz[0]) / scale_b
                            b_hi = zmin_b + float(_bz[1]) / scale_b
                            m_bot_gt = (gt[:, :, 2] >= b_lo) & (gt[:, :, 2] < b_hi)
                            _recon_bm = []
                            _bot_terms = []
                            for _src in (pred_coarse_bm, pred_fine_bm):
                                m_keep_pr = ~((_src[:, :, 2] > w_lo) & (_src[:, :, 2] < w_hi))
                                m_bot_pr = (_src[:, :, 2] >= b_lo) & (_src[:, :, 2] < b_hi)
                                _cd_i = []
                                for _g, _p, _mg, _mp, _bg, _bp in zip(gt, _src, m_keep_gt, m_keep_pr, m_bot_gt, m_bot_pr):
                                    _gk = _g[_mg].unsqueeze(0).float()
                                    _pk = _p[_mp].unsqueeze(0).float()
                                    assert _gk.shape[1] > 0 and _pk.shape[1] > 0
                                    _cd_i.append(model.loss_func(_pk, _gk))
                                    if _bw > 0.0:
                                        _gk2 = _g[_bg].unsqueeze(0).float()
                                        _pk2 = _p[_bp].unsqueeze(0).float()
                                        if _gk2.shape[1] > 0 and _pk2.shape[1] > 0:
                                            _bot_terms.append(model.loss_func(_pk2, _gk2))
                                _recon_bm.append(torch.stack(_cd_i).mean())
                            loss = loss_denoised + _recon_bm[0] + _recon_bm[1]
                            if _bot_terms:
                                loss = loss + _bw * torch.stack(_bot_terms).mean()
                        if lc.get("coarse_supervise", False) or lc.get("fine_gt2pred_weighted", False):
                            from extensions.chamfer_dist import ChamferFunction
                            from losses.adapt_modules import z_density_weights
                            if len(ret) == 4:
                                pred_coarse, _, _, pred_fine = ret
                            else:
                                pred_coarse, pred_fine = ret
                            _zb = int(lc.get("zbins", 60)); _wm = float(lc.get("wmax", 50))
                            if lc.get("coarse_supervise", False):
                                lam_c = float(lc.get("coarse_lambda", 1.0))
                                if lc.get("coarse_weighted", False):
                                    w_c = z_density_weights(gt.float(), nbins=_zb, wmax=_wm)
                                    d1c, d2c = ChamferFunction.apply(pred_coarse, gt)
                                    cd_c = (torch.sqrt(d1c).mean() + (w_c * torch.sqrt(d2c)).sum() / w_c.sum()) / 2
                                else:
                                    cd_c = model.loss_func(pred_coarse, gt)
                                loss = loss - model.loss_func(pred_coarse, gt) + lam_c * cd_c
                            if lc.get("fine_gt2pred_weighted", False):
                                w_f = z_density_weights(gt.float(), nbins=_zb, wmax=_wm)
                                d1f, d2f = ChamferFunction.apply(pred_fine, gt)
                                cd_f = (torch.sqrt(d1f).mean() + (w_f * torch.sqrt(d2f)).sum() / w_f.sum()) / 2
                                loss = loss - model.loss_func(pred_fine, gt) + cd_f
                    else:
                        loss = ret[-1].float().sum() * 0.0  # 可反传零损失，仅闭环
                    if adapt_losses is not None:
                        loss = loss + adapt_losses(ret, partial, gt)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu().item()))

            scheduler.step()
            val_cd = validate(model, val_loader, device, cd_l2)
            loss_mean = float(np.mean(losses)) if losses else float("nan")
            lr_now = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            w.writerow([epoch, f"{loss_mean:.8f}", f"{val_cd:.8f}", f"{lr_now:.3e}", f"{dt:.1f}", int(args.loss_backend == "dummy")])
            f.flush()
            print(f"[epoch {epoch:03d}] train_loss={loss_mean:.6f} val_cd_l2={val_cd:.6f} lr={lr_now:.3e} time={dt:.1f}s")

            save_ckpt(ckptdir / f"{run_name}_last.pt", model, optimizer, epoch, val_cd, cfg)
            if args.save_every and epoch % args.save_every == 0 and epoch >= args.save_every:
                save_ckpt(ckptdir / f"{run_name}_ep{epoch}.pt", model, optimizer, epoch, val_cd, cfg)
            if val_cd < best:
                best = val_cd
                save_ckpt(ckptdir / f"{run_name}_best.pt", model, optimizer, epoch, val_cd, cfg)

    summary = {"run": run_name, "best_val_cd_l2": best, "log": str(log_path)}
    with open(logdir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] best_val_cd_l2={best:.6f} (dummy 数值无效)" if args.loss_backend == "dummy" else f"[done] best_val_cd_l2={best:.6f}")
    print("TRAIN_LOOP_PASS")


if __name__ == "__main__":
    main()