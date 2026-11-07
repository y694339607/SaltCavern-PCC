import torch
import torch.nn as nn


def _cd_l2_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 近似 CD-L2（逐样本计算，控制显存）。数值仅用于本地闭环。"""
    vals = []
    for i in range(pred.size(0)):
        d = torch.cdist(pred[i].float(), gt[i].float()) ** 2
        vals.append(d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean())
    return torch.stack(vals).mean()


class _CDBackend:
    def __init__(self, backend: str):
        self.backend = backend
        self.criterion = None
        if backend == "chamfer":
            import os, sys
            vendor = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor_src", "PoinTr-master"))
            sys.path.insert(0, vendor)
            from extensions.chamfer_dist import ChamferDistanceL2
            self.criterion = ChamferDistanceL2()

    def __call__(self, pred, gt):
        if self.backend == "chamfer":
            return self.criterion(pred, gt)
        return _cd_l2_torch(pred, gt)


class AdaptLossModule(nn.Module):
    """ 损失适配模块。
    B1: loss = 上游损失 + lambda_known * CD(pred_fine, partial)
    B2: B1 + lambda_missing * CD(pred_fine, gt_missing)
    """

    def __init__(self, backend: str, stage: str,
                 lambda_known: float = 0.5, lambda_missing: float = 0.5,
                 missing_thresh: float = 0.02):
        super().__init__()
        assert stage in ("B1", "B2")
        self.stage = stage
        self.lambda_known = float(lambda_known)
        self.lambda_missing = float(lambda_missing)
        self.missing_thresh = float(missing_thresh)
        self.cd = _CDBackend(backend)

    def _missing_gt(self, gt: torch.Tensor, partial: torch.Tensor):
        """逐样本筛出到 partial 最近距离大于阈值的 gt 点（缺失区近似）。"""
        outs = []
        for i in range(gt.size(0)):
            dmin = torch.cdist(gt[i].float(), partial[i].float()).min(dim=1).values
            keep = dmin > self.missing_thresh
            outs.append(gt[i][keep])
        return outs

    def forward(self, ret, partial: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_fine = ret[-1]
        loss = self.cd(pred_fine, gt) + self.cd(pred_fine, partial) * self.lambda_known
        if self.stage == "B2":
            missing = self._missing_gt(gt, partial)
            vals = []
            cap = 2048
            for i in range(pred_fine.size(0)):
                if missing[i].numel() == 0:
                    continue
                p = pred_fine[i].unsqueeze(0)
                m = missing[i].unsqueeze(0)
                if self.cd.backend == "torch":
                    if p.size(1) > cap:
                        idx = torch.linspace(0, p.size(1) - 1, cap).long().to(p.device)
                        p = p[:, idx, :]
                    if m.size(1) > cap:
                        idx = torch.linspace(0, m.size(1) - 1, min(cap, m.size(1))).long().to(m.device)
                        m = m[:, idx, :]
                vals.append(self.cd(p, m))
            if vals:
                loss = loss + torch.stack(vals).mean() * self.lambda_missing
        return loss


def build_adapt_losses(cfg: dict, backend: str, device: torch.device, stage: str) -> AdaptLossModule:
    lc = cfg.get("loss", {}) or {}
    module = AdaptLossModule(
        backend=backend,
        stage=stage,
        lambda_known=float(lc.get("lambda_known", 0.5)),
        lambda_missing=float(lc.get("lambda_missing", 0.5)),
        missing_thresh=float(lc.get("missing_thresh", 0.02)),
    )
    return module.to(device)