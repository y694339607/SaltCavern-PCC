# pcn_loss.py
# chamfer 损失：优先用 extensions/chamfer_distance 编译扩展（GPU、显存安全），
# 不可用时按样本分块回退 torch.cdist。两路均为双向 chamfer 均值（L1 口径，扩展返回平方距离开根号）。
import torch

try:
    from extensions.chamfer_distance.chamfer_distance import ChamferDistance
    _EXT = ChamferDistance()
    _HAS_EXT = True
except Exception:
    _HAS_EXT = False


def cd_l1(a, b):
    """a, b: (B, N, 3) / (B, M, 3)，返回标量（双向均值）。"""
    if _HAS_EXT:
        d1, d2 = _EXT(a.contiguous(), b.contiguous())
        return torch.sqrt(d1).mean() + torch.sqrt(d2).mean()
    loss = a.new_zeros(())
    for i in range(a.shape[0]):
        d = torch.cdist(a[i], b[i])
        loss = loss + d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()
    return loss / a.shape[0]


HAS_EXT = _HAS_EXT
