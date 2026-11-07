# dataset_pc.py
# PCv2 数据集适配层：与 E2/BM 完全同一数据链（PairedDataset、同划分同种子同归一化），
# 仅暴露 (obs, gt) 元组以对接 PCN 复现的训练循环。
import sys
import torch
from pathlib import Path

import os
_ROOT = os.environ.get("SCC_ROOT", "/data")            # dataset root: contains dataset/ and validate/
_ARCHIVE = os.environ.get("SCC_ARCHIVE", _ROOT)         # optional separate root for figure source archives

ROOT = Path(_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from paired_data.dataset import PairedDataset


class PCv2(torch.utils.data.Dataset):
    def __init__(self, split):
        assert split in ('train', 'val', 'test')
        self.base = PairedDataset(root=str(ROOT/'dataset/pc_v2'), split=split,
                                    input_points=1024, gt_points=16384, downsample_seed=20260921)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        _, _, (obs, gt) = self.base[i]
        return obs, gt
