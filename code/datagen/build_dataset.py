#!/usr/bin/env python3
"""Reference PyTorch Dataset for the SaltCavern-PCC benchmark (pc_v2 layout).

Supplemental, clean-room reference implementation authored for the public
repository release (NOT one of the archived study scripts; those live in
code/training/ and code/evaluation/). It documents the on-disk format:

    data/pc_v2/<split>/partial/cav_NNN_Lk.npy    (8192, 3) float32
    data/pc_v2/<split>/complete/cav_NNN_Lk.npy   (16384, 3) float32   (val/test)
    data/pc_v2/splits/<split>.txt                cavity names, one per line

Usage:
    python code/supplemental/build_dataset.py --root data/pc_v2 --split val
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TIERS = ("L0", "L1", "L2")
N_INCOMPLETE = 8192
N_COMPLETE = 16384


class SaltCavernPCV2(Dataset):
    """Paired (incomplete, complete) salt-cavity point clouds.

    Coordinates are in meters in the original, unnormalized frame.
    Normalize before Chamfer-style metric computation; use float64 kernels
    for distance computation to avoid cancellation on large coordinates.
    """

    def __init__(self, root: str | Path, split: str = "val"):
        self.root = Path(root)
        self.split = split
        self.samples = sorted((self.root / split / "partial").glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(
                f"no partial samples under {self.root / split / 'partial'}; "
                "see data/README.md (dataset ships as a release asset)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        partial = np.load(self.samples[idx])          # (8192, 3) float32
        complete_path = self.root / self.split / "complete" / self.samples[idx].name
        complete = np.load(complete_path) if complete_path.exists() else None
        item = {
            "id": self.samples[idx].stem,
            "partial": torch.from_numpy(partial.astype(np.float32)),
        }
        if complete is not None:
            item["complete"] = torch.from_numpy(complete.astype(np.float32))
        return item


def list_cavities(root: str | Path, split: str) -> list[str]:
    """Cavity names (group-level split units) for a split."""
    partial = Path(root) / split / "partial"
    names = {p.stem.rsplit("_", 1)[0] for p in partial.glob("*.npy")}
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pc_v2")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--list-splits", action="store_true", help="print cavity names for the split")
    args = ap.parse_args()

    ds = SaltCavernPCV2(args.root, args.split)
    print(f"{args.split}: {len(ds)} samples, {len(list_cavities(args.root, args.split))} cavities")
    sample = ds[0]
    print(f"example {sample['id']}: partial {tuple(sample['partial'].shape)}")
    if "complete" in sample:
        print(f"           complete {tuple(sample['complete'].shape)}")


if __name__ == "__main__":
    main()
