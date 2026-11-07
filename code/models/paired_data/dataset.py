from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import make_sample_rng, normalize_pair, seeded_random_sample, to_tensor


class PairedDataset(Dataset):
    EXPECTED_COLUMNS = {
        "name", "num", "group", "split", "level", "combo",
        "coverage", "spacing_m", "p_loss", "shadow_limited", "seed",
    }

    def __init__(
        self,
        root: str = "dataset/pc_v2",
        split: str = "train",
        input_points: int = 1024,
        gt_points: int = 16384,
        downsample_seed: int = 20260921,
        csv_name: str = "samples.csv",
    ):
        self.root = Path(root)
        self.split = split
        self.input_points = int(input_points)
        self.gt_points = int(gt_points)
        self.downsample_seed = int(downsample_seed)

        csv_path = self.root / csv_name
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = self.EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"samples.csv missing columns: {sorted(missing)}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows for split={split}")

        self.partial_dir = self.root / split / "partial"
        self.complete_dir = self.root / split / "complete"

    def __len__(self) -> int:
        return len(self.df)

    def _paths(self, row: pd.Series) -> Tuple[Path, Path]:
        name = str(row["name"])
        level = int(row["level"])
        partial_path = self.partial_dir / f"{name}_L{level}.npy"
        complete_path = self.complete_dir / f"{name}_L{level}.npy"
        if not partial_path.exists():
            raise FileNotFoundError(partial_path)
        if not complete_path.exists():
            raise FileNotFoundError(complete_path)
        return partial_path, complete_path

    def get_meta(self, idx: int, include_transform: bool = False) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        partial_path, complete_path = self._paths(row)
        meta = {
            "taxonomy_id": "saltcavern_pcc",
            "model_id": str(row["name"]),
            "name": str(row["name"]),
            "num": int(row["num"]),
            "split": self.split,
            "level": int(row["level"]),
            "group": str(row["group"]),
            "combo": str(row["combo"]),
            "coverage": float(row["coverage"]),
            "spacing_m": float(row["spacing_m"]),
            "p_loss": float(row["p_loss"]),
            "shadow_limited": bool(row["shadow_limited"]),
            "seed": str(row["seed"]),
            "partial_path": str(partial_path),
            "complete_path": str(complete_path),
            "downsample_seed": self.downsample_seed,
        }

        if include_transform:
            partial_raw = np.load(meta["partial_path"]).astype(np.float32)
            complete_raw = np.load(meta["complete_path"]).astype(np.float32)
            rng = make_sample_rng(meta["num"], meta["level"])
            partial_ds, sample_idx = seeded_random_sample(partial_raw, self.input_points, rng)
            _, _, params = normalize_pair(partial_ds, complete_raw)
            meta["sample_idx"] = sample_idx
            meta["transform_params"] = params

        return meta

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        meta = self.get_meta(idx)

        partial_raw = np.load(meta["partial_path"]).astype(np.float32)
        complete_raw = np.load(meta["complete_path"]).astype(np.float32)

        if partial_raw.ndim != 2 or partial_raw.shape[1] != 3:
            raise ValueError(f"bad partial raw shape {partial_raw.shape}")
        if complete_raw.shape != (self.gt_points, 3):
            raise ValueError(f"bad complete raw shape {complete_raw.shape}, expected {(self.gt_points, 3)}")

        rng = make_sample_rng(meta["num"], meta["level"])
        partial_ds, sample_idx = seeded_random_sample(partial_raw, self.input_points, rng)
        partial_n, complete_n, params = normalize_pair(partial_ds, complete_raw)

        if partial_n.shape != (self.input_points, 3):
            raise ValueError(f"bad partial shape {partial_n.shape}")
        if complete_n.shape != (self.gt_points, 3):
            raise ValueError(f"bad complete shape {complete_n.shape}")

        meta["sample_idx"] = sample_idx
        meta["transform_params"] = params

        return (
            "saltcavern_pcc",
            meta["model_id"],
            (to_tensor(partial_n), to_tensor(complete_n)),
        )