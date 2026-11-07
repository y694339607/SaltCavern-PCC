from typing import Dict, Tuple

import numpy as np
import torch

MASTER_SEED = 20260921          # same seed the benchmark build uses
TAG_DOWN = 2                    # spawn tag for the 1024-point training subsample


def make_rng(master: int, tags) -> np.random.Generator:
    """Seed-derived stream: SeedSequence(master, spawn_key=tags) -> Philox."""
    sq = np.random.SeedSequence(master, spawn_key=list(tags))
    return np.random.Generator(np.random.Philox(sq))


def make_sample_rng(num: int, level: int, master: int = MASTER_SEED) -> np.random.Generator:
    """Per-cavity stream: make_rng(20260921, [cavity, level, tag])."""
    return make_rng(master, [int(num), int(level), TAG_DOWN])


def seeded_random_sample(points: np.ndarray, n_points: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points (N,3), got {points.shape}")
    if points.shape[0] < n_points:
        raise ValueError(f"cannot sample {n_points} from {points.shape[0]} points")
    idx = rng.permutation(points.shape[0])[:n_points]
    return points[idx].astype(np.float32, copy=True), idx.astype(np.int64)


def normalize_pair(partial: np.ndarray, complete: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    partial64 = partial.astype(np.float64, copy=False)
    complete64 = complete.astype(np.float64, copy=False)

    centroid = partial64.mean(axis=0)
    centered = partial64 - centroid[None, :]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid normalization scale: {scale}")

    partial_n = (centered / scale).astype(np.float32)
    complete_n = ((complete64 - centroid[None, :]) / scale).astype(np.float32)
    params = {
        "centroid": centroid.astype(np.float64),
        "scale": np.array([scale], dtype=np.float64),
    }
    return partial_n, complete_n, params


def denormalize(points: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    arr = points.astype(np.float64, copy=False)
    scale = np.asarray(params["scale"], dtype=np.float64).reshape(1, -1)
    centroid = np.asarray(params["centroid"], dtype=np.float64).reshape(1, -1)
    return arr * scale + centroid


def to_tensor(points: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    if t.ndim != 2 or t.size(1) != 3:
        raise ValueError(f"expected (N,3) tensor, got {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("non-finite points")
    return t