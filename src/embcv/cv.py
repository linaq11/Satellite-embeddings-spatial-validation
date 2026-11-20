"""Random and spatial-block cross-validation with training-size curves."""
from __future__ import annotations

import numpy as np


def random_folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id (0..k-1) for each of n points, balanced and shuffled."""
    rng = np.random.default_rng(seed)
    f = np.arange(n) % k
    rng.shuffle(f)
    return f


def block_ids(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    """Integer id of the square block of side `size` (same units as x and y) holding each point."""
    bx = np.floor((x - x.min()) / size).astype(np.int64)
    by = np.floor((y - y.min()) / size).astype(np.int64)
    return bx * 100_000 + by


def block_folds(x: np.ndarray, y: np.ndarray, size: float, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per point; every point of a block shares its fold, blocks are dealt to folds at random."""
    ids = block_ids(x, y, size)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of_block = dict(zip(uniq[order], np.arange(len(uniq)) % k))
    return np.array([fold_of_block[b] for b in ids])


def subsample_per_class(labels: np.ndarray, idx: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Up to n positions of `idx` per class (all of them when n is None or larger than the class)."""
    if n is None:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels[idx]):
        pool = idx[labels[idx] == c]
        out.append(pool if len(pool) <= n else rng.choice(pool, n, replace=False))
    return np.concatenate(out)


def bootstrap_mean_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of d and its 95% percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, d.size).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
