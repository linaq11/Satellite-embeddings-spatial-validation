import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embcv.cv import block_folds, block_ids, bootstrap_mean_ci, random_folds, subsample_per_class  # noqa: E402


def test_random_folds_are_balanced():
    f = random_folds(103, 5, seed=1)
    counts = np.bincount(f)
    assert counts.max() - counts.min() <= 1 and set(f) == set(range(5))


def test_points_of_one_block_share_a_fold():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 100_000, 2000), rng.uniform(0, 100_000, 2000)
    f = block_folds(x, y, 20_000, k=5, seed=0)
    ids = block_ids(x, y, 20_000)
    for b in np.unique(ids):
        assert len(np.unique(f[ids == b])) == 1
    assert len(np.unique(f)) == 5


def test_block_ids_split_at_the_block_edge():
    ids = block_ids(np.array([0.0, 19_999.0, 20_000.0]), np.array([0.0, 0.0, 0.0]), 20_000)
    assert ids[0] == ids[1] != ids[2]


def test_subsample_per_class_caps_each_class():
    labels = np.array([0] * 50 + [1] * 5)
    idx = np.arange(55)
    s = subsample_per_class(labels, idx, 10, seed=0)
    assert (labels[s] == 0).sum() == 10 and (labels[s] == 1).sum() == 5
    assert len(subsample_per_class(labels, idx, None, seed=0)) == 55


def test_bootstrap_interval_contains_the_mean_and_detects_a_shift():
    rng = np.random.default_rng(0)
    m, lo, hi = bootstrap_mean_ci(rng.normal(0.1, 0.05, 50))
    assert lo < m < hi and lo > 0
    m0, lo0, hi0 = bootstrap_mean_ci(rng.normal(0.0, 0.05, 50))
    assert lo0 < 0 < hi0
