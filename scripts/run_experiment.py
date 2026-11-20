"""Run the preregistered grid: feature sets x models x validation schemes x training sizes x draws.

Writes data/results.csv (one row per fitted model) and data/folds.json (class counts per spatial fold).
USE: python scripts/run_experiment.py                                   (Part 1, California)
     python scripts/run_experiment.py --sample sample_fr.csv --crs 2154 --tag _fr   (Part 2, France)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embcv.cv import block_folds, random_folds, subsample_per_class  # noqa: E402

K, DRAWS, SIZES = 5, 10, [10, 25, 50, 100, 200, None]


def model(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)   # parallelism is over folds and draws
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def fit_one(X, y, classes, tr, te, size, m, draw):
    idx = subsample_per_class(y, tr, size, seed=draw)
    pred = model(m, draw).fit(X[idx], y[idx]).predict(X[te])
    return len(idx), f1_score(y[te], pred, labels=classes, average="macro", zero_division=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="sample.csv")
    p.add_argument("--crs", type=int, default=32611)
    p.add_argument("--tag", default="")
    p.add_argument("--jobs", type=int, default=12)
    a = p.parse_args()
    s = pd.read_csv(ROOT / "data" / a.sample)
    y = s.crop.to_numpy()
    classes = np.unique(y)
    cols = {"A": [c for c in s.columns if c[0] == "A" and c[1:].isdigit()],
            "B": [c for c in s.columns if c.startswith("S2_")]}
    cols["C"] = cols["A"] + cols["B"]
    X = {k: s[v].to_numpy(dtype=float) for k, v in cols.items()}
    x, yy = Transformer.from_crs(4326, a.crs, always_xy=True).transform(s.lon.to_numpy(), s.lat.to_numpy())
    folds = {"random": random_folds(len(s), K, 0)}
    for km in (10, 20, 40):
        folds[f"block{km}"] = block_folds(np.asarray(x), np.asarray(yy), km * 1000.0, K, 0)
    diag = {k: {int(f): {str(c): int(((v == f) & (y == c)).sum()) for c in classes} for f in range(K)}
            for k, v in folds.items() if k != "random"}
    (ROOT / "data" / f"folds{a.tag}.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")

    plan = []
    for scheme in ("random", "block20"):
        for feats in ("A", "B", "C"):
            for m in ("rf", "logit"):
                for size in SIZES:
                    plan.append((scheme, feats, m, size))
    for scheme in ("block10", "block40"):
        for feats in ("A", "B"):
            for size in (25, None):
                plan.append((scheme, feats, "rf", size))

    rows, t0 = [], time.time()
    for i, (scheme, feats, m, size) in enumerate(plan):
        f = folds[scheme]
        # a fold with no points (40 km blocks in the French box, which only holds four) is left out
        jobs = [(fold, draw) for fold in range(K) if (f == fold).any() for draw in range(DRAWS)]
        res = Parallel(n_jobs=a.jobs)(delayed(fit_one)(X[feats], y, classes, np.where(f != fold)[0], np.where(f == fold)[0],
                                                       size, m, draw) for fold, draw in jobs)
        for (fold, draw), (n_train, f1) in zip(jobs, res):
            te = np.where(f == fold)[0]
            rows.append({"scheme": scheme, "features": feats, "model": m, "size": size if size else "all",
                         "fold": fold, "draw": draw, "n_train": n_train, "n_test": len(te),
                         "test_classes": len(np.unique(y[te])), "macro_f1": f1})
        pd.DataFrame(rows).to_csv(ROOT / "data" / f"results{a.tag}.csv", index=False)
        print(f"{i + 1}/{len(plan)} {scheme} {feats} {m} {size}: "
              f"{np.mean([r['macro_f1'] for r in rows[-K * DRAWS:]]):.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
