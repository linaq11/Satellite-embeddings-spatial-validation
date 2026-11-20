"""Run the preregistered grid: feature sets x models x validation schemes x training sizes x draws.

Writes data/results.csv (one row per fitted model) and data/folds.json (class counts per spatial fold).
USE: python scripts/run_experiment.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
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
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def main() -> None:
    s = pd.read_csv(ROOT / "data" / "sample.csv")
    y = s.crop.to_numpy()
    classes = np.unique(y)
    cols = {"A": [c for c in s.columns if c[0] == "A" and c[1:].isdigit()],
            "B": [c for c in s.columns if c.startswith("S2_")]}
    cols["C"] = cols["A"] + cols["B"]
    X = {k: s[v].to_numpy(dtype=float) for k, v in cols.items()}
    x, yy = Transformer.from_crs(4326, 32611, always_xy=True).transform(s.lon.to_numpy(), s.lat.to_numpy())
    folds = {"random": random_folds(len(s), K, 0)}
    for km in (10, 20, 40):
        folds[f"block{km}"] = block_folds(np.asarray(x), np.asarray(yy), km * 1000.0, K, 0)
    diag = {k: {int(f): {int(c): int(((v == f) & (y == c)).sum()) for c in classes} for f in range(K)}
            for k, v in folds.items() if k != "random"}
    (ROOT / "data" / "folds.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")

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
        for fold in range(K):
            tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
            for draw in range(DRAWS):
                idx = subsample_per_class(y, tr, size, seed=draw)
                clf = model(m, draw).fit(X[feats][idx], y[idx])
                pred = clf.predict(X[feats][te])
                rows.append({"scheme": scheme, "features": feats, "model": m, "size": size if size else "all",
                             "fold": fold, "draw": draw, "n_train": len(idx), "n_test": len(te),
                             "test_classes": len(np.unique(y[te])),
                             "macro_f1": f1_score(y[te], pred, labels=classes, average="macro", zero_division=0)})
        pd.DataFrame(rows).to_csv(ROOT / "data" / "results.csv", index=False)
        print(f"{i + 1}/{len(plan)} {scheme} {feats} {m} {size}: "
              f"{np.mean([r['macro_f1'] for r in rows[-K * DRAWS:]]):.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
