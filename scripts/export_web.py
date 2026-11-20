"""Small data files for the dashboard, one set per part (docs/data/{ca,fr}). USE: python scripts/export_web.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embcv.cv import block_folds, block_ids, random_folds  # noqa: E402

PARTS = {"ca": ("sample.csv", "classes.json", "summary.json", "results.csv", 32611),
         "fr": ("sample_fr.csv", "classes_fr.json", "summary_fr.json", "results_fr.csv", 2154)}


def export(part, sample, classes_f, summary_f, results_f, crs):
    out = ROOT / "docs" / "data" / part
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "data" / sample)
    classes = json.loads((ROOT / "data" / classes_f).read_text(encoding="utf-8"))["classes"]
    code_ix = {str(c["code"]): i for i, c in enumerate(classes)}
    to_p = Transformer.from_crs(4326, crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    x, y = (np.asarray(v) for v in to_p.transform(s.lon.to_numpy(), s.lat.to_numpy()))
    fr, fb = random_folds(len(s), 5, 0), block_folds(x, y, 20_000.0, 5, 0)
    pts = np.column_stack([s.lon.round(5), s.lat.round(5), s.crop.astype(str).map(code_ix), fr, fb]).tolist()
    (out / "points.json").write_text(json.dumps(pts, separators=(",", ":")), encoding="utf-8")
    ids = block_ids(x, y, 20_000.0)
    blocks = []
    for b in np.unique(ids):
        bx, by = divmod(int(b), 100_000)
        xs = [x.min() + bx * 20_000, x.min() + (bx + 1) * 20_000]
        ys = [y.min() + by * 20_000, y.min() + (by + 1) * 20_000]
        lon, lat = to_ll.transform([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]])
        blocks.append({"fold": int(fb[ids == b][0]), "ring": [[round(o, 5), round(t, 5)] for o, t in zip(lon, lat)]})
    (out / "blocks.json").write_text(json.dumps(blocks, separators=(",", ":")), encoding="utf-8")
    summ = json.loads((ROOT / "data" / summary_f).read_text(encoding="utf-8"))
    r = pd.read_csv(ROOT / "data" / results_f, dtype={"size": str})
    curves = {}
    for (sch, m, f), g in r.groupby(["scheme", "model", "features"]):
        if sch in ("random", "block20"):
            q = g.groupby("size").macro_f1.mean()
            curves[f"{sch}|{m}|{f}"] = {z: [round(float(q.loc[z]), 4)] for z in ["10", "25", "50", "100", "200", "all"]}
    web = {"classes": [c["name"] for c in classes], "n_points": len(s), "curves": curves, "H1": summ["H1"], "H2": summ["H2"]}
    (out / "summary.json").write_text(json.dumps(web, indent=1, ensure_ascii=False), encoding="utf-8")
    print(part, {f.name: round(f.stat().st_size / 1024) for f in out.iterdir()}, len(blocks), "blocks")


if __name__ == "__main__":
    for part, args in PARTS.items():
        export(part, *args)
