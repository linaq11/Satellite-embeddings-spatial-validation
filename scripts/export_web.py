"""Small data files for the dashboard (docs/data). USE: python scripts/export_web.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embcv.cv import block_folds, block_ids, random_folds  # noqa: E402

OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(ROOT / "data" / "sample.csv")
classes = json.loads((ROOT / "data" / "classes.json").read_text(encoding="utf-8"))["classes"]
code_ix = {c["code"]: i for i, c in enumerate(classes)}
to_utm = Transformer.from_crs(4326, 32611, always_xy=True)
to_ll = Transformer.from_crs(32611, 4326, always_xy=True)
x, y = (np.asarray(v) for v in to_utm.transform(s.lon.to_numpy(), s.lat.to_numpy()))
fr = random_folds(len(s), 5, 0)
fb = block_folds(x, y, 20_000.0, 5, 0)
pts = np.column_stack([s.lon.round(5), s.lat.round(5), s.crop.map(code_ix), fr, fb]).tolist()
(OUT / "points.json").write_text(json.dumps(pts, separators=(",", ":")), encoding="utf-8")

# 20 km blocks that hold at least one point, with their fold
ids = block_ids(x, y, 20_000.0)
x0, y0 = x.min(), y.min()
blocks = []
for b in np.unique(ids):
    bx, by = divmod(int(b), 100_000)
    xs = [x0 + bx * 20_000, x0 + (bx + 1) * 20_000]
    ys = [y0 + by * 20_000, y0 + (by + 1) * 20_000]
    lon, lat = to_ll.transform([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]])
    blocks.append({"fold": int(fb[ids == b][0]), "ring": [[round(o, 5), round(t, 5)] for o, t in zip(lon, lat)]})
(OUT / "blocks.json").write_text(json.dumps(blocks, separators=(",", ":")), encoding="utf-8")

summ = json.loads((ROOT / "data" / "summary.json").read_text(encoding="utf-8"))
r = pd.read_csv(ROOT / "data" / "results.csv", dtype={"size": str})
curves = {}
for (sch, m, f), g in r.groupby(["scheme", "model", "features"]):
    if sch in ("random", "block20"):
        q = g.groupby("size").macro_f1.agg(["mean", "std"])
        curves[f"{sch}|{m}|{f}"] = {z: [round(float(q.loc[z, "mean"]), 4), round(float(q.loc[z, "std"]), 4)]
                                    for z in ["10", "25", "50", "100", "200", "all"]}
web = {"classes": [c["name"] for c in classes], "n_points": len(s), "curves": curves,
       "H1": summ["H1"], "H2": summ["H2"], "sensitivity": summ["sensitivity"]}
(OUT / "summary.json").write_text(json.dumps(web, indent=1), encoding="utf-8")
print({f.name: round(f.stat().st_size / 1024) for f in OUT.iterdir()}, len(blocks), "blocks")
