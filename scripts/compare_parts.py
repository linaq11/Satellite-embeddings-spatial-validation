"""Figure: AlphaEarth minus Sentinel-2 macro-F1 by training size, California (labels seen) against France (independent).

USE: python scripts/compare_parts.py  ->  docs/fig_parts.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embcv.cv import bootstrap_mean_ci  # noqa: E402

SIZES = ["10", "25", "50", "100", "200", "all"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def gap(results):
    r = pd.read_csv(ROOT / "data" / results, dtype={"size": str}).query("scheme == 'block20' and model == 'rf'")
    out = []
    for z in SIZES:
        a = r.query("size == @z and features == 'A'").set_index(["fold", "draw"]).macro_f1
        b = r.query("size == @z and features == 'B'").set_index(["fold", "draw"]).macro_f1
        out.append(bootstrap_mean_ci((a - b.loc[a.index]).to_numpy()))
    return np.array(out)


ca, fr = gap("results.csv"), gap("results_fr.csv")
xs = np.arange(len(SIZES))
fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
for d, col, lab in [(ca, "#c98a10", "California, CDL labels (seen in AlphaEarth training)"),
                    (fr, "#1b8a9a", "France, RPG labels (independent)")]:
    ax.plot(xs, d[:, 0], "-o", color=col, lw=2, ms=4, label=lab)
    ax.fill_between(xs, d[:, 1], d[:, 2], color=col, alpha=0.2, lw=0)
ax.axhline(0, color="0.35", lw=0.8)
ax.set_xticks(xs, SIZES)
ax.set_xlabel("Training points per class")
ax.set_ylabel("Macro-F1, AlphaEarth minus Sentinel-2")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=9.5)
fig.savefig(ROOT / "docs" / "fig_parts.png", bbox_inches="tight", facecolor="white")
print(json.dumps({"california": ca.round(4).tolist(), "france": fr.round(4).tolist()}))
