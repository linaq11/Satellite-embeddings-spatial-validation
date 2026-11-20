"""Apply the preregistered decision rules to data/results.csv and draw the figures.

Writes data/summary.json and docs/fig_*.png. USE: python scripts/analyze.py
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
COL = {"A": "#c98a10", "B": "#1b8a9a", "C": "#5a6b62"}
LAB = {"A": "AlphaEarth embeddings (64)", "B": "Sentinel-2 quarterly medians (40)", "C": "Both (104)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def paired(r, scheme, model, size, fa, fb):
    """Paired differences fa minus fb over the same fold and draw."""
    k = ["fold", "draw"]
    a = r.query("scheme == @scheme and model == @model and size == @size and features == @fa").set_index(k).macro_f1
    b = r.query("scheme == @scheme and model == @model and size == @size and features == @fb").set_index(k).macro_f1
    return (a - b.loc[a.index]).to_numpy()


def main() -> None:
    r = pd.read_csv(ROOT / "data" / "results.csv", dtype={"size": str})
    out = {"means": {}, "H1": {}, "H2": {}, "sensitivity": {}}
    g = r.groupby(["scheme", "model", "features", "size"]).macro_f1.mean()
    out["means"] = {f"{s}|{m}|{f}|{z}": round(float(v), 4) for (s, m, f, z), v in g.items()}

    # H1: spatial validation, RF, A minus B at 10 and 25 per class
    ok = True
    for z in ("10", "25"):
        mean, lo, hi = bootstrap_mean_ci(paired(r, "block20", "rf", z, "A", "B"))
        out["H1"][z] = {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        ok &= mean >= 0.05 and lo > 0
    out["H1"]["holds"] = bool(ok)

    # H2: all training points, random minus spatial, each feature set and model
    h2_all = True
    for f in ("A", "B", "C"):
        for m in ("rf", "logit"):
            k = ["fold", "draw"]
            a = r.query("scheme == 'random' and model == @m and size == 'all' and features == @f").set_index(k).macro_f1
            b = r.query("scheme == 'block20' and model == @m and size == 'all' and features == @f").set_index(k).macro_f1
            # folds differ between schemes, so pair by draw over the fold-averaged score
            d = (a.groupby("draw").mean() - b.groupby("draw").mean()).to_numpy()
            mean, lo, hi = bootstrap_mean_ci(d)
            holds = mean >= 0.02 and lo > 0
            # the draws only vary the forest's seed, so also resample the five folds of each scheme (unpaired)
            fa, fb = a.groupby("fold").mean().to_numpy(), b.groupby("fold").mean().to_numpy()
            rng = np.random.default_rng(0)
            bs = [rng.choice(fa, fa.size).mean() - rng.choice(fb, fb.size).mean() for _ in range(2000)]
            out["H2"][f"{f}|{m}"] = {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "holds": bool(holds),
                                     "ci95_folds": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
            h2_all &= holds
    out["H2"]["holds_for_all"] = bool(h2_all)

    for sch in ("block10", "block20", "block40"):
        for z in ("25", "all"):
            if ((r.scheme == sch) & (r["size"] == z) & (r.model == "rf")).any():
                d = paired(r, sch, "rf", z, "A", "B")
                out["sensitivity"][f"{sch}|{z}"] = {"A_minus_B": round(float(d.mean()), 4),
                                                   "A": round(float(g[(sch, "rf", "A", z)]), 4), "B": round(float(g[(sch, "rf", "B", z)]), 4)}
    (ROOT / "data" / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("H1", "H2", "sensitivity")}, indent=1))

    # Figure 1: learning curves, random forest, random against spatial validation
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    xs = np.arange(len(SIZES))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    for f in ("A", "B", "C"):
        for sch, ls in (("random", "--"), ("block20", "-")):
            ys = [g[(sch, "rf", f, z)] for z in SIZES]
            ax.plot(xs, ys, ls, color=COL[f], lw=2, marker="o", ms=4,
                    label=f"{LAB[f]}, {'random' if sch == 'random' else 'spatial 20 km'}")
    ax.set_xticks(xs, ["10", "25", "50", "100", "200", "all"])
    ax.set_xlabel("Training points per class")
    ax.set_ylabel("Macro-F1 (random forest)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(docs / "fig_learning_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 2: inflation, random minus spatial with all training points
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    keys = [(f, m) for f in ("A", "B", "C") for m in ("rf", "logit")]
    vals = [out["H2"][f"{f}|{m}"]["mean_diff"] for f, m in keys]
    errs = np.array([[v - out["H2"][f"{f}|{m}"]["ci95_folds"][0], out["H2"][f"{f}|{m}"]["ci95_folds"][1] - v] for (f, m), v in zip(keys, vals)]).T
    ax.bar(range(len(keys)), vals, color=[COL[f] for f, _ in keys], yerr=errs, capsize=3)
    ax.set_xticks(range(len(keys)), [f"{f}\n{'RF' if m == 'rf' else 'Logistic'}" for f, m in keys])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("Macro-F1, random minus spatial validation")
    fig.savefig(docs / "fig_inflation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
