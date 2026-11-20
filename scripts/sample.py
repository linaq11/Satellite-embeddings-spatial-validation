"""Draw the labelled sample and extract both feature sets in Earth Engine (PREREGISTRATION.md).

Writes data/sample.csv: one row per point with lon, lat, crop code and name, the 64 AlphaEarth
bands (A00..A63) and the 40 Sentinel-2 quarterly medians (S2_q1_B2 ... S2_q4_B12).

USE: python scripts/sample.py --project my-ee-project
"""
import argparse
import json
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOX = [-120.6, 35.8, -118.9, 37.1]
YEAR, PER_CLASS, N_CLASSES, SEED, MIN_CONF = 2023, 600, 8, 42, 90
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def cdl_labels(box):
    cdl = ee.ImageCollection("USDA/NASS/CDL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").first()
    ok = cdl.select("cultivated").eq(2).And(cdl.select("confidence").gte(MIN_CONF))
    return cdl.select("cropland").updateMask(ok).rename("crop"), cdl


def top_classes(crop, box):
    hist = crop.reduceRegion(ee.Reducer.frequencyHistogram(), box, 30, maxPixels=1e10, tileScale=8).get("crop").getInfo()
    counts = sorted(((int(k), int(v)) for k, v in hist.items()), key=lambda kv: -kv[1])
    return counts


def s2_quarters(box):
    def mask(img):
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(clear).select(S2_BANDS)
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box).map(mask)
    out = []
    for q, (a, b) in enumerate([("01-01", "04-01"), ("04-01", "07-01"), ("07-01", "10-01"), ("10-01", "01-01")], start=1):
        end = f"{YEAR + 1}-{b}" if q == 4 else f"{YEAR}-{b}"
        med = col.filterDate(f"{YEAR}-{a}", end).median()
        out.append(med.rename([f"S2_q{q}_{bd}" for bd in S2_BANDS]))
    return ee.Image.cat(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    a = p.parse_args()
    ee.Initialize(project=a.project)
    box = ee.Geometry.Rectangle(BOX)
    crop, cdl = cdl_labels(box)
    counts = top_classes(crop, box)
    classes = [c for c, _ in counts[:N_CLASSES]]
    # the CDL stores its legend as two comma-separated strings
    names = {int(v): n for v, n in zip(cdl.get("cropland_class_values").getInfo().split(","),
                                       cdl.get("cropland_class_names").getInfo().split(","))}
    meta = {"classes": [{"code": c, "name": names.get(c, str(c)), "cells": n} for c, n in counts[:N_CLASSES]],
            "next_classes": [{"code": c, "name": names.get(c, str(c)), "cells": n} for c, n in counts[N_CLASSES:N_CLASSES + 4]]}
    print(json.dumps(meta, indent=1))

    pts = crop.stratifiedSample(numPoints=0, classBand="crop", region=box, scale=30, seed=SEED, geometries=True,
                                classValues=classes, classPoints=[PER_CLASS] * N_CLASSES, tileScale=8)
    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").filterBounds(box).mosaic()
    feats = emb.addBands(s2_quarters(box))
    rows = []
    for c in classes:
        sub = pts.filter(ee.Filter.eq("crop", c))
        sampled = feats.reduceRegions(sub, ee.Reducer.first(), scale=10, tileScale=8).getInfo()["features"]
        for f in sampled:
            lon, lat = f["geometry"]["coordinates"]
            rows.append({"lon": lon, "lat": lat, **f["properties"]})
        print(c, names.get(c), len(sampled), flush=True)
    df = pd.DataFrame(rows)
    df["crop_name"] = df.crop.map(names)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "sample.csv", index=False)
    (out / "classes.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    emb_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    s2_cols = [c for c in df.columns if c.startswith("S2_")]
    print(f"rows {len(df)}, embedding columns {len(emb_cols)}, S2 columns {len(s2_cols)}, "
          f"rows with any missing value {int(df[emb_cols + s2_cols].isna().any(axis=1).sum())}")


if __name__ == "__main__":
    main()
