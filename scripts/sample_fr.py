"""Part 2: draw the French sample from the RPG 2023 parcels and extract both feature sets (PREREGISTRATION.md).

Writes data/sample_fr.csv and data/classes_fr.json. USE: python scripts/sample_fr.py --project my-ee-project
"""
import argparse
import json
import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample import s2_quarters  # noqa: E402  same Sentinel-2 composites as Part 1

BOX = [1.2, 48.0, 2.2, 48.6]
PER_CLASS, N_CLASSES, SEED, MIN_HA = 600, 8, 42, 1.0


def crop_names():
    r = requests.get("https://data.geopf.fr/wfs/ows", params={"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                     "TYPENAMES": "RPG.2023:codes_cultures", "OUTPUTFORMAT": "application/json"}, timeout=120)
    r.raise_for_status()
    props = [f["properties"] for f in r.json()["features"]]
    code = next(k for k in props[0] if "code" in k.lower() and "group" not in k.lower())
    name = next(k for k in props[0] if "libell" in k.lower() and "group" not in k.lower())
    return {p[code]: p[name] for p in props}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    a = p.parse_args()
    parcels = pd.concat([gpd.read_file(f) for f in sorted((ROOT / "data" / "rpg_raw").glob("page_*.geojson"))], ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=2154)
    big = parcels[parcels.surf_parc >= MIN_HA]
    area = big.groupby("code_cultu").surf_parc.sum().sort_values(ascending=False)
    classes = list(area.index[:N_CLASSES])
    names = crop_names()
    meta = {"classes": [{"code": c, "name": names.get(c, c), "declared_ha": round(float(area[c]), 1),
                         "parcels": int((big.code_cultu == c).sum())} for c in classes],
            "next_classes": [{"code": c, "name": names.get(c, c), "declared_ha": round(float(area[c]), 1)} for c in area.index[N_CLASSES:N_CLASSES + 4]],
            "parcels_in_box": len(parcels), "parcels_1ha_or_more": len(big)}
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    pick = pd.concat([big[big.code_cultu == c].sample(n=min(PER_CLASS, int((big.code_cultu == c).sum())), random_state=SEED)
                      for c in classes])
    pts = pick.copy()
    pts["geometry"] = pick.geometry.representative_point()      # always inside the parcel
    pts = pts.to_crs(4326)

    ee.Initialize(project=a.project)
    box = ee.Geometry.Rectangle(BOX)
    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2023-01-01", "2024-01-01").filterBounds(box).mosaic()
    feats = emb.addBands(s2_quarters(box))
    rows = []
    for c in classes:
        sub = pts[pts.code_cultu == c]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([g.x, g.y]), {"crop": c, "id_parcel": i})
                                   for g, i in zip(sub.geometry, sub.id_parcel)])
        for f in feats.reduceRegions(fc, ee.Reducer.first(), scale=10, tileScale=8).getInfo()["features"]:
            lon, lat = f["geometry"]["coordinates"]
            rows.append({"lon": lon, "lat": lat, **f["properties"]})
        print(c, names.get(c, c), len(sub), flush=True)
    df = pd.DataFrame(rows)
    df["crop_name"] = df.crop.map(lambda c: names.get(c, c))
    df.to_csv(ROOT / "data" / "sample_fr.csv", index=False)
    (ROOT / "data" / "classes_fr.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = [c for c in df.columns if (c[0] == "A" and c[1:].isdigit()) or c.startswith("S2_")]
    print(f"rows {len(df)}, feature columns {len(cols)}, rows with a missing value {int(df[cols].isna().any(axis=1).sum())}")


if __name__ == "__main__":
    main()
