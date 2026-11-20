"""Download the RPG 2023 parcels of the Beauce box from the Géoplateforme WFS, page by page (resumable).

Writes data/rpg_raw/page_XXX.geojson in Lambert-93 (EPSG:2154). USE: python scripts/download_rpg.py
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rpg_raw"
URL = "https://data.geopf.fr/wfs/ows"
BBOX = "48.0,1.2,48.6,2.2,urn:ogc:def:crs:EPSG::4326"       # lat/lon order for EPSG:4326 in WFS 2.0
PAGE = 5000


def params(**extra):
    return {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": "RPG.2023:parcelles_graphiques",
            "BBOX": BBOX, **extra}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hits = requests.get(URL, params=params(RESULTTYPE="hits"), timeout=120).text
    total = int(hits.split('numberMatched="')[1].split('"')[0])
    print("parcels in the box:", total, flush=True)
    for k, start in enumerate(range(0, total, PAGE)):
        dst = OUT / f"page_{k:03d}.geojson"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        for attempt in range(5):
            r = requests.get(URL, params=params(COUNT=PAGE, STARTINDEX=start, SORTBY="id_parcel", SRSNAME="EPSG:2154",
                                                OUTPUTFORMAT="application/json"), timeout=300)
            if r.ok:
                break
            time.sleep(20 * (attempt + 1))
        r.raise_for_status()
        dst.write_bytes(r.content)
        print(f"page {k}: {len(json.loads(r.content)['features'])} parcels, {len(r.content) / 1e6:.1f} MB", flush=True)
    n = sum(len(json.loads(p.read_bytes())["features"]) for p in OUT.glob("page_*.geojson"))
    print(f"downloaded {n} of {total} parcels, {sum(p.stat().st_size for p in OUT.glob('page_*.geojson')) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
