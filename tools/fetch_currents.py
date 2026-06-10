#!/usr/bin/env python3
"""
MaraMap — Copernicus Marmara yüzey akıntısı çekici
GitHub Actions üzerinde günlük çalışır; currents.json üretir.

Gerekli ortam değişkenleri (GitHub Secrets):
  COPERNICUSMARINE_SERVICE_USERNAME
  COPERNICUSMARINE_SERVICE_PASSWORD
"""
import json, os, sys, datetime

import copernicusmarine
import xarray as xr

DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
BBOX = dict(minimum_longitude=26.0, maximum_longitude=30.0,
            minimum_latitude=40.0, maximum_latitude=41.3)
OUT = os.path.join(os.path.dirname(__file__), "..", "currents.json")
NC  = "/tmp/marmara_cur.nc"

def main():
    today = datetime.date.today()
    start = f"{today}T00:00:00"

    copernicusmarine.subset(
        dataset_id=DATASET,
        variables=["uo", "vo"],
        start_datetime=start, end_datetime=start,
        minimum_depth=1.0, maximum_depth=1.5,
        output_filename=NC, overwrite=True,
        **BBOX,
    )

    ds = xr.open_dataset(NC)
    uo = ds["uo"].squeeze()
    vo = ds["vo"].squeeze()

    cells = []
    lats = ds["latitude"].values
    lngs = ds["longitude"].values
    # 4.2 km ızgarayı ~0.1 dereceye seyrelt (her 3. nokta)
    for i in range(0, len(lats), 3):
        for j in range(0, len(lngs), 3):
            u = float(uo.values[i, j]); v = float(vo.values[i, j])
            if u != u or v != v:          # NaN = kara
                continue
            cells.append({
                "lat": round(float(lats[i]), 3),
                "lng": round(float(lngs[j]), 3),
                "uo":  round(u, 4),
                "vo":  round(v, 4),
            })

    out = {
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        "dataset": DATASET,
        "count": len(cells),
        "cells": cells,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"OK: {len(cells)} hucre -> currents.json")

if __name__ == "__main__":
    sys.exit(main())

