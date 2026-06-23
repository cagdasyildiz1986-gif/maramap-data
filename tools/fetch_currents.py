"""
fetch_currents.py — MaraMap v4.1 (vectorized)
Copernicus Mediterranean Model: yüzey akıntısı + karışık tabaka derinliği (MLD)
Dataset: cmems_mod_med_phy-cur_anfc_4.2km_P1D-m  (4.2km — Akdeniz/Ege/Marmara)
Çıktı: currents.json → GitHub maramap-data repo'ya push edilir

HIZ: nokta nokta okuma yerine numpy ile toplu (vectorized) okuma.
     Binlerce hücre saniyeler içinde işlenir.
"""
import copernicusmarine
import json, os, sys
import numpy as np
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), '..', 'currents.json')

# Türkiye kıyı bölgesi: Marmara + Ege + Akdeniz
BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.0,
    minimum_longitude = 25.0,
    maximum_longitude = 36.25,
)

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"[{datetime.now(timezone.utc).isoformat()}] Veri cekiliyor — {today}")

result = {'updated': datetime.now(timezone.utc).isoformat(), 'currents': [], 'mld': []}
errors = []

# Veri inceltme: her N. hücre (4.2km ince; 2 = ~8.4km, harita için yeterli)
STRIDE = 2

# ── 1. AKINTILAR (yüzey ~1m) ─────────────────────────────────
try:
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        variables=["uo", "vo"],
        start_datetime=today,
        end_datetime=today,
        minimum_depth=1.0,
        maximum_depth=1.5,
        **BBOX
    )

    da_uo = ds["uo"].isel(time=0)
    da_vo = ds["vo"].isel(time=0)
    if "depth" in da_uo.dims:
        da_uo = da_uo.isel(depth=0)
        da_vo = da_vo.isel(depth=0)

    lats = ds["latitude"].values[::STRIDE]
    lons = ds["longitude"].values[::STRIDE]

    # Tüm grid tek seferde numpy'a (vectorized)
    uo = da_uo.values[::STRIDE, ::STRIDE]
    vo = da_vo.values[::STRIDE, ::STRIDE]

    spd = np.sqrt(uo**2 + vo**2)
    dir_deg = (np.degrees(np.arctan2(uo, vo)) + 360) % 360

    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            u, v, s = uo[i, j], vo[i, j], spd[i, j]
            if np.isnan(u) or np.isnan(v) or s < 0.02:
                continue
            rows.append({
                'lat': round(float(lats[i]), 3),
                'lng': round(float(lons[j]), 3),
                'uo': round(float(u), 3),
                'vo': round(float(v), 3),
                'speed': round(float(s), 3),
                'dir_deg': round(float(dir_deg[i, j]), 1),
            })

    result['currents'] = rows
    print(f"  Akinti: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] Akinti: {e}", file=sys.stderr)

# ── 2. KARISIK TABAKA DERINLIGI (MLD) ────────────────────────
try:
    dsm = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-mld_anfc_4.2km_P1D-m",
        variables=["mlotst"],
        start_datetime=today,
        end_datetime=today,
        **BBOX
    )

    da_m = dsm["mlotst"].isel(time=0)
    if "depth" in da_m.dims:
        da_m = da_m.isel(depth=0)

    mlats = dsm["latitude"].values[::STRIDE]
    mlons = dsm["longitude"].values[::STRIDE]
    mld = da_m.values[::STRIDE, ::STRIDE]

    mld_rows = []
    for i in range(len(mlats)):
        for j in range(len(mlons)):
            v = mld[i, j]
            if np.isnan(v) or v <= 0:
                continue
            mld_rows.append({
                'lat': round(float(mlats[i]), 3),
                'lng': round(float(mlons[j]), 3),
                'depth': round(float(v), 1),
            })

    result['mld'] = mld_rows
    print(f"  MLD: {len(mld_rows)} nokta")
    dsm.close()
except Exception as e:
    errors.append(f"MLD: {e}")
    print(f"  [HATA] MLD: {e}", file=sys.stderr)

# ── Kaydet ───────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"Kaydedildi: {OUT} ({kb} KB) — "
      f"{len(result['currents'])} akinti, {len(result['mld'])} MLD noktasi")
