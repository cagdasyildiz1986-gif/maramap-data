"""
fetch_currents.py — MaraMap v4.0
Copernicus Mediterranean Model: akıntı + çok-derinlik sıcaklık + karışık tabaka derinliği
Dataset: cmems_mod_med_phy-cur_anfc_4.2km_P1D-m  (4.2km, Akdeniz/Ege/Marmara)
Çıktı: currents.json → GitHub maramap-data repo'ya push edilir
"""
import copernicusmarine
import json, os, sys
from datetime import datetime, timedelta, timezone

OUT = os.path.join(os.path.dirname(__file__), '..', 'currents.json')

# Türkiye kıyı bölgesi: Marmara + Ege + Akdeniz
BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.0,
    minimum_longitude = 25.0,
    maximum_longitude = 36.25,
)

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"[{datetime.now(timezone.utc).isoformat()}] Veri çekiliyor — {today}")

result = {'updated': datetime.now(timezone.utc).isoformat(), 'currents': [], 'mld': []}
errors = []

# ── 1. AKINTILAR (günlük, yüzey) ─────────────────────────────
try:
    import numpy as np
    ds_cur = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        variables=["uo","vo"],
        start_datetime=today,
        end_datetime=today,
        minimum_depth=1.0,
        maximum_depth=2.0,
        **BBOX
    )
    # İlk zaman adımını al
    t0  = ds_cur.isel(time=0)
    lats = t0.latitude.values
    lons = t0.longitude.values

    rows = []
    for ilat, lat in enumerate(lats):
        for ilon, lon in enumerate(lons):
            try:
                uo = float(t0['uo'].isel(latitude=ilat, longitude=ilon).values.flat[0])
                vo = float(t0['vo'].isel(latitude=ilat, longitude=ilon).values.flat[0])
                if np.isnan(uo) or np.isnan(vo): continue
                spd = float(np.sqrt(uo**2 + vo**2))
                if spd < 0.02: continue  # çok yavaş akıntıları atla
                import math
                dir_deg = float(math.degrees(math.atan2(uo, vo))) % 360
                rows.append({'lat':round(float(lat),3), 'lng':round(float(lon),3),
                             'uo':round(uo,3), 'vo':round(vo,3),
                             'speed':round(spd,3), 'dir_deg':round(dir_deg,1)})
            except: continue

    result['currents'] = rows
    print(f"  Akıntı: {len(rows)} nokta")
    ds_cur.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] Akıntı: {e}", file=sys.stderr)

# ── 2. KARISIK TABAKA DERİNLİĞİ (Mixed Layer Depth) ─────────
try:
    ds_mld = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-mld_anfc_4.2km_P1D-m",
        variables=["mlotst"],
        start_datetime=today,
        end_datetime=today,
        **BBOX
    )
    t0m = ds_mld.isel(time=0)
    mld_rows = []
    lats_m = t0m.latitude.values
    lons_m = t0m.longitude.values
    for ilat, lat in enumerate(lats_m[::2]):  # her 2. satır (hız için)
        for ilon, lon in enumerate(lons_m[::2]):
            try:
                import numpy as np
                v = float(t0m['mlotst'].isel(latitude=ilat*2, longitude=ilon*2).values.flat[0])
                if np.isnan(v) or v <= 0: continue
                mld_rows.append({'lat':round(float(lat),3), 'lng':round(float(lon),3), 'depth':round(v,1)})
            except: continue
    result['mld'] = mld_rows
    print(f"  MLD: {len(mld_rows)} nokta")
    ds_mld.close()
except Exception as e:
    errors.append(f"MLD: {e}")
    print(f"  [HATA] MLD: {e}", file=sys.stderr)

# ── Kaydet ──────────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',',':'))

kb = os.path.getsize(OUT) // 1024
print(f"Kaydedildi: {OUT} ({kb} KB) — {len(result['currents'])} akıntı, {len(result['mld'])} MLD noktası")
