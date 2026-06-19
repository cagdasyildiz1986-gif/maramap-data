"""
fetch_currents.py — MaraMap v4.2
Copernicus Mediterranean Model — günlük veri çekimi

Çekilen veriler:
  1. Yüzey akıntısı (uo, vo) — cmems_mod_med_phy-cur
  2. Karışık tabaka derinliği (MLD) — cmems_mod_med_phy-mld
  3. Dip sıcaklığı (bottomT) — cmems_mod_med_phy-tem  [YENİ]
  4. Yüzey tuzluluğu (so) — cmems_mod_med_phy-sal    [YENİ]

Kapsam: Türkiye kıyıları — Marmara + Ege + Akdeniz
"""
import copernicusmarine
import json, os, sys
import numpy as np
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), '..', 'currents.json')

# Türkiye kıyı bölgesi: Marmara + Ege + Akdeniz + Karadeniz batı
BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.5,
    minimum_longitude = 25.0,
    maximum_longitude = 36.5,
)

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"[{datetime.now(timezone.utc).isoformat()}] Veri cekiliyor — {today}")

result = {
    'updated':  datetime.now(timezone.utc).isoformat(),
    'currents': [],
    'mld':      [],
    'bottom_t': [],   # Dip sıcaklığı
    'salinity': [],   # Yüzey tuzluluğu
}
errors = []
STRIDE = 2  # 4.2km × 2 = ~8.4km çözünürlük

# ── 1. YÜzey Akıntısı ────────────────────────────────────────
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
    uo   = da_uo.values[::STRIDE, ::STRIDE]
    vo   = da_vo.values[::STRIDE, ::STRIDE]
    spd  = np.sqrt(uo**2 + vo**2)
    dir_deg = (np.degrees(np.arctan2(uo, vo)) + 360) % 360

    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            u, v, s = uo[i,j], vo[i,j], spd[i,j]
            if np.isnan(u) or np.isnan(v) or s < 0.02: continue
            rows.append({
                'lat': round(float(lats[i]), 3),
                'lng': round(float(lons[j]), 3),
                'uo':  round(float(u), 3),
                'vo':  round(float(v), 3),
                'speed':   round(float(s), 3),
                'dir_deg': round(float(dir_deg[i,j]), 1),
            })
    result['currents'] = rows
    print(f"  Akinti: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] Akinti: {e}", file=sys.stderr)

# ── 2. Karışık Tabaka Derinliği (MLD) ────────────────────────
try:
    dsm = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-mld_anfc_4.2km_P1D-m",
        variables=["mlotst"],
        start_datetime=today,
        end_datetime=today,
        **BBOX
    )
    da_m  = dsm["mlotst"].isel(time=0)
    if "depth" in da_m.dims: da_m = da_m.isel(depth=0)
    mlats = dsm["latitude"].values[::STRIDE]
    mlons = dsm["longitude"].values[::STRIDE]
    mld   = da_m.values[::STRIDE, ::STRIDE]
    mld_rows = []
    for i in range(len(mlats)):
        for j in range(len(mlons)):
            v = mld[i,j]
            if np.isnan(v) or v <= 0: continue
            mld_rows.append({
                'lat':   round(float(mlats[i]), 3),
                'lng':   round(float(mlons[j]), 3),
                'depth': round(float(v), 1),
            })
    result['mld'] = mld_rows
    print(f"  MLD: {len(mld_rows)} nokta")
    dsm.close()
except Exception as e:
    errors.append(f"MLD: {e}")
    print(f"  [HATA] MLD: {e}", file=sys.stderr)

# ── 3. DİP SICAKLIĞI (bottomT) ── [YENİ] ────────────────────
try:
    dst = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        variables=["bottomT"],
        start_datetime=today,
        end_datetime=today,
        **BBOX
    )
    da_bt  = dst["bottomT"].isel(time=0)
    blats  = dst["latitude"].values[::STRIDE]
    blons  = dst["longitude"].values[::STRIDE]
    bt_arr = da_bt.values[::STRIDE, ::STRIDE]
    bt_rows = []
    for i in range(len(blats)):
        for j in range(len(blons)):
            v = bt_arr[i,j]
            if np.isnan(v) or v < -2 or v > 40: continue
            bt_rows.append({
                'lat': round(float(blats[i]), 3),
                'lng': round(float(blons[j]), 3),
                'bt':  round(float(v), 2),   # Dip sıcaklığı °C
            })
    result['bottom_t'] = bt_rows
    print(f"  Dip sicakligi: {len(bt_rows)} nokta")
    dst.close()
except Exception as e:
    errors.append(f"BottomT: {e}")
    print(f"  [HATA] Dip sicakligi: {e}", file=sys.stderr)

# ── 4. YÜZEY TUZLULUĞU (so) ── [YENİ] ───────────────────────
try:
    dss = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
        variables=["so"],
        start_datetime=today,
        end_datetime=today,
        minimum_depth=1.0,
        maximum_depth=1.5,
        **BBOX
    )
    da_s  = dss["so"].isel(time=0)
    if "depth" in da_s.dims: da_s = da_s.isel(depth=0)
    slats  = dss["latitude"].values[::STRIDE]
    slons  = dss["longitude"].values[::STRIDE]
    sal    = da_s.values[::STRIDE, ::STRIDE]
    sal_rows = []
    for i in range(len(slats)):
        for j in range(len(slons)):
            v = sal[i,j]
            if np.isnan(v) or v < 1 or v > 45: continue
            sal_rows.append({
                'lat': round(float(slats[i]), 3),
                'lng': round(float(slons[j]), 3),
                'sal': round(float(v), 2),   # PSU
            })
    result['salinity'] = sal_rows
    print(f"  Tuzluluk: {len(sal_rows)} nokta")
    dss.close()
except Exception as e:
    errors.append(f"Salinity: {e}")
    print(f"  [HATA] Tuzluluk: {e}", file=sys.stderr)

# ── Kaydet ───────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"\nKaydedildi: {OUT} ({kb} KB)")
print(f"  {len(result['currents'])} akinti")
print(f"  {len(result['mld'])} MLD")
print(f"  {len(result['bottom_t'])} dip sicakligi")
print(f"  {len(result['salinity'])} tuzluluk")
if errors:
    print(f"  HATALAR: {errors}")
