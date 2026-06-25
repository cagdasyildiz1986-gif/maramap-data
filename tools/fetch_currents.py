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

# ── KARADENİZ (Black Sea) modeli — Akdeniz modelinin kapsamadığı
#    Karadeniz kıyısı + İstanbul Boğazı + iç Marmara kuzeyi ──
BBOX_BLK = dict(
    minimum_latitude  = 40.5,
    maximum_latitude  = 43.2,
    minimum_longitude = 27.0,
    maximum_longitude = 42.0,
)
DS_CUR_BLK = "cmems_mod_blk_phy-cur_anfc_2.5km_P1D-m"   # uo, vo
DS_TEM_BLK = "cmems_mod_blk_phy-tem_anfc_2.5km_P1D-m"   # bottomT
DS_SAL_BLK = "cmems_mod_blk_phy-sal_anfc_2.5km_P1D-m"   # so
DS_MLD_BLK = "cmems_mod_blk_phy-mld_anfc_2.5km_P1D-m"   # mlotst
STRIDE_BLK = 3

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

# ── 1b. DİP SICAKLIK (bottomT) + TUZLULUK (so) ───────────────
# Akıntı hücrelerine 'bt' (dip sıcaklık °C) ve 'sal' (tuzluluk PSU) ekler.
# Hızlı eşleştirme için "lat_lng" anahtarlı indeks kullanılır.
def _idx_key(la, lo):
    return f"{round(float(la),3)}_{round(float(lo),3)}"

cur_index = {}
for r in result['currents']:
    cur_index[_idx_key(r['lat'], r['lng'])] = r

# Dip sıcaklık
try:
    dsb = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        variables=["bottomT"],
        start_datetime=today, end_datetime=today, **BBOX
    )
    da_b = dsb["bottomT"].isel(time=0)
    if "depth" in da_b.dims:
        da_b = da_b.isel(depth=0)
    blats = dsb["latitude"].values[::STRIDE]
    blons = dsb["longitude"].values[::STRIDE]
    bvals = da_b.values[::STRIDE, ::STRIDE]
    n = 0
    for i in range(len(blats)):
        for j in range(len(blons)):
            v = bvals[i, j]
            if v is None or not np.isfinite(float(v)):
                continue
            v = float(v)
            if v > 250:           # Kelvin → °C güvence
                v -= 273.15
            if v < -3 or v > 35:
                continue
            k = _idx_key(blats[i], blons[j])
            if k in cur_index:
                cur_index[k]['bt'] = round(v, 1); n += 1
    print(f"  Dip sicaklik: {n} nokta eslesti")
    dsb.close()
except Exception as e:
    errors.append(f"BottomT: {e}")
    print(f"  [HATA] Dip sicaklik: {e}", file=sys.stderr)

# Tuzluluk (yüzey)
try:
    dss = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
        variables=["so"],
        start_datetime=today, end_datetime=today,
        minimum_depth=1.0, maximum_depth=1.5, **BBOX
    )
    da_s = dss["so"].isel(time=0)
    if "depth" in da_s.dims:
        da_s = da_s.isel(depth=0)
    slats = dss["latitude"].values[::STRIDE]
    slons = dss["longitude"].values[::STRIDE]
    svals = da_s.values[::STRIDE, ::STRIDE]
    n = 0
    for i in range(len(slats)):
        for j in range(len(slons)):
            v = svals[i, j]
            if v is None or not np.isfinite(float(v)):
                continue
            v = float(v)
            if v < 5 or v > 45:    # makul tuzluluk PSU aralığı
                continue
            k = _idx_key(slats[i], slons[j])
            if k in cur_index:
                cur_index[k]['sal'] = round(v, 1); n += 1
    print(f"  Tuzluluk: {n} nokta eslesti")
    dss.close()
except Exception as e:
    errors.append(f"Salinity: {e}")
    print(f"  [HATA] Tuzluluk: {e}", file=sys.stderr)

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

# ── 3. KARADENİZ (Black Sea) — akıntı + dip sıcaklık + tuzluluk + MLD ──
# Akdeniz modelinin boş bıraktığı Karadeniz + Boğaz + iç Marmara kuzeyini
# doldurur. Akıntı satırlarına bt/sal gömülür (currents formatıyla aynı).
def _blk_key(la, lo):
    return f"{round(float(la),3)}_{round(float(lo),3)}"

try:
    blk_rows = {}
    # Akıntı (uo/vo)
    dsc = copernicusmarine.open_dataset(
        dataset_id=DS_CUR_BLK, variables=["uo", "vo"],
        start_datetime=today, end_datetime=today,
        minimum_depth=1.0, maximum_depth=1.5, **BBOX_BLK
    )
    da_uo = dsc["uo"].isel(time=0)
    da_vo = dsc["vo"].isel(time=0)
    if "depth" in da_uo.dims:
        da_uo = da_uo.isel(depth=0); da_vo = da_vo.isel(depth=0)
    clats = dsc["latitude"].values[::STRIDE_BLK]
    clons = dsc["longitude"].values[::STRIDE_BLK]
    uo = da_uo.values[::STRIDE_BLK, ::STRIDE_BLK]
    vo = da_vo.values[::STRIDE_BLK, ::STRIDE_BLK]
    spd = np.sqrt(uo**2 + vo**2)
    ddeg = (np.degrees(np.arctan2(uo, vo)) + 360) % 360
    for i in range(len(clats)):
        for j in range(len(clons)):
            u, v, s = uo[i, j], vo[i, j], spd[i, j]
            if np.isnan(u) or np.isnan(v) or s < 0.02:
                continue
            k = _blk_key(clats[i], clons[j])
            blk_rows[k] = {
                'lat': round(float(clats[i]), 3),
                'lng': round(float(clons[j]), 3),
                'uo': round(float(u), 3), 'vo': round(float(v), 3),
                'speed': round(float(s), 3),
                'dir_deg': round(float(ddeg[i, j]), 1),
            }
    dsc.close()

    # Dip sıcaklık (bottomT) → blk_rows'a 'bt'
    try:
        dsb = copernicusmarine.open_dataset(
            dataset_id=DS_TEM_BLK, variables=["bottomT"],
            start_datetime=today, end_datetime=today, **BBOX_BLK)
        da_b = dsb["bottomT"].isel(time=0)
        if "depth" in da_b.dims: da_b = da_b.isel(depth=0)
        blats = dsb["latitude"].values[::STRIDE_BLK]
        blons = dsb["longitude"].values[::STRIDE_BLK]
        bvals = da_b.values[::STRIDE_BLK, ::STRIDE_BLK]
        for i in range(len(blats)):
            for j in range(len(blons)):
                v = bvals[i, j]
                if v is None or not np.isfinite(float(v)): continue
                v = float(v)
                if v > 250: v -= 273.15
                if v < -3 or v > 35: continue
                k = _blk_key(blats[i], blons[j])
                if k in blk_rows: blk_rows[k]['bt'] = round(v, 1)
        dsb.close()
    except Exception as e:
        errors.append(f"BLK bottomT: {e}")

    # Tuzluluk (so) → blk_rows'a 'sal'
    try:
        dss = copernicusmarine.open_dataset(
            dataset_id=DS_SAL_BLK, variables=["so"],
            start_datetime=today, end_datetime=today,
            minimum_depth=1.0, maximum_depth=1.5, **BBOX_BLK)
        da_s = dss["so"].isel(time=0)
        if "depth" in da_s.dims: da_s = da_s.isel(depth=0)
        slats = dss["latitude"].values[::STRIDE_BLK]
        slons = dss["longitude"].values[::STRIDE_BLK]
        svals = da_s.values[::STRIDE_BLK, ::STRIDE_BLK]
        for i in range(len(slats)):
            for j in range(len(slons)):
                v = svals[i, j]
                if v is None or not np.isfinite(float(v)): continue
                v = float(v)
                if v < 5 or v > 45: continue
                k = _blk_key(slats[i], slons[j])
                if k in blk_rows: blk_rows[k]['sal'] = round(v, 1)
        dss.close()
    except Exception as e:
        errors.append(f"BLK salinity: {e}")

    result['currents'].extend(blk_rows.values())
    _b = sum(1 for r in blk_rows.values() if 'bt' in r)
    _s = sum(1 for r in blk_rows.values() if 'sal' in r)
    print(f"  [Karadeniz] {len(blk_rows)} akinti, {_b} dip-sic, {_s} tuzluluk eklendi")

    # MLD (Karadeniz)
    try:
        dsm = copernicusmarine.open_dataset(
            dataset_id=DS_MLD_BLK, variables=["mlotst"],
            start_datetime=today, end_datetime=today, **BBOX_BLK)
        da_m = dsm["mlotst"].isel(time=0)
        if "depth" in da_m.dims: da_m = da_m.isel(depth=0)
        mlats = dsm["latitude"].values[::STRIDE_BLK]
        mlons = dsm["longitude"].values[::STRIDE_BLK]
        mvals = da_m.values[::STRIDE_BLK, ::STRIDE_BLK]
        nm = 0
        for i in range(len(mlats)):
            for j in range(len(mlons)):
                v = mvals[i, j]
                if v is None or not np.isfinite(float(v)) or float(v) <= 0: continue
                result['mld'].append({'lat': round(float(mlats[i]), 3),
                                      'lng': round(float(mlons[j]), 3),
                                      'depth': round(float(v), 1)})
                nm += 1
        print(f"  [Karadeniz] MLD: {nm} nokta eklendi")
        dsm.close()
    except Exception as e:
        errors.append(f"BLK MLD: {e}")
except Exception as e:
    errors.append(f"BLK currents: {e}")
    print(f"  [HATA] Karadeniz akinti: {e}", file=sys.stderr)

# ── Kaydet ───────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
_bt = sum(1 for r in result['currents'] if 'bt' in r)
_sal = sum(1 for r in result['currents'] if 'sal' in r)
print(f"Kaydedildi: {OUT} ({kb} KB) — "
      f"{len(result['currents'])} akinti, {len(result['mld'])} MLD, "
      f"{_bt} dip-sicaklik, {_sal} tuzluluk noktasi")
