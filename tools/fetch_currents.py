"""
fetch_ocean.py — MaraMap v5.0
Copernicus Marine'den SST + Klorofil + Dalga + Rüzgar çeker (vectorized).
Çıktı: ocean.json → GitHub maramap-data repo'ya push edilir.

Akıntı + MLD ayrı script'tedir (fetch_currents.py → currents.json), DOKUNULMAZ.

KAPSAM: Akdeniz Copernicus modeli — Marmara + Ege + Akdeniz kıyıları.
        (Karadeniz bu modelde yoktur; o hücreler NaN gelir, atlanır —
         uygulamanın hedef kapsamı zaten Karadeniz'i dışlıyor.)

═══════════════════════════════════════════════════════════════════════
ÖNEMLİ — DATASET ID DOĞRULAMASI
Aşağıdaki dataset_id değerleri, çalışan fetch_currents.py'deki
'cmems_mod_med_phy-cur_anfc_4.2km_P1D-m' adlandırma kalıbına göre türetildi.
Copernicus zaman zaman ID'leri sürümler. Bir fetch başarısız olursa,
https://data.marine.copernicus.eu/products üzerinden güncel ID'yi doğrulayın.
Her fetch BAĞIMSIZDIR: biri çökse bile diğerleri çalışır ve ocean.json yazılır.
═══════════════════════════════════════════════════════════════════════
"""
import copernicusmarine
import json, os, sys
import numpy as np
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), '..', 'ocean.json')

# Türkiye kıyı bölgesi (Akdeniz modeli kapsamı)
BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.0,
    minimum_longitude = 25.0,
    maximum_longitude = 37.0,
)

# Dataset ID'leri — gerekirse Copernicus kataloğundan doğrulayın
DS_SST  = "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"   # deniz sıcaklığı (thetao)
DS_CHL  = "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m"   # klorofil (chl)
DS_WAVE = "cmems_mod_med_wav_anfc_4.2km_PT1H-i"      # dalga (VHM0 = anlamlı dalga yük.)
# Rüzgar: global L4 blended (Akdeniz dahil her yeri kapsar, 0.125°)
DS_WIND = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"

STRIDE = 2   # 4.2km × 2 ≈ 8.4km — harita için yeterli, JSON küçük kalır

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"[{datetime.now(timezone.utc).isoformat()}] ocean.json — {today}")

result = {
    'updated': datetime.now(timezone.utc).isoformat(),
    'sst': [], 'chl': [], 'waves': [], 'wind': [],
}
errors = []


def grid_rows(ds, var, value_key, stride=STRIDE, surface=True, transform=None):
    """Bir değişkeni vectorized okuyup [{lat,lng,<value_key>}] satırları üretir."""
    da = ds[var].isel(time=0)
    if surface and "depth" in da.dims:
        da = da.isel(depth=0)
    lats = ds["latitude"].values[::stride]
    lons = ds["longitude"].values[::stride]
    vals = da.values[::stride, ::stride]
    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            v = vals[i, j]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v = float(v)
            if transform:
                v = transform(v)
            if v is None:
                continue
            rows.append({
                'lat': round(float(lats[i]), 3),
                'lng': round(float(lons[j]), 3),
                value_key: round(v, 3),
            })
    return rows


# ── 1. SST (deniz yüzey sıcaklığı) ───────────────────────────
try:
    ds = copernicusmarine.open_dataset(
        dataset_id=DS_SST, variables=["thetao"],
        start_datetime=today, end_datetime=today,
        minimum_depth=1.0, maximum_depth=1.5, **BBOX
    )
    # Kelvin gelirse °C'ye çevir (Copernicus thetao zaten °C, ama güvence)
    def _sst(v):
        if v > 250 and v < 320:
            v = v - 273.15
        if v < -5 or v > 40:
            return None
        return v
    result['sst'] = grid_rows(ds, "thetao", "sst", transform=_sst)
    print(f"  SST: {len(result['sst'])} nokta")
    ds.close()
except Exception as e:
    errors.append(f"SST: {e}")
    print(f"  [HATA] SST: {e}", file=sys.stderr)

# ── 2. KLOROFİL ──────────────────────────────────────────────
try:
    ds = copernicusmarine.open_dataset(
        dataset_id=DS_CHL, variables=["chl"],
        start_datetime=today, end_datetime=today,
        minimum_depth=1.0, maximum_depth=1.5, **BBOX
    )
    def _chl(v):
        if v <= 0 or v > 50:
            return None
        return v
    result['chl'] = grid_rows(ds, "chl", "chl", transform=_chl)
    print(f"  Klorofil: {len(result['chl'])} nokta")
    ds.close()
except Exception as e:
    errors.append(f"CHL: {e}")
    print(f"  [HATA] Klorofil: {e}", file=sys.stderr)

# ── 3. DALGA (anlamlı dalga yüksekliği) ──────────────────────
try:
    ds = copernicusmarine.open_dataset(
        dataset_id=DS_WAVE, variables=["VHM0"],
        start_datetime=today, end_datetime=today, **BBOX
    )
    def _wave(v):
        if v < 0 or v > 20:
            return None
        return v
    result['waves'] = grid_rows(ds, "VHM0", "wave", surface=False, transform=_wave)
    print(f"  Dalga: {len(result['waves'])} nokta")
    ds.close()
except Exception as e:
    errors.append(f"WAVE: {e}")
    print(f"  [HATA] Dalga: {e}", file=sys.stderr)

# ── 4. RÜZGAR (10m, u/v → hız km/h) ──────────────────────────
try:
    ds = copernicusmarine.open_dataset(
        dataset_id=DS_WIND,
        variables=["eastward_wind", "northward_wind"],
        start_datetime=today, end_datetime=today, **BBOX
    )
    da_u = ds["eastward_wind"].isel(time=0)
    da_v = ds["northward_wind"].isel(time=0)
    if "depth" in da_u.dims:
        da_u = da_u.isel(depth=0); da_v = da_v.isel(depth=0)
    # Rüzgar gridi daha kaba (0.125°) — stride 1 yeterli
    wlats = ds["latitude"].values
    wlons = ds["longitude"].values
    u = da_u.values
    v = da_v.values
    spd_ms = np.sqrt(u**2 + v**2)
    rows = []
    for i in range(len(wlats)):
        for j in range(len(wlons)):
            s = spd_ms[i, j]
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            rows.append({
                'lat': round(float(wlats[i]), 3),
                'lng': round(float(wlons[j]), 3),
                'wind': round(float(s) * 3.6, 1),    # m/s → km/h
            })
    result['wind'] = rows
    print(f"  Rüzgar: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"WIND: {e}")
    print(f"  [HATA] Rüzgar: {e}", file=sys.stderr)

# ── Kaydet ───────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"Kaydedildi: {OUT} ({kb} KB) — "
      f"{len(result['sst'])} SST, {len(result['chl'])} CHL, "
      f"{len(result['waves'])} dalga, {len(result['wind'])} rüzgar")

# Hiçbir veri çekilemezse hata koduyla çık (workflow kırmızı görünsün)
if not any([result['sst'], result['chl'], result['waves'], result['wind']]):
    print("Hiçbir veri çekilemedi!", file=sys.stderr)
    sys.exit(1)
