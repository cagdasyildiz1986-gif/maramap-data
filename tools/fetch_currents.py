"""
fetch_currents.py — MaviMera v4.3
copernicusmarine.subset() kullanır — open_dataset() storage_options sorununu önler.

Veri setleri:
  1. Yüzey akıntısı (uo, vo)
  2. Karışık tabaka derinliği (MLD)
  3. Dip sıcaklığı (bottomT)
  4. Yüzey tuzluluğu (so)
"""
import copernicusmarine
import xarray as xr
import json, os, sys, tempfile
import numpy as np
from datetime import datetime, timezone, timedelta

OUT  = os.path.join(os.path.dirname(__file__), '..', 'currents.json')
TMPD = tempfile.mkdtemp()

today = datetime.now(timezone.utc).strftime('%Y-%m-%dT00:00:00')
end   = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime('%Y-%m-%dT01:00:00')

BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.5,
    minimum_longitude = 25.0,
    maximum_longitude = 36.5,
)

print(f"[{datetime.now(timezone.utc).isoformat()}] Basliyor — {today[:10]}")

result = {'updated': datetime.now(timezone.utc).isoformat(),
          'currents': [], 'mld': [], 'bottom_t': [], 'salinity': []}
errors = []
STRIDE = 2


def fetch(dataset_id, variables, fname, **kwargs):
    """subset() ile indir, xarray ile oku."""
    path = os.path.join(TMPD, fname)
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        start_datetime=today,
        end_datetime=end,
        output_filename=fname,
        output_directory=TMPD,
        force_download=True,
        **BBOX,
        **kwargs
    )
    return xr.open_dataset(path)


# ── 1. Yüzey Akıntısı ──────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m',
               ['uo', 'vo'], 'cur.nc',
               minimum_depth=0.5, maximum_depth=1.5)

    uo = ds['uo'].isel(time=0)
    vo = ds['vo'].isel(time=0)
    if 'depth' in uo.dims: uo, vo = uo.isel(depth=0), vo.isel(depth=0)

    lats = ds['latitude'].values[::STRIDE]
    lons = ds['longitude'].values[::STRIDE]
    U    = uo.values[::STRIDE, ::STRIDE]
    V    = vo.values[::STRIDE, ::STRIDE]
    spd  = np.sqrt(U**2 + V**2)
    dir_ = (np.degrees(np.arctan2(U, V)) + 360) % 360

    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            u, v, s = U[i,j], V[i,j], spd[i,j]
            if np.isnan(u) or np.isnan(v) or s < 0.02: continue
            rows.append({'lat': round(float(lats[i]),3),
                         'lng': round(float(lons[j]),3),
                         'uo':  round(float(u),3),
                         'vo':  round(float(v),3),
                         'speed':   round(float(s),3),
                         'dir_deg': round(float(dir_[i,j]),1)})
    result['currents'] = rows
    print(f"  Akinti: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] Akinti: {e}", file=sys.stderr)

# ── 2. Karışık Tabaka (MLD) ────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-mld_anfc_4.2km_P1D-m',
               ['mlotst'], 'mld.nc')
    da = ds['mlotst'].isel(time=0)
    if 'depth' in da.dims: da = da.isel(depth=0)
    mlats = ds['latitude'].values[::STRIDE]
    mlons = ds['longitude'].values[::STRIDE]
    M     = da.values[::STRIDE, ::STRIDE]
    rows  = []
    for i in range(len(mlats)):
        for j in range(len(mlons)):
            v = M[i,j]
            if np.isnan(v) or v <= 0: continue
            rows.append({'lat': round(float(mlats[i]),3),
                         'lng': round(float(mlons[j]),3),
                         'depth': round(float(v),1)})
    result['mld'] = rows
    print(f"  MLD: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"MLD: {e}")
    print(f"  [HATA] MLD: {e}", file=sys.stderr)

# ── 3. Dip Sıcaklığı (bottomT) ────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-tem_anfc_4.2km_P1D-m',
               ['bottomT'], 'bt.nc')
    da = ds['bottomT'].isel(time=0)
    blats = ds['latitude'].values[::STRIDE]
    blons = ds['longitude'].values[::STRIDE]
    B     = da.values[::STRIDE, ::STRIDE]
    rows  = []
    for i in range(len(blats)):
        for j in range(len(blons)):
            v = B[i,j]
            if np.isnan(v) or v < -2 or v > 40: continue
            rows.append({'lat': round(float(blats[i]),3),
                         'lng': round(float(blons[j]),3),
                         'bt':  round(float(v),2)})
    result['bottom_t'] = rows
    print(f"  Dip sicakligi: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"BottomT: {e}")
    print(f"  [HATA] Dip sicakligi: {e}", file=sys.stderr)

# ── 4. Tuzluluk (so) ──────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-sal_anfc_4.2km_P1D-m',
               ['so'], 'sal.nc',
               minimum_depth=0.5, maximum_depth=1.5)
    da = ds['so'].isel(time=0)
    if 'depth' in da.dims: da = da.isel(depth=0)
    slats = ds['latitude'].values[::STRIDE]
    slons = ds['longitude'].values[::STRIDE]
    S     = da.values[::STRIDE, ::STRIDE]
    rows  = []
    for i in range(len(slats)):
        for j in range(len(slons)):
            v = S[i,j]
            if np.isnan(v) or v < 1 or v > 45: continue
            rows.append({'lat': round(float(slats[i]),3),
                         'lng': round(float(slons[j]),3),
                         'sal': round(float(v),2)})
    result['salinity'] = rows
    print(f"  Tuzluluk: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Salinity: {e}")
    print(f"  [HATA] Tuzluluk: {e}", file=sys.stderr)

# ── Kaydet ────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"\nKaydedildi ({kb} KB): {len(result['currents'])} akinti, "
      f"{len(result['mld'])} MLD, {len(result['bottom_t'])} dipT, "
      f"{len(result['salinity'])} tuzluluk")
if errors:
    print(f"HATALAR: {errors}")
