"""
fetch_currents.py — MaviMera v4.5
Daha küçük bbox + timeout — GitHub Actions timeout sorununu önler
"""
import copernicusmarine
import xarray as xr
import json, os, sys, tempfile
import numpy as np
from datetime import datetime, timezone, timedelta

OUT  = os.path.join(os.path.dirname(__file__), '..', 'currents.json')
TMPD = tempfile.mkdtemp()

now   = datetime.now(timezone.utc)
today = now.strftime('%Y-%m-%dT00:00:00')
end   = now.strftime('%Y-%m-%dT06:00:00')

# Sadece Türkiye kıyısı — daha küçük alan = daha hızlı indirme
BBOX = dict(
    minimum_latitude  = 35.5,
    maximum_latitude  = 42.0,
    minimum_longitude = 25.5,
    maximum_longitude = 36.0,
)

print(f"[{now.isoformat()}] Basliyor — {today[:10]}")
try:
    print(f"Versiyon: {copernicusmarine.__version__}")
except:
    pass

result = {'updated': now.isoformat(),
          'currents': [], 'mld': [], 'bottom_t': [], 'salinity': []}
errors = []
STRIDE = 3  # 4.2km × 3 = ~12.6km — daha az nokta, daha hızlı

def fetch(dataset_id, variables, fname, **extra):
    path = os.path.join(TMPD, fname)
    if os.path.exists(path):
        os.remove(path)
    
    kw = dict(dataset_id=dataset_id, variables=variables,
              start_datetime=today, end_datetime=end,
              output_filename=fname, output_directory=TMPD,
              **BBOX, **extra)
    
    # v2.x → overwrite_output_data, v1.x → force_download
    for param in [{'overwrite_output_data': True},
                  {'force_download': True}, {}]:
        try:
            copernicusmarine.subset(**kw, **param)
            break
        except TypeError:
            continue
    
    return xr.open_dataset(path)

def to_rows(ds, var, stride, key, vmin, vmax):
    da = ds[var].isel(time=0)
    if 'depth' in da.dims: da = da.isel(depth=0)
    lats = ds['latitude'].values[::stride]
    lons = ds['longitude'].values[::stride]
    A    = da.values[::stride, ::stride]
    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            v = A[i,j]
            if np.isnan(v) or v < vmin or v > vmax: continue
            rows.append({'lat': round(float(lats[i]),3),
                         'lng': round(float(lons[j]),3),
                         key:   round(float(v),3)})
    return rows

# ── 1. Akıntı ──────────────────────────────────────────────
try:
    ds  = fetch('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m',
                ['uo','vo'], 'cur.nc',
                minimum_depth=0.5, maximum_depth=1.5)
    u   = ds['uo'].isel(time=0)
    v_  = ds['vo'].isel(time=0)
    if 'depth' in u.dims: u, v_ = u.isel(depth=0), v_.isel(depth=0)
    lats = ds['latitude'].values[::STRIDE]
    lons = ds['longitude'].values[::STRIDE]
    U,V  = u.values[::STRIDE,::STRIDE], v_.values[::STRIDE,::STRIDE]
    spd  = np.sqrt(U**2+V**2)
    dir_ = (np.degrees(np.arctan2(U,V))+360)%360
    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            s = spd[i,j]
            if np.isnan(U[i,j]) or s < 0.02: continue
            rows.append({'lat':round(float(lats[i]),3),'lng':round(float(lons[j]),3),
                         'uo':round(float(U[i,j]),3),'vo':round(float(V[i,j]),3),
                         'speed':round(float(s),3),'dir_deg':round(float(dir_[i,j]),1)})
    result['currents'] = rows
    print(f"  Akinti: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] {e}", file=sys.stderr)

# ── 2. MLD ─────────────────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-mld_anfc_4.2km_P1D-m', ['mlotst'], 'mld.nc')
    result['mld'] = to_rows(ds, 'mlotst', STRIDE, 'depth', 0, 500)
    print(f"  MLD: {len(result['mld'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MLD: {e}"); print(f"  [HATA] {e}", file=sys.stderr)

# ── 3. Dip Sıcaklığı ───────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-tem_anfc_4.2km_P1D-m', ['bottomT'], 'bt.nc')
    result['bottom_t'] = to_rows(ds, 'bottomT', STRIDE, 'bt', -2, 40)
    print(f"  DipT: {len(result['bottom_t'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"BottomT: {e}"); print(f"  [HATA] {e}", file=sys.stderr)

# ── 4. Tuzluluk ────────────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-sal_anfc_4.2km_P1D-m', ['so'], 'sal.nc',
               minimum_depth=0.5, maximum_depth=1.5)
    result['salinity'] = to_rows(ds, 'so', STRIDE, 'sal', 1, 45)
    print(f"  Tuzluluk: {len(result['salinity'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"Salinity: {e}"); print(f"  [HATA] {e}", file=sys.stderr)

# ── Kaydet ─────────────────────────────────────────────────
if errors: result['errors'] = errors
with open(OUT,'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',',':'))
kb = os.path.getsize(OUT)//1024
print(f"\nTamamlandi ({kb} KB): {len(result['currents'])} akinti "
      f"| {len(result['mld'])} MLD | {len(result['bottom_t'])} dipT "
      f"| {len(result['salinity'])} tuzluluk")
if errors: print(f"HATALAR: {errors}")
