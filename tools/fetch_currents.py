"""
fetch_currents.py — MaviMera v4.4
copernicusmarine latest (v2.x uyumlu) — subset() tabanlı
"""
import copernicusmarine
import xarray as xr
import json, os, sys, tempfile
import numpy as np
from datetime import datetime, timezone, timedelta

OUT  = os.path.join(os.path.dirname(__file__), '..', 'currents.json')
TMPD = tempfile.mkdtemp()

today = datetime.now(timezone.utc).strftime('%Y-%m-%dT00:00:00')
end   = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime('%Y-%m-%dT03:00:00')

BBOX = dict(
    minimum_latitude  = 35.0,
    maximum_latitude  = 42.5,
    minimum_longitude = 25.0,
    maximum_longitude = 36.5,
)

print(f"[{datetime.now(timezone.utc).isoformat()}] Basliyor — {today[:10]}")
print(f"copernicusmarine versiyon: {copernicusmarine.__version__}")

result = {'updated': datetime.now(timezone.utc).isoformat(),
          'currents': [], 'mld': [], 'bottom_t': [], 'salinity': []}
errors = []
STRIDE = 2


def fetch(dataset_id, variables, fname, **kwargs):
    """subset() ile indir (v2.x uyumlu), xarray ile oku."""
    path = os.path.join(TMPD, fname)
    if os.path.exists(path):
        os.remove(path)

    subset_kwargs = dict(
        dataset_id=dataset_id,
        variables=variables,
        start_datetime=today,
        end_datetime=end,
        output_filename=fname,
        output_directory=TMPD,
        **BBOX,
        **kwargs
    )
    # v2.x'te overwrite_output_data, v1.x'te force_download
    try:
        copernicusmarine.subset(**subset_kwargs, overwrite_output_data=True)
    except TypeError:
        try:
            copernicusmarine.subset(**subset_kwargs, force_download=True)
        except TypeError:
            copernicusmarine.subset(**subset_kwargs)

    return xr.open_dataset(path)


def grid_to_rows(ds, var, stride, value_key, val_min=-999, val_max=999,
                 extra_fn=None):
    da = ds[var].isel(time=0)
    if 'depth' in da.dims:
        da = da.isel(depth=0)
    lats = ds['latitude'].values[::stride]
    lons = ds['longitude'].values[::stride]
    A    = da.values[::stride, ::stride]
    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            v = A[i, j]
            if np.isnan(v) or v < val_min or v > val_max:
                continue
            row = {'lat': round(float(lats[i]), 3),
                   'lng': round(float(lons[j]), 3),
                   value_key: round(float(v), 3)}
            if extra_fn:
                extra_fn(row, A, i, j, lats, lons)
            rows.append(row)
    return rows


# ── 1. Yüzey Akıntısı ──────────────────────────────────────
try:
    ds = fetch('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m',
               ['uo', 'vo'], 'cur.nc',
               minimum_depth=0.5, maximum_depth=1.5)

    uo_da = ds['uo'].isel(time=0)
    vo_da = ds['vo'].isel(time=0)
    if 'depth' in uo_da.dims:
        uo_da = uo_da.isel(depth=0)
        vo_da = vo_da.isel(depth=0)

    lats = ds['latitude'].values[::STRIDE]
    lons = ds['longitude'].values[::STRIDE]
    U    = uo_da.values[::STRIDE, ::STRIDE]
    V    = vo_da.values[::STRIDE, ::STRIDE]
    spd  = np.sqrt(U**2 + V**2)
    dir_ = (np.degrees(np.arctan2(U, V)) + 360) % 360

    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            u, v, s = U[i,j], V[i,j], spd[i,j]
            if np.isnan(u) or np.isnan(v) or s < 0.02:
                continue
            rows.append({'lat': round(float(lats[i]), 3),
                         'lng': round(float(lons[j]), 3),
                         'uo':  round(float(u), 3),
                         'vo':  round(float(v), 3),
                         'speed':   round(float(s), 3),
                         'dir_deg': round(float(dir_[i,j]), 1)})
    result['currents'] = rows
    print(f"  Akinti: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Currents: {e}")
    print(f"  [HATA] Akinti: {e}", file=sys.stderr)

# ── 2. MLD ─────────────────────────────────────────────────
try:
    ds   = fetch('cmems_mod_med_phy-mld_anfc_4.2km_P1D-m', ['mlotst'], 'mld.nc')
    rows = grid_to_rows(ds, 'mlotst', STRIDE, 'depth', 0, 500)
    result['mld'] = rows
    print(f"  MLD: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"MLD: {e}")
    print(f"  [HATA] MLD: {e}", file=sys.stderr)

# ── 3. Dip Sıcaklığı ───────────────────────────────────────
try:
    ds   = fetch('cmems_mod_med_phy-tem_anfc_4.2km_P1D-m', ['bottomT'], 'bt.nc')
    rows = grid_to_rows(ds, 'bottomT', STRIDE, 'bt', -2, 40)
    result['bottom_t'] = rows
    print(f"  Dip sicakligi: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"BottomT: {e}")
    print(f"  [HATA] Dip sicakligi: {e}", file=sys.stderr)

# ── 4. Tuzluluk ────────────────────────────────────────────
try:
    ds   = fetch('cmems_mod_med_phy-sal_anfc_4.2km_P1D-m', ['so'], 'sal.nc',
                 minimum_depth=0.5, maximum_depth=1.5)
    rows = grid_to_rows(ds, 'so', STRIDE, 'sal', 1, 45)
    result['salinity'] = rows
    print(f"  Tuzluluk: {len(rows)} nokta")
    ds.close()
except Exception as e:
    errors.append(f"Salinity: {e}")
    print(f"  [HATA] Tuzluluk: {e}", file=sys.stderr)

# ── Kaydet ─────────────────────────────────────────────────
if errors:
    result['errors'] = errors

with open(OUT, 'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"\nKaydedildi ({kb} KB): {len(result['currents'])} akinti, "
      f"{len(result['mld'])} MLD, {len(result['bottom_t'])} dipT, "
      f"{len(result['salinity'])} tuzluluk")
