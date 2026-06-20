"""
fetch_currents.py — MaviMera v4.6
İKİ MODEL: Akdeniz/Ege (med) + Marmara/Karadeniz (blk mrm-500m)
- Akdeniz datası: cmems_mod_med_phy-* (Ege + Akdeniz)
- Marmara datası: cmems_mod_blk_phy-*_mrm-500m (Marmara Denizi 500m U-TSS modeli)
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

# Akdeniz + Ege kutusu (Marmara HARİÇ — onu blk modeli kapsar)
MED_BBOX = dict(minimum_latitude=35.5, maximum_latitude=40.5,
                minimum_longitude=25.5, maximum_longitude=36.0)
# Marmara kutusu — mrm-500m dataset sınırları: lat 39.5-41.5, lng 25-30
BLK_BBOX = dict(minimum_latitude=40.0, maximum_latitude=41.4,
                minimum_longitude=26.5, maximum_longitude=29.9)

print(f"[{now.isoformat()}] Basliyor — {today[:10]}")
try: print(f"Versiyon: {copernicusmarine.__version__}")
except: pass

result = {'updated': now.isoformat(),
          'currents': [], 'mld': [], 'bottom_t': [], 'salinity': []}
errors = []
STRIDE = 3

def fetch(dataset_id, variables, fname, bbox, **extra):
    path = os.path.join(TMPD, fname)
    if os.path.exists(path): os.remove(path)
    kw = dict(dataset_id=dataset_id, variables=variables,
              start_datetime=today, end_datetime=end,
              output_filename=fname, output_directory=TMPD,
              **bbox, **extra)
    for param in [{'overwrite_output_data':True},{'force_download':True},{}]:
        try:
            copernicusmarine.subset(**kw, **param); break
        except TypeError: continue
    return xr.open_dataset(path)

def add_currents(ds, stride, out_list):
    u = ds['uo'].isel(time=0); v_ = ds['vo'].isel(time=0)
    if 'depth' in u.dims: u, v_ = u.isel(depth=0), v_.isel(depth=0)
    lats = ds['latitude'].values[::stride]; lons = ds['longitude'].values[::stride]
    U,V = u.values[::stride,::stride], v_.values[::stride,::stride]
    spd = np.sqrt(U**2+V**2); dir_ = (np.degrees(np.arctan2(U,V))+360)%360
    n = 0
    for i in range(len(lats)):
        for j in range(len(lons)):
            s = spd[i,j]
            if np.isnan(U[i,j]) or s < 0.02: continue
            out_list.append({'lat':round(float(lats[i]),3),'lng':round(float(lons[j]),3),
                             'uo':round(float(U[i,j]),3),'vo':round(float(V[i,j]),3),
                             'speed':round(float(s),3),'dir_deg':round(float(dir_[i,j]),1)})
            n += 1
    return n

def add_scalar(ds, var, stride, key, vmin, vmax, out_list):
    da = ds[var].isel(time=0)
    if 'depth' in da.dims: da = da.isel(depth=0)
    lats = ds['latitude'].values[::stride]; lons = ds['longitude'].values[::stride]
    A = da.values[::stride,::stride]; n = 0
    for i in range(len(lats)):
        for j in range(len(lons)):
            v = A[i,j]
            if np.isnan(v) or v<vmin or v>vmax: continue
            out_list.append({'lat':round(float(lats[i]),3),'lng':round(float(lons[j]),3),
                             key:round(float(v),3)}); n += 1
    return n

# ═══ AKDENİZ + EGE (med modeli) ═══════════════════════════
print("\n── Akdeniz/Ege (med) ──")
try:
    ds = fetch('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m', ['uo','vo'], 'med_cur.nc',
               MED_BBOX, minimum_depth=1.0, maximum_depth=2.0)
    print(f"  Akinti: {add_currents(ds, STRIDE, result['currents'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MED-cur: {e}"); print(f"  [HATA] {e}", file=sys.stderr)
try:
    ds = fetch('cmems_mod_med_phy-mld_anfc_4.2km_P1D-m', ['mlotst'], 'med_mld.nc', MED_BBOX)
    print(f"  MLD: {add_scalar(ds,'mlotst',STRIDE,'depth',0,500,result['mld'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MED-mld: {e}"); print(f"  [HATA] {e}", file=sys.stderr)
try:
    ds = fetch('cmems_mod_med_phy-tem_anfc_4.2km_P1D-m', ['bottomT'], 'med_bt.nc', MED_BBOX)
    print(f"  DipT: {add_scalar(ds,'bottomT',STRIDE,'bt',-2,40,result['bottom_t'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MED-bt: {e}"); print(f"  [HATA] {e}", file=sys.stderr)
try:
    ds = fetch('cmems_mod_med_phy-sal_anfc_4.2km_P1D-m', ['so'], 'med_sal.nc',
               MED_BBOX, minimum_depth=1.0, maximum_depth=2.0)
    print(f"  Tuzluluk: {add_scalar(ds,'so',STRIDE,'sal',1,45,result['salinity'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MED-sal: {e}"); print(f"  [HATA] {e}", file=sys.stderr)

# ═══ MARMARA (blk mrm-500m modeli) ════════════════════════
print("\n── Marmara (blk mrm-500m) ──")
# Marmara 500m datasetleri daha ince — stride büyük tutalım (çok nokta olmasın)
MRM_STRIDE = 8
try:
    ds = fetch('cmems_mod_blk_phy-cur_anfc_mrm-500m_P1D-m', ['uo','vo'], 'mrm_cur.nc',
               BLK_BBOX, minimum_depth=1.0, maximum_depth=2.0)
    print(f"  Marmara akinti: {add_currents(ds, MRM_STRIDE, result['currents'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MRM-cur: {e}"); print(f"  [HATA] {e}", file=sys.stderr)
try:
    ds = fetch('cmems_mod_blk_phy-sal_anfc_mrm-500m_P1D-m', ['so'], 'mrm_sal.nc',
               BLK_BBOX, minimum_depth=1.0, maximum_depth=2.0)
    print(f"  Marmara tuzluluk: {add_scalar(ds,'so',MRM_STRIDE,'sal',1,45,result['salinity'])} nokta"); ds.close()
except Exception as e:
    errors.append(f"MRM-sal: {e}"); print(f"  [HATA] {e}", file=sys.stderr)
try:
    # Marmara mrm modeli bottomT yerine thetao (sıcaklık) kullanır — en derin katmanı dip kabul et
    ds = fetch('cmems_mod_blk_phy-tem_anfc_mrm-500m_P1D-m', ['thetao'], 'mrm_bt.nc', BLK_BBOX)
    da = ds['thetao'].isel(time=0)
    if 'depth' in da.dims:
        da = da.isel(depth=-1)  # en derin katman = dip sıcaklığı
    blats = ds['latitude'].values[::MRM_STRIDE]
    blons = ds['longitude'].values[::MRM_STRIDE]
    B = da.values[::MRM_STRIDE, ::MRM_STRIDE]
    n = 0
    for i in range(len(blats)):
        for j in range(len(blons)):
            v = B[i,j]
            if np.isnan(v) or v < -2 or v > 40: continue
            result['bottom_t'].append({'lat':round(float(blats[i]),3),
                                       'lng':round(float(blons[j]),3),
                                       'bt':round(float(v),2)})
            n += 1
    print(f"  Marmara dipT: {n} nokta"); ds.close()
except Exception as e:
    errors.append(f"MRM-bt: {e}"); print(f"  [HATA] {e}", file=sys.stderr)

# ═══ Kaydet ═══════════════════════════════════════════════
if errors: result['errors'] = errors
with open(OUT,'w') as f:
    json.dump(result, f, ensure_ascii=False, separators=(',',':'))
kb = os.path.getsize(OUT)//1024
print(f"\nTamamlandi ({kb} KB): {len(result['currents'])} akinti "
      f"| {len(result['mld'])} MLD | {len(result['bottom_t'])} dipT "
      f"| {len(result['salinity'])} tuzluluk")
if errors: print(f"HATALAR: {errors}")
