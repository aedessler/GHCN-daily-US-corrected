#!/usr/bin/env python3
"""
Reproduce "Conterminous U.S. Observed Number of Daily Temperature Records"
using GHCND data adjusted with monthly FLs.52j offsets.

adjusted = raw/10 + monthly_offset (both in °C)

Results are cached to records_cache.npz so replot.py can regenerate
the figure without repeating the ~50-minute data loading step.
"""

import time
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import urllib.request
from collections import defaultdict

DATA_DIR     = Path("/Volumes/adessler_lab/GHCND/by_year")
OFFSETS_FILE = Path("/Volumes/adessler_lab/GHCND/monthly_data/processed/monthly_offsets.nc")
OUT_DIR      = Path("/Users/adessler/Desktop/recHighs")
CACHE_FILE   = OUT_DIR / "records_cache.npz"
CKPT_TMAX    = OUT_DIR / "checkpoint_tmax.dat"
CKPT_TMIN    = OUT_DIR / "checkpoint_tmin.dat"
CKPT_META    = OUT_DIR / "checkpoint_meta.npz"

STUDY_START   = 1900
STUDY_END     = 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80   # 80% of possible station-days must have data

# CONUS lat/lon bounds (excludes AK, HI, PR)
LAT_MIN, LAT_MAX =  24.5,  49.5
LON_MIN, LON_MAX = -125.0, -66.0

# ---------------------------------------------------------------
# 1. Download station metadata and identify CONUS candidates
# ---------------------------------------------------------------
print("Downloading station metadata …", flush=True)
STA_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
INV_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"

sta_lines = urllib.request.urlopen(STA_URL).read().decode().split('\n')
sta_meta = {}
for line in sta_lines:
    if len(line) < 30:
        continue
    sid = line[:11].strip()
    try:
        lat = float(line[12:20])
        lon = float(line[21:30])
    except ValueError:
        continue
    sta_meta[sid] = (lat, lon)

conus_ids = {sid for sid, (lat, lon) in sta_meta.items()
             if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX}
print(f"  CONUS stations (lat/lon filter): {len(conus_ids)}", flush=True)

inv_lines = urllib.request.urlopen(INV_URL).read().decode().split('\n')
inv = defaultdict(dict)
for line in inv_lines:
    parts = line.split()
    if len(parts) < 6:
        continue
    sid, el = parts[0], parts[3]
    try:
        fy, ly = int(parts[4]), int(parts[5])
    except ValueError:
        continue
    inv[sid][el] = (fy, ly)

candidates = []
for sid in sorted(conus_ids):
    if 'TMAX' not in inv[sid] or 'TMIN' not in inv[sid]:
        continue
    tmax_fy, tmax_ly = inv[sid]['TMAX']
    tmin_fy, tmin_ly = inv[sid]['TMIN']
    avail = min(
        max(0, min(tmax_ly, STUDY_END) - max(tmax_fy, STUDY_START) + 1),
        max(0, min(tmin_ly, STUDY_END) - max(tmin_fy, STUDY_START) + 1),
    )
    if avail >= MIN_YEARS:
        candidates.append(sid)

candidate_set = set(candidates)
n_cands = len(candidates)
cand_idx = {sid: i for i, sid in enumerate(candidates)}
print(f"  Candidate stations (inventory ≥{MIN_YEARS} yrs): {n_cands}", flush=True)

# ---------------------------------------------------------------
# 2. Load monthly offsets for candidate stations
# ---------------------------------------------------------------
print("Loading monthly offsets …", flush=True)
ds_off = xr.open_dataset(OFFSETS_FILE)
off_station_ids = ds_off.station_id.values
off_times = pd.DatetimeIndex(ds_off.time.values)
off_idx_map = {sid: i for i, sid in enumerate(off_station_ids)}

t0 = pd.Timestamp(f'{STUDY_START}-01-01')
t1 = pd.Timestamp(f'{STUDY_END}-12-01')
time_mask = (off_times >= t0) & (off_times <= t1)
study_times = off_times[time_mask]
n_study_months = len(study_times)
time_to_idx = {(t.year, t.month): i for i, t in enumerate(study_times)}

tmax_off = np.zeros((n_cands, n_study_months), dtype=np.float32)
tmin_off = np.zeros((n_cands, n_study_months), dtype=np.float32)

for ci, sid in enumerate(candidates):
    if sid in off_idx_map:
        oi = off_idx_map[sid]
        tmax_vals = ds_off.tmax_offset.values[oi, time_mask]
        tmin_vals = ds_off.tmin_offset.values[oi, time_mask]
        tmax_off[ci] = np.where(np.isnan(tmax_vals), 0.0, tmax_vals)
        tmin_off[ci] = np.where(np.isnan(tmin_vals), 0.0, tmin_vals)

ds_off.close()
print(f"  Offsets loaded for {n_cands} candidates.", flush=True)

# ---------------------------------------------------------------
# 3. Allocate data arrays as memory-mapped files for crash recovery
#    Shape: (n_cands × N_YEARS × 366)
# ---------------------------------------------------------------
n_doys = 366
years  = np.arange(STUDY_START, STUDY_END + 1)
year_to_yidx = {y: i for i, y in enumerate(years)}
array_shape = (n_cands, N_YEARS, n_doys)

# Check for a compatible existing checkpoint
years_done = set()
ckpt_ok = False
if CKPT_META.exists() and CKPT_TMAX.exists() and CKPT_TMIN.exists():
    try:
        meta = np.load(CKPT_META, allow_pickle=True)
        if (tuple(meta['shape']) == array_shape and
                list(meta['candidates']) == candidates):
            years_done = set(int(y) for y in meta['years_done'])
            ckpt_ok = True
            print(f"  Resuming checkpoint: {len(years_done)} of {N_YEARS} years already done.", flush=True)
    except Exception as e:
        print(f"  Checkpoint unreadable ({e}), starting fresh.", flush=True)

if ckpt_ok:
    tmax_data = np.memmap(CKPT_TMAX, dtype='float32', mode='r+', shape=array_shape)
    tmin_data = np.memmap(CKPT_TMIN, dtype='float32', mode='r+', shape=array_shape)
else:
    tmax_data = np.memmap(CKPT_TMAX, dtype='float32', mode='w+', shape=array_shape)
    tmin_data = np.memmap(CKPT_TMIN, dtype='float32', mode='w+', shape=array_shape)
    tmax_data[:] = np.nan
    tmin_data[:] = np.nan
    np.savez(CKPT_META, shape=array_shape, candidates=candidates, years_done=np.array([], dtype=int))
    print(f"  New checkpoint created ({2 * tmax_data.nbytes / 1e9:.1f} GB on disk).", flush=True)

# ---------------------------------------------------------------
# 4. Read each year file, apply offsets, fill arrays
# ---------------------------------------------------------------
print("Reading year files …", flush=True)
COLS   = ['station', 'date', 'element', 'value', 'mflag', 'qflag', 'sflag', 'extra']
DTYPES = {'station': str, 'date': str, 'element': str,
          'value': 'Int32', 'mflag': str, 'qflag': str, 'sflag': str, 'extra': str}

for year in years:
    if year in years_done:
        continue

    yfile = DATA_DIR / f"{year}.csv.gz"

    # Retry up to 10 times in case of transient network-drive disconnect
    for attempt in range(10):
        try:
            exists = yfile.exists()
            break
        except OSError as e:
            print(f"  {year}: drive error ({e}), retrying in 30 s …", flush=True)
            time.sleep(30)
    else:
        print(f"  {year}: drive unavailable after retries, skipping", flush=True)
        continue

    if not exists:
        print(f"  {year}: file missing, skipping", flush=True)
        continue

    yidx = year_to_yidx[year]

    for attempt in range(10):
        try:
            df = pd.read_csv(
                yfile, header=None, names=COLS, dtype=DTYPES,
                compression='gzip', low_memory=True,
                on_bad_lines='skip',
            )
            break
        except OSError as e:
            print(f"  {year}: read error ({e}), retrying in 30 s …", flush=True)
            time.sleep(30)
    else:
        print(f"  {year}: could not read after retries, skipping", flush=True)
        continue

    df = df[df['station'].isin(candidate_set) & df['element'].isin(['TMAX', 'TMIN'])].copy()
    if df.empty:
        continue

    qflag_bad = df['qflag'].notna() & (df['qflag'].str.strip() != '')
    df = df[~qflag_bad].copy()
    df = df[df['value'].notna() & (df['value'] != -9999)].copy()
    if df.empty:
        continue

    dates = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df[dates.notna()].copy()
    dates = dates[dates.notna()]
    df['doy']   = dates.dt.day_of_year.values - 1   # 0-based
    df['month'] = dates.dt.month.values
    df['temp_c'] = df['value'].astype(np.float32) / 10.0
    df['ci'] = df['station'].map(cand_idx)
    df = df[df['ci'].notna()].copy()
    df['ci'] = df['ci'].astype(int)
    df['tidx'] = df['month'].map(lambda m: time_to_idx.get((year, m), -1))

    for el, arr, off in [('TMAX', tmax_data, tmax_off), ('TMIN', tmin_data, tmin_off)]:
        sub = df[df['element'] == el]
        if sub.empty:
            continue

        ci_arr   = sub['ci'].values
        doy_arr  = sub['doy'].values
        tidx_arr = sub['tidx'].values
        temp_arr = sub['temp_c'].values

        valid = (doy_arr >= 0) & (doy_arr < n_doys)
        ci_arr, doy_arr, tidx_arr, temp_arr = (
            ci_arr[valid], doy_arr[valid], tidx_arr[valid], temp_arr[valid]
        )

        tidx_safe = np.clip(tidx_arr, 0, n_study_months - 1)
        off_vals  = np.where(tidx_arr >= 0, off[ci_arr, tidx_safe], 0.0)
        adj_temp  = temp_arr + off_vals.astype(np.float32)

        keys = ci_arr.astype(np.int64) * n_doys + doy_arr
        _, first = np.unique(keys, return_index=True)
        ci_u, doy_u, temp_u = ci_arr[first], doy_arr[first], adj_temp[first]

        existing = arr[ci_u, yidx, doy_u]
        fill = np.isnan(existing)
        arr[ci_u[fill], yidx, doy_u[fill]] = temp_u[fill]

    # Flush memmap to disk and record this year as complete
    tmax_data.flush()
    tmin_data.flush()
    years_done.add(year)
    np.savez(CKPT_META, shape=array_shape, candidates=candidates,
             years_done=np.array(sorted(years_done), dtype=int))

    if year % 10 == 0 or year == STUDY_START:
        print(f"  {year} done  ({len(years_done)}/{N_YEARS})", flush=True)

print("Year files processed.", flush=True)

# ---------------------------------------------------------------
# 5. Completeness filter
# ---------------------------------------------------------------
n_possible = N_YEARS * 365.25
tmax_years = np.sum(~np.all(np.isnan(tmax_data), axis=2), axis=1)
tmin_years = np.sum(~np.all(np.isnan(tmin_data), axis=2), axis=1)
tmax_days  = np.sum(~np.isnan(tmax_data), axis=(1, 2))
tmin_days  = np.sum(~np.isnan(tmin_data), axis=(1, 2))

good = (
    (tmax_years >= MIN_YEARS) & (tmin_years >= MIN_YEARS) &
    (tmax_days / n_possible >= MIN_DATA_FRAC) &
    (tmin_days / n_possible >= MIN_DATA_FRAC)
)
good_stations = [candidates[i] for i in range(n_cands) if good[i]]
tmax_data = tmax_data[good]
tmin_data = tmin_data[good]
n_good = len(good_stations)
print(f"Stations after completeness filter: {n_good}", flush=True)

# ---------------------------------------------------------------
# 6. Compute all-time record highs and lows per year
# ---------------------------------------------------------------
print("Computing records …", flush=True)

tmax_flat = tmax_data.transpose(0, 2, 1).reshape(-1, N_YEARS)
tmin_flat = tmin_data.transpose(0, 2, 1).reshape(-1, N_YEARS)
valid_tmax = ~np.all(np.isnan(tmax_flat), axis=1)
valid_tmin = ~np.all(np.isnan(tmin_flat), axis=1)

rec_highs = np.zeros(N_YEARS, dtype=int)
if valid_tmax.any():
    tmax_max  = np.nanmax(tmax_flat[valid_tmax], axis=1, keepdims=True)
    max_match = (tmax_flat[valid_tmax] == tmax_max)
    rec_highs = np.sum(max_match, axis=0)

rec_lows = np.zeros(N_YEARS, dtype=int)
if valid_tmin.any():
    tmin_min  = np.nanmin(tmin_flat[valid_tmin], axis=1, keepdims=True)
    min_match = (tmin_flat[valid_tmin] == tmin_min)
    rec_lows  = np.sum(min_match, axis=0)

print(f"Total record highs: {rec_highs.sum():,}  (expected ≈ {n_good*365:.0f})", flush=True)
print(f"Total record lows:  {rec_lows.sum():,}",  flush=True)

np.savez(CACHE_FILE, years=years, rec_highs=rec_highs, rec_lows=rec_lows, n_good=np.array(n_good))
print(f"Cached to {CACHE_FILE}", flush=True)
print("Done. Run replot.py to generate the figure.", flush=True)
