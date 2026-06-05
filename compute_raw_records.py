#!/usr/bin/env python3
"""
Raw (UNADJUSTED) records, reconstructed from the existing adjusted checkpoints.

The checkpoint memmaps hold adjusted temps, where  adjusted = raw + monthly_offset
(the same offset for every day in a calendar month). So the raw temperature is
recovered exactly by subtracting the monthly offset back out:

    raw = round((adjusted - monthly_offset) * 10) / 10

The round-to-0.1° snap makes it bit-exact: raw temps are always whole tenths of
a degree, and the float32 round-trip error is ~1e-5 °C, far below 0.05.

This avoids re-reading the ~50 min of year files — it only needs the (small)
offsets NetCDF. Station availability is unchanged by an additive offset, so the
completeness filter yields the same 1266 stations as the adjusted run, making
the two figures a clean apples-to-apples comparison.

Outputs records_cache_raw.npz (parallel to records_cache.npz).
"""

import calendar
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

OUT_DIR      = Path("/Users/adessler/Desktop/recHighs")
OFFSETS_FILE = Path("/Volumes/adessler_lab/GHCND/monthly_data/processed/monthly_offsets.nc")
CKPT_TMAX    = OUT_DIR / "checkpoint_tmax.dat"        # adjusted TMAX
CKPT_TMIN    = OUT_DIR / "checkpoint_tmin.dat"        # adjusted TMIN
CKPT_META    = OUT_DIR / "checkpoint_meta.npz"
RAW_TMAX     = OUT_DIR / "checkpoint_tmax_raw.dat"
RAW_TMIN     = OUT_DIR / "checkpoint_tmin_raw.dat"

STUDY_START, STUDY_END = 1900, 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80
N_DOY         = 366

# ---------------------------------------------------------------
# 1. Adjusted checkpoints + candidate list
# ---------------------------------------------------------------
meta  = np.load(CKPT_META, allow_pickle=True)
shape = tuple(int(x) for x in meta['shape'])
candidates = list(meta['candidates'])
n_cands = len(candidates)
adj_tmax = np.memmap(CKPT_TMAX, dtype='float32', mode='r', shape=shape)
adj_tmin = np.memmap(CKPT_TMIN, dtype='float32', mode='r', shape=shape)
print(f"Adjusted checkpoints: {n_cands} candidates, shape {shape}")

# ---------------------------------------------------------------
# 2. Reload monthly offsets EXACTLY as compute_adjusted_records.py did
# ---------------------------------------------------------------
print("Loading monthly offsets …")
ds_off = xr.open_dataset(OFFSETS_FILE)
off_station_ids = ds_off.station_id.values
off_idx_map = {sid: i for i, sid in enumerate(off_station_ids)}
off_times = pd.DatetimeIndex(ds_off.time.values)

t0, t1 = pd.Timestamp(f'{STUDY_START}-01-01'), pd.Timestamp(f'{STUDY_END}-12-01')
time_mask = (off_times >= t0) & (off_times <= t1)
study_times = off_times[time_mask]
n_study_months = len(study_times)
time_to_idx = {(t.year, t.month): i for i, t in enumerate(study_times)}

tmax_all = ds_off.tmax_offset.values            # (station, time)
tmin_all = ds_off.tmin_offset.values
tmax_off = np.zeros((n_cands, n_study_months), dtype=np.float32)
tmin_off = np.zeros((n_cands, n_study_months), dtype=np.float32)
for ci, sid in enumerate(candidates):
    if sid in off_idx_map:
        oi = off_idx_map[sid]
        tv, nv = tmax_all[oi, time_mask], tmin_all[oi, time_mask]
        tmax_off[ci] = np.where(np.isnan(tv), 0.0, tv)
        tmin_off[ci] = np.where(np.isnan(nv), 0.0, nv)
ds_off.close()
print(f"  offsets reloaded for {n_cands} candidates ({n_study_months} months)")

# ---------------------------------------------------------------
# 3. Reconstruct raw temps into new memmaps
# ---------------------------------------------------------------
print("Reconstructing raw temperatures …")
years = np.arange(STUDY_START, STUDY_END + 1)
raw_tmax = np.memmap(RAW_TMAX, dtype='float32', mode='w+', shape=shape)
raw_tmin = np.memmap(RAW_TMIN, dtype='float32', mode='w+', shape=shape)

for adj_arr, raw_arr, off in [(adj_tmax, raw_tmax, tmax_off),
                              (adj_tmin, raw_tmin, tmin_off)]:
    raw_arr[:] = np.nan                       # unfilled slots (e.g. doy 365 in non-leap) stay NaN
    for yidx, year in enumerate(years):
        ndays  = 366 if calendar.isleap(year) else 365
        months = pd.date_range(f'{year}-01-01', periods=ndays, freq='D').month.values
        for m in range(1, 13):
            doys = np.where(months == m)[0]   # 0-based day-of-year indices in month m
            tidx = time_to_idx.get((year, m), -1)
            block = adj_arr[:, yidx, doys]
            if tidx >= 0:                      # mirror the where(tidx>=0, off, 0) of the original
                block = block - off[:, tidx][:, None]
            raw_arr[:, yidx, doys] = np.round(block * 10.0) / 10.0   # snap -> exact 0.1°
    raw_arr.flush()

# ---------------------------------------------------------------
# 4. Sanity: data availability must be identical to the adjusted run
# ---------------------------------------------------------------
assert np.array_equal(np.isnan(raw_tmax), np.isnan(adj_tmax)), "TMAX NaN mask drifted"
assert np.array_equal(np.isnan(raw_tmin), np.isnan(adj_tmin)), "TMIN NaN mask drifted"
print("NaN masks match adjusted (data availability identical).")

# ---------------------------------------------------------------
# 5. Completeness filter (identical to compute_adjusted_records.py)
# ---------------------------------------------------------------
npos = N_YEARS * 365.25
good = (
    (np.sum(~np.all(np.isnan(raw_tmax), axis=2), axis=1) >= MIN_YEARS) &
    (np.sum(~np.all(np.isnan(raw_tmin), axis=2), axis=1) >= MIN_YEARS) &
    (np.sum(~np.isnan(raw_tmax), axis=(1, 2)) / npos >= MIN_DATA_FRAC) &
    (np.sum(~np.isnan(raw_tmin), axis=(1, 2)) / npos >= MIN_DATA_FRAC)
)
good_idx = np.where(good)[0]
n_good = len(good_idx)
n_good_adj = int(np.load(OUT_DIR / "records_cache.npz")['n_good'])
print(f"Stations after completeness filter: {n_good} (adjusted run: {n_good_adj})")
assert n_good == n_good_adj, "station set differs from adjusted run"

# ---------------------------------------------------------------
# 6. Records per year (identical logic to the adjusted run)
# ---------------------------------------------------------------
def yearly_records(data, mode):
    flat = np.ascontiguousarray(data[good_idx].transpose(0, 2, 1).reshape(-1, N_YEARS))
    valid = ~np.all(np.isnan(flat), axis=1)
    fv = flat[valid]
    ext = (np.nanmax if mode == 'max' else np.nanmin)(fv, axis=1, keepdims=True)
    return np.sum(fv == ext, axis=0)          # ties counted in each tied year (as original)

rec_highs = yearly_records(raw_tmax, 'max')
rec_lows  = yearly_records(raw_tmin, 'min')

np.savez(OUT_DIR / "records_cache_raw.npz",
         years=years, rec_highs=rec_highs, rec_lows=rec_lows, n_good=n_good)
adj = np.load(OUT_DIR / "records_cache.npz")
print(f"Saved records_cache_raw.npz")
print(f"  raw      total highs={rec_highs.sum():,}  lows={rec_lows.sum():,}")
print(f"  adjusted total highs={int(adj['rec_highs'].sum()):,}  lows={int(adj['rec_lows'].sum()):,}")

# The raw checkpoints are large (~830 MB) and reconstruct in seconds, so we
# don't keep them — everything downstream uses records_cache_raw.npz.
del raw_tmax, raw_tmin
for f in (RAW_TMAX, RAW_TMIN):
    f.unlink(missing_ok=True)
print("Removed raw checkpoint memmaps.")
