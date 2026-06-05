#!/usr/bin/env python3
"""
Shared helpers for the record-counting scripts.

Two independent utilities live here:

1. Station coordinates (`fetch_station_coords`, `coords_for`) — download and
   parse NOAA's fixed-width ghcnd-stations.txt. Anything that needs coordinates
   for the good stations (area weighting, the station map, the E/W density
   stats) gets them from the live metadata for the current run, rather than from
   a cached CSV that could drift out of sync with the checkpoints.

2. Raw reconstruction (`reconstruct_raw`) — recover raw (unadjusted) temps from
   the adjusted arrays by removing the monthly offsets.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Station coordinates
# ---------------------------------------------------------------------------
# GHCN-daily fixed-width spec: ID in [0:11], latitude in [12:20], longitude in
# [21:30].
STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"


def fetch_station_coords(url=STATIONS_URL):
    """Return {station_id: (lat, lon)} for every station in ghcnd-stations.txt."""
    lines = urllib.request.urlopen(url).read().decode().split('\n')
    coords = {}
    for ln in lines:
        if len(ln) < 30:
            continue
        try:
            coords[ln[:11].strip()] = (float(ln[12:20]), float(ln[21:30]))
        except ValueError:
            pass
    return coords


def coords_for(station_ids, url=STATIONS_URL):
    """Return parallel (lats, lons) numpy arrays aligned to `station_ids`.

    Raises SystemExit listing any IDs absent from the metadata, so a station
    that can't be located fails loudly instead of being silently dropped from
    the weighting."""
    allc = fetch_station_coords(url)
    missing = [str(s) for s in station_ids if s not in allc]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(station_ids)} stations have no coordinates "
            f"in ghcnd-stations.txt (e.g. {missing[:3]}). NOAA's station "
            f"metadata may have changed; cannot assign area weights.")
    lats = np.array([allc[s][0] for s in station_ids])
    lons = np.array([allc[s][1] for s in station_ids])
    return lats, lons


# ---------------------------------------------------------------------------
# Raw reconstruction
# ---------------------------------------------------------------------------
# adjusted = raw + monthly_offset, so raw = round((adjusted - offset) * 10) / 10
# (snap to 0.1° -> bit-exact, since raw temps are whole tenths of a degree).
# Needs the offsets NetCDF on the external drive; no year files required.
OFFSETS_FILE = Path("/Volumes/adessler_lab/GHCND/monthly_data/processed/monthly_offsets.nc")


def reconstruct_raw(adj_tmax_good, adj_tmin_good, good_station_ids,
                    offsets_file=OFFSETS_FILE, study_start=1900, study_end=2024):
    """Return (raw_tmax, raw_tmin) for the good stations, same shape as the
    adjusted arrays, by removing the monthly offsets."""
    offsets_file = Path(offsets_file)
    if not offsets_file.exists():
        raise SystemExit(
            f"Offsets file not found: {offsets_file}\n"
            "Mount the external drive (adessler_lab) to reconstruct raw temps.")

    n_good, n_years, _ = adj_tmax_good.shape
    ds = xr.open_dataset(offsets_file)
    off_idx_map = {sid: i for i, sid in enumerate(ds.station_id.values)}
    off_times   = pd.DatetimeIndex(ds.time.values)
    t0, t1 = pd.Timestamp(f'{study_start}-01-01'), pd.Timestamp(f'{study_end}-12-01')
    time_mask   = (off_times >= t0) & (off_times <= t1)
    study_times = off_times[time_mask]
    time_to_idx = {(t.year, t.month): i for i, t in enumerate(study_times)}

    tmax_all = ds.tmax_offset.values
    tmin_all = ds.tmin_offset.values
    tmax_off = np.zeros((n_good, len(study_times)), np.float32)
    tmin_off = np.zeros((n_good, len(study_times)), np.float32)
    for gi, sid in enumerate(good_station_ids):
        oi = off_idx_map.get(sid)
        if oi is not None:
            tv, nv = tmax_all[oi, time_mask], tmin_all[oi, time_mask]
            tmax_off[gi] = np.where(np.isnan(tv), 0.0, tv)
            tmin_off[gi] = np.where(np.isnan(nv), 0.0, nv)
    ds.close()

    years = np.arange(study_start, study_end + 1)
    raw_tmax = np.full_like(adj_tmax_good, np.nan)
    raw_tmin = np.full_like(adj_tmin_good, np.nan)
    for adj, raw, off in [(adj_tmax_good, raw_tmax, tmax_off),
                          (adj_tmin_good, raw_tmin, tmin_off)]:
        for yidx, year in enumerate(years):
            ndays  = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            months = pd.date_range(f'{year}-01-01', periods=ndays, freq='D').month.values
            for m in range(1, 13):
                doys = np.where(months == m)[0]
                tidx = time_to_idx.get((year, m), -1)
                block = adj[:, yidx, doys]
                if tidx >= 0:
                    block = block - off[:, tidx][:, None]
                raw[:, yidx, doys] = np.round(block * 10.0) / 10.0   # snap -> exact 0.1°
    return raw_tmax, raw_tmin
