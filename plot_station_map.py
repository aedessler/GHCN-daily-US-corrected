#!/usr/bin/env python3
"""
Cartopy map of the stations used in adjusted_records.png.

The figure's station set is the `n_good` stations that pass the completeness
filter in compute_adjusted_records.py. records_cache.npz only stores the annual
counts, so we recover the exact set by re-applying the same filter to the
checkpoint memmaps, then join NOAA lat/lon metadata and plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from helpers import fetch_station_coords

OUT_DIR   = Path("/Users/adessler/Desktop/recHighs")
DATA_DIR  = OUT_DIR / "data"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_TMAX = DATA_DIR / "checkpoint_tmax.dat"
CKPT_TMIN = DATA_DIR / "checkpoint_tmin.dat"
CKPT_META = DATA_DIR / "checkpoint_meta.npz"

# Must match compute_adjusted_records.py
STUDY_START, STUDY_END = 1900, 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80
LAT_MIN, LAT_MAX =  24.5,  49.5
LON_MIN, LON_MAX = -125.0, -66.0

# ---------------------------------------------------------------
# 1. Recover the "good" station set from the checkpoint memmaps
# ---------------------------------------------------------------
meta        = np.load(CKPT_META, allow_pickle=True)
shape       = tuple(int(x) for x in meta['shape'])
candidates  = list(meta['candidates'])
print(f"Checkpoint: {len(candidates)} candidate stations, shape {shape}")

tmax_data = np.memmap(CKPT_TMAX, dtype='float32', mode='r', shape=shape)
tmin_data = np.memmap(CKPT_TMIN, dtype='float32', mode='r', shape=shape)

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
good_stations = [candidates[i] for i in range(len(candidates)) if good[i]]
print(f"Stations passing completeness filter: {len(good_stations)}")

# ---------------------------------------------------------------
# 2. NOAA lat/lon metadata for the good stations
# ---------------------------------------------------------------
print("Downloading station metadata …")
sta_meta = fetch_station_coords()

lats = np.array([sta_meta[s][0] for s in good_stations if s in sta_meta])
lons = np.array([sta_meta[s][1] for s in good_stations if s in sta_meta])
print(f"Plotting {len(lats)} stations with coordinates")

# ---------------------------------------------------------------
# 3. Map
# ---------------------------------------------------------------
proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                             standard_parallels=(33, 45))
fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=proj)
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f5f5f0')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dcebf7')
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='black', linewidth=0.7)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.7)

ax.scatter(lons, lats, transform=ccrs.PlateCarree(),
           s=14, c='tab:red', edgecolor='black', linewidth=0.25,
           alpha=0.8, zorder=5)

ax.set_title(
    f"GHCND stations used in the adjusted records figure\n"
    f"{len(lats)} CONUS stations with ≥{MIN_YEARS} yrs and ≥"
    f"{MIN_DATA_FRAC*100:.0f}% data, {STUDY_START}–{STUDY_END}",
    fontsize=12, fontweight='bold')

out = FIG_DIR / "station_map.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
