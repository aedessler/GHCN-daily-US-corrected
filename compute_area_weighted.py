#!/usr/bin/env python3
"""
Area-weighted version of the adjusted records computation.

The original figure SUMS raw record-setting station-days nationally, so the
denser Eastern network dominates the totals. Here we re-aggregate on an
equal-area grid: each station gets weight a_cell / n_stations_in_cell (with
a_cell = cos(latitude), the relative area of a lat/lon cell), so every occupied
grid cell contributes in proportion to its AREA, not its station count.
Weights are normalized to mean 1 so the y-axis stays comparable to the original.

Reads the checkpoint memmaps (full adjusted-temperature arrays), not the cache,
because area weighting needs the per-station record assignments.
Validates by reproducing the unweighted cached totals exactly.
"""

import numpy as np
import csv
from pathlib import Path

OUT_DIR   = Path("/Users/adessler/Desktop/recHighs")
CKPT_TMAX = OUT_DIR / "checkpoint_tmax.dat"
CKPT_TMIN = OUT_DIR / "checkpoint_tmin.dat"
CKPT_META = OUT_DIR / "checkpoint_meta.npz"
COORDS    = OUT_DIR / "good_stations.csv"   # written by station_density.py
CACHE     = OUT_DIR / "records_cache.npz"

STUDY_START, STUDY_END = 1900, 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80
N_DOY         = 366
GRID_DEG      = 2.0   # equal-area grid resolution (tunable knob)

# ---------------------------------------------------------------
# 1. Recover good stations + their adjusted-temperature arrays
# ---------------------------------------------------------------
meta  = np.load(CKPT_META, allow_pickle=True)
shape = tuple(int(x) for x in meta['shape'])
cands = list(meta['candidates'])
tmax  = np.memmap(CKPT_TMAX, dtype='float32', mode='r', shape=shape)
tmin  = np.memmap(CKPT_TMIN, dtype='float32', mode='r', shape=shape)

npos = N_YEARS * 365.25
good = (
    (np.sum(~np.all(np.isnan(tmax), axis=2), axis=1) >= MIN_YEARS) &
    (np.sum(~np.all(np.isnan(tmin), axis=2), axis=1) >= MIN_YEARS) &
    (np.sum(~np.isnan(tmax), axis=(1, 2)) / npos >= MIN_DATA_FRAC) &
    (np.sum(~np.isnan(tmin), axis=(1, 2)) / npos >= MIN_DATA_FRAC)
)
good_idx      = np.where(good)[0]
good_stations = [cands[i] for i in good_idx]
n_good        = len(good_stations)
tmax_g        = np.asarray(tmax[good_idx])   # (n_good, N_YEARS, N_DOY)
tmin_g        = np.asarray(tmin[good_idx])
print(f"Good stations: {n_good}")

# ---------------------------------------------------------------
# 2. Coordinates -> equal-area grid cells -> per-station weights
# ---------------------------------------------------------------
coord = {}
with open(COORDS) as f:
    for r in csv.DictReader(f):
        coord[r['station']] = (float(r['lat']), float(r['lon']))
lats = np.array([coord[s][0] for s in good_stations])
lons = np.array([coord[s][1] for s in good_stations])

ilat = np.floor(lats / GRID_DEG).astype(int)
ilon = np.floor(lons / GRID_DEG).astype(int)
cell_center_lat = (ilat + 0.5) * GRID_DEG
a_cell = np.cos(np.radians(cell_center_lat))           # relative cell area

# stations per cell
cell_key = ilat.astype(np.int64) * 100000 + ilon       # unique per (ilat,ilon)
uniq, inv, counts = np.unique(cell_key, return_inverse=True, return_counts=True)
n_in_cell = counts[inv]

w_raw = a_cell / n_in_cell                              # each cell sums to a_cell
w = w_raw * n_good / w_raw.sum()                        # normalize to mean 1
print(f"Occupied grid cells ({GRID_DEG}deg): {len(uniq)}")
print(f"Weight range: {w.min():.3f}..{w.max():.3f}  (mean {w.mean():.3f})")

# ---------------------------------------------------------------
# 3. Records per year: unweighted (validation) + area-weighted
# ---------------------------------------------------------------
def yearly_records(data, mode):
    """mode: 'max' for highs, 'min' for lows. Returns (unweighted, weighted)."""
    flat = np.ascontiguousarray(data.transpose(0, 2, 1).reshape(-1, N_YEARS))
    row_station = np.arange(flat.shape[0]) // N_DOY
    w_rows = w[row_station]
    valid = ~np.all(np.isnan(flat), axis=1)
    fv, wv = flat[valid], w_rows[valid]
    ext = (np.nanmax if mode == 'max' else np.nanmin)(fv, axis=1, keepdims=True)
    match = (fv == ext)                 # ties counted in each tied year (as original)
    return match.sum(axis=0), np.dot(wv, match)

hi_u, hi_w = yearly_records(tmax_g, 'max')
lo_u, lo_w = yearly_records(tmin_g, 'min')

# ---------------------------------------------------------------
# 4. Validate against the original cache
# ---------------------------------------------------------------
c = np.load(CACHE)
ok_hi = np.array_equal(hi_u, c['rec_highs'])
ok_lo = np.array_equal(lo_u, c['rec_lows'])
print(f"Unweighted reproduces cache:  highs={ok_hi}  lows={ok_lo}")
if not (ok_hi and ok_lo):
    print("  WARNING: unweighted recompute does not match cache exactly.")

years = np.arange(STUDY_START, STUDY_END + 1)
np.savez(OUT_DIR / "records_cache_areaweighted.npz",
         years=years, rec_highs=hi_w, rec_lows=lo_w,
         rec_highs_unw=hi_u, rec_lows_unw=lo_u,
         n_good=n_good, grid_deg=GRID_DEG)
print(f"Saved: records_cache_areaweighted.npz")
print(f"  weighted total highs={hi_w.sum():,.0f} lows={lo_w.sum():,.0f}"
      f"  (unweighted {hi_u.sum():,} / {lo_u.sum():,})")
