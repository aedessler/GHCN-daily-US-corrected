#!/usr/bin/env python3
"""
Cartopy map of the stations used in adjusted_records.png, over a color map of
the 1934-1936 JJA mean surface-temperature anomaly (the Dust Bowl heat).

The station set is the `n_good` stations that pass the completeness filter in
compute_adjusted_records.py. records_cache.npz only stores the annual counts,
so we recover the exact set by re-applying the same filter to the checkpoint
memmaps, then join NOAA lat/lon metadata and plot.

The background field is the Berkeley Earth gridded TMAX anomaly
(Complete_TMAX_LatLong1.nc), averaged over June/July/August of 1934-1936.
"""

import warnings
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from helpers import fetch_station_coords

# Cosmetic, data-safe warnings:
#  - Berkeley Earth stores valid_min/valid_max as float64 on a float32 var,
#    so netCDF4 declines to apply them as a mask (missing values are NaN anyway).
#  - cartopy's '50m' geometries trip shapely topology checks during buffering.
warnings.filterwarnings("ignore", message=r"WARNING: valid_m(in|ax) not used")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

OUT_DIR   = Path("/Users/adessler/Desktop/recHighs")
DATA_DIR  = OUT_DIR / "data"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_TMAX = DATA_DIR / "checkpoint_tmax.dat"
CKPT_TMIN = DATA_DIR / "checkpoint_tmin.dat"
CKPT_META = DATA_DIR / "checkpoint_meta.npz"
TMAX_NC   = OUT_DIR / "Complete_TMAX_LatLong1.nc"

# Must match compute_adjusted_records.py
STUDY_START, STUDY_END = 1900, 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80
LAT_MIN, LAT_MAX =  24.5,  49.5
LON_MIN, LON_MAX = -125.0, -66.0

# Background temperature field
FIELD_YEARS  = list(np.arange(1930,1940))
FIELD_MONTHS = [6, 7, 8]   # JJA

# Coarse grid (deg) for the station-density vs. anomaly scatter
DENSITY_BIN_DEG = 2.0
# Set True to also produce figures/density_vs_anomaly.png (see section 5).
RUN_DENSITY_SCATTER = False

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
# 3. 1934-1936 JJA mean TMAX anomaly from Berkeley Earth
# ---------------------------------------------------------------
print(f"Reading {TMAX_NC.name} …")
ds      = nc.Dataset(TMAX_NC)
grid_lat = ds.variables['latitude'][:]
grid_lon = ds.variables['longitude'][:]
time    = ds.variables['time'][:]                       # decimal years A.D.
yr      = np.floor(time).astype(int)
mo      = np.round((time - yr) * 12 + 0.5).astype(int)  # 1..12

sel = np.isin(yr, FIELD_YEARS) & np.isin(mo, FIELD_MONTHS)
print(f"Averaging {sel.sum()} monthly fields "
      f"(JJA {FIELD_YEARS[0]}-{FIELD_YEARS[-1]})")
anom = np.nanmean(ds.variables['temperature'][sel, :, :], axis=0)

# Keep land only so the ocean stays a clean blue under the coastlines.
land_mask = ds.variables['land_mask'][:]
anom = np.ma.masked_where(land_mask < 0.5, anom)
ds.close()

# Symmetric color limits from the CONUS subset we actually display.
la = (grid_lat >= LAT_MIN) & (grid_lat <= LAT_MAX)
lo = (grid_lon >= LON_MIN) & (grid_lon <= LON_MAX)
vmax = np.ceil(np.nanmax(np.abs(anom[np.ix_(la, lo)])))

# ---------------------------------------------------------------
# 4. Map
# ---------------------------------------------------------------
proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                             standard_parallels=(33, 45))
fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=proj)
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f5f5f0', zorder=0)

# Temperature anomaly field beneath everything else.
mesh = ax.pcolormesh(grid_lon, grid_lat, anom, transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     shading='auto', zorder=1)

ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dcebf7', zorder=2)
ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5, zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='black', linewidth=0.7, zorder=3)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.7, zorder=3)

ax.scatter(lons, lats, transform=ccrs.PlateCarree(),
           s=14, c='k', edgecolor='white', linewidth=0.3,
           alpha=0.9, zorder=5)

cbar = fig.colorbar(mesh, ax=ax, orientation='vertical',
                    shrink=0.7, pad=0.02, extend='both')
cbar.set_label("JJA TMAX anomaly (°C, vs 1951–1980)", fontsize=10)

ax.set_title(
    f"{len(lats)} GHCND stations used in the adjusted records figure\n"
    f"over the {FIELD_YEARS[0]}–{FIELD_YEARS[-1]} JJA mean temperature anomaly",
    fontsize=12, fontweight='bold')

out = FIG_DIR / "station_map.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------
# 5. Station density vs. temperature anomaly  (optional; RUN_DENSITY_SCATTER)
# ---------------------------------------------------------------
if RUN_DENSITY_SCATTER:
    # Aggregate both fields onto a coarse CONUS grid: station counts -> density
    # (area-corrected, stations per 10^4 km^2), and the 1-deg anomaly -> bin mean.
    lon_edges = np.arange(LON_MIN, LON_MAX + DENSITY_BIN_DEG, DENSITY_BIN_DEG)
    lat_edges = np.arange(LAT_MIN, LAT_MAX + DENSITY_BIN_DEG, DENSITY_BIN_DEG)

    counts, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])  # (nlon, nlat)

    # Bin-mean anomaly from the 1-deg land cells whose centers fall in the box.
    gx, gy = np.meshgrid(grid_lon, grid_lat)            # (nlat, nlon)
    a      = np.ma.filled(anom, np.nan)
    inbox  = ((gx >= LON_MIN) & (gx <= LON_MAX) &
              (gy >= LAT_MIN) & (gy <= LAT_MAX) & np.isfinite(a))
    sum_a, _, _ = np.histogram2d(gx[inbox], gy[inbox],
                                 bins=[lon_edges, lat_edges], weights=a[inbox])
    n_a,   _, _ = np.histogram2d(gx[inbox], gy[inbox], bins=[lon_edges, lat_edges])
    with np.errstate(invalid='ignore'):
        mean_anom = sum_a / n_a                          # NaN where bin has no land

    # Bin area (km^2) varies with latitude; normalize counts to stations / 10^4 km^2.
    lat_ctr = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    km_per_deg = 111.32
    bin_area = (DENSITY_BIN_DEG * km_per_deg) * \
               (DENSITY_BIN_DEG * km_per_deg * np.cos(np.radians(lat_ctr)))  # (nlat,)
    density = counts / (bin_area[None, :] / 1e4)

    keep = np.isfinite(mean_anom)                        # CONUS land bins
    x, y = density[keep], mean_anom[keep]

    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)
    xfit = np.array([x.min(), x.max()])

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(x, y, s=18, c='tab:red', edgecolor='black', linewidth=0.25, alpha=0.7)
    ax2.plot(xfit, slope * xfit + intercept, 'k--', linewidth=1.2,
             label=f"fit: {slope:+.3f} °C per (10⁴ km²)⁻¹")
    ax2.axhline(0, color='gray', linewidth=0.6, zorder=0)
    ax2.set_xlabel("Station density (stations per 10⁴ km²)")
    ax2.set_ylabel(f"{FIELD_YEARS[0]}–{FIELD_YEARS[-1]} JJA TMAX anomaly (°C)")
    ax2.set_title(f"Station density vs. temperature anomaly\n"
                  f"{DENSITY_BIN_DEG:g}° bins over CONUS  (Pearson r = {r:+.2f}, "
                  f"n = {keep.sum()})", fontsize=12, fontweight='bold')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.3)

    out2 = FIG_DIR / "density_vs_anomaly.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}  (r = {r:+.3f}, slope = {slope:+.4f} °C per 10⁴ km⁻²)")
    plt.close()
