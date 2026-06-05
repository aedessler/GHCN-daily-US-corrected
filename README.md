# U.S. Daily Temperature Records — Adjusted Data

Reproduces the "Conterminous U.S. Observed Number of Daily Temperature Records" figure (originally by Chris Mortz / NOAA) using **homogeneity-adjusted** GHCND daily data rather than raw observations.

## What it produces

Three-panel figure (`adjusted_records.png`) and a CSV of annual counts (`adjusted_records.csv`):

1. **Daily Record Highs** — number of station–day combinations that set an all-time TMAX record in each year
2. **Daily Record Lows** — same for all-time TMIN records
3. **Ratio of Highs to Lows** — values > 1 indicate more record-breaking heat than cold

![Adjusted daily temperature records](adjusted_records.png)

Additional outputs:

| File | Produced by | Description |
|------|-------------|-------------|
| `station_map.png` | `plot_station_map.py` | Cartopy map of the stations used in the figure |
| `good_stations.csv` | `station_density.py` | Station IDs and lat/lon of the stations used |
| `adjusted_records_areaweighted.png` | `compute_area_weighted.py` + `replot_areaweighted.py` | Equal-area-weighted version of the three-panel figure (see [Spatial weighting](#spatial-weighting)) |

## Data sources

| Data | Path |
|------|------|
| GHCND daily observations (by year) | `/Volumes/adessler_lab/GHCND/by_year/YYYY.csv.gz` |
| Monthly FLs.52j adjustment offsets | `/Volumes/adessler_lab/GHCND/monthly_data/processed/monthly_offsets.nc` |
| Station metadata | Downloaded at runtime from NOAA NCEI |

The adjustment offsets are from the [NOAA North American Dataset (FLs.52j)](https://www.ncei.noaa.gov/data/north-american-dataset/), which corrects for station moves, time-of-observation changes, and instrument changes. The adjustment is applied as:

```
adjusted_temp = raw_temp_C + monthly_offset_C
```

where `monthly_offset = FLs.52j − raw` (from the NetCDF file), and the same offset is applied to all days within a calendar month. The `monthly_offsets.nc` file is produced by a companion repository: [aedessler/GHCN-monthly-offsets](https://github.com/aedessler/GHCN-monthly-offsets).

## Station selection

Matches the methodology of the original figure:
- **CONUS only**: latitude 24.5–49.5°N, longitude −125 to −66°W
- **≥100 years** of TMAX and TMIN data within the 1900–2024 study period
- **≥80%** of all possible station-days have valid, unflagged observations

This yields **1,266 stations**. (The stations actually used are not stored in `records_cache.npz`, which holds only the annual totals; they are recovered by re-applying the completeness filter to the checkpoint memmaps — see `plot_station_map.py` and `station_density.py`.)

![Map of stations used](station_map.png)

## Record definition

For each station × calendar day-of-year pair, the year with the **highest adjusted TMAX** across all years receives one record high, and the year with the **lowest adjusted TMIN** receives one record low. Ties are counted in all tied years. Each station–DOY pair contributes at most ~1 record per type per year, so the expected count in any given year ≈ (N stations × 365) / N years ≈ 2,000.

## Spatial weighting

The GHCNd network is much denser in the eastern U.S. than the west: 68% of the 1,266 stations lie east of 100°W, although the west is roughly half of CONUS by area. Because the main figure **sums** record-setting station-days nationally, the totals are weighted toward eastern climate.

`compute_area_weighted.py` re-aggregates the records on an equal-area grid to remove this bias: each station is weighted by `cos(lat) / (stations in its 2° cell)`, normalized to mean 1, so every occupied grid cell contributes in proportion to its **area** rather than its station count. The unweighted recompute reproduces `records_cache.npz` exactly as a check.

**Result:** area-weighting barely changes the curves — the high-to-low ratio shifts from 0.67→0.63 (1900–1929) and 5.77→5.61 (2010–2024). The eastern over-sampling is real but does not drive the trend, so the original figure's conclusion is robust. `GRID_DEG` (default 2.0°) is the main tunable knob.

![Area-weighted adjusted records](adjusted_records_areaweighted.png)

## Usage

```bash
python compute_adjusted_records.py    # reads year files -> records_cache.npz (slow)
python replot.py                       # records_cache.npz -> adjusted_records.png + .csv

python plot_station_map.py             # cartopy map of stations -> station_map.png
python station_density.py              # E/W density stats -> good_stations.csv
python compute_area_weighted.py        # area-weighted records -> records_cache_areaweighted.npz
python replot_areaweighted.py          # -> adjusted_records_areaweighted.png
```

Requires: `numpy`, `pandas`, `xarray`, `matplotlib`, and `cartopy` (all in the Miniconda base environment).

`compute_adjusted_records.py` runs approximately 50–60 minutes, dominated by reading and filtering the compressed year files from the external drive. The remaining scripts run in seconds from the cached checkpoint memmaps (`compute_area_weighted.py` re-reads them, `plot_station_map.py` / `station_density.py` also download station metadata from NOAA NCEI).

## Original figure

`HJuIBQRWUAM7SPU.png` — unadjusted version, computed from 711 stations, 1895–2024.
