#!/usr/bin/env python3
"""Quantify the E/W station-density imbalance and cache good-station coords."""
import numpy as np, csv
from pathlib import Path
from helpers import fetch_station_coords

OUT = Path("/Users/adessler/Desktop/recHighs")
DATA = OUT / "data"
meta = np.load(DATA / "checkpoint_meta.npz", allow_pickle=True)
shape = tuple(int(x) for x in meta['shape']); cands = list(meta['candidates'])
N_YEARS = 125; MIN_YEARS = 100; FRAC = 0.80

tmax = np.memmap(DATA/"checkpoint_tmax.dat", dtype='float32', mode='r', shape=shape)
tmin = np.memmap(DATA/"checkpoint_tmin.dat", dtype='float32', mode='r', shape=shape)
npos = N_YEARS * 365.25
good = ((np.sum(~np.all(np.isnan(tmax),axis=2),axis=1) >= MIN_YEARS) &
        (np.sum(~np.all(np.isnan(tmin),axis=2),axis=1) >= MIN_YEARS) &
        (np.sum(~np.isnan(tmax),axis=(1,2))/npos >= FRAC) &
        (np.sum(~np.isnan(tmin),axis=(1,2))/npos >= FRAC))
gs = [cands[i] for i in range(len(cands)) if good[i]]

m = fetch_station_coords()
coords = [(s, *m[s]) for s in gs if s in m]
lons = np.array([c[2] for c in coords])
with open(DATA/"good_stations.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["station","lat","lon"]); w.writerows(coords)

# CONUS spans -125..-66 (59° lon). 100°W is the conventional E/W divide.
east = (lons > -100).sum(); west = (lons <= -100).sum()
# fraction of CONUS *width* that lies west of 100W
west_frac_area = (-100 - (-125)) / (-66 - (-125))   # ~0.42 of lon-span
print(f"Total good stations: {len(coords)}")
print(f"East of 100W: {east} ({east/len(coords)*100:.0f}%)")
print(f"West of 100W: {west} ({west/len(coords)*100:.0f}%)")
print(f"...but the West is ~{west_frac_area*100:.0f}% of CONUS longitudinal span")
for lo, hi in [(-125,-115),(-115,-105),(-105,-95),(-95,-85),(-85,-66)]:
    n = ((lons > lo) & (lons <= hi)).sum()
    print(f"  {abs(lo)}W..{abs(hi)}W: {n:4d} stations")
print("Saved good_stations.csv")
