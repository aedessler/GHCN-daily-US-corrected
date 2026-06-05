#!/usr/bin/env python3
"""Quantify the E/W station-density imbalance and cache good-station coords."""
import numpy as np, urllib.request, csv
from pathlib import Path

OUT = Path("/Users/adessler/Desktop/recHighs")
meta = np.load(OUT / "checkpoint_meta.npz", allow_pickle=True)
shape = tuple(int(x) for x in meta['shape']); cands = list(meta['candidates'])
N_YEARS = 125; MIN_YEARS = 100; FRAC = 0.80

tmax = np.memmap(OUT/"checkpoint_tmax.dat", dtype='float32', mode='r', shape=shape)
tmin = np.memmap(OUT/"checkpoint_tmin.dat", dtype='float32', mode='r', shape=shape)
npos = N_YEARS * 365.25
good = ((np.sum(~np.all(np.isnan(tmax),axis=2),axis=1) >= MIN_YEARS) &
        (np.sum(~np.all(np.isnan(tmin),axis=2),axis=1) >= MIN_YEARS) &
        (np.sum(~np.isnan(tmax),axis=(1,2))/npos >= FRAC) &
        (np.sum(~np.isnan(tmin),axis=(1,2))/npos >= FRAC))
gs = [cands[i] for i in range(len(cands)) if good[i]]

lines = urllib.request.urlopen(
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt").read().decode().split('\n')
m = {}
for ln in lines:
    if len(ln) < 30: continue
    try: m[ln[:11].strip()] = (float(ln[12:20]), float(ln[21:30]))
    except ValueError: pass

coords = [(s, *m[s]) for s in gs if s in m]
lons = np.array([c[2] for c in coords])
with open(OUT/"good_stations.csv", "w", newline="") as f:
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
