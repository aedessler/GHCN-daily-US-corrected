#!/usr/bin/env python3
"""
General records time-series figure: overlay any subset of {raw, adjusted,
weighted} on the standard three-panel layout (record highs, record lows, and
their ratio), each drawn as a centered running mean.

All series are computed from the adjusted checkpoint memmaps for the same 1266
complete stations, so any selection is mutually consistent:

  raw       unadjusted GHCNd, reconstructed by removing the FLs.52j monthly
            offsets (needs the external offsets file; only loaded if selected)
  adjusted  FLs.52j-adjusted, summed nationally
  weighted  FLs.52j-adjusted, equal-area gridded (cos(lat) / stations-in-cell)

Examples
  python plot_records.py                         # all three
  python plot_records.py adjusted                # baseline only
  python plot_records.py raw adjusted            # adjusted vs raw
  python plot_records.py adjusted weighted -o figures/my.png
  python plot_records.py raw adjusted weighted --csv data/records_compare.csv

The station map (plot_station_map.py) and the MJJAS figures are separate and
left untouched.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import coords_for, reconstruct_raw, OFFSETS_FILE

OUT_DIR   = Path("/Users/adessler/Desktop/recHighs")
DATA_DIR  = OUT_DIR / "data"
FIG_DIR   = OUT_DIR / "figures"
CKPT_TMAX = DATA_DIR / "checkpoint_tmax.dat"
CKPT_TMIN = DATA_DIR / "checkpoint_tmin.dat"
CKPT_META = DATA_DIR / "checkpoint_meta.npz"

STUDY_START, STUDY_END = 1900, 2024
N_YEARS       = STUDY_END - STUDY_START + 1
MIN_YEARS     = 100
MIN_DATA_FRAC = 0.80
N_DOY         = 366

ALL_SERIES = ['raw', 'adjusted', 'weighted']          # canonical draw/legend order
SHORT      = {'raw': 'raw', 'adjusted': 'adj', 'weighted': 'wtd'}
LABEL      = {'raw': 'Raw (unadjusted)',
              'adjusted': 'Adjusted',
              'weighted': 'Adjusted, area-weighted'}
# per-series line style; 'adjusted' inherits each panel's accent color, the
# others use fixed colors so the three are always distinguishable.
STYLE      = {'raw':      dict(ls='--', lw=2.0, color='0.5'),
              'adjusted': dict(ls='-',  lw=2.0, color=None),
              'weighted': dict(ls='-.', lw=2.0, color='tab:green')}


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_good():
    """Recover the 1266 complete stations and their adjusted TMAX/TMIN arrays
    by re-applying the completeness filter to the checkpoint memmaps."""
    meta  = np.load(CKPT_META, allow_pickle=True)
    shape = tuple(int(x) for x in meta['shape'])
    cands = list(meta['candidates'])
    tmax  = np.memmap(CKPT_TMAX, dtype='float32', mode='r', shape=shape)
    tmin  = np.memmap(CKPT_TMIN, dtype='float32', mode='r', shape=shape)
    npos  = N_YEARS * 365.25
    good = (
        (np.sum(~np.all(np.isnan(tmax), axis=2), axis=1) >= MIN_YEARS) &
        (np.sum(~np.all(np.isnan(tmin), axis=2), axis=1) >= MIN_YEARS) &
        (np.sum(~np.isnan(tmax), axis=(1, 2)) / npos >= MIN_DATA_FRAC) &
        (np.sum(~np.isnan(tmin), axis=(1, 2)) / npos >= MIN_DATA_FRAC)
    )
    gidx     = np.where(good)[0]
    good_ids = [cands[i] for i in gidx]
    return good_ids, np.asarray(tmax[gidx]), np.asarray(tmin[gidx])


def yearly(data, mode, n_sel, weights=None):
    """Per-year record count with fractional tie-splitting; optional per-station
    weights (one value per station, broadcast over its n_sel day-of-year rows).
    n_sel is the number of day-of-year columns in `data` (366, or fewer for a
    seasonal subset)."""
    flat = np.ascontiguousarray(data.transpose(0, 2, 1).reshape(-1, N_YEARS))
    valid = ~np.all(np.isnan(flat), axis=1)
    fv = flat[valid]
    ext = (np.nanmax if mode == 'max' else np.nanmin)(fv, axis=1, keepdims=True)
    match = (fv == ext).astype(np.float64)
    match /= match.sum(axis=1, keepdims=True)   # split ties: N tied years each get 1/N
    if weights is None:
        return match.sum(axis=0)
    wr = weights[np.arange(flat.shape[0]) // n_sel][valid]
    return np.dot(wr, match)


def area_weights(good_ids, grid_deg):
    """Equal-area per-station weights (mean 1) on a grid_deg° grid; coordinates
    pulled live from NOAA metadata for exactly these stations."""
    lats, lons = coords_for(good_ids)
    ilat = np.floor(lats / grid_deg).astype(int)
    ilon = np.floor(lons / grid_deg).astype(int)
    a_cell = np.cos(np.radians((ilat + 0.5) * grid_deg))
    _, inv, counts = np.unique(ilat.astype(np.int64) * 100000 + ilon,
                               return_inverse=True, return_counts=True)
    w_raw = a_cell / counts[inv]
    return w_raw * len(good_ids) / w_raw.sum()


def centered_mean(a, win):
    half = win // 2
    out = np.full_like(a, np.nan)
    for i in range(half, len(a) - half):
        out[i] = np.nanmean(a[i - half:i + half + 1])
    return out


def compute(selection, grid_deg, doy_sel=None):
    """Return (years, {series: (highs, lows)}, n_good) for the selected series.

    doy_sel chooses which day-of-year columns to count (default: all 366); pass a
    subset (e.g. the May–Sep indices) for a seasonal figure."""
    good_ids, adj_tmax, adj_tmin = load_good()
    n_good = len(good_ids)
    if doy_sel is None:
        doy_sel = np.arange(N_DOY)
    n_sel = len(doy_sel)
    print(f"Stations: {n_good}" + (f"   DOY slots: {n_sel}" if n_sel != N_DOY else ""))
    at, an = adj_tmax[:, :, doy_sel], adj_tmin[:, :, doy_sel]
    out = {}
    if 'adjusted' in selection:
        out['adjusted'] = (yearly(at, 'max', n_sel), yearly(an, 'min', n_sel))
    if 'weighted' in selection:
        w = area_weights(good_ids, grid_deg)
        print(f"Area weights ({grid_deg:g}° grid): {w.min():.3f}..{w.max():.3f}")
        out['weighted'] = (yearly(at, 'max', n_sel, w), yearly(an, 'min', n_sel, w))
    if 'raw' in selection:
        # raw needs the full arrays (offsets are monthly) -> reconstruct, then
        # subset. only here do we touch the external drive.
        raw_tmax, raw_tmin = reconstruct_raw(adj_tmax, adj_tmin, good_ids,
                                             OFFSETS_FILE, STUDY_START, STUDY_END)
        rt, rn = raw_tmax[:, :, doy_sel], raw_tmin[:, :, doy_sel]
        out['raw'] = (yearly(rt, 'max', n_sel), yearly(rn, 'min', n_sel))
    years = np.arange(STUDY_START, STUDY_END + 1)
    for name in selection:
        hi, lo = out[name]
        print(f"  {name:9s} total highs={hi.sum():,.0f} lows={lo.sum():,.0f}")
    return years, out, n_good


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
def make_figure(years, series, selection, n_good, smooth, grid_deg, outpath,
                title=None, season=None):
    ratio = {k: np.where(lo > 0, hi / lo, np.nan) for k, (hi, lo) in series.items()}

    def draw(ax, idx, accent, title, ylabel):
        get = (lambda k: series[k][idx]) if idx in (0, 1) else (lambda k: ratio[k])
        for name in selection:                       # canonical order
            st = dict(STYLE[name])
            if st['color'] is None:
                st['color'] = accent
            ax.plot(years, centered_mean(get(name), smooth), label=LABEL[name], **st)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='upper left')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 3, 2]})
    shown = ", ".join(LABEL[s] for s in selection)
    head = title or "Conterminous U.S. Daily Temperature Records"
    fig.suptitle(f"{head}\n{STUDY_START}–{STUDY_END}  ·  {shown}",
                 fontsize=13, fontweight='bold', y=0.99)
    draw(ax1, 0, 'tab:red',  "Daily Record Highs", "Number of Records")
    draw(ax2, 1, 'tab:blue', "Daily Record Lows",  "Number of Records")
    draw(ax3, 2, 'black',    "Ratio of Record Highs to Record Lows", "Highs / Lows")
    ax3.axhline(1.0, color='k', lw=0.8, ls=':')
    ax3.set_xlabel("Year")
    note = f"Same {n_good} stations"
    if season:
        note += f" ({season})"
    note += f"; {smooth}-yr centered means"
    if 'weighted' in selection:
        note += f"; area weighting on a {grid_deg:g}° grid"
    ax1.text(0.02, 0.74, note, transform=ax1.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(STUDY_START - 0.5, STUDY_END + 0.5)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.text(0.01, 0.01, "Data: NOAA GHCN-daily. Raw = unadjusted; Adjusted = FLs.52j; "
             "area-weighted = equal-area grid.", fontsize=7, color='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")

    def era(a, lo, hi):
        s = (years >= lo) & (years <= hi)
        return np.nanmean(a[s])
    for lo, hi in [(1900, 1929), (2010, 2024)]:
        bits = "  ".join(f"{SHORT[k]}={era(ratio[k], lo, hi):.2f}" for k in selection)
        print(f"  ratio {lo}–{hi}:  {bits}")


def write_csv(years, series, selection, path):
    cols = {'year': years}
    for name in selection:
        hi, lo = series[name]
        cols[f'{SHORT[name]}_highs'] = np.round(hi, 3)
        cols[f'{SHORT[name]}_lows']  = np.round(lo, 3)
        cols[f'{SHORT[name]}_ratio'] = np.where(lo > 0, hi / lo, np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"Saved CSV: {path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay any subset of {raw, adjusted, weighted} records.")
    p.add_argument('series', nargs='*', default=ALL_SERIES, choices=ALL_SERIES,
                   help="series to overlay (default: all three)")
    p.add_argument('-o', '--out', default=None,
                   help="output PNG path (default: figures/records_<sel>.png)")
    p.add_argument('--csv', default=None,
                   help="also write the plotted series to this CSV path")
    p.add_argument('--smooth', type=int, default=15,
                   help="centered running-mean window in years (default 15)")
    p.add_argument('--grid', type=float, default=2.0,
                   help="area-weight grid size in degrees (default 2.0)")
    a = p.parse_args()
    sel = [s for s in ALL_SERIES if s in set(a.series)]   # canonical order, dedup
    out = (Path(a.out) if a.out
           else FIG_DIR / f"records_{'_'.join(SHORT[s] for s in sel)}.png")
    return sel, out, (Path(a.csv) if a.csv else None), a.smooth, a.grid


if __name__ == "__main__":
    selection, outpath, csvpath, smooth, grid_deg = parse_args()
    print(f"Series: {', '.join(selection)}")
    years, series, n_good = compute(selection, grid_deg)
    make_figure(years, series, selection, n_good, smooth, grid_deg, outpath)
    if csvpath is not None:
        write_csv(years, series, selection, csvpath)
