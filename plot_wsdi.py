#!/usr/bin/env python3
"""
Warm Spell Duration Index (WSDI) figure, following the ETCCDI / climpact
definition (https://climpact-sci.org/indices/).

For each station:

  1. Build a calendar-day 90th-percentile TMAX threshold from a reference period
     (default 1961-1990). For each calendar day the percentile is taken over all
     reference-year values within a centered 5-day window (so 1 Jul also uses
     29 Jun..3 Jul), which enlarges the sample and smooths the seasonal cycle.
     The window wraps circularly across the year boundary.

  2. In each analysis year, mark every day with TMAX > its calendar-day
     threshold as "hot" (a 0/1 sequence).

  3. Find runs of >= 6 consecutive hot days ("warm spells"); a 5-day run is
     ignored, a 6-day run counts, a 10-day run contributes all 10 days.

  4. WSDI for that station-year = total number of days lying in qualifying runs.

Per-station WSDI is then averaged across stations to a national annual series
(area-weighted for the "weighted" view). Everything is computed from the same
1266 complete stations and adjusted checkpoint memmaps as plot_records.py /
plot_hot_days.py, so the three views are mutually consistent:

  raw       unadjusted GHCNd, reconstructed by removing the FLs.52j monthly
            offsets (needs the external offsets file; only loaded if selected)
  adjusted  FLs.52j-adjusted
  weighted  FLs.52j-adjusted, equal-area gridded (cos(lat) / stations-in-cell)

Note: thresholds use the in-base sample directly; the ETCCDI out-of-base
bootstrap (which de-biases reference-period years) is not applied. The stored
day-of-year columns are sequential days within each year, so leap vs non-leap
years differ by up to one calendar day after 28 Feb; with the 5-day window this
is absorbed into the threshold smoothing, as in the records scripts.

Each selected series gets its own stacked panel (annual = light line, centered
running mean = heavy line), sharing axes so the panels compare directly.

Examples
  python plot_wsdi.py                       # raw, adjusted, weighted (3 panels)
  python plot_wsdi.py adjusted              # just the adjusted panel
  python plot_wsdi.py raw adjusted          # raw and adjusted panels
  python plot_wsdi.py adjusted --ref 1981 2010
  python plot_wsdi.py adjusted --csv data/wsdi_adj.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_records import (load_good, area_weights, reconstruct_raw, OFFSETS_FILE,
                          centered_mean, STUDY_START, STUDY_END, N_DOY,
                          ALL_SERIES, SHORT, LABEL, STYLE, FIG_DIR)

REF_START, REF_END = 1961, 1990     # ETCCDI reference period
WINDOW             = 5              # centered moving window (days) for the percentile
PCTILE             = 90            # "hot day" percentile
MIN_RUN            = 6             # minimum consecutive hot days for a warm spell
MIN_FRAC           = 0.90          # min fraction of a year present to score it
CHUNK              = 300           # stations per chunk in the percentile step

ACCENT = 'tab:red'                  # panel accent (heat) for the 'adjusted' series


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


# ---------------------------------------------------------------
# Step 1: calendar-day 90th-percentile thresholds
# ---------------------------------------------------------------
def calendar_thresholds(tmax, pctile=PCTILE, window=WINDOW,
                        ref_start=REF_START, ref_end=REF_END, chunk=CHUNK):
    """Per-station calendar-day threshold, shape (S, N_DOY): the `pctile`-th
    percentile of reference-period TMAX within a centered `window`-day circular
    moving window. Days with no reference data get NaN (and so are never hot)."""
    S, _, D = tmax.shape
    yrs  = np.arange(STUDY_START, STUDY_END + 1)
    refm = (yrs >= ref_start) & (yrs <= ref_end)
    half = window // 2
    thr  = np.empty((S, D), np.float32)
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        ref = tmax[a:b][:, refm, :]                       # (c, n_ref, D)
        # stack the +/-half day window by rolling the day axis, concatenating
        # the rolled copies onto the year axis -> one big sample per calendar day
        win = np.concatenate([np.roll(ref, sh, axis=2)
                              for sh in range(-half, half + 1)], axis=1)
        with warnings.catch_warnings():                   # all-NaN days -> NaN
            warnings.simplefilter('ignore', category=RuntimeWarning)
            thr[a:b] = np.nanpercentile(win, pctile, axis=1)
    return thr


# ---------------------------------------------------------------
# Steps 2-4: hot days, warm spells, WSDI
# ---------------------------------------------------------------
def days_in_runs(hot, min_run):
    """Days lying in runs of >= min_run consecutive True, per row of a 2-D
    boolean array (S, D); returns shape (S,)."""
    D   = hot.shape[1]
    pos = np.broadcast_to(np.arange(D, dtype=np.int32), hot.shape)
    # last False index at/before each position -> length of the True-run ending here
    lf  = np.maximum.accumulate(np.where(hot, np.int32(-1), pos), axis=1)
    run = np.where(hot, pos - lf, 0)
    nxt = np.zeros_like(hot)
    nxt[:, :-1] = hot[:, 1:]
    is_end = hot & ~nxt                       # last day of each run
    qual   = is_end & (run >= min_run)        # at run ends, `run` == full run length
    return (run * qual).sum(axis=1)


def wsdi_per_station_year(tmax, thr, min_run=MIN_RUN, min_frac=MIN_FRAC):
    """Per-station, per-year WSDI in days, shape (S, N_YEARS). Station-years with
    fewer than `min_frac` of their days present are set NaN (excluded from the
    national mean) since missing days both shorten and break warm spells."""
    S, Y, _ = tmax.shape
    yrs = np.arange(STUDY_START, STUDY_END + 1)
    out = np.full((S, Y), np.nan)
    for yi, yr in enumerate(yrs):
        tx     = tmax[:, yi, :]                       # (S, D)
        ndays  = 366 if is_leap(yr) else 365
        valid  = ~np.isnan(tx)
        hot    = valid & (tx > thr)                   # NaN tx or NaN thr -> not hot
        days   = days_in_runs(hot, min_run).astype(float)
        frac   = valid[:, :ndays].sum(axis=1) / ndays
        out[:, yi] = np.where(frac >= min_frac, days, np.nan)
    return out


def national(wsdi_sy, weights=None):
    """Collapse (S, N_YEARS) per-station WSDI to a national annual mean, ignoring
    NaN station-years; optional per-station area weights."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if weights is None:
            return np.nanmean(wsdi_sy, axis=0)
        valid = ~np.isnan(wsdi_sy)
        w   = weights[:, None]
        num = np.nansum(np.where(valid, wsdi_sy * w, 0.0), axis=0)
        den = np.sum(np.where(valid, w, 0.0), axis=0)
        return num / np.where(den > 0, den, np.nan)


def compute(selection, grid_deg, ref, window, pctile, min_run, min_frac):
    """Return (years, {series: annual_wsdi}, n_good)."""
    good_ids, adj_tmax, _ = load_good()
    n_good = len(good_ids)
    years  = np.arange(STUDY_START, STUDY_END + 1)
    print(f"Stations: {n_good}   reference {ref[0]}-{ref[1]}   "
          f"p{pctile}, {window}-day window, >= {min_run}-day spells")
    out = {}

    # adjusted per-station WSDI feeds both 'adjusted' and 'weighted'
    if {'adjusted', 'weighted'} & set(selection):
        thr  = calendar_thresholds(adj_tmax, pctile, window, ref[0], ref[1])
        wsdi = wsdi_per_station_year(adj_tmax, thr, min_run, min_frac)
        if 'adjusted' in selection:
            out['adjusted'] = national(wsdi)
        if 'weighted' in selection:
            w = area_weights(good_ids, grid_deg)
            print(f"Area weights ({grid_deg:g}° grid): {w.min():.3f}..{w.max():.3f}")
            out['weighted'] = national(wsdi, w)

    if 'raw' in selection:
        # offsets are monthly -> reconstruct the full array (touches the external
        # drive), then thresholds + WSDI on the raw temps. TMIN is unused; pass
        # adj_tmax as a harmless stand-in for the reconstructor's TMIN slot.
        raw_tmax, _ = reconstruct_raw(adj_tmax, adj_tmax, good_ids,
                                      OFFSETS_FILE, STUDY_START, STUDY_END)
        thr  = calendar_thresholds(raw_tmax, pctile, window, ref[0], ref[1])
        wsdi = wsdi_per_station_year(raw_tmax, thr, min_run, min_frac)
        out['raw'] = national(wsdi)

    for name in selection:
        print(f"  {name:9s} mean WSDI: {np.nanmean(out[name]):.2f} days/yr")
    return years, out, n_good


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
def make_figure(years, series, selection, n_good, smooth, grid_deg, ref, outpath):
    # one stacked panel per series, shared axes so the panels are directly
    # comparable; light line = annual value, heavy line = centered running mean
    n = len(selection)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n + 1.3),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for ax, name in zip(axes, selection):
        col = STYLE[name]['color'] or ACCENT
        ax.plot(years, series[name], color=col, lw=1.0, alpha=0.30, zorder=1)
        ax.plot(years, centered_mean(series[name], smooth),
                color=col, lw=2.4, zorder=3)
        ax.set_title(LABEL[name], fontsize=11, loc='left')
        ax.set_ylabel("WSDI (days/yr)")
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].legend([Line2D([0], [0], color='0.4', lw=1.0, alpha=0.5),
                    Line2D([0], [0], color='0.2', lw=2.4)],
                   ['annual', f'{smooth}-yr centered mean'],
                   fontsize=8, loc='upper left', framealpha=0.9)
    axes[0].set_xlim(STUDY_START - 1, STUDY_END + 1)
    axes[0].set_ylim(0, None)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Conterminous U.S. Warm Spell Duration Index (WSDI)\n"
                 f"{STUDY_START}–{STUDY_END}  ·  "
                 f"days/yr in >= {MIN_RUN}-day TX > p{PCTILE} spells "
                 f"({ref[0]}-{ref[1]} baseline)", fontsize=13, fontweight='bold')
    note = f"Data: NOAA GHCN-daily. {n_good} CONUS stations, station mean"
    if 'weighted' in selection:
        note += f"; area weighting on a {grid_deg:g}° grid"
    fig.text(0.01, 0.005, note, fontsize=7.5, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")

    def era(a, lo, hi):
        s = (years >= lo) & (years <= hi)
        return np.nanmean(a[s])
    for lo, hi in [(1900, 1929), (2010, 2024)]:
        bits = "  ".join(f"{SHORT[k]}={era(series[k], lo, hi):.2f}" for k in selection)
        print(f"  mean WSDI {lo}-{hi}:  {bits} days/yr")


def write_csv(years, series, selection, path):
    cols = {'year': years}
    for name in selection:
        cols[f'{SHORT[name]}_wsdi'] = np.round(series[name], 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"Saved CSV: {path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Warm Spell Duration Index (WSDI) for raw / adjusted / weighted.")
    p.add_argument('series', nargs='*', default=ALL_SERIES, choices=ALL_SERIES,
                   help="series to overlay (default: all three)")
    p.add_argument('-o', '--out', default=None,
                   help="output PNG path (default: figures/wsdi_<sel>.png)")
    p.add_argument('--csv', default=None,
                   help="also write the annual WSDI series to this CSV path")
    p.add_argument('--smooth', type=int, default=15,
                   help="centered running-mean window in years (default 15)")
    p.add_argument('--grid', type=float, default=2.0,
                   help="area-weight grid size in degrees (default 2.0)")
    p.add_argument('--ref', type=int, nargs=2, default=[REF_START, REF_END],
                   metavar=('START', 'END'),
                   help=f"baseline period (default {REF_START} {REF_END})")
    p.add_argument('--window', type=int, default=WINDOW,
                   help=f"centered percentile window in days (default {WINDOW})")
    p.add_argument('--pctile', type=float, default=PCTILE,
                   help=f"hot-day percentile (default {PCTILE})")
    p.add_argument('--min-run', type=int, default=MIN_RUN,
                   help=f"min consecutive hot days for a spell (default {MIN_RUN})")
    p.add_argument('--min-frac', type=float, default=MIN_FRAC,
                   help=f"min fraction of a year present to score it (default {MIN_FRAC})")
    a = p.parse_args()
    sel = [s for s in ALL_SERIES if s in set(a.series)]   # canonical order, dedup
    out = (Path(a.out) if a.out
           else FIG_DIR / f"wsdi_{'_'.join(SHORT[s] for s in sel)}.png")
    return sel, out, (Path(a.csv) if a.csv else None), a


if __name__ == "__main__":
    selection, outpath, csvpath, a = parse_args()
    print(f"Series: {', '.join(selection)}")
    years, series, n_good = compute(selection, a.grid, a.ref, a.window,
                                    a.pctile, a.min_run, a.min_frac)
    make_figure(years, series, selection, n_good, a.smooth, a.grid, a.ref, outpath)
    if csvpath is not None:
        write_csv(years, series, selection, csvpath)
