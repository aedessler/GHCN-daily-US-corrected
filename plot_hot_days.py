#!/usr/bin/env python3
"""
"Very hot days per year" figure: for each year, the percent of station-days
whose daily high reaches a set of nested thresholds (default >=95, >=100,
>=105 F), drawn as overlapping bars.

Because >=105 F days are a subset of >=100 F days, which are a subset of >=95 F
days, the bars are drawn from zero, largest behind smallest, giving a nested
look.

Everything is computed from the same 1266 complete stations and the same
adjusted checkpoint memmaps as plot_records.py, so the three views are mutually
consistent:

  raw       unadjusted GHCNd, reconstructed by removing the FLs.52j monthly
            offsets (needs the external offsets file; only loaded if selected)
  adjusted  FLs.52j-adjusted
  weighted  FLs.52j-adjusted, equal-area gridded (cos(lat) / stations-in-cell)

One PNG is written per selected series.

Examples
  python plot_hot_days.py                          # raw, adjusted, weighted
  python plot_hot_days.py adjusted                 # just the adjusted figure
  python plot_hot_days.py raw adjusted             # raw and adjusted figures
  python plot_hot_days.py weighted --grid 2.5
  python plot_hot_days.py adjusted --thresholds 90 95 100 105
  python plot_hot_days.py adjusted --csv data/hot_days_adj.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from matplotlib.lines import Line2D

from plot_records import (load_good, area_weights, reconstruct_raw, OFFSETS_FILE,
                          centered_mean, STUDY_START, STUDY_END, SHORT, FIG_DIR)

ALL_SERIES = ['raw', 'adjusted', 'weighted']          # canonical order
LABEL      = {'raw': 'Raw (unadjusted)',
              'adjusted': 'Adjusted (FLs.52j)',
              'weighted': 'Adjusted, area-weighted'}

DEFAULT_THRESHOLDS_F = [95, 100, 105]
# color ramp for thresholds, coolest (lowest) -> hottest (highest): a
# coral / dark-red / black heat ramp. Extra thresholds borrow intermediate
# reds so any count from 2..5 still reads as a heat ramp.
THRESH_COLORS = ['#fb8b5f', '#d24a3a', '#8c1d1d', 'black', '#2b0a0a']


def f_to_c(f):
    return (f - 32) * 5.0 / 9.0


def trend_line_style(barcolor):
    """Pick a smoothed-trend line color + halo that stays readable over a bar of
    `barcolor`: a dark tint of the bar over light bars, white over dark bars."""
    r, g, b = mcolors.to_rgb(barcolor)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 0.4:                                  # dark bar -> light line, dark halo
        return 'white', 'black'
    return (r * 0.45, g * 0.45, b * 0.45), 'white'  # light bar -> dark line, light halo


# ---------------------------------------------------------------
# Counting
# ---------------------------------------------------------------
def hot_day_percent(tmax, thresholds_c, weights=None):
    """Per-year percent of (optionally weighted) station-days with TMAX >= each
    threshold. Returns an array of shape (n_thresholds, N_YEARS); the denominator
    is every valid TMAX station-day that year, so winter days count too, exactly
    as in a 'percent of days' chart."""
    valid = ~np.isnan(tmax)                       # (S, Y, D)
    if weights is None:
        den = valid.sum(axis=(0, 2)).astype(float)
        num = np.stack([(valid & (tmax >= tc)).sum(axis=(0, 2)) for tc in thresholds_c])
    else:
        w = weights[:, None, None]
        den = (valid * w).sum(axis=(0, 2))
        num = np.stack([((valid & (tmax >= tc)) * w).sum(axis=(0, 2)) for tc in thresholds_c])
    return 100.0 * num / np.where(den > 0, den, np.nan)


def compute(selection, thresholds_f, grid_deg):
    """Return (years, {series: pct_array}, n_good) where pct_array is
    (n_thresholds, N_YEARS)."""
    thresholds_c = [f_to_c(f) for f in thresholds_f]
    good_ids, adj_tmax, _ = load_good()
    n_good = len(good_ids)
    years = np.arange(STUDY_START, STUDY_END + 1)
    print(f"Stations: {n_good}")
    out = {}
    if 'adjusted' in selection:
        out['adjusted'] = hot_day_percent(adj_tmax, thresholds_c)
    if 'weighted' in selection:
        w = area_weights(good_ids, grid_deg)
        print(f"Area weights ({grid_deg:g}° grid): {w.min():.3f}..{w.max():.3f}")
        out['weighted'] = hot_day_percent(adj_tmax, thresholds_c, w)
    if 'raw' in selection:
        # raw offsets are monthly -> reconstruct the full array (touches the
        # external drive), then count. TMIN is unused here but the reconstructor
        # wants both; pass adj_tmax as a harmless stand-in for the TMIN slot.
        raw_tmax, _ = reconstruct_raw(adj_tmax, adj_tmax, good_ids,
                                      OFFSETS_FILE, STUDY_START, STUDY_END)
        out['raw'] = hot_day_percent(raw_tmax, thresholds_c)
    for name in selection:
        means = out[name].mean(axis=1)
        bits = "  ".join(f"{f:g}F={m:.2f}%" for f, m in zip(thresholds_f, means))
        print(f"  {name:9s} mean: {bits}")
    return years, out, n_good


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
def make_figure(years, pct, name, thresholds_f, n_good, grid_deg, smooth, outpath):
    fig, ax = plt.subplots(figsize=(9, 6))
    # draw largest (lowest threshold) first so the hotter, smaller bars sit on top
    order = np.argsort(thresholds_f)[::-1]
    handles = []
    for k in order:
        col = THRESH_COLORS[k % len(THRESH_COLORS)]
        h = ax.bar(years, pct[k], width=0.85, color=col,
                   label=f"Days ≥{thresholds_f[k]:g}°F", zorder=2 + k, edgecolor='none')
        handles.append(h)
        # centered running-mean trend on top of this threshold's bars
        lc, halo = trend_line_style(col)
        ax.plot(years, centered_mean(pct[k], smooth), color=lc, lw=2.0,
                zorder=20 + k, solid_capstyle='round',
                path_effects=[mpe.Stroke(linewidth=3.5, foreground=halo),
                              mpe.Normal()])
    ax.set_title("Conterminous U.S. Observed Number of Very Hot Days Per Year\n"
                 f"{STUDY_START} to {STUDY_END}  ·  {LABEL[name]}",
                 fontsize=13)
    ax.set_ylabel("Percent of Days")
    ax.set_xlim(STUDY_START - 1, STUDY_END + 1)
    ax.set_ylim(0, None)
    # legend hottest-first, with a proxy for the trend lines
    labels = [h.get_label() for h in handles]
    o2 = np.argsort([float(l.split('≥')[1].rstrip('°F')) for l in labels])[::-1]
    leg_h = [handles[i] for i in o2] + [
        Line2D([0], [0], color='0.3', lw=2.0,
               path_effects=[mpe.Stroke(linewidth=3.5, foreground='white'), mpe.Normal()])]
    leg_l = [labels[i] for i in o2] + [f"{smooth}-yr centered mean"]
    ax.legend(leg_h, leg_l, fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    src = "Data: NOAA GHCN-daily, FLs.52j-adjusted"
    if name == 'raw':
        src = "Data: NOAA GHCN-daily, raw (unadjusted)"
    note = f"{src}; {n_good} CONUS stations"
    if name == 'weighted':
        note += f"; equal-area weighting on a {grid_deg:g}° grid"
    fig.text(0.01, 0.005, note, fontsize=7.5, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def write_csv(years, pct, thresholds_f, path):
    cols = {'year': years}
    for k, f in enumerate(thresholds_f):
        cols[f'pct_ge_{f:g}F'] = np.round(pct[k], 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"Saved CSV: {path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Very-hot-days-per-year figure for raw / adjusted / weighted.")
    p.add_argument('series', nargs='*', default=ALL_SERIES, choices=ALL_SERIES,
                   help="series to plot, one figure each (default: all three)")
    p.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS_F,
                   metavar='F', help="TMAX thresholds in °F (default 95 100 105)")
    p.add_argument('-o', '--out', default=None,
                   help="output PNG path; only valid with a single series "
                        "(default: figures/hot_days_<series>.png)")
    p.add_argument('--csv', default=None,
                   help="also write the plotted percentages to this CSV "
                        "(single series only)")
    p.add_argument('--smooth', type=int, default=15,
                   help="centered running-mean window in years for the trend "
                        "lines (default 15)")
    p.add_argument('--grid', type=float, default=2.0,
                   help="area-weight grid size in degrees (default 2.0)")
    a = p.parse_args()
    sel = [s for s in ALL_SERIES if s in set(a.series)]      # canonical order, dedup
    if (a.out or a.csv) and len(sel) != 1:
        p.error("-o/--csv require selecting exactly one series")
    return sel, a.thresholds, a.out, a.csv, a.smooth, a.grid


if __name__ == "__main__":
    selection, thresholds_f, out, csv, smooth, grid_deg = parse_args()
    print(f"Series: {', '.join(selection)}   thresholds(°F): "
          f"{', '.join(f'{f:g}' for f in thresholds_f)}")
    years, allpct, n_good = compute(selection, thresholds_f, grid_deg)
    for name in selection:
        outpath = (Path(out) if out
                   else FIG_DIR / f"hot_days_{SHORT[name]}.png")
        make_figure(years, allpct[name], name, thresholds_f, n_good, grid_deg,
                    smooth, outpath)
    if csv is not None:
        write_csv(years, allpct[selection[0]], thresholds_f, Path(csv))
