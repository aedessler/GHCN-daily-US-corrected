#!/usr/bin/env python3
"""
Generate adjusted_records.png and adjusted_records.csv from records_cache.npz.
Run compute_adjusted_records.py first to produce the cache.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

OUT_DIR    = Path("/Users/adessler/Desktop/recHighs")
DATA_DIR   = OUT_DIR / "data"
FIG_DIR    = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "records_cache.npz"
SMOOTH_YRS = 15   # window for centered running average (must be odd)

if not CACHE_FILE.exists():
    raise FileNotFoundError(
        f"Cache not found: {CACHE_FILE}\nRun compute_adjusted_records.py first."
    )

cache     = np.load(CACHE_FILE)
years     = cache['years']
rec_highs = cache['rec_highs'].astype(float)
rec_lows  = cache['rec_lows'].astype(float)
n_good    = int(cache['n_good'])
ratio     = np.where(rec_lows > 0, rec_highs / rec_lows, np.nan)

STUDY_START = int(years[0])
STUDY_END   = int(years[-1])

print(f"Loaded: {n_good} stations, {len(years)} years ({STUDY_START}–{STUDY_END})")
print(f"Total record highs: {rec_highs.sum():,.0f}   record lows: {rec_lows.sum():,.0f}")


def centered_mean(arr, window):
    """Centered running mean with given (odd) window; NaN at edges where window is incomplete."""
    half = window // 2
    out = np.full_like(arr, np.nan)
    for i in range(half, len(arr) - half):
        out[i] = np.nanmean(arr[i - half : i + half + 1])
    return out


trail_highs = centered_mean(rec_highs, SMOOTH_YRS)
trail_lows  = centered_mean(rec_lows,  SMOOTH_YRS)
trail_ratio = centered_mean(ratio,     SMOOTH_YRS)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={'height_ratios': [3, 3, 2]})
fig.suptitle(
    f"Conterminous U.S. Adjusted Number of Daily Temperature Records\n"
    f"{STUDY_START}–{STUDY_END}",
    fontsize=13, fontweight='bold', y=0.98,
)

ax1, ax2, ax3 = axes
ylim_top = max(np.nanmax(rec_highs), np.nanmax(rec_lows)) * 1.15

# ── Panel 1: Record Highs ────────────────────────────────────────
ax1.bar(years, rec_highs, color='tab:red', width=0.8, alpha=0.5)
ax1.plot(years, trail_highs, color='darkred', lw=1.8,
         label=f'{SMOOTH_YRS}-yr trailing mean')
ax1.set_title("Daily Record Highs", fontsize=11)
ax1.set_ylabel("Number of Records")
ax1.set_ylim(0, ylim_top)
ax1.text(0.02, 0.93,
         f"Computed from {n_good} NOAA GHCND stations with ≥100 years\n"
         f"of data and ≥80% observational data",
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.legend(fontsize=8, loc='upper right')

# ── Panel 2: Record Lows ─────────────────────────────────────────
ax2.bar(years, rec_lows, color='tab:blue', width=0.8, alpha=0.5)
ax2.plot(years, trail_lows, color='darkblue', lw=1.8,
         label=f'{SMOOTH_YRS}-yr trailing mean')
ax2.set_title("Daily Record Lows", fontsize=11)
ax2.set_ylabel("Number of Records")
ax2.set_ylim(0, ylim_top)
ax2.legend(fontsize=8, loc='upper right')

# ── Panel 3: Ratio ───────────────────────────────────────────────
colors3 = ['tab:red' if (not np.isnan(r) and r >= 1) else 'tab:blue' for r in ratio]
ax3.bar(years, ratio, color=colors3, width=0.8, alpha=0.5)
ax3.plot(years, trail_ratio, color='black', lw=1.8,
         label=f'{SMOOTH_YRS}-yr trailing mean')
ax3.axhline(1.0, color='k', lw=0.8, ls='--')
ax3.set_title("Ratio of Record Highs to Record Lows", fontsize=11)
ax3.set_ylabel("Highs / Lows")
ax3.set_xlabel("Year")
from matplotlib.lines import Line2D
legend_handles = [
    mpatches.Patch(color='tab:red',  alpha=0.5, label='More highs than lows'),
    mpatches.Patch(color='tab:blue', alpha=0.5, label='More lows than highs'),
    Line2D([0], [0], color='black', lw=1.8,  label=f'{SMOOTH_YRS}-yr trailing mean'),
]
ax3.legend(handles=legend_handles, fontsize=8, loc='upper left')

for ax in axes:
    ax.set_xlim(STUDY_START - 0.5, STUDY_END + 0.5)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.01, 0.01,
         "Data source: NOAA Global Historical Climatology Network-daily (GHCNd)\n"
         "Adjusted with North American FLs.52j monthly offsets",
         fontsize=7, color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(FIG_DIR / "adjusted_records.png", dpi=150, bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'adjusted_records.png'}")
plt.close()

# ---------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------
pd.DataFrame({
    'year':              years,
    'rec_highs':         rec_highs.round(3),
    'rec_lows':          rec_lows.round(3),
    'ratio':             ratio,
    f'centered{SMOOTH_YRS}_highs': trail_highs,
    f'centered{SMOOTH_YRS}_lows':  trail_lows,
    f'centered{SMOOTH_YRS}_ratio': trail_ratio,
}).to_csv(DATA_DIR / "adjusted_records.csv", index=False)
print(f"Saved CSV: {DATA_DIR / 'adjusted_records.csv'}")
