#!/usr/bin/env python3
"""
Area-weighted adjusted-records figure.
Run compute_area_weighted.py first to produce records_cache_areaweighted.npz.
Solid lines = area-weighted 15-yr mean; dashed gray = original (unweighted) mean.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR    = Path("/Users/adessler/Desktop/recHighs")
CACHE      = OUT_DIR / "records_cache_areaweighted.npz"
SMOOTH_YRS = 15

c          = np.load(CACHE)
years      = c['years']
hi_w, lo_w = c['rec_highs'].astype(float),     c['rec_lows'].astype(float)
hi_u, lo_u = c['rec_highs_unw'].astype(float), c['rec_lows_unw'].astype(float)
grid_deg   = float(c['grid_deg']); n_good = int(c['n_good'])
ratio_w = np.where(lo_w > 0, hi_w / lo_w, np.nan)
ratio_u = np.where(lo_u > 0, hi_u / lo_u, np.nan)
S0, S1 = int(years[0]), int(years[-1])


def centered_mean(a, win):
    half = win // 2
    out = np.full_like(a, np.nan)
    for i in range(half, len(a) - half):
        out[i] = np.nanmean(a[i - half:i + half + 1])
    return out


th_w, tl_w, tr_w = (centered_mean(x, SMOOTH_YRS) for x in (hi_w, lo_w, ratio_w))
th_u, tl_u, tr_u = (centered_mean(x, SMOOTH_YRS) for x in (hi_u, lo_u, ratio_u))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 3, 2]})
fig.suptitle(f"Conterminous U.S. Adjusted Daily Temperature Records — "
             f"AREA-WEIGHTED\n{S0}–{S1}", fontsize=13, fontweight='bold', y=0.99)
ytop = max(np.nanmax(hi_w), np.nanmax(lo_w)) * 1.15

# Panel 1: highs
ax1.bar(years, hi_w, color='tab:red', width=0.8, alpha=0.5)
ax1.plot(years, th_w, color='darkred', lw=1.8, label=f'{SMOOTH_YRS}-yr mean (area-weighted)')
ax1.plot(years, th_u, color='gray', lw=1.3, ls='--', label=f'{SMOOTH_YRS}-yr mean (unweighted)')
ax1.set_title("Daily Record Highs", fontsize=11)
ax1.set_ylabel("Records (area-weighted)"); ax1.set_ylim(0, ytop)
ax1.text(0.02, 0.93,
         f"{n_good} stations, equal-area weighted on a {grid_deg:g}° grid\n"
         f"(each occupied cell weighted by area, not station count)",
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.legend(fontsize=8, loc='upper right')

# Panel 2: lows
ax2.bar(years, lo_w, color='tab:blue', width=0.8, alpha=0.5)
ax2.plot(years, tl_w, color='darkblue', lw=1.8, label=f'{SMOOTH_YRS}-yr mean (area-weighted)')
ax2.plot(years, tl_u, color='gray', lw=1.3, ls='--', label=f'{SMOOTH_YRS}-yr mean (unweighted)')
ax2.set_title("Daily Record Lows", fontsize=11)
ax2.set_ylabel("Records (area-weighted)"); ax2.set_ylim(0, ytop)
ax2.legend(fontsize=8, loc='upper right')

# Panel 3: ratio
colors3 = ['tab:red' if (not np.isnan(r) and r >= 1) else 'tab:blue' for r in ratio_w]
ax3.bar(years, ratio_w, color=colors3, width=0.8, alpha=0.5)
ax3.plot(years, tr_w, color='black', lw=1.8, label=f'{SMOOTH_YRS}-yr mean (area-weighted)')
ax3.plot(years, tr_u, color='gray', lw=1.3, ls='--', label=f'{SMOOTH_YRS}-yr mean (unweighted)')
ax3.axhline(1.0, color='k', lw=0.8, ls='--')
ax3.set_title("Ratio of Record Highs to Record Lows", fontsize=11)
ax3.set_ylabel("Highs / Lows"); ax3.set_xlabel("Year")
ax3.legend(handles=[
    mpatches.Patch(color='tab:red', alpha=0.5, label='More highs than lows'),
    mpatches.Patch(color='tab:blue', alpha=0.5, label='More lows than highs'),
    Line2D([0], [0], color='black', lw=1.8, label=f'{SMOOTH_YRS}-yr mean (area-weighted)'),
    Line2D([0], [0], color='gray', lw=1.3, ls='--', label=f'{SMOOTH_YRS}-yr mean (unweighted)'),
], fontsize=8, loc='upper left')

for ax in (ax1, ax2, ax3):
    ax.set_xlim(S0 - 0.5, S1 + 0.5)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.text(0.01, 0.01,
         "Data: NOAA GHCNd, adjusted with North American FLs.52j monthly offsets. "
         f"Area-weighted on {grid_deg:g}° equal-area grid.",
         fontsize=7, color='gray')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = OUT_DIR / "adjusted_records_areaweighted.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# quick numeric summary of the effect
def m(a, lo, hi):
    s = (years >= lo) & (years <= hi)
    return np.nanmean(a[s])
print(f"Mean ratio 2010–2024:  unweighted={m(ratio_u,2010,2024):.2f}  "
      f"area-weighted={m(ratio_w,2010,2024):.2f}")
print(f"Mean ratio 1900–1929:  unweighted={m(ratio_u,1900,1929):.2f}  "
      f"area-weighted={m(ratio_w,1900,1929):.2f}")
