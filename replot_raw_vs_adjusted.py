#!/usr/bin/env python3
"""
Adjusted-vs-raw comparison overlay.
Run compute_adjusted_records.py (-> records_cache.npz) and
compute_raw_records.py (-> records_cache_raw.npz) first.

Each panel overlays the homogeneity-ADJUSTED series (solid, colored) against the
RAW/unadjusted series (dashed, gray): faint thin = annual, bold = 15-yr mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR    = Path("/Users/adessler/Desktop/recHighs")
SMOOTH_YRS = 15
RAW_CACHE  = OUT_DIR / "records_cache_raw.npz"

if not RAW_CACHE.exists():
    raise SystemExit("records_cache_raw.npz not found — run compute_raw_records.py first.")

adj = np.load(OUT_DIR / "records_cache.npz")
raw = np.load(RAW_CACHE)
years = adj['years']
hi_a, lo_a = adj['rec_highs'].astype(float), adj['rec_lows'].astype(float)
hi_r, lo_r = raw['rec_highs'].astype(float), raw['rec_lows'].astype(float)
ratio_a = np.where(lo_a > 0, hi_a / lo_a, np.nan)
ratio_r = np.where(lo_r > 0, hi_r / lo_r, np.nan)
S0, S1 = int(years[0]), int(years[-1])


def centered_mean(a, win):
    half = win // 2
    out = np.full_like(a, np.nan)
    for i in range(half, len(a) - half):
        out[i] = np.nanmean(a[i - half:i + half + 1])
    return out


def panel(ax, ya, yr, color, title, ylabel):
    ax.plot(years, ya, color=color, lw=0.7, alpha=0.30)
    ax.plot(years, yr, color='gray', lw=0.7, alpha=0.30, ls='--')
    ax.plot(years, centered_mean(ya, SMOOTH_YRS), color=color, lw=2.0,
            label='Adjusted (15-yr mean)')
    ax.plot(years, centered_mean(yr, SMOOTH_YRS), color='dimgray', lw=2.0, ls='--',
            label='Raw / unadjusted (15-yr mean)')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc='upper left')


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 3, 2]})
fig.suptitle("Conterminous U.S. Daily Temperature Records — "
             f"Adjusted vs Raw\n{S0}–{S1}", fontsize=13, fontweight='bold', y=0.99)

panel(ax1, hi_a, hi_r, 'tab:red',  "Daily Record Highs", "Number of Records")
panel(ax2, lo_a, lo_r, 'tab:blue', "Daily Record Lows",  "Number of Records")
panel(ax3, ratio_a, ratio_r, 'black', "Ratio of Record Highs to Record Lows", "Highs / Lows")
ax3.axhline(1.0, color='k', lw=0.8, ls=':')
ax3.set_xlabel("Year")
ax1.text(0.02, 0.78,
         "Homogeneity adjustment (FLs.52j) vs raw GHCNd, same 1266 stations",
         transform=ax1.transAxes, fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

for ax in (ax1, ax2, ax3):
    ax.set_xlim(S0 - 0.5, S1 + 0.5)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.text(0.01, 0.01,
         "Data: NOAA GHCNd. Adjusted = raw + FLs.52j monthly offsets; "
         "raw reconstructed by removing those offsets.",
         fontsize=7, color='gray')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = OUT_DIR / "adjusted_vs_raw_records.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# records_cache_raw.npz is a transient intermediate (reconstructs in seconds), so
# we don't keep it once the figure exists. Re-run compute_raw_records.py to remake.
RAW_CACHE.unlink(missing_ok=True)
print("Removed records_cache_raw.npz.")


def m(a, lo, hi):
    s = (years >= lo) & (years <= hi)
    return np.nanmean(a[s])
print(f"Mean ratio 1900–1929:  adjusted={m(ratio_a,1900,1929):.2f}  raw={m(ratio_r,1900,1929):.2f}")
print(f"Mean ratio 2010–2024:  adjusted={m(ratio_a,2010,2024):.2f}  raw={m(ratio_r,2010,2024):.2f}")
