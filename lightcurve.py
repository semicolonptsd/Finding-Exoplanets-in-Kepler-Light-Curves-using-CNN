"""
lightcurve.py
-------------
Everything that happens to a light curve before the CNN sees it:

    raw quarters  ->  clean + stitch  ->  detrend  ->  phase fold  ->  global / local views

All of it is plain numpy/pandas so every step can be explained line by line.
Times are in days (Kepler's BKJD = BJD - 2454833), durations come in hours from the
KOI table so we convert them to days first.
"""

import numpy as np
import pandas as pd

GLOBAL_BINS = 2001   # whole orbit
LOCAL_BINS = 201     # zoom on the transit
LOCAL_WIDTH = 4      # local view covers +- 4 transit durations


# ---------------------------------------------------------------------------
# 1. Cleaning and stitching the quarters
# ---------------------------------------------------------------------------
def stitch_quarters(quarters):
    """
    quarters: list of (time, flux, quality) arrays, one tuple per Kepler quarter.

    Kepler rolls by 90 degrees every ~3 months, so each quarter the star lands on a
    different CCD with a slightly different sensitivity. That shows up as a jump in
    the flux level between quarters. Dividing each quarter by its own median puts
    every quarter on the same scale (median = 1), which removes the jumps.
    """
    all_t, all_f = [], []
    for t, f, q in quarters:
        t, f, q = np.asarray(t, float), np.asarray(f, float), np.asarray(q)
        good = (q == 0) & np.isfinite(t) & np.isfinite(f)   # drop flagged cadences
        if good.sum() < 100:
            continue
        t, f = t[good], f[good]
        all_t.append(t)
        all_f.append(f / np.median(f))                     # quarter-wise normalisation
    if not all_t:
        return None, None
    t = np.concatenate(all_t)
    f = np.concatenate(all_f)
    order = np.argsort(t)
    return t[order], f[order]


def transit_mask(t, period, t0, duration_days, factor=1.5):
    """True for points within factor * duration of a transit centre."""
    phase = phase_time(t, period, t0)
    return np.abs(phase) < factor * duration_days


def detrend(t, f, period, t0, duration_days):
    """
    Remove slow stellar variability (spots rotating, etc.) with a running median.

    Trick: we hide the in-transit points from the running median (set them to NaN).
    Otherwise the median would partly 'follow' the dip and eat into the transit.
    The window is at least 1 day and at least 3 transit durations long, so a dip
    lasting a few hours can never be absorbed by the trend.
    """
    in_transit = transit_mask(t, period, t0, duration_days)
    window_days = max(1.0, 3 * duration_days)

    f_masked = f.copy()
    f_masked[in_transit] = np.nan
    series = pd.Series(f_masked, index=pd.to_timedelta(t, unit="D"))
    trend = series.rolling(pd.Timedelta(days=window_days), center=True, min_periods=10).median()
    trend = trend.interpolate(limit_direction="both").to_numpy()

    return f / trend


def clip_outliers(t, f, n_sigma=3.0):
    """
    Remove upward outliers only (cosmic rays, flares). We never clip downward,
    because a downward point might be part of a transit or of a secondary eclipse,
    and those are exactly the things we want the network to see.
    Sigma is estimated robustly with the MAD (median absolute deviation).
    """
    med = np.median(f)
    sigma = 1.4826 * np.median(np.abs(f - med))   # MAD -> sigma for gaussian noise
    keep = f < med + n_sigma * sigma
    return t[keep], f[keep]


def clean_light_curve(quarters, period, t0, duration_hours):
    """Full cleaning: stitch -> detrend -> clip. Returns (t, f) with baseline ~ 1."""
    t, f = stitch_quarters(quarters)
    if t is None:
        return None, None
    dur = duration_hours / 24.0
    f = detrend(t, f, period, t0, dur)
    good = np.isfinite(f)
    t, f = clip_outliers(t[good], f[good])
    return t, f


# ---------------------------------------------------------------------------
# 2. Phase folding
# ---------------------------------------------------------------------------
def phase_time(t, period, t0):
    """
    Time measured from the nearest transit centre, in days, in [-P/2, P/2).
    This is the whole idea of folding: every transit lands on phase 0.
    """
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def bin_median(x, y, lo, hi, n_bins):
    """
    Split [lo, hi] into n_bins equal bins and take the median flux in each bin.
    Median (not mean) so that a single leftover outlier can't drag a bin around.
    Empty bins are filled by linear interpolation from their neighbours.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(x, edges) - 1
    inside = (idx >= 0) & (idx < n_bins)
    out = np.full(n_bins, np.nan)
    df = pd.DataFrame({"i": idx[inside], "y": y[inside]})
    med = df.groupby("i")["y"].median()
    out[med.index.to_numpy()] = med.to_numpy()

    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(out)
    if ok.sum() < 2:
        return None
    out[~ok] = np.interp(centres[~ok], centres[ok], out[ok])
    return out


def normalise_view(v):
    """
    Shift and scale so that the out-of-transit baseline is 0 and the deepest point
    is -1. Every view then has the same scale, so the network compares SHAPES
    rather than absolute depths (a 50 ppm and a 5000 ppm transit look alike).
    """
    v = v - np.median(v)
    depth = -v.min()
    if depth <= 0:
        return None
    return v / depth


def make_views(t, f, period, t0, duration_hours):
    """Return (global_view, local_view) or (None, None) if something is off."""
    dur = duration_hours / 24.0
    x = phase_time(t, period, t0)

    g = bin_median(x, f, -0.5 * period, 0.5 * period, GLOBAL_BINS)
    half = min(LOCAL_WIDTH * dur, 0.5 * period)
    l = bin_median(x, f, -half, half, LOCAL_BINS)
    if g is None or l is None:
        return None, None

    g, l = normalise_view(g), normalise_view(l)
    if g is None or l is None:
        return None, None
    return g.astype(np.float32), l.astype(np.float32)
