"""Robust multi-year growth and base-year distortion checks.

A plain CAGR uses only two years. When the starting year is depressed by a write-off,
impairment, hedge mark-to-market or a cyclical trough (or inflated by a one-off gain or a
cyclical peak), the CAGR mostly measures that one year. Two general tools fix this:

* ``trend_growth`` fits a Theil-Sen line (the median of all pairwise slopes) through log values
  of up to ``TREND_WINDOW`` fiscal years and annualizes the slope. Every year contributes, and
  roughly 29% of the points can be distorted before the estimate breaks down, so with 8 years
  two distorted years (sometimes three) do not move it materially.
* ``base_vs_trend`` fits the same line through every year in the window EXCEPT the CAGR's
  starting year and compares that year with the line. A ratio outside
  [1/DISTORTION_RATIO, DISTORTION_RATIO] means the starting year is clearly abnormal and the
  plain CAGR is marked not meaningful (its value is kept for reference).

Both operate on a fiscal-year panel frame: index = ticker, columns = offsets 0 (latest) .. N.
"""
from typing import Tuple

import numpy as np
import pandas as pd

TREND_WINDOW = 8            # fiscal years (FY0 .. FY-7) used by the robust trend
TREND_MIN_POINTS = 5        # positive years required inside the window
TREND_MAX_NONPOSITIVE = 1   # a second loss year makes log growth not meaningful
DISTORTION_RATIO = 1.5      # starting year more than 1.5x above / below the trend of the other years
BASE_MIN_POINTS = 4         # other positive years needed to judge the starting year


def _theil_sen(k: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Median pairwise slope and median intercept of y against offset k."""
    i, j = np.triu_indices(len(k), 1)
    slope = float(np.median((y[j] - y[i]) / (k[j] - k[i])))
    intercept = float(np.median(y - slope * k))
    return slope, intercept


def trend_growth(frame: pd.DataFrame, window: int = TREND_WINDOW, min_points: int = TREND_MIN_POINTS,
                 max_nonpositive: int = TREND_MAX_NONPOSITIVE) -> Tuple[pd.Series, pd.Series]:
    """Annualized robust growth (%) and status (ok / nm / short / missing) per ticker."""
    cols = [c for c in range(window) if c in frame.columns]
    vals = frame[cols].to_numpy(dtype=float)
    k_all = np.array(cols, dtype=float)
    out = np.full(len(frame), np.nan)
    status = np.full(len(frame), "missing", dtype=object)
    for r in range(len(frame)):
        v = vals[r]
        have = ~np.isnan(v)
        if not have.any():
            continue
        pos = have & (v > 0)
        if not have[0]:
            continue                         # no latest year: a trend "to now" can't be measured
        if not (v[0] > 0):
            status[r] = "nm"                 # latest year is a loss: growth rate not meaningful
            continue
        if (have & (v <= 0)).sum() > max_nonpositive:
            status[r] = "nm"
            continue
        if pos.sum() < min_points:
            status[r] = "short" if have.sum() < min_points else "nm"
            continue
        slope, _ = _theil_sen(k_all[pos], np.log(v[pos]))
        out[r] = (np.exp(-slope) - 1.0) * 100.0   # offsets count backwards in time
        status[r] = "ok"
    return pd.Series(out, index=frame.index), pd.Series(status, index=frame.index)


def base_vs_trend(frame: pd.DataFrame, base: int, window: int = TREND_WINDOW,
                  min_points: int = BASE_MIN_POINTS) -> pd.Series:
    """Starting-year value / value implied by the trend of the other years (NaN if not judgeable)."""
    cols = [c for c in range(window) if c in frame.columns and c != base]
    if base not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    vals = frame[cols].to_numpy(dtype=float)
    b = frame[base].to_numpy(dtype=float)
    k_all = np.array(cols, dtype=float)
    out = np.full(len(frame), np.nan)
    for r in range(len(frame)):
        if not (b[r] > 0):
            continue
        v = vals[r]
        pos = ~np.isnan(v) & (v > 0)
        if pos.sum() < min_points:
            continue
        slope, intercept = _theil_sen(k_all[pos], np.log(v[pos]))
        out[r] = b[r] / np.exp(intercept + slope * base)
    return pd.Series(out, index=frame.index)


def is_distorted(ratio: pd.Series, threshold: float = DISTORTION_RATIO) -> pd.Series:
    return (ratio < 1.0 / threshold) | (ratio > threshold)


# ---- level series (margins, in percentage points) ------------------------------------------------
MARGIN_MIN_POINTS = 5        # years with a margin inside the window
MARGIN_DISTORTION_PP = 3.0   # a starting-year margin is distorted when it sits more than
MARGIN_DISTORTION_MADS = 3.0 #   max(3pp, 3 x the typical deviation of the other years) off their trend


def linear_trend(frame: pd.DataFrame, window: int = TREND_WINDOW,
                 min_points: int = MARGIN_MIN_POINTS) -> Tuple[pd.Series, pd.Series]:
    """Theil-Sen slope per year (units of the series, e.g. pp/yr) and status per ticker."""
    cols = [c for c in range(window) if c in frame.columns]
    vals = frame[cols].to_numpy(dtype=float)
    k_all = np.array(cols, dtype=float)
    out = np.full(len(frame), np.nan)
    status = np.full(len(frame), "missing", dtype=object)
    for r in range(len(frame)):
        v = vals[r]
        have = ~np.isnan(v)
        if not have.any() or not have[0]:
            continue
        if have.sum() < min_points:
            status[r] = "short"
            continue
        slope, _ = _theil_sen(k_all[have], v[have])
        out[r] = -slope                       # offsets count backwards in time
        status[r] = "ok"
    return pd.Series(out, index=frame.index), pd.Series(status, index=frame.index)


def base_residual_linear(frame: pd.DataFrame, base: int, window: int = TREND_WINDOW,
                         min_points: int = BASE_MIN_POINTS) -> Tuple[pd.Series, pd.Series]:
    """(starting-year value minus the trend of the other years, distortion tolerance) per ticker."""
    cols = [c for c in range(window) if c in frame.columns and c != base]
    resid = np.full(len(frame), np.nan)
    tol = np.full(len(frame), np.nan)
    if base not in frame.columns:
        return pd.Series(resid, index=frame.index), pd.Series(tol, index=frame.index)
    vals = frame[cols].to_numpy(dtype=float)
    b = frame[base].to_numpy(dtype=float)
    k_all = np.array(cols, dtype=float)
    for r in range(len(frame)):
        v = vals[r]
        have = ~np.isnan(v)
        if np.isnan(b[r]) or have.sum() < min_points:
            continue
        slope, intercept = _theil_sen(k_all[have], v[have])
        others = v[have] - (intercept + slope * k_all[have])
        mad = float(np.median(np.abs(others)))
        resid[r] = b[r] - (intercept + slope * base)
        tol[r] = max(MARGIN_DISTORTION_PP, MARGIN_DISTORTION_MADS * mad)
    return pd.Series(resid, index=frame.index), pd.Series(tol, index=frame.index)


# ---- through-the-cycle averages ------------------------------------------------------------------
CYCLE_WINDOW = 8             # fiscal years averaged for normalized (through-the-cycle) margins
CYCLE_MIN_POINTS = 5


def cycle_average(frame: pd.DataFrame, window: int = CYCLE_WINDOW,
                  min_points: int = CYCLE_MIN_POINTS) -> Tuple[pd.Series, pd.Series]:
    """Mean over up to `window` fiscal years (losses included), requiring the latest year."""
    cols = [c for c in range(window) if c in frame.columns]
    sub = frame[cols]
    n = sub.notna().sum(axis=1)
    avg = sub.mean(axis=1).where((n >= min_points) & sub[cols[0]].notna())
    status = np.where(avg.notna(), "ok", np.where((n >= 1) & (n < min_points), "short", "missing"))
    return avg, pd.Series(status, index=frame.index)
