"""Stock-split adjustment of multi-year share counts and per-share figures.

Why this exists
---------------
Data sources do not restate history consistently after a split. SEC filings,
for example, restate only the years shown in each new 10-K, so a company's
FY2017 EPS and dividends per share may still be on the pre-split basis while
FY2018+ are restated. Without adjustment a 4-for-1 split looks like a 75%
dividend cut or a 300% share issuance.

Method (per company, provider-neutral)
--------------------------------------
1. Share counts. For each share series (diluted weighted average, shares
   outstanding) compare each year with the next older year. A ratio that snaps
   to a standard split factor (3:2, 2:1, 3:1 ... 50:1, or a reverse split) is
   treated as a split, unless (a) total assets (or revenue) scaled up nearly as
   much — a stock-funded acquisition — or (b) the company's other share series
   has data around that year and shows no matching split (mergers phase into
   weighted-average shares over two years and rarely produce a clean ratio in
   both series). Older years are multiplied by the cumulative factor so the
   whole series is on today's share basis.
2. Per-share figures (diluted EPS, dividends per share). Restatement status can
   differ field by field (Apple's FY2018 DPS is restated while its FY2018 share
   count is not), so each value is checked against a figure implied by the
   adjusted totals: EPS vs net income / adjusted diluted shares, DPS vs dividends
   paid / adjusted shares outstanding. If the reported value is about N times the
   implied one, where N is a product of the detected splits, it is divided by N.
   Values that already match are left alone. When totals are missing, the
   per-share value follows its share series' adjustment.

Reported figures are only rescaled by detected split factors — never replaced
with estimates.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SPLIT_FACTORS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
REVERSE_FACTORS = [1.0 / f for f in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)]
CANDIDATES = SPLIT_FACTORS + REVERSE_FACTORS
PER_SHARE_TOLERANCE = math.log(1.35)


def _valid(x) -> bool:
    return x is not None and np.isfinite(x) and x > 0


def snap_split_factor(ratio: float) -> Optional[float]:
    """Return the standard split factor closest to `ratio`, or None if none is close enough."""
    if not _valid(ratio):
        return None
    best, best_err = None, None
    for f in CANDIDATES:
        tol = math.log(1.06) if f == 1.5 else math.log(1.15)
        err = abs(math.log(ratio / f))
        if err < tol and (best_err is None or err < best_err):
            best, best_err = f, err
    return best


def _not_an_acquisition(f: float, scale_ratio: Optional[float]) -> bool:
    """A share jump with a matching jump in assets/revenue is an acquisition, not a split."""
    if f >= 2:
        return scale_ratio is None or scale_ratio <= max(1.5, 0.5 * f)
    if f == 1.5:
        return scale_ratio is not None and scale_ratio <= 1.2
    return scale_ratio is None or scale_ratio >= 0.67  # reverse split: the business did not shrink 50%+


def _candidate_events(shares: np.ndarray, assets: np.ndarray, revenue: np.ndarray) -> List[Tuple[int, int, float]]:
    """Share-count jumps that look like splits: [(newer index a, older index b, factor), ...]."""
    events = []
    idx = [k for k in range(len(shares)) if _valid(shares[k])]
    for a, b in zip(idx, idx[1:]):  # a = newer year, b = next older year with data
        f = snap_split_factor(shares[a] / shares[b])
        if f is None:
            continue
        scale = None
        if _valid(assets[a]) and _valid(assets[b]):
            scale = assets[a] / assets[b]
        elif _valid(revenue[a]) and _valid(revenue[b]):
            scale = revenue[a] / revenue[b]
        if _not_an_acquisition(f, scale):
            events.append((a, b, f))
    return events


def _corroborated(event: Tuple[int, int, float], other: np.ndarray, other_events: List[Tuple[int, int, float]]) -> bool:
    """A real split shows up in BOTH share series (possibly one year apart, because filings restate
    them separately). A stock-funded merger usually doesn't: weighted-average shares phase in over
    two years, so neither year's jump matches a clean split ratio."""
    a, b, f = event
    if any(abs(b2 - b) <= 1 and f2 == f for _, b2, f2 in other_events):
        return True
    # if the other series has no data around this boundary it can't contradict the event
    informative = (_valid(other[a]) and _valid(other[b])) or \
        (b + 1 < len(other) and _valid(other[b]) and _valid(other[b + 1])) or \
        (a - 1 >= 0 and _valid(other[a - 1]) and _valid(other[a]))
    return not informative


def detect_splits(shares: np.ndarray, assets: np.ndarray, revenue: np.ndarray,
                  other_shares: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """shares/assets/revenue are arrays ordered newest (index 0) -> oldest. `other_shares` is the
    company's other share-count series, used to corroborate each split.

    Returns (cumulative factor per index, [(index of first pre-split year, factor), ...]).
    """
    n = len(shares)
    events = _candidate_events(shares, assets, revenue)
    if other_shares is not None:
        other_events = _candidate_events(other_shares, assets, revenue)
        events = [e for e in events if _corroborated(e, other_shares, other_events)]
    step = np.ones(n)
    for _, b, f in events:
        step[b] = f
    return np.cumprod(step), [(b, f) for _, b, f in events]


def _adjust_per_share(reported: np.ndarray, numer: np.ndarray, denom_adj: np.ndarray,
                      chain: List[float], fallback_cum: np.ndarray) -> np.ndarray:
    cands = [1.0]
    prod = 1.0
    for f in chain:
        prod *= f
        cands.append(prod)
    out = reported.copy()
    for j in range(len(reported)):
        rep = reported[j]
        if rep is None or not np.isfinite(rep) or rep == 0:
            continue
        implied = None
        if numer[j] is not None and np.isfinite(numer[j]) and numer[j] != 0 and _valid(denom_adj[j]):
            implied = numer[j] / denom_adj[j]
        if implied is not None:
            m = rep / implied
            if m <= 0:
                continue
            c = min(cands, key=lambda c_: abs(math.log(m / c_)))
            if abs(math.log(m / c)) < PER_SHARE_TOLERANCE:
                out[j] = rep / c
            # no close match -> leave the reported value untouched
        elif fallback_cum[j] != 1.0:
            out[j] = rep / fallback_cum[j]
    return out


def split_adjust_panel(P: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Adjust share counts, EPS and DPS in a fiscal-year panel. Returns (panel, splits_detected)."""
    P = dict(P)
    tickers = P["shares_diluted_wavg"].index
    new = {f: P[f].copy() for f in ("shares_diluted_wavg", "shares_outstanding", "eps_diluted", "dps")}
    n_splits = pd.Series(0.0, index=tickers)
    for t in tickers:
        assets = P["total_assets"].loc[t].to_numpy(dtype=float)
        revenue = P["revenue"].loc[t].to_numpy(dtype=float)
        wavg = P["shares_diluted_wavg"].loc[t].to_numpy(dtype=float)
        so = P["shares_outstanding"].loc[t].to_numpy(dtype=float)
        cum_w, ev_w = detect_splits(wavg, assets, revenue, other_shares=so)
        cum_o, ev_o = detect_splits(so, assets, revenue, other_shares=wavg)
        if not ev_w and not ev_o:
            continue
        wavg_adj, so_adj = wavg * cum_w, so * cum_o
        chain = [f for _, f in sorted(ev_w if len(ev_w) >= len(ev_o) else ev_o)]
        n_splits[t] = float(len(chain))
        # denominators on today's basis, each falling back to the other share series
        den_eps = np.where(np.isfinite(wavg_adj), wavg_adj, so_adj)
        den_dps = np.where(np.isfinite(so_adj), so_adj, wavg_adj)
        fb_eps = np.where(np.isfinite(wavg), cum_w, cum_o)
        fb_dps = np.where(np.isfinite(so), cum_o, cum_w)
        eps = P["eps_diluted"].loc[t].to_numpy(dtype=float)
        dps = P["dps"].loc[t].to_numpy(dtype=float)
        ni = P["net_income"].loc[t].to_numpy(dtype=float)
        div = P["dividends_paid"].loc[t].to_numpy(dtype=float)
        new["shares_diluted_wavg"].loc[t] = wavg_adj
        new["shares_outstanding"].loc[t] = so_adj
        new["eps_diluted"].loc[t] = _adjust_per_share(eps, ni, den_eps, chain, fb_eps)
        new["dps"].loc[t] = _adjust_per_share(dps, div, den_dps, chain, fb_dps)
    P.update(new)
    return P, n_splits
