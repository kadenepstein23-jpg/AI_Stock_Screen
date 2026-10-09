"""Universe validation for a freshly downloaded snapshot (run BEFORE any final screens).

`python -m screener universe --provider sharadar` answers: did the bulk download contain the
whole market with the expected history, is the investable universe a plausible size, why were
companies excluded, and are known reference companies present with the right industry profile?
Only counts and labels are printed: no licensed values are reproduced.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .pipeline import ensure_metrics, load_tables
from .profiles import assign_profiles
from .universe import build_universe

# Reference companies: one or more per industry profile, plus an index-sized name and a
# delisted name (survivorship). Profile is what the screener should assign.
DEFAULT_SPOT_CHECKS: Dict[str, Optional[str]] = {
    "AAPL": "default", "MSFT": "default", "XOM": "cyclical", "NUE": "cyclical", "LEN": "cyclical",
    "JPM": "bank", "USB": "bank", "TRV": "insurance", "CB": "insurance", "O": "reit", "PLD": "reit",
    "NEE": "utility", "DUK": "utility", "UNH": "default", "GS": "bank",
}


def _share(n: int, d: int) -> str:
    return f"{n:,} ({100.0 * n / d:.0f}%)" if d else f"{n:,}"


def universe_report(cfg: Config, provider: str, spot: Optional[List[str]] = None,
                    expect: Optional[Dict[str, Optional[str]]] = None) -> Tuple[List[str], List[tuple]]:
    """(report lines, checks[(name, status PASS/WARN/FAIL, detail)])."""
    snap = ensure_metrics(cfg, provider)
    t = load_tables(snap)
    sec, fin, prices = t["securities"], t["financials"], t["prices"]
    values = snap.read("metrics").set_index("ticker")
    ucfg = cfg.settings["universe"]
    profile = assign_profiles(sec, cfg.profiles, values)
    universe, excluded = build_universe(sec, fin, values, ucfg)
    as_of = snap.as_of
    years_tier = int(cfg.provider_settings(provider).get("bulk_years", 10)) if cfg.is_licensed(provider) else None
    mode = snap.manifest.get("mode", "?")
    L: List[str] = []
    checks: List[tuple] = []

    def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
        checks.append((name, "PASS" if ok else ("WARN" if warn_only else "FAIL"), detail))

    # --- what was downloaded -------------------------------------------------------------
    L.append(f"Snapshot {snap.snapshot_id}  mode={mode}  data as of {as_of.date()}  "
             f"daily scale={snap.manifest.get('daily_value_scale', '?')}")
    s = sec.drop_duplicates("ticker")
    n_sec, n_del = len(s), int(s["is_delisted"].astype(bool).sum())
    L.append(f"\nSECURITIES: {n_sec:,} tickers, {_share(n_del, n_sec)} delisted (kept for survivorship)")
    L.append("   by type: " + ", ".join(f"{k} {v:,}" for k, v in s["security_type"].value_counts().items()))
    listed = s[~s["is_delisted"].astype(bool)]
    L.append("   listed, by exchange: " + ", ".join(
        f"{k} {v:,}" for k, v in listed["exchange"].fillna("?").value_counts().head(8).items()))

    fy = fin[fin["period_type"] == "FY"]
    ttm = fin[fin["period_type"] == "TTM"]
    fy_years = fy.groupby("ticker")["period_end"].nunique()
    L.append(f"\nFUNDAMENTALS: {fy['ticker'].nunique():,} tickers with annual data, "
             f"{ttm['ticker'].nunique():,} with a trailing-twelve-month row")
    if not fy.empty:
        L.append(f"   fiscal years span FY{fy['period_end'].min().year}-FY{fy['period_end'].max().year}; "
                 f"latest filing {pd.to_datetime(fin['filing_date']).max().date()}")
    pdates = prices.groupby("ticker")["date"].max() if not prices.empty else pd.Series(dtype="datetime64[ns]")
    L.append(f"PRICES: {prices['ticker'].nunique():,} tickers, {prices['date'].min().date() if len(prices) else '-'}"
             f" to {prices['date'].max().date() if len(prices) else '-'}")

    # --- universe funnel ---------------------------------------------------------------
    L.append(f"\nINVESTABLE UNIVERSE: {len(universe):,} of {n_sec:,} securities")
    L.append("   excluded, by first failing rule:")
    for reason, n in excluded["reason"].value_counts().items():
        L.append(f"      {n:>7,}  {reason}")
    # among listed common stocks only (the interesting part of the funnel)
    lc = excluded[excluded["ticker"].isin(listed.loc[listed["security_type"] == "common", "ticker"])]
    if not lc.empty:
        L.append("   listed common stocks excluded, by first failing rule:")
        for reason, n in lc["reason"].value_counts().items():
            L.append(f"      {n:>7,}  {reason}")

    u = universe
    mc = values["market_cap_musd"].reindex(u)
    buckets = pd.cut(mc, [0, 2_000, 10_000, 200_000, float("inf")],
                     labels=["small $0.5-2B", "mid $2-10B", "large $10-200B", "mega >$200B"])
    L.append("   by size: " + ", ".join(f"{k} {v:,}" for k, v in buckets.value_counts().sort_index().items()))
    L.append("   by profile: " + ", ".join(f"{k} {v:,}" for k, v in profile.reindex(u).value_counts().items()))
    sectors = s.set_index("ticker")["sector"].reindex(u).fillna("Unknown")
    L.append("   by sector: " + ", ".join(f"{k} {v:,}" for k, v in sectors.value_counts().items()))
    yrs_u = fy_years.reindex(u)
    if len(u):
        L.append(f"   fiscal years per company: median {yrs_u.median():.0f}, "
                 f">=8 years {_share(int((yrs_u >= 8).sum()), len(u))}, "
                 f">=10 years {_share(int((yrs_u >= 10).sum()), len(u))}")

    # --- spot checks -----------------------------------------------------------------------
    expect = dict(DEFAULT_SPOT_CHECKS if expect is None else expect)
    for tk in spot or []:
        expect.setdefault(tk.upper(), None)
    L.append("\nSPOT CHECKS (ticker: in universe? profile, fiscal years; reason if excluded)")
    ex = excluded.set_index("ticker")
    missing, wrong = [], []
    for tk, want in expect.items():
        if tk not in set(s["ticker"]):
            L.append(f"   {tk:<6} NOT IN DATA")
            missing.append(tk)
            continue
        prof = profile.get(tk, "?")
        yrs = int(fy_years.get(tk, 0))
        if tk in u:
            note = "" if want in (None, prof) else f"   <- expected {want}"
            L.append(f"   {tk:<6} yes   {prof:<10} {yrs:>2}y{note}")
        else:
            L.append(f"   {tk:<6} NO    {prof:<10} {yrs:>2}y   {ex.at[tk, 'all_reasons'] if tk in ex.index else ''}")
        if want is not None and prof != want:
            wrong.append(f"{tk} {prof} (expected {want})")

    # --- checks --------------------------------------------------------------------------
    full = mode == "full"
    if cfg.is_licensed(provider):
        check("full bulk download (not the sample)", full, f"mode={mode}")
    age = (pd.Timestamp.today().normalize() - as_of).days
    check("data is current", age <= 7, f"as of {as_of.date()} ({age} days old)")
    check("no duplicate tickers", sec["ticker"].duplicated().sum() == 0,
          f"{int(sec['ticker'].duplicated().sum())} duplicates")
    if full:
        check("whole market downloaded", n_sec >= 10_000 and len(listed) >= 4_000,
              f"{n_sec:,} securities, {len(listed):,} listed")
        check("delisted companies included (survivorship)", n_del >= 1_000, f"{n_del:,} delisted")
        check("universe size plausible (about 1,500-4,000)", 1_000 <= len(u) <= 5_000, f"{len(u):,}")
    if years_tier and len(u):
        med = float(yrs_u.median())
        check(f"history depth matches the {years_tier}-year tier", med >= years_tier - 2,
              f"median {med:.0f} fiscal years in the universe")
    if len(u):
        fresh = pdates.reindex(u)
        stale = int((fresh.isna() | (fresh < as_of - pd.Timedelta(days=7))).sum())
        check("every universe company has a recent price", stale == 0, f"{stale} without a price in the last 7 days")
        no_ttm = int((~pd.Index(u).isin(ttm["ticker"])).sum())
        check("every universe company has a TTM row", no_ttm == 0, f"{no_ttm} missing", warn_only=True)
        unk = int((sectors == "Unknown").sum())
        check("sector labels present", unk <= 0.02 * len(u), f"{unk} without a sector", warn_only=True)
    check("reference companies present", not missing, ", ".join(missing) or "all present", warn_only=not full)
    check("reference companies get the expected profile", not wrong, "; ".join(wrong) or "all as expected")
    return L, checks
