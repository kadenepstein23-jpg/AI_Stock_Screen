"""Investable-universe filter. Produces the eligible ticker list and an exclusion log."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def build_universe(securities: pd.DataFrame, financials: pd.DataFrame, metrics: pd.DataFrame,
                   ucfg: Dict) -> Tuple[pd.Index, pd.DataFrame]:
    """Return (eligible tickers, exclusions DataFrame[ticker, name, reason, all_reasons])."""
    sec = securities.drop_duplicates("ticker").set_index("ticker")
    m = metrics.reindex(sec.index)
    reasons: Dict[str, List[str]] = {t: [] for t in sec.index}

    def fail(mask: pd.Series, reason: str) -> None:
        for t in mask[mask.fillna(True)].index:
            reasons[t].append(reason)

    allowed = set(ucfg.get("allowed_security_types", ["common"]))
    if ucfg.get("include_adrs"):
        allowed.add("adr")
    fail(~sec["security_type"].isin(allowed), "security type not allowed")
    if ucfg.get("exclude_secondary_share_classes", True):
        fail(sec["share_class"] == "secondary", "secondary share class")
    fail(sec["is_delisted"].astype(bool), "delisted")
    exch = set(ucfg.get("exchanges", []))
    if exch:
        fail(~sec["exchange"].isin(exch), "exchange not allowed")
    sic_ex = set(ucfg.get("exclude_sic_codes", []))
    if sic_ex:
        fail(sec["sic_code"].isin(sic_ex).fillna(False).astype(bool) & True, "SPAC / blank check (SIC)")
    pats = [p.lower() for p in ucfg.get("exclude_name_patterns", [])]
    if pats:
        names = sec["name"].astype("string").str.lower().fillna("")
        fail(names.apply(lambda n: any(p in n for p in pats)), "SPAC / blank check (name)")

    def below(col: str, thresh: float, reason: str, scale: float = 1.0) -> None:
        v = m[col].astype(float) * scale
        fail(v.isna() | (v < thresh), reason)

    below("market_cap_musd", ucfg.get("min_market_cap_usd", 0), "market cap below minimum or missing", 1e6)
    below("price", ucfg.get("min_price", 0), "price below minimum or missing")
    below("median_dollar_volume_60d_musd", ucfg.get("min_median_dollar_volume_60d", 0),
          "illiquid (median $ volume) or missing", 1e6)
    below("fiscal_years_available", ucfg.get("min_fiscal_years", 0), "insufficient fiscal-year history")
    age = m["filing_age_days"].astype(float)
    fail(age.isna() | (age > ucfg.get("max_filing_age_days", 10_000)), "stale or missing filings")
    if ucfg.get("require_positive_revenue", True):
        fy = financials[financials["period_type"] == "FY"].sort_values("period_end")
        rev = fy.groupby("ticker")["revenue"].last().reindex(sec.index)
        fail(rev.isna() | (rev <= 0), "no positive revenue")

    rows = [(t, sec.at[t, "name"], r[0], "; ".join(r)) for t, r in reasons.items() if r]
    excluded = pd.DataFrame(rows, columns=["ticker", "name", "reason", "all_reasons"])
    eligible = pd.Index(sorted(t for t, r in reasons.items() if not r), name="ticker")
    return eligible, excluded
