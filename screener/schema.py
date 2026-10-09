"""Canonical, provider-neutral data schema.

Every provider adapter converts its raw downloads into these tables. Everything
downstream (universe, metrics, scoring, outputs) reads ONLY these tables, so a
provider can be swapped or purged without touching the screening logic.

Units: all monetary values in US dollars (not millions); per-share values in
USD per share; share counts in shares. Sign conventions are normalized:
capex, dividends_paid and share_repurchases are POSITIVE numbers when cash
leaves the company.
"""
from __future__ import annotations

from typing import Dict, List

SECURITY_TYPES = ["common", "adr", "etf", "fund", "preferred", "warrant", "unit", "right", "other"]

SECURITIES_COLUMNS: Dict[str, str] = {
    "ticker": "string",
    "name": "string",
    "exchange": "string",          # normalized: NYSE, NASDAQ, NYSEMKT, NYSEARCA, BATS, OTC, ...
    "security_type": "string",     # one of SECURITY_TYPES
    "share_class": "string",       # primary | secondary | single
    "is_delisted": "bool",
    "sector": "string",
    "industry": "string",
    "sic_code": "Int64",
    "cik": "Int64",
    "location": "string",
    "currency": "string",
    "source_category": "string",   # provider's raw classification, kept for audit
}

# period_type: FY (fiscal year) or TTM (trailing twelve months)
FINANCIALS_COLUMNS: Dict[str, str] = {
    "ticker": "string",
    "period_type": "string",
    "period_end": "datetime64[ns]",
    "filing_date": "datetime64[ns]",
    # income statement
    "revenue": "float64",
    "cost_of_revenue": "float64",
    "gross_profit": "float64",
    "operating_income": "float64",
    "ebit": "float64",
    "ebitda": "float64",
    "d_and_a": "float64",
    "interest_expense": "float64",
    "real_estate_impairment": "float64",   # provisions for impairment (REIT FFO add-back)
    "gain_on_sale_real_estate": "float64", # gains on property sales, positive = gain (REIT FFO removal)
    "fx_gain_loss": "float64",             # foreign-currency transaction gain (+) / loss (-), pretax
    "pretax_income": "float64",
    "income_tax": "float64",
    "net_income": "float64",       # attributable to common shareholders
    "eps_diluted": "float64",
    "shares_diluted_wavg": "float64",
    "shares_outstanding": "float64",
    # balance sheet
    "cash": "float64",
    "current_assets": "float64",
    "current_liabilities": "float64",
    "total_assets": "float64",
    "total_liabilities": "float64",
    "total_debt": "float64",
    "equity": "float64",           # total shareholders' equity attributable to the parent
    "preferred_equity": "float64", # carrying value of preferred stock (if reported)
    "aoci": "float64",             # accumulated other comprehensive income (unrealized gains/losses etc.)
    "intangibles": "float64",      # goodwill + other intangibles
    # cash flow
    "operating_cash_flow": "float64",
    "capex": "float64",            # positive = spending
    "free_cash_flow": "float64",
    "sbc": "float64",
    "dividends_paid": "float64",   # positive = paid out
    "share_repurchases": "float64",  # gross, positive = bought back (may be missing)
    "net_common_issuance": "float64",  # issuance minus repurchases (negative = net buyback)
    "dps": "float64",              # dividends declared per share
}

# Raw statement fields (everything except identifiers) — used by the licensing
# guard to recognize files that contain licensed raw values.
RAW_VALUE_FIELDS: List[str] = [c for c in FINANCIALS_COLUMNS if c not in
                               ("ticker", "period_type", "period_end", "filing_date")]

MARKET_COLUMNS: Dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns]",
    "market_cap": "float64",
    "enterprise_value": "float64",   # optional; computed if missing
}

PRICES_COLUMNS: Dict[str, str] = {
    "ticker": "string",
    "date": "datetime64[ns]",
    "close": "float64",
    "close_adj": "float64",          # split/dividend adjusted (for returns)
    "close_split_adj": "float64",    # split-adjusted only (optional; with close_adj it reveals spin-offs)
    "volume": "float64",
}

ESTIMATES_COLUMNS: Dict[str, str] = {
    "ticker": "string",
    "fiscal_period_end": "datetime64[ns]",
    "eps_estimate": "float64",
    "revenue_estimate": "float64",
}

TABLES: Dict[str, Dict[str, str]] = {
    "securities": SECURITIES_COLUMNS,
    "financials": FINANCIALS_COLUMNS,
    "market": MARKET_COLUMNS,
    "prices": PRICES_COLUMNS,
    "estimates": ESTIMATES_COLUMNS,
}

REQUIRED_TABLES = ["securities", "financials"]


def conform(df, table: str):
    """Return df with exactly the canonical columns (missing ones added as NA) and dtypes."""
    import pandas as pd

    cols = TABLES[table]
    out = pd.DataFrame(index=df.index)
    for col, dtype in cols.items():
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series(pd.NA, index=df.index)
        try:
            if dtype.startswith("datetime"):
                s = pd.to_datetime(s, errors="coerce")
            elif dtype == "float64":
                s = pd.to_numeric(s, errors="coerce").astype("float64")
            elif dtype == "Int64":
                s = pd.to_numeric(s, errors="coerce").round().astype("Int64")
            elif dtype == "bool":
                s = s.map(lambda x: bool(x) if not pd.isna(x) else False).astype(bool)
            else:
                s = s.astype("string")
        except (TypeError, ValueError):
            s = s.astype("object")
        out[col] = s
    return out.reset_index(drop=True)
