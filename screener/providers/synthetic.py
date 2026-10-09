"""Synthetic provider: a deterministic universe of FICTIONAL companies.

Purpose
-------
* lets the entire pipeline (universe -> metrics -> flags -> scoring -> outputs)
  run and be tested with zero API access and zero licensed data;
* every archetype is designed to exercise a specific rule: compounders,
  yield traps, dilutive SBC, negative equity, banks, REITs, utilities,
  cyclicals at peak, stale filers, ETFs, preferreds, warrants, SPACs, etc.;
* safe to commit and publish — no real company data is involved.

All tickers start with "SY" and all names contain "(Synthetic)".
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import schema
from ..store import Snapshot
from .base import Provider

AS_OF = date(2026, 9, 30)
FY_YEARS = list(range(2016, 2026))  # FY2016 .. FY2025 (10 fiscal years)

SECTORS = {
    "tech": ("Technology", "Software - Application", 7372),
    "semis": ("Technology", "Semiconductors", 3674),
    "health": ("Healthcare", "Drug Manufacturers - General", 2834),
    "staples": ("Consumer Defensive", "Packaged Foods", 2000),
    "industrial": ("Industrials", "Specialty Industrial Machinery", 3560),
    "retail": ("Consumer Cyclical", "Apparel Retail", 5651),
    "media": ("Communication Services", "Entertainment", 4841),
    "bank": ("Financial Services", "Banks - Regional", 6022),
    "insurer": ("Financial Services", "Insurance - Property & Casualty", 6331),
    "reit": ("Real Estate", "REIT - Industrial", 6798),
    "utility": ("Utilities", "Utilities - Regulated Electric", 4911),
    "energy": ("Energy", "Oil & Gas E&P", 1311),
    "chem": ("Basic Materials", "Specialty Chemicals", 2821),
}

# Archetype parameters. g = revenue growth, om = operating margin path (start, end),
# lev = net-debt/EBITDA-style debt multiple, pe = valuation multiple, r1 = 1-yr price return.
BASE = dict(g=0.08, g_sd=0.02, gm=(0.45, 0.45), om=(0.18, 0.18), da=0.04, lev=1.0,
            cash=0.10, eq=0.55, cr=1.8, capex=0.04, sbc=0.02, share_g=-0.01,
            payout=0.0, dps_g=0.0, buyback=0.3, issuance=0.0, pe=20.0, val="pe",
            r1=0.10, turnover=0.006, rev0=2.0e9, kind="normal", tax=0.21, rate=0.05)

ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "compounder": dict(g=0.20, g_sd=0.03, gm=(0.62, 0.70), om=(0.18, 0.29), sbc=0.04,
                       lev=0.3, pe=34, r1=0.25, buyback=0.2, cash=0.18, ttm=True),
    "steady_grower": dict(g=0.10, gm=(0.48, 0.50), om=(0.19, 0.21), payout=0.20, dps_g=0.08, pe=22),
    "hypergrowth": dict(g=0.42, g_sd=0.05, gm=(0.70, 0.74), om=(-0.30, -0.06), sbc=0.15,
                        share_g=0.06, lev=0.0, cash=0.45, issuance=0.05, buyback=0.0,
                        val="ev_sales", pe=9.0, r1=0.40, rev0=4.0e8),
    "aristocrat": dict(g=0.05, g_sd=0.01, gm=(0.40, 0.41), om=(0.21, 0.22), payout=0.55,
                       dps_g=0.07, lev=1.8, pe=20, r1=0.08, buyback=0.3, rev0=8.0e9),
    "yield_trap": dict(g=-0.06, g_sd=0.02, gm=(0.38, 0.30), om=(0.16, 0.06), payout=0.45,
                       dps_g=0.03, lev=2.0, lev_end=4.6, pe=13, pe_now=15.0, yield_now=0.095, r1=-0.38,
                       buyback=0.0, rev0=6.0e9),
    "high_leverage": dict(g=0.06, om=(0.20, 0.19), lev=6.5, pe=12, eq=0.30, payout=0.25, dps_g=0.03),
    "negative_equity": dict(g=0.07, gm=(0.55, 0.56), om=(0.30, 0.31), lev=2.8, eq=-0.25,
                            payout=0.45, dps_g=0.08, buyback=0.9, share_g=-0.03, pe=24),
    "diluter": dict(g=0.22, gm=(0.72, 0.73), om=(0.05, 0.09), sbc=0.14, share_g=0.07,
                    issuance=0.04, buyback=0.0, lev=0.0, cash=0.35, pe=60, r1=0.15),
    "decliner": dict(g=-0.04, om=(0.15, 0.07), gm=(0.40, 0.34), pe=9, r1=-0.20, payout=0.30, dps_g=0.0),
    "bank": dict(kind="bank", g=0.06, pe=1.25, val="pb", payout=0.35, dps_g=0.07, r1=0.12, rev0=5.0e9),
    "insurer": dict(kind="insurer", g=0.07, pe=1.4, val="pb", payout=0.30, dps_g=0.06, r1=0.10, rev0=1.2e10),
    "reit": dict(kind="reit", g=0.05, om=(0.34, 0.36), da=0.36, lev=6.0, eq=1.6, tax=0.0,
                 payout=0.0, ffo_payout=0.75, dps_g=0.04, val="p_ffo", pe=15, r1=0.05, rate=0.035,
                 capex=0.10, buyback=0.0, rev0=1.5e9),
    "utility": dict(kind="normal", g=0.04, g_sd=0.01, gm=(0.40, 0.40), om=(0.22, 0.22), da=0.14,
                    lev=5.2, capex=0.45, payout=0.65, dps_g=0.05, pe=18, eq=0.9, buyback=0.0, rate=0.035,
                    r1=0.06, rev0=7.0e9),
    "cyclical_peak": dict(g=0.03, g_sd=0.08, gm=(0.30, 0.30), om=(0.08, 0.08), spike=(0.26, 0.30),
                          lev=1.5, capex=0.12, da=0.10, pe=7, payout=0.25, dps_g=0.05, r1=0.30),
    "cyclical_normal": dict(g=0.04, g_sd=0.06, gm=(0.30, 0.30), om=(0.11, 0.12), lev=1.4,
                            capex=0.10, da=0.09, pe=12, payout=0.30, dps_g=0.04),
}

# (ticker, name, archetype, sector_key, overrides, security overrides)
UNIVERSE: List[Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any]]] = [
    ("SYCMA", "Alderbrook Software", "compounder", "tech", {}, {}),
    ("SYCMB", "Brightwater Cloud", "compounder", "tech", {"g": 0.24, "pe": 42}, {}),
    ("SYCMC", "Cobalt Devices", "compounder", "semis", {"g": 0.17, "om": (0.22, 0.31)}, {}),
    ("SYCMD", "Dunmore Diagnostics", "compounder", "health", {"g": 0.15, "pe": 30}, {}),
    ("SYCME", "Everline Payments", "compounder", "tech", {"g": 0.19, "om": (0.25, 0.27), "pe": 28}, {}),
    ("SYCMF", "Foxglove Medical", "compounder", "health", {"g": 0.13, "om": (0.20, 0.24), "pe": 26}, {}),
    ("SYSGA", "Granite Industrial", "steady_grower", "industrial", {}, {}),
    ("SYSGB", "Harbor Tools", "steady_grower", "industrial", {"g": 0.09, "pe": 19}, {}),
    ("SYSGC", "Ironleaf Foods", "steady_grower", "staples", {"g": 0.085}, {}),
    ("SYSGD", "Juniper Retail", "steady_grower", "retail", {"g": 0.11, "pe": 18}, {}),
    ("SYSGE", "Keystone Media", "steady_grower", "media", {"g": 0.095, "pe": 21}, {}),
    ("SYSGF", "Lakemont Systems", "steady_grower", "tech", {"g": 0.12, "pe": 25}, {}),
    ("SYHGA", "Meridian AI Labs", "hypergrowth", "tech", {}, {}),
    ("SYHGB", "Nimbus Genomics", "hypergrowth", "health", {"g": 0.35}, {}),
    ("SYHGC", "Orbit Robotics", "hypergrowth", "industrial", {"g": 0.38, "om": (-0.25, 0.02)}, {}),
    ("SYARA", "Pinecrest Brands", "aristocrat", "staples", {}, {}),
    ("SYARB", "Quarry Household", "aristocrat", "staples", {"payout": 0.62, "pe": 22}, {}),
    ("SYARC", "Redwood Pharma", "aristocrat", "health", {"g": 0.06, "om": (0.28, 0.29)}, {}),
    ("SYARD", "Summit Railworks", "aristocrat", "industrial", {"payout": 0.40, "dps_g": 0.09}, {}),
    ("SYARE", "Tidewater Beverages", "aristocrat", "staples", {"payout": 0.68, "pe": 24}, {}),
    ("SYARF", "Upland Insurance Services", "aristocrat", "industrial", {"payout": 0.45}, {}),
    ("SYYTA", "Vantage Print & Paper", "yield_trap", "industrial", {}, {}),
    ("SYYTB", "Westfield Telecom", "yield_trap", "media", {"g": -0.04, "yield_now": 0.11}, {}),
    ("SYHLA", "Xavier Leisure", "high_leverage", "retail", {}, {}),
    ("SYHLB", "Yardley Packaging", "high_leverage", "industrial", {"lev": 7.5}, {}),
    ("SYNEQ", "Zenith Quick Service", "negative_equity", "retail", {}, {}),
    ("SYDIL", "Apex Social", "diluter", "tech", {}, {}),
    ("SYDCA", "Bramble Department Stores", "decliner", "retail", {}, {}),
    ("SYDCB", "Cinder Legacy Media", "decliner", "media", {"g": -0.03}, {}),
    ("SYBKA", "Delta Valley Bancorp", "bank", "bank", {}, {}),
    ("SYBKB", "Eastgate Financial", "bank", "bank", {"pe": 1.05, "g": 0.05}, {}),
    ("SYBKC", "Fairhaven Bank", "bank", "bank", {"pe": 1.6, "g": 0.08}, {}),
    ("SYBKD", "Glenwood Bancshares", "bank", "bank", {"pe": 0.95, "payout": 0.45}, {}),
    ("SYBKE", "Highland Savings", "bank", "bank", {"g": 0.04, "pe": 1.15}, {}),
    ("SYINA", "Ivory Mutual Holdings", "insurer", "insurer", {}, {}),
    ("SYINB", "Jasper Specialty Insurance", "insurer", "insurer", {"g": 0.10, "pe": 1.9}, {}),
    ("SYRTA", "Kingsley Logistics REIT", "reit", "reit", {}, {}),
    ("SYRTB", "Larkspur Self Storage REIT", "reit", "reit", {"g": 0.07, "pe": 18}, {}),
    ("SYRTC", "Maple Office REIT", "reit", "reit", {"g": -0.02, "lev": 8.5, "ffo_payout": 0.95, "pe": 8, "r1": -0.25}, {}),
    ("SYRTD", "Northstar Residential REIT", "reit", "reit", {"g": 0.05}, {}),
    ("SYUTA", "Oakridge Power & Light", "utility", "utility", {}, {}),
    ("SYUTB", "Prairie Electric", "utility", "utility", {"g": 0.05}, {}),
    ("SYUTC", "Quinton Water & Gas", "utility", "utility", {"lev": 5.8, "payout": 0.75}, {}),
    ("SYUTD", "Riverbend Energy Delivery", "utility", "utility", {"g": 0.035}, {}),
    ("SYCYA", "Sagebrush Petroleum", "cyclical_peak", "energy", {}, {}),
    ("SYCYB", "Tamarack Chemicals", "cyclical_peak", "chem", {"spike": (0.22, 0.24)}, {}),
    ("SYCYC", "Umber Resources", "cyclical_normal", "energy", {}, {}),
    ("SYCYD", "Verdant Materials", "cyclical_normal", "chem", {}, {}),
    # ----- rule exercisers ---------------------------------------------------
    ("SYSTL", "Willow Stale Filer", "steady_grower", "industrial", {"stale": True}, {}),
    ("SYIPO", "Xenon Recent IPO", "compounder", "tech", {"years": 3}, {}),
    ("SYMIS", "Yonder Patchy Data", "steady_grower", "health", {"missing": True}, {}),
    ("SYSML", "Zephyr Small Cap", "steady_grower", "industrial", {"rev0": 2.0e7}, {}),
    ("SYILQ", "Amber Thinly Traded", "steady_grower", "industrial", {"turnover": 0.00001}, {}),
    ("SYPNY", "Basalt Penny Stock", "steady_grower", "retail", {"rev0": 9.0e8, "price_target": 3.0}, {}),
    ("SYETF", "Synthetic Total Market ETF", "steady_grower", "tech", {}, {"security_type": "etf"}),
    ("SYPRA", "Alderbrook Software 6% Preferred", "steady_grower", "tech", {}, {"security_type": "preferred"}),
    ("SYWTS", "Meridian AI Labs Warrants", "steady_grower", "tech", {}, {"security_type": "warrant"}),
    ("SYSPC", "Coastal Acquisition Corp", "steady_grower", "industrial", {}, {"sic_code": 6770}),
    ("SYCMA.B", "Alderbrook Software Class B", "compounder", "tech", {}, {"share_class": "secondary"}),
    ("SYADR", "Danube Holdings ADR", "steady_grower", "industrial", {}, {"security_type": "adr"}),
]


def _ticker_seed(ticker: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(ticker))


def _company(ticker: str, name: str, arche: str, sector_key: str, ov: Dict[str, Any],
             sec_ov: Dict[str, Any]) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    p = dict(BASE)
    p.update(ARCHETYPES[arche])
    p.update(ov)
    rng = np.random.default_rng(_ticker_seed(ticker))
    sector, industry, sic = SECTORS[sector_key]
    n_years = int(p.get("years", len(FY_YEARS)))
    years = FY_YEARS[-n_years:]
    if p.get("stale"):
        years = [y - 2 for y in years]
    n = len(years)
    t = np.linspace(0, 1, n)

    # ---- income statement --------------------------------------------------
    growth = p["g"] + rng.normal(0, p["g_sd"], n)
    rev = np.empty(n)
    rev[0] = p["rev0"] * (1 + rng.normal(0, 0.05))
    for i in range(1, n):
        rev[i] = rev[i - 1] * (1 + growth[i])
    gm = p["gm"][0] + (p["gm"][1] - p["gm"][0]) * t + rng.normal(0, 0.005, n)
    om = p["om"][0] + (p["om"][1] - p["om"][0]) * t + rng.normal(0, 0.006, n)
    if "spike" in p:  # cyclical: margins spike in the last two years
        om[-2], om[-1] = p["spike"]
    da = p["da"] * rev
    ebit = om * rev
    ebitda = ebit + da
    lev = p["lev"] + (p.get("lev_end", p["lev"]) - p["lev"]) * t
    debt = np.maximum(lev * np.maximum(ebitda, 0.15 * rev), 0.0)
    if p["lev"] == 0:
        debt = np.zeros(n)
    cash = p["cash"] * rev
    interest = p["rate"] * debt
    pretax = ebit - interest
    tax = np.where(pretax > 0, pretax * p["tax"], 0.0)
    ni = pretax - tax
    shares0 = rev[0] / 40.0
    shares = shares0 * (1 + p["share_g"]) ** np.arange(n)
    sbc = p["sbc"] * rev
    ocf = ni + da + sbc + rng.normal(0, 0.01, n) * rev
    capex = p["capex"] * rev
    fcf = ocf - capex

    kind = p["kind"]
    preferred = np.full(n, np.nan)
    if kind == "bank":
        ni = rev * (0.28 + rng.normal(0, 0.01, n))
        pretax = ni / 0.79
        tax = pretax - ni
        total_assets = rev * 20.0
        equity = total_assets * 0.09
        preferred = equity * 0.06
        debt = total_assets * 0.07
        cash = total_assets * 0.05
        ocf = ni * (1.1 + rng.normal(0, 0.15, n))
        capex = rev * 0.03
        fcf = ocf - capex
        da = rev * 0.03
        sbc = rev * 0.01
    elif kind == "insurer":
        ni = rev * (0.10 + rng.normal(0, 0.02, n))
        pretax = ni / 0.79
        tax = pretax - ni
        ebit = pretax + 0.01 * rev
        total_assets = rev * 3.0
        equity = rev * 0.8
        debt = rev * 0.15
        cash = rev * 0.10
        ocf = ni * 1.3
        capex = rev * 0.01
        fcf = ocf - capex

    if p["payout"] > 0 or p.get("ffo_payout"):
        if p.get("ffo_payout"):
            base_div = p["ffo_payout"] * (ni + da)
        else:
            base_div = p["payout"] * ni
        dps0 = max(base_div[0], 0) / shares[0]
        dps = dps0 * (1 + p["dps_g"]) ** np.arange(n)
        if p.get("ffo_payout"):
            dps = p["ffo_payout"] * (ni + da) / shares
        if arche == "yield_trap":
            dps[-1] = dps[-2] * 1.0  # dividend held flat while earnings collapse
        div = dps * shares
    else:
        dps = np.zeros(n)
        div = np.zeros(n)
    repurch = p["buyback"] * np.maximum(fcf - div, 0)
    issuance = p["issuance"] * rev
    net_issuance = issuance - repurch

    if kind not in ("bank", "insurer"):
        cl = 0.20 * rev
        ca = cl * p["cr"]
        equity = p["eq"] * rev
        other_liab = 0.10 * rev
        total_liab = debt + cl + other_liab
        total_assets = equity + total_liab
    else:
        ca = np.full(n, np.nan)
        cl = np.full(n, np.nan)
        total_liab = total_assets - equity
    intangibles = 0.12 * np.abs(total_assets)

    fin_rows = []
    for i, y in enumerate(years):
        row = dict(
            ticker=ticker, period_type="FY",
            period_end=pd.Timestamp(f"{y}-12-31"), filing_date=pd.Timestamp(f"{y + 1}-02-20"),
            revenue=rev[i], cost_of_revenue=rev[i] * (1 - gm[i]), gross_profit=rev[i] * gm[i],
            operating_income=ebit[i], ebit=ebit[i], ebitda=ebitda[i], d_and_a=da[i],
            interest_expense=interest[i], pretax_income=pretax[i], income_tax=tax[i],
            net_income=ni[i], eps_diluted=ni[i] / shares[i], shares_diluted_wavg=shares[i],
            shares_outstanding=shares[i], cash=cash[i], current_assets=ca[i],
            current_liabilities=cl[i], total_assets=total_assets[i], total_liabilities=total_liab[i],
            total_debt=debt[i], equity=equity[i], intangibles=intangibles[i],
            preferred_equity=preferred[i] if kind == "bank" else np.nan,
            real_estate_impairment=0.03 * rev[i] if kind == "reit" else np.nan,
            gain_on_sale_real_estate=0.02 * rev[i] if kind == "reit" else np.nan,
            operating_cash_flow=ocf[i], capex=capex[i], free_cash_flow=fcf[i], sbc=sbc[i],
            dividends_paid=div[i], share_repurchases=repurch[i],
            net_common_issuance=net_issuance[i], dps=dps[i],
        )
        if kind == "bank":
            for k in ("cost_of_revenue", "gross_profit", "operating_income", "ebit", "ebitda",
                      "interest_expense"):
                row[k] = np.nan
        if kind == "insurer":
            for k in ("cost_of_revenue", "gross_profit", "ebitda"):
                row[k] = np.nan
        if p.get("missing"):
            for k in ("operating_cash_flow", "capex", "free_cash_flow", "sbc", "gross_profit",
                      "cost_of_revenue", "current_assets", "current_liabilities", "d_and_a", "ebitda"):
                row[k] = np.nan
        fin_rows.append(row)
    if p.get("ttm"):
        last = dict(fin_rows[-1])
        last.update(period_type="TTM", period_end=pd.Timestamp("2026-06-30"),
                    filing_date=pd.Timestamp("2026-08-05"))
        for k in ("revenue", "cost_of_revenue", "gross_profit", "operating_income", "ebit", "ebitda",
                  "d_and_a", "pretax_income", "income_tax", "net_income", "operating_cash_flow",
                  "capex", "free_cash_flow", "sbc", "dividends_paid", "share_repurchases",
                  "net_common_issuance", "eps_diluted", "dps"):
            if last.get(k) is not None and not pd.isna(last[k]):
                last[k] = last[k] * (1 + p["g"] / 2)
        fin_rows.append(last)

    # ---- market values at each fiscal year end + today --------------------------
    def valuation(i: int, multiple: float) -> float:
        if p["val"] == "pb":
            return multiple * equity[i]
        if p["val"] == "p_ffo":
            return multiple * (ni[i] + da[i])
        if p["val"] == "ev_sales":
            return multiple * rev[i]
        return multiple * max(ni[i], 0.02 * rev[i])

    mkt_rows = []
    for i, y in enumerate(years):
        mult = p["pe"] * (1 + rng.normal(0, 0.08))
        mkt_rows.append(dict(ticker=ticker, date=pd.Timestamp(f"{y}-12-31"),
                             market_cap=valuation(i, mult), enterprise_value=np.nan))
    mcap_now = valuation(n - 1, p.get("pe_now", p["pe"]))
    if p.get("yield_now") and div[-1] > 0:
        mcap_now = div[-1] / p["yield_now"]
    if p.get("pe_now"):
        mcap_now *= 1.0
    else:
        mcap_now *= (1 + p["r1"] * 0.5)
    shares_now = shares[-1]
    if "price_target" in p:
        shares_now = mcap_now / p["price_target"]
    price_now = mcap_now / shares_now
    if not p.get("stale"):
        mkt_rows.append(dict(ticker=ticker, date=pd.Timestamp(AS_OF), market_cap=mcap_now,
                             enterprise_value=np.nan))

    # ---- daily prices (last ~300 trading days) --------------------------------
    days = pd.bdate_range(end=pd.Timestamp(AS_OF), periods=300)
    # Brownian bridge: the 1-yr return equals the archetype's r1 exactly
    start_price = price_now / (1 + p["r1"])
    n_d = len(days)
    anchor = n_d - 253                      # index of the price ~1 year before AS_OF
    walk = np.cumsum(rng.normal(0, 0.010, n_d))
    tt = np.arange(n_d, dtype=float)
    bridge = walk - walk[anchor] - (walk[-1] - walk[anchor]) * (tt - anchor) / (n_d - 1 - anchor)
    logp = np.log(start_price) + (np.log(price_now) - np.log(start_price)) * (tt - anchor) / (n_d - 1 - anchor)
    path = np.exp(logp + bridge)
    if p.get("stale"):
        days = days[:-120]
        path = path[:-120]
    vol = (mcap_now * p["turnover"]) / path * (1 + rng.normal(0, 0.2, len(path))).clip(0.3)
    px_rows = [dict(ticker=ticker, date=d, close=c, close_adj=c, volume=v)
               for d, c, v in zip(days, path, vol)]

    sec = dict(ticker=ticker, name=f"{name} (Synthetic)",
               exchange="NASDAQ" if sector_key in ("tech", "semis", "health") else "NYSE",
               security_type="common", share_class="single", is_delisted=False,
               sector=sector, industry=industry, sic_code=sic, cik=None,
               location="Synthetic, USA", currency="USD", source_category=f"synthetic:{arche}")
    sec.update(sec_ov)
    if sec_ov.get("share_class") == "secondary":
        pass
    return sec, fin_rows, mkt_rows, px_rows


def build_synthetic_tables() -> Dict[str, pd.DataFrame]:
    secs, fins, mkts, pxs = [], [], [], []
    for spec in UNIVERSE:
        s, f, m, px = _company(*spec)
        secs.append(s)
        fins.extend(f)
        mkts.extend(m)
        pxs.extend(px)
    # the primary class of the dual-class company is marked "primary"
    for s in secs:
        if s["ticker"] == "SYCMA":
            s["share_class"] = "primary"
    return {
        "securities": schema.conform(pd.DataFrame(secs), "securities"),
        "financials": schema.conform(pd.DataFrame(fins), "financials"),
        "market": schema.conform(pd.DataFrame(mkts), "market"),
        "prices": schema.conform(pd.DataFrame(pxs), "prices"),
    }


class SyntheticProvider(Provider):
    name = "synthetic"

    def refresh(self, **options: Any) -> Snapshot:
        snap = self.store.new_snapshot(self.name, as_of=AS_OF,
                                       extra={"source": "screener.providers.synthetic",
                                              "companies": len(UNIVERSE)})
        for table, df in build_synthetic_tables().items():
            snap.write(table, df)
        self.store.activate(snap)
        return snap
