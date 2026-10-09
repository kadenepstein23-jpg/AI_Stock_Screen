"""Utility handling, validated against NextEra Energy (NEE) and Duke Energy (DUK).

Inputs are the companies' SEC company-facts values (public; $ millions unless noted).
Reported comparisons (Q4 2025 releases): NEE GAAP EPS $3.30 (2025) / $3.37 (2024), adjusted EPS $3.71;
DUK reported EPS $6.31 / $5.71, interest expense $3,634M, net income to common $4,912M.
NEE quarterly dividends: $0.515 (2024) and $0.5666 (2025) -> $2.06 and $2.27 annual (roughly 10%/yr target).
"""
import numpy as np
import pandas as pd
import pytest

from screener import schema
from screener.config import load_config
from screener.flags import detect_flags
from screener.metrics.engine import compute_metrics
from screener.profiles import assign_profiles, mask_not_applicable, peer_group
from screener.scoring import pillar_specs_for_profile
from tests.conftest import fy_rows

AS_OF = pd.Timestamp("2026-06-30")
M = 1e6
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def nextera_rows():
    return fy_rows(
        "NEE", YEARS,
        revenue=[17000 * M, 18800 * M, 23000 * M, 24800 * M, 23500 * M, 25800 * M],
        operating_income=[5116 * M, 2913 * M, 4081 * M, 10237 * M, 7479 * M, 8280 * M],
        d_and_a=[4052 * M, 3924 * M, 4503 * M, 5879 * M, 5462 * M, 6580 * M],
        pretax_income=[2413 * M, 3175 * M, 3832 * M, 7288 * M, 6037 * M, 4530 * M],
        interest_expense=[1432 * M, 1323 * M, 1375 * M, 2463 * M, 2737 * M, 3501 * M],   # cash interest paid
        income_tax=[44 * M, 348 * M, 586 * M, 1006 * M, 339 * M, -802 * M],
        net_income=[2919 * M, 3573 * M, 4147 * M, 7310 * M, 6946 * M, 6835 * M],
        eps_diluted=[1.48, 1.81, 2.10, 3.60, 3.37, 3.30],
        dps=[1.40, 1.54, 1.70, 1.87, 2.06, 2.27],
        dividends_paid=[2743 * M, 3024 * M, 3352 * M, 3782 * M, 4235 * M, 4680 * M],
        operating_cash_flow=[7983 * M, 7553 * M, 8262 * M, 11301 * M, 13260 * M, 12485 * M],
        capex=[np.nan] * 6,                                  # NEE tags capex with company-specific tags
        total_debt=[47633 * M, 54127 * M, 63598 * M, 72956 * M, 82116 * M, 95011 * M],
        cash=[1105 * M, 639 * M, 1601 * M, 2690 * M, 1487 * M, 2812 * M],
        equity=[36513 * M, 37202 * M, 39229 * M, 47468 * M, 50101 * M, 54608 * M],
        total_assets=[140912 * M, 158935 * M, 158935 * M, 177489 * M, 190144 * M, 212721 * M],
        current_assets=[7.5e9, 9.0e9, 11.0e9, 11.4e9, 11.0e9, 12.0e9],
        current_liabilities=[13e9, 15e9, 18e9, 19e9, 19e9, 20e9],
        shares_diluted_wavg=[1968.8 * M, 1972.2 * M, 1978.6 * M, 2030.8 * M, 2059.2 * M, 2070.6 * M],
        shares_outstanding=[1960 * M, 1963 * M, 1987 * M, 2052 * M, 2057 * M, 2083 * M],
    )


def duke_rows():
    return fy_rows(
        "DUK", YEARS,
        revenue=[23253 * M, 24486 * M, 28675 * M, 28674 * M, 30050 * M, 31741 * M],
        operating_income=[4571 * M, 5500 * M, 6012 * M, 7070 * M, 7926 * M, 8626 * M],
        d_and_a=[5486 * M, 5663 * M, 5843 * M, 6084 * M, 6419 * M, 7704 * M],
        pretax_income=[920 * M, 3991 * M, 4078 * M, 4767 * M, 5194 * M, 5712 * M],
        interest_expense=[2097 * M, 2207 * M, 2439 * M, 3014 * M, 3384 * M, 3634 * M],
        income_tax=[-169 * M, 268 * M, 300 * M, 438 * M, 590 * M, 642 * M],
        net_income=[1270 * M, 3802 * M, 2444 * M, 2735 * M, 4402 * M, 4912 * M],
        eps_diluted=[1.72, 4.94, 3.17, 3.54, 5.71, 6.31],
        dps=[3.82, 3.90, 3.98, 4.06, 4.14, 4.22],
        dividends_paid=[2812 * M, 3114 * M, 3179 * M, np.nan, np.nan, np.nan],   # untagged after 2022
        operating_cash_flow=[8856 * M, 8290 * M, 5927 * M, 9878 * M, 12328 * M, 12330 * M],
        capex=[9907 * M, 9715 * M, 11367 * M, 12604 * M, 12280 * M, 14024 * M],
        total_debt=[62736 * M, 67139 * M, 73703 * M, 79540 * M, 84273 * M, 89836 * M],
        cash=[259 * M, 341 * M, 409 * M, 253 * M, 314 * M, 245 * M],
        equity=[47964 * M, 49296 * M, 49322 * M, 49112 * M, 50127 * M, 51842 * M],
        total_assets=[162e9, 169e9, 178e9, 177e9, 186e9, 196e9],
        shares_diluted_wavg=[738 * M, 769 * M, 770 * M, 771 * M, 772 * M, 777 * M],
    )


def _tables(rows, mcap=None):
    sic = {"NEE": 4911, "DUK": 4931}
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sector="Utilities", industry="Utilities - Regulated Electric",
                              sic_code=sic.get(t, 4911)) for t in tickers])
    mkt = pd.DataFrame([dict(ticker=t, date=AS_OF, market_cap=(mcap or {}).get(t)) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(mkt, "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


@pytest.fixture(scope="module")
def util():
    cfg = load_config()
    t = _tables(nextera_rows() + duke_rows(), mcap={"NEE": 150e9, "DUK": 95e9})
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(mv, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["DUK", "NEE"]))
    return cfg, t, mv, ms, prof, fl


# ---- leverage ------------------------------------------------------------------------------------
def test_utility_leverage_metrics(util):
    cfg, t, v, s, prof, fl = util
    assert prof["NEE"] == "utility" and prof["DUK"] == "utility"
    assert v.at["NEE", "net_debt_to_ebitda_x"] == pytest.approx((95011 - 2812) / (8280 + 6580))
    assert v.at["NEE", "cfo_to_debt_pct"] == pytest.approx(12485 / 95011 * 100)
    assert v.at["DUK", "cfo_to_debt_pct"] == pytest.approx(12330 / 89836 * 100)
    assert v.at["DUK", "debt_to_capital_pct"] == pytest.approx(89836 / (89836 + 51842) * 100)


def test_utility_leverage_flag_uses_cfo_to_debt(util):
    cfg, t, v, s, prof, fl = util
    # NEE's 6.2x net debt/EBITDA is normal for a utility; CFO/debt of 13% is above the 12% trigger
    assert v.at["NEE", "net_debt_to_ebitda_x"] > 6.0
    assert "high_leverage" not in set(fl[fl["ticker"] == "NEE"]["flag"])
    weak = [dict(r, ticker="WEAK") for r in duke_rows()]
    for r in weak:
        if r["period_end"].year == 2025:
            r["operating_cash_flow"] = 6000 * M            # CFO/debt 6.7% -> severe
    tb = _tables(duke_rows() + weak)
    vv, ss = compute_metrics(tb, AS_OF)
    pr = assign_profiles(tb["securities"], cfg.profiles)
    vv, ss = mask_not_applicable(vv, ss, pr, cfg.profiles)
    f2 = detect_flags(vv, pr, peer_group(tb["securities"], pr, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["DUK", "WEAK"]))
    hl = f2[(f2["ticker"] == "WEAK") & (f2["flag"] == "high_leverage")]
    assert len(hl) == 1 and hl.iloc[0]["default_severity"] == "severe" and "CFO/debt 7%" in hl.iloc[0]["detail"]
    assert "high_leverage" not in set(f2[f2["ticker"] == "DUK"]["flag"])


# ---- capex-heavy FCF -----------------------------------------------------------------------------
UTIL_NA = ["fcf_margin_pct", "fcf_cagr_3y_pct", "fcf_cagr_5y_pct", "fcf_margin_chg_3y_pp", "fcf_yield_pct",
           "p_fcf_x", "fcf_payout_ratio_pct", "fcf_payout_3y_pct", "fcf_conversion_3y_pct",
           "fcf_positive_share_5y_pct", "gross_margin_pct", "gross_margin_chg_3y_pp", "current_ratio_x"]


def test_fcf_and_working_capital_metrics_not_applicable(util):
    cfg, t, v, s, prof, fl = util
    for m in UTIL_NA:
        assert s.at["DUK", m] == "na", m
        assert s.at["NEE", m] == "na", m
    # Duke's FCF really is negative (capex > operating cash flow) but it is not flagged or scored
    assert 12330 - 14024 < 0
    duk = set(fl[fl["ticker"] == "DUK"]["flag"])
    assert not duk & {"negative_fcf", "poor_fcf_conversion", "weak_dividend_coverage"}


def test_utility_pillars():
    cfg = load_config()
    for screen in ("growth", "income"):
        specs = pillar_specs_for_profile(cfg.screens[screen], cfg.profiles["utility"])
        used = {m["name"] for items in specs.values() for m in items}
        assert not used & set(UTIL_NA), (screen, used & set(UTIL_NA))
        assert "cfo_to_debt_pct" in used
    inc = {m["name"] for items in pillar_specs_for_profile(cfg.screens["income"], cfg.profiles["utility"]).values()
           for m in items}
    assert {"payout_ratio_pct", "pe_x", "pb_x", "ev_ebitda_x", "interest_coverage_x"} <= inc


# ---- dividends ------------------------------------------------------------------------------------
def test_utility_dividend_payout_and_growth(util):
    cfg, t, v, s, prof, fl = util
    assert v.at["NEE", "payout_ratio_pct"] == pytest.approx(4680 / 6835 * 100)          # GAAP basis
    assert v.at["NEE", "dps_cagr_3y_pct"] == pytest.approx(((2.27 / 1.70) ** (1 / 3) - 1) * 100)
    assert v.at["NEE", "dps_cagr_3y_pct"] == pytest.approx(10.0, abs=0.5)               # ~10%/yr policy
    assert v.at["NEE", "dividend_growth_streak_years"] == 5
    # Duke stopped tagging dividends paid: DPS x diluted shares is used instead
    assert v.at["DUK", "payout_ratio_pct"] == pytest.approx(4.22 * 777 / 4912 * 100)
    assert v.at["DUK", "dps_cagr_5y_pct"] == pytest.approx(((4.22 / 3.82) ** (1 / 5) - 1) * 100)


def test_sec_parser_ordinary_dividends_tag():
    from screener.providers.sec import parse_companyfacts

    def dur(v):
        return {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": v, "filed": "2026-02-27",
                                   "form": "10-K", "fy": 2025, "fp": "FY"}]}}

    facts = {"facts": {"us-gaap": {"Revenues": dur(31741.0), "NetIncomeLoss": dur(5071.0),
                                   "PaymentsOfOrdinaryDividends": dur(3300.0), "Dividends": dur(3290.0)}}}
    assert parse_companyfacts(facts, "DUK").iloc[0]["dividends_paid"] == 3300.0


# ---- interest coverage, earnings, valuation -----------------------------------------------------------
def test_interest_coverage_and_gaap_earnings(util):
    cfg, t, v, s, prof, fl = util
    assert v.at["DUK", "interest_coverage_x"] == pytest.approx(8626 / 3634)            # reported interest
    assert v.at["NEE", "interest_coverage_x"] == pytest.approx(8280 / 3501)
    assert "low_interest_coverage" not in set(fl["flag"])                            # utility threshold 2.0x
    assert v.at["NEE", "eps_cagr_1y_pct"] == pytest.approx((3.30 / 3.37 - 1) * 100)   # GAAP, not adjusted
    assert v.at["DUK", "eps_cagr_1y_pct"] == pytest.approx((6.31 / 5.71 - 1) * 100)


def test_utility_valuation_metrics(util):
    cfg, t, v, s, prof, fl = util
    assert v.at["NEE", "pe_x"] == pytest.approx(150e9 / (6835 * M))
    assert v.at["NEE", "pb_x"] == pytest.approx(150e9 / (54608 * M))
    ev = 150e9 + (95011 - 2812) * M
    assert v.at["NEE", "ev_ebitda_x"] == pytest.approx(ev / ((8280 + 6580) * M))
    assert v.at["NEE", "dividend_yield_pct"] == pytest.approx(4680 * M / 150e9 * 100)
