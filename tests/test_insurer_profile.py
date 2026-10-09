"""Insurer handling, validated against Travelers (TRV) and Chubb (CB).

Inputs are the companies' SEC company-facts values (public). Reported comparisons:
  Travelers Q4 2025 release: BVPS $151.21 (2025) / $122.97 (2024); ROE 21.0%; total debt $9,267M;
    debt-to-capital 22.0%; diluted EPS $27.43 / $21.47.
  Chubb Q4 2025 release: BVPS $188.59, up 18.0% in 2025; BVPS excluding AOCI up 11.0%.
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


def travelers_rows():
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    return fy_rows(
        "TRV", years,
        revenue=[31581 * M, 31981 * M, 34816 * M, 36884 * M, 41364 * M, 46423 * M, 48828 * M],
        pretax_income=[3138 * M, 3237 * M, 4458 * M, 3354 * M, 3371 * M, 6180 * M, 7796 * M],
        interest_expense=[344 * M, 339 * M, 340 * M, 351 * M, 376 * M, 392 * M, 425 * M],
        net_income=[2603 * M, 2678 * M, 3635 * M, 2822 * M, 2969 * M, 4961 * M, 6242 * M],
        eps_diluted=[9.92, 10.52, 14.49, 11.77, 12.79, 21.47, 27.43],
        equity=[25943 * M, 29201 * M, 28887 * M, 21560 * M, 24921 * M, 27864 * M, 32894 * M],
        aoci=[640 * M, 2502 * M, 1193 * M, -6445 * M, -4471 * M, -4967 * M, -2500 * M],
        shares_outstanding=[255.5 * M, 252.4 * M, 241.2 * M, 232.1 * M, 228.2 * M, 226.6 * M, 217.5 * M],
        shares_diluted_wavg=[262.3 * M, 254.6 * M, 250.8 * M, 239.7 * M, 232.2 * M, 231.1 * M, 227.6 * M],
        total_debt=[6558 * M, 6550 * M, 7290 * M, 7292 * M, 8031 * M, 8033 * M, 9267 * M],
        total_assets=[110122 * M, 116764 * M, 120466 * M, 115717 * M, 125978 * M, 133189 * M, 143708 * M],
        intangibles=[4291 * M, 4293 * M, 4314 * M, 4239 * M, 4253 * M, 4593 * M, 4402 * M],
        cash=[494 * M, 721 * M, 761 * M, 799 * M, 650 * M, 699 * M, 842 * M],
        dividends_paid=[844 * M, 861 * M, 869 * M, 875 * M, 908 * M, 951 * M, 979 * M],
        operating_cash_flow=[6.0e9, 7.0e9, 7.4e9, 7.0e9, 7.3e9, 9.0e9, 10.0e9],
    )


def chubb_rows():
    return fy_rows(
        "CB", [2024, 2025],
        revenue=[55753 * M, 59402 * M], pretax_income=[11455 * M, 13044 * M],
        interest_expense=[741 * M, 764 * M], net_income=[9272 * M, 10310 * M], eps_diluted=[22.70, 25.68],
        equity=[64021 * M, 73757 * M], aoci=[-8644 * M, -4975 * M],
        shares_outstanding=[400.703663 * M, 391.101227 * M], shares_diluted_wavg=[408.5 * M, 401.5 * M],
        total_debt=[15179 * M, 17227 * M], total_assets=[246548 * M, 272327 * M],
        intangibles=[25956 * M, 26448 * M], dividends_paid=[1436 * M, 1505 * M], dps=[3.59, 3.82],
    )


def _tables(rows, mcap=None):
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sector="Financial Services",
                              industry="Insurance - Property & Casualty", sic_code=6331) for t in tickers])
    mkt = pd.DataFrame([dict(ticker=t, date=AS_OF, market_cap=(mcap or {}).get(t)) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(mkt, "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


@pytest.fixture(scope="module")
def ins():
    cfg = load_config()
    t = _tables(travelers_rows() + chubb_rows(), mcap={"TRV": 65e9, "CB": 115e9})
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    return cfg, t, mv, ms, prof


# ---- ROE ---------------------------------------------------------------------------------------
def test_insurer_roe_uses_average_common_equity(ins):
    cfg, t, v, s, prof = ins
    assert prof["TRV"] == "insurance" and prof["CB"] == "insurance"
    assert v.at["TRV", "roe_avg_common_pct"] == pytest.approx(6242 / ((32894 + 27864) / 2) * 100)
    assert v.at["TRV", "roe_avg_common_pct"] == pytest.approx(21.0, abs=0.6)   # reported (quarterly averages)
    assert s.at["TRV", "roe_pct"] == "na"                                      # ending-equity ROE not used


# ---- book value ----------------------------------------------------------------------------------
def test_book_value_per_share_matches_reported(ins):
    cfg, t, v, s, prof = ins
    assert v.at["TRV", "bvps"] == pytest.approx(151.21, abs=0.05)
    assert v.at["TRV", "bvps_growth_1y_pct"] == pytest.approx((151.21 / 122.97 - 1) * 100, abs=0.1)
    assert v.at["CB", "bvps"] == pytest.approx(188.59, abs=0.01)
    assert v.at["CB", "bvps_growth_1y_pct"] == pytest.approx(18.0, abs=0.05)          # reported +18.0%
    assert v.at["CB", "bvps_ex_aoci_growth_1y_pct"] == pytest.approx(11.0, abs=0.05)  # reported +11.0%


def test_book_value_growth_ex_aoci_removes_rate_swings(ins):
    cfg, t, v, s, prof = ins
    # 2022 bond losses pushed TRV's reported BVPS down to $92.89, so the 3-yr reported CAGR (2022 base)
    # is inflated by the rebound; ex-AOCI growth reflects retained earnings instead
    bv22 = 21560 / 232.1
    assert v.at["TRV", "bvps_cagr_3y_pct"] == pytest.approx(((151.24 / bv22) ** (1 / 3) - 1) * 100, abs=0.05)
    ex25, ex22 = (32894 + 2500) / 217.5, (21560 + 6445) / 232.1
    assert v.at["TRV", "bvps_ex_aoci_cagr_3y_pct"] == pytest.approx(((ex25 / ex22) ** (1 / 3) - 1) * 100)
    assert v.at["TRV", "bvps_ex_aoci_cagr_3y_pct"] < v.at["TRV", "bvps_cagr_3y_pct"]


def test_price_to_book_uses_common_equity():
    rows = fy_rows("PFD", [2024, 2025], equity=[1000.0, 1100.0], preferred_equity=[100.0, 100.0],
                   intangibles=[200.0, 200.0], net_income=[90.0, 100.0], revenue=[500.0, 520.0],
                   total_assets=[9000.0, 9500.0])
    v, _ = compute_metrics(_tables(rows, mcap={"PFD": 1500.0}), AS_OF)
    assert v.at["PFD", "pb_x"] == pytest.approx(1500 / 1000)
    assert v.at["PFD", "ptbv_x"] == pytest.approx(1500 / 800)


# ---- leverage -------------------------------------------------------------------------------------
def test_debt_to_capital_matches_reported(ins):
    cfg, t, v, s, prof = ins
    assert v.at["TRV", "debt_to_capital_pct"] == pytest.approx(22.0, abs=0.05)
    assert v.at["TRV", "interest_coverage_x"] == pytest.approx((7796 + 425) / 425)   # holding-company debt service


def test_sec_debt_combined_tag_vs_parts():
    from screener.providers.sec import parse_companyfacts

    def inst(v):
        return {"units": {"USD": [{"end": "2025-12-31", "val": v, "filed": "2026-02-12", "form": "10-K",
                                   "fy": 2025, "fp": "FY"}]}}

    def dur(v):
        return {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": v, "filed": "2026-02-12",
                                   "form": "10-K", "fy": 2025, "fp": "FY"}]}}

    # Travelers: only commercial paper + an all-in combined total are tagged
    trv = {"facts": {"us-gaap": {"Revenues": dur(48828.0), "NetIncomeLoss": dur(6288.0),
                                 "CommercialPaper": inst(100.0),
                                 "DebtLongtermAndShorttermCombinedAmount": inst(9267.0)}}}
    assert parse_companyfacts(trv, "TRV").iloc[0]["total_debt"] == 9267.0
    # Honeywell: the "combined" tag leaves out $5.9B of commercial paper -> sum of parts is larger
    hon = {"facts": {"us-gaap": {"Revenues": dur(40000.0), "NetIncomeLoss": dur(5000.0),
                                 "LongTermDebt": inst(29046.0), "ShortTermBorrowings": inst(5893.0),
                                 "DebtLongtermAndShorttermCombinedAmount": inst(28687.0)}}}
    assert parse_companyfacts(hon, "HON").iloc[0]["total_debt"] == pytest.approx(29046.0 + 5893.0)


def test_sec_parser_reads_aoci():
    from screener.providers.sec import parse_companyfacts
    from tests.test_providers import companyfacts_fixture

    f = companyfacts_fixture()
    f["facts"]["us-gaap"]["AccumulatedOtherComprehensiveIncomeLossNetOfTax"] = {"units": {"USD": [
        {"end": "2024-12-31", "val": -25.0, "filed": "2025-02-15", "form": "10-K", "fy": 2024, "fp": "FY"}]}}
    assert parse_companyfacts(f, "ZZZ").set_index("period_end").loc["2024-12-31", "aoci"] == -25.0


# ---- not-applicable metrics and substitutes ----------------------------------------------------------
INS_NA = ["roe_pct", "roic_pct", "gross_margin_pct", "ebitda_margin_pct", "current_ratio_x", "cash_musd",
          "net_debt_musd", "debt_to_equity_x", "net_debt_to_ebitda_x", "enterprise_value_musd", "ev_ebitda_x",
          "ev_sales_x", "fcf_margin_pct", "fcf_yield_pct", "p_fcf_x", "fcf_payout_3y_pct",
          "fcf_conversion_3y_pct", "fcf_cagr_3y_pct", "fcf_positive_share_5y_pct"]


def test_insurer_not_applicable_metrics(ins):
    cfg, t, v, s, prof = ins
    for m in INS_NA:
        assert s.at["TRV", m] == "na", m
    for m in ["roe_avg_common_pct", "bvps", "bvps_ex_aoci_cagr_3y_pct", "debt_to_capital_pct", "pb_x",
              "ptbv_x", "payout_ratio_pct", "interest_coverage_x", "total_debt_musd",
              "earnings_positive_share_5y_pct", "operating_margin_stdev_5y_pp", "eps_cagr_3y_pct"]:
        assert s.at["TRV", m] == "ok", m


def test_insurer_pillars():
    cfg = load_config()
    for screen in ("growth", "income"):
        specs = pillar_specs_for_profile(cfg.screens[screen], cfg.profiles["insurance"])
        used = {m["name"] for items in specs.values() for m in items}
        assert not used & set(INS_NA), (screen, used & set(INS_NA))
        assert {"roe_avg_common_pct", "debt_to_capital_pct", "bvps_ex_aoci_cagr_5y_pct"} <= used
    g = {m["name"] for items in pillar_specs_for_profile(cfg.screens["growth"], cfg.profiles["insurance"]).values()
         for m in items}
    assert {"bvps_ex_aoci_cagr_3y_pct", "pb_x", "net_margin_pct"} <= g
    i = {m["name"] for items in pillar_specs_for_profile(cfg.screens["income"], cfg.profiles["insurance"]).values()
         for m in items}
    assert {"pb_x", "ptbv_x", "payout_ratio_pct"} <= i


def test_insurance_brokers_are_not_balance_sheet_insurers():
    cfg = load_config()
    secs = pd.DataFrame([
        dict(ticker="BRK1", name="Broker A", sector="Financial Services", industry="Insurance Brokers", sic_code=6411),
        dict(ticker="BRK2", name="Broker B", sector=None, industry="Insurance Agents, Brokers & Service", sic_code=6411),
        dict(ticker="CAR", name="Carrier", sector="Financial Services", industry="Insurance - Property & Casualty",
             sic_code=6331),
        dict(ticker="LIF", name="Life Co", sector="Financial Services", industry="Insurance - Life", sic_code=6311),
    ])
    p = assign_profiles(schema.conform(secs, "securities"), cfg.profiles)
    assert p["BRK1"] == "default" and p["BRK2"] == "default"
    assert p["CAR"] == "insurance" and p["LIF"] == "insurance"


# ---- flags -------------------------------------------------------------------------------------------
def test_insurer_leverage_flag_uses_debt_to_capital(ins):
    cfg = load_config()
    lev = [dict(r, ticker="LEV") for r in travelers_rows()]
    for r in lev:
        if r["period_end"].year == 2025:
            r["total_debt"] = 32894 * M            # 50% debt / capital
    t = _tables(travelers_rows() + lev)
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["LEV", "TRV"]))
    hl = fl[(fl["ticker"] == "LEV") & (fl["flag"] == "high_leverage")]
    assert len(hl) == 1 and hl.iloc[0]["default_severity"] == "severe" and "debt/capital 50%" in hl.iloc[0]["detail"]
    trv = set(fl[fl["ticker"] == "TRV"]["flag"])
    assert not trv & {"high_leverage", "negative_fcf", "poor_fcf_conversion", "weak_dividend_coverage"}
