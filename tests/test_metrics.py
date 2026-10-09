import numpy as np
import pandas as pd
import pytest

from screener import schema
from screener.metrics.engine import compute_metrics
from tests.conftest import fy_rows

AS_OF = pd.Timestamp("2026-06-30")
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def _tables(fin_rows, mcap=10e9, extra_secs=()):
    secs = [dict(ticker=r, name=r, exchange="NYSE", security_type="common", share_class="single",
                 is_delisted=False, sector="Industrials", industry="Machinery", sic_code=3560)
            for r in sorted({row["ticker"] for row in fin_rows}) + list(extra_secs)]
    mkt = [dict(ticker=t, date=AS_OF, market_cap=mcap) for t in {s["ticker"] for s in secs}]
    return {"securities": schema.conform(pd.DataFrame(secs), "securities"),
            "financials": schema.conform(pd.DataFrame(fin_rows), "financials"),
            "market": schema.conform(pd.DataFrame(mkt), "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


def base_company(ticker="AAA", **over):
    s = dict(
        revenue=[100, 110, 121, 133.1, 146.41, 161.051],          # exactly +10%/yr
        gross_profit=[40, 44, 48.4, 53.24, 58.564, 64.4204],
        operating_income=[10, 12, 14, 16, 18, 24.1577],
        d_and_a=[5] * 6, interest_expense=[2] * 6,
        pretax_income=[8, 10, 12, 14, 16, 22.1577], income_tax=[1.68, 2.1, 2.52, 2.94, 3.36, 4.6531],
        net_income=[6.32, 7.9, 9.48, 11.06, 12.64, 17.5046],
        eps_diluted=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0], shares_diluted_wavg=[10, 10, 10, 10, 10, 9.5],
        shares_outstanding=[10] * 5 + [9.5],
        cash=[20] * 6, total_debt=[40] * 6, equity=[60] * 6, total_assets=[150] * 6,
        current_assets=[50] * 6, current_liabilities=[25] * 6, intangibles=[10] * 6,
        operating_cash_flow=[12] * 5 + [20], capex=[4] * 6, sbc=[1] * 6,
        dividends_paid=[2, 2.2, 2.4, 2.6, 2.8, 3.0], dps=[0.2, 0.22, 0.24, 0.26, 0.28, 0.30],
        net_common_issuance=[0] * 5 + [-1.0],
    )
    s.update(over)
    return fy_rows(ticker, YEARS, **s)


def test_growth_margins_and_returns():
    v, st = compute_metrics(_tables(base_company(), mcap=350.0), AS_OF)
    r = v.loc["AAA"]
    assert r["revenue_cagr_1y_pct"] == pytest.approx(10.0)
    assert r["revenue_cagr_3y_pct"] == pytest.approx(10.0)
    assert r["revenue_cagr_5y_pct"] == pytest.approx(10.0)
    assert r["eps_cagr_5y_pct"] == pytest.approx((2.0 / 1.0) ** (1 / 5) * 100 - 100)
    assert r["gross_margin_pct"] == pytest.approx(40.0)
    assert r["operating_margin_pct"] == pytest.approx(24.1577 / 161.051 * 100)
    assert r["fcf_margin_pct"] == pytest.approx(16 / 161.051 * 100)
    # ROIC = EBIT*(1-t) / avg(debt + equity - cash);  t = 4.6531/22.1577 = 21%
    assert r["roic_pct"] == pytest.approx(24.1577 * (1 - 4.6531 / 22.1577) / 80 * 100)
    assert r["net_debt_to_ebitda_x"] == pytest.approx(20 / (24.1577 + 5))
    assert r["current_ratio_x"] == pytest.approx(2.0)
    assert r["interest_coverage_x"] == pytest.approx(24.1577 / 2)
    assert r["pe_x"] == pytest.approx(350 / 17.5046)
    assert r["dividend_yield_pct"] == pytest.approx(3.0 / 350 * 100)
    assert r["dividend_paying_years"] == 6
    assert r["dividend_growth_streak_years"] == 5
    assert r["dividend_cut_5y"] == 0
    assert r["share_change_1y_pct"] == pytest.approx(-5.0)
    assert r["net_buyback_yield_pct"] == pytest.approx(1 / 350 * 100)
    assert r["operating_margin_chg_3y_pp"] == pytest.approx(24.1577 / 161.051 * 100 - 14 / 121 * 100)
    assert st.loc["AAA", "revenue_cagr_5y_pct"] == "ok"


def test_not_meaningful_and_short_history():
    rows = base_company("NEG", eps_diluted=[-1.0, 0.5, 0.8, 1.0, 1.2, 1.5],
                        net_income=[-5, 3, 4, 5, 6, -2.0], operating_income=[-1, 1, 2, 3, 4, -3.0])
    rows += base_company("NEW")[-2:]                       # only 2 fiscal years
    v, st = compute_metrics(_tables(rows), AS_OF)
    assert st.loc["NEG", "eps_cagr_5y_pct"] == "nm"         # negative base
    assert np.isnan(v.loc["NEG", "eps_cagr_5y_pct"])
    assert st.loc["NEG", "pe_x"] == "nm"                    # losses
    assert st.loc["NEG", "payout_ratio_pct"] == "nm"        # paying dividends while losing money
    assert st.loc["NEG", "net_debt_to_ebitda_x"] == "ok"    # EBITDA = -3 + 5 = 2 > 0
    assert st.loc["NEW", "revenue_cagr_3y_pct"] == "short"
    assert st.loc["NEW", "revenue_cagr_1y_pct"] == "ok"


def test_negative_ebitda_and_negative_equity_are_nm():
    rows = base_company("BAD", operating_income=[1] * 5 + [-20.0], equity=[10] * 5 + [-5.0])
    v, st = compute_metrics(_tables(rows), AS_OF)
    assert st.loc["BAD", "net_debt_to_ebitda_x"] == "nm"
    assert st.loc["BAD", "roe_pct"] == "nm"
    assert st.loc["BAD", "pb_x"] == "nm"


def test_dividend_cut_detected_and_no_dividend_is_zero_payout():
    rows = base_company("CUT", dps=[0.2, 0.22, 0.24, 0.26, 0.28, 0.14], dividends_paid=[2, 2.2, 2.4, 2.6, 2.8, 1.4])
    rows += base_company("NOD", dps=[0.0] * 6, dividends_paid=[0.0] * 6)
    v, st = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["CUT", "dividend_cut_5y"] == 1
    assert v.loc["CUT", "dividend_growth_streak_years"] == 0
    assert v.loc["NOD", "payout_ratio_pct"] == 0
    assert v.loc["NOD", "dividend_paying_years"] == 0
    assert st.loc["NOD", "dps_cagr_3y_pct"] == "na"           # a non-payer has no dividend growth (was nm)


def test_missing_inputs_are_missing_not_zero():
    rows = base_company("GAP", operating_cash_flow=[np.nan] * 6, capex=[np.nan] * 6)
    v, st = compute_metrics(_tables(rows), AS_OF)
    assert st.loc["GAP", "fcf_margin_pct"] == "missing"
    assert st.loc["GAP", "fcf_cagr_3y_pct"] == "missing"
    assert np.isnan(v.loc["GAP", "fcf_yield_pct"])


def test_gap_year_does_not_shift_series():
    rows = [r for r in base_company("HOLE") if r["period_end"].year != 2023]
    v, st = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["HOLE", "revenue_cagr_5y_pct"] == pytest.approx(10.0)   # FY2025 vs FY2020 still aligned
    assert st.loc["HOLE", "revenue_cagr_1y_pct"] == "ok"
