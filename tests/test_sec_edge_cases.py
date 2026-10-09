"""Remaining SEC edge cases: FCF starting-year distortion, basic-EPS fallback, current debt
maturities, FX in the EBIT fallback, dividend streaks with calculated / merger-year / mis-tagged DPS,
and non-payer dividend growth shown as n/a.

Figures are SEC company-facts values ($ millions) unless noted; reported comparisons in comments.
"""
import numpy as np
import pandas as pd
import pytest

from screener import schema
from screener.cli import main
from screener.config import load_config
from screener.metrics.engine import compute_metrics
from screener.providers.sec import parse_companyfacts
from screener.scoring import pillar_specs_for_profile
from tests.conftest import fy_rows
from tests.test_cyclical_profile import _facts
from tests.test_growth_profile import nvda_rows

M = 1e6
AS_OF = pd.Timestamp("2026-06-30")


def _tables(rows, sic=7372):
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sic_code=sic) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(pd.DataFrame(), "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


# ---- 1) free-cash-flow growth ----------------------------------------------------------------------
def test_fcf_growth_from_a_trough_year_is_not_scored_as_194_pct():
    """Nvidia FY2023 FCF $3,808M (CFO 5,641 - capex 1,833) vs FY2026 $96,676M: the plain 3-yr FCF
    CAGR is 194%/yr. Its base is 3.9% of the latest value and far below the trend of the other years."""
    v, s = compute_metrics(_tables(nvda_rows()), AS_OF)
    assert v.at["NVDA", "fcf_cagr_3y_pct"] == pytest.approx(((96676 / 3808) ** (1 / 3) - 1) * 100, rel=1e-3)
    assert s.at["NVDA", "fcf_cagr_3y_pct"] == "nm"                     # value kept, reference only
    assert s.at["NVDA", "fcf_trend_growth_pct"] == "ok"
    assert v.at["NVDA", "fcf_growth_scored_pct"] == pytest.approx(v.at["NVDA", "fcf_trend_growth_pct"])
    assert v.at["NVDA", "fcf_growth_scored_pct"] < v.at["NVDA", "fcf_cagr_3y_pct"] / 2


def test_clean_fcf_growth_is_blended():
    rows = fy_rows("FCF", list(range(2018, 2026)), revenue=[1e10 * 1.1 ** i for i in range(8)],
                   operating_cash_flow=[3e9 * 1.1 ** i for i in range(8)], capex=[1e9 * 1.1 ** i for i in range(8)],
                   net_income=[1e9] * 8, shares_diluted_wavg=[1e9] * 8)
    v, s = compute_metrics(_tables(rows), AS_OF)
    assert s.at["FCF", "fcf_cagr_3y_pct"] == "ok"
    assert v.at["FCF", "fcf_growth_scored_pct"] == pytest.approx(10.0, abs=1e-6)


def test_fcf_growth_scoring_by_profile():
    cfg = load_config()
    names = lambda prof: {m["name"]: m for items in pillar_specs_for_profile(cfg.screens["growth"],  # noqa: E731
                                                                              cfg.profiles[prof]).values()
                          for m in items}
    assert "fcf_growth_scored_pct" in names("default") and "fcf_cagr_3y_pct" not in names("default")
    assert names("insurance")["bvps_ex_aoci_cagr_3y_pct"]["substituted_for"] == "fcf_growth_scored_pct"
    assert "fcf_growth_scored_pct" not in names("cyclical")        # phase-dependent, like EPS trends
    assert "fcf_growth_scored_pct" not in names("utility") and "fcf_growth_scored_pct" not in names("bank")


# ---- 2) basic EPS / shares fill years with no diluted figure ---------------------------------------
def test_basic_eps_fills_missing_diluted():
    """Lennar FY2025 tagged only basic EPS ($7.98) and basic weighted shares (257.7M)."""
    f = _facts(duration={"Revenues": 34e9, "EarningsPerShareBasic": 7.98})
    f["facts"]["us-gaap"]["WeightedAverageNumberOfSharesOutstandingBasic"] = {"units": {"shares": [
        {"end": "2025-12-31", "start": "2025-01-01", "val": 257746000, "form": "10-K", "accn": "x",
         "filed": "2026-01-20", "fy": 2025, "fp": "FY"}]}}
    row = parse_companyfacts(f, "LEN").iloc[-1]
    assert row["eps_diluted"] == pytest.approx(7.98) and row["shares_diluted_wavg"] == 257746000
    # a diluted figure, when present, always wins
    f2 = _facts(duration={"Revenues": 1e10, "EarningsPerShareBasic": 2.00, "EarningsPerShareDiluted": 1.95})
    assert parse_companyfacts(f2, "X").iloc[-1]["eps_diluted"] == pytest.approx(1.95)


# ---- 3) current maturities next to short-term borrowings -------------------------------------------
def test_current_maturities_added_when_no_current_debt_total():
    """Nucor: long-term debt & finance leases due after one year $6,909M + short-term debt $122M +
    current portion of long-term debt & finance leases $90M = $7,121M (release balance sheet)."""
    f = _facts(instant={"LongTermDebtAndCapitalLeaseObligations": 6909 * M, "ShortTermBorrowings": 122 * M,
                        "LongTermDebtAndCapitalLeaseObligationsCurrent": 90 * M, "LongTermDebtCurrent": 66 * M},
               duration={"Revenues": 32494 * M})
    assert parse_companyfacts(f, "NUE").iloc[-1]["total_debt"] == pytest.approx(7121 * M)


def test_short_term_line_already_includes_maturities_when_a_current_total_is_tagged():
    """Chevron tags DebtCurrent ($10,918M gross, before $9.9B is reclassified to long-term) next to its
    balance-sheet short-term debt ($977M, which includes current maturities): total = $40,758M."""
    f = _facts(instant={"LongTermDebtAndCapitalLeaseObligations": 39781 * M, "ShortTermBorrowings": 977 * M,
                        "DebtCurrent": 10918 * M, "LongTermDebtCurrent": 2345 * M, "CommercialPaper": 4642 * M},
               duration={"Revenues": 184432 * M})
    assert parse_companyfacts(f, "CVX").iloc[-1]["total_debt"] == pytest.approx(40758 * M)


# ---- 4) FX remeasurement is not operating income ----------------------------------------------------
def test_ebit_fallback_excludes_fx_gains_and_losses():
    """Public Storage tags no operating-income line; its 2025 pretax income ($1,789.8M) includes a
    $216M FX loss on euro-denominated notes, which used to cut its EBIT margin by 4.5 points."""
    rows = fy_rows("PSA", [2024, 2025], revenue=[4695.6 * M, 4824.1 * M], pretax_income=[2089.1 * M, 1789.8 * M],
                   interest_expense=[287.4 * M, 304.5 * M], real_estate_impairment=[0.0, 4.3 * M],
                   gain_on_sale_real_estate=[1.5 * M, 1.1 * M], fx_gain_loss=[0.0, -216.0 * M],
                   net_income=[1872.7 * M, 1585.6 * M], shares_diluted_wavg=[176 * M, 175.9 * M])
    v, _ = compute_metrics(_tables(rows, sic=6798), AS_OF)
    expected = (1789.8 + 304.5 + 4.3 - 1.1 + 216.0) / 4824.1 * 100
    assert v.at["PSA", "operating_margin_pct"] == pytest.approx(expected, rel=1e-4)       # 48.0%, not 43.5%


def test_operating_income_line_is_never_overridden_by_the_fallback():
    rows = fy_rows("OPI", [2024, 2025], revenue=[1e10, 1e10], operating_income=[2e9, 2e9],
                   pretax_income=[1.5e9, 1.5e9], fx_gain_loss=[-5e8, -5e8], net_income=[1e9, 1e9],
                   shares_diluted_wavg=[1e9, 1e9])
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.at["OPI", "operating_margin_pct"] == pytest.approx(20.0)


# ---- 5) dividend streaks -----------------------------------------------------------------------------
def _div_company(ticker, dps, paid, shares, years=None):
    years = years or list(range(2025 - len(paid) + 1, 2026))
    return fy_rows(ticker, years, dps=dps, dividends_paid=[p * M for p in paid],
                   shares_diluted_wavg=[x * M for x in shares], revenue=[1e10] * len(years),
                   net_income=[1e9] * len(years))


def test_calculated_dps_noise_does_not_break_a_streak():
    """Chevron 2015-18 has no DPS tag; dividends paid / shares gives 4.262, 4.288, 4.285, 4.442 - the
    0.07% dip in 2017 is share-count timing, not a cut (Chevron raised its dividend every year).
    The 2015-17 paid / share pairs below are reconstructed to reproduce those calculated values."""
    paid = [8015, 8032, 8132, 8502, 8959, 9651, 10179, 10968, 11336, 11801, 12751]
    shares = [1880.6, 1873.2, 1897.8, 1914.0, 1895.0, 1870.0, 1920.0, 1940.0, 1880.0, 1817.0, 1856.0]
    dps = [np.nan] * 4 + [4.76, 5.16, 5.31, 5.68, 6.04, 6.52, 6.84]
    v, _ = compute_metrics(_tables(_div_company("CVX", dps, paid, shares)), AS_OF)
    assert v.at["CVX", "dividend_growth_streak_years"] == 10
    assert v.at["CVX", "dividend_cut_5y"] == 0


def test_merger_year_calculated_dps_is_bridged():
    """Chubb 2016 (ACE-Chubb merger, shares +42%): dividends paid / shares = 2.52 vs ~2.72 actually
    paid. The unreliable year is replaced by the midpoint of its neighbours, so no false dip."""
    paid = [862, 1173, 1308]
    shares = [328.8, 465.9, 471.2]
    v, _ = compute_metrics(_tables(_div_company("CB", [2.66, np.nan, np.nan], paid, shares,
                                                years=[2015, 2016, 2017])), AS_OF)
    assert v.at["CB", "dividend_growth_streak_years"] == 2


def test_outlier_reported_dps_is_replaced():
    """P&G FY2017: the DPS tag holds 3.00 while dividends paid / shares is 2.64 (reported 2.66 and
    2.79 in the years either side agree with paid / shares within 2%)."""
    dps = [2.66, 3.00, 2.786, 2.897]
    paid = [2.614 * 2900, 2.64 * 2700, 2.752 * 2650, 2.953 * 2600]
    shares = [2900, 2700, 2650, 2600]
    v, _ = compute_metrics(_tables(_div_company("PG", dps, paid, shares, years=[2016, 2017, 2018, 2019])), AS_OF)
    assert v.at["PG", "dividend_growth_streak_years"] == 3
    assert v.at["PG", "dividend_cut_5y"] == 0


def test_real_cut_still_detected_with_calculated_dps():
    paid = [100, 105, 110, 60]
    v, _ = compute_metrics(_tables(_div_company("CUT", [np.nan] * 4, paid, [100] * 4)), AS_OF)
    assert v.at["CUT", "dividend_cut_5y"] == 1 and v.at["CUT", "dividend_growth_streak_years"] == 0


# ---- 6) non-payers ------------------------------------------------------------------------------------
def test_non_payer_dividend_growth_is_na(project, capsys):
    rows = _div_company("NOD", [0.0] * 6, [0.0] * 6, [100] * 6)
    v, s = compute_metrics(_tables(rows), AS_OF)
    assert s.at["NOD", "dps_cagr_3y_pct"] == "na" and s.at["NOD", "dps_cagr_5y_pct"] == "na"
    assert main(["demo"]) == 0
    capsys.readouterr()
    assert main(["company", "SYCMA"]) == 0                       # synthetic compounder pays no dividend
    line = next(x for x in capsys.readouterr().out.splitlines() if x.strip().startswith("Dividend/share CAGR 3y"))
    assert "[n/a: no dividend]" in line


def test_special_dividend_survives_the_outlier_rule():
    """Public Storage 2022: $21.15 declared (incl. a $13.15 special); dividends paid are untagged from
    2023, so filled DPS x shares values are not evidence that the 2022 tag is an outlier."""
    dps = [8.0, 8.0, 21.15, 12.0, 12.0, 12.0]
    paid = [8.0 * 175, 8.0 * 175, 13.047 * 175, np.nan, np.nan, np.nan]
    v, s = compute_metrics(_tables(_div_company("PSA", dps, paid, [175] * 6)), AS_OF)
    assert v.at["PSA", "special_dividends_detected"] == 1
    assert v.at["PSA", "dividend_cut_5y"] == 0
    assert s.at["PSA", "dps_cagr_3y_pct"] != "ok"           # base year is the special-dividend year


def test_dps_tag_consistent_with_dividends_paid_is_chosen():
    """Prologis FY2025 tags $0.03 as 'declared' but $4.04 paid ($3.8B on ~0.93B shares)."""
    import numpy as np
    from screener.providers.sec import _pick_dps

    assert _pick_dps(0.03, 4.04, 3.76e9, 0.93e9) == 4.04
    assert _pick_dps(5.80, 5.55, 15.6e9, 2.8e9) == 5.80          # both plausible: declared kept
    assert np.isnan(_pick_dps(0.03, np.nan, 3.76e9, 0.93e9))      # lone implausible figure dropped
    assert _pick_dps(1.0, np.nan, np.nan, np.nan) == 1.0          # nothing to check against


# ---- revenue: total line beats the partial contract-revenue tag --------------------------------
@pytest.mark.parametrize("case, duration, expected", [
    # Target Hospitality FY2025: contract revenue $274.7M (services only) vs total revenue $320.6M
    ("TH", {"RevenueFromContractWithCustomerExcludingAssessedTax": 274.7 * M, "Revenues": 320.635 * M}, 320.635 * M),
    # Evergy FY2025: contract revenue $5,705.1M excludes alternative-revenue programs; total $5,961.6M
    ("EVRG", {"RevenueFromContractWithCustomerExcludingAssessedTax": 5705.1 * M, "Revenues": 5961.6 * M}, 5961.6 * M),
    # Piper Sandler FY2025: contract revenue $1,834.1M; gross Revenues $1,904.2M; NET revenues $1,899.4M (headline)
    ("PIPR", {"RevenueFromContractWithCustomerExcludingAssessedTax": 1834.1 * M, "Revenues": 1904.2 * M,
              "RevenuesNetOfInterestExpense": 1899.4 * M}, 1899.4 * M),
    # only the contract tag -> it is used
    ("ONLY", {"RevenueFromContractWithCustomerExcludingAssessedTax": 1000 * M}, 1000 * M),
    # a generic 'Revenues' fact far below contract revenue is not the consolidated total
    ("SUB", {"RevenueFromContractWithCustomerExcludingAssessedTax": 1000 * M, "Revenues": 200 * M}, 1000 * M),
])
def test_total_revenue_preferred_over_partial_contract_revenue(case, duration, expected):
    f = _facts(duration=dict(duration, NetIncomeLoss=10 * M))
    assert parse_companyfacts(f, case).iloc[-1]["revenue"] == pytest.approx(expected)
