"""REIT handling, validated against Realty Income (O).

Inputs are Realty Income's SEC / press-release figures (public): FY2025 and FY2024 FFO
reconciliation items (net income to common, D&A, impairments, gains on sale), diluted FFO share
counts, dividends paid, total debt and interest expense.
Reported (Q4 2025 release): FFO/share $4.25 (2025) and $4.01 (2024); FFO $3,860.3M;
dividends paid $3.2170/share; net debt / annualized pro forma adjusted EBITDAre 5.4x.
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


def realty_income_rows():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    return fy_rows(
        "O", years,
        revenue=[1647.087 * M, 2080.463 * M, 3343.681 * M, 4078.993 * M, 5271.142 * M, 5749.377 * M],
        net_income=[395.486 * M, 359.456 * M, 869.408 * M, 872.309 * M, 847.893 * M, 1058.590 * M],
        d_and_a=[677.038 * M, 897.835 * M, 1670.389 * M, 1895.177 * M, 2395.644 * M, 2524.200 * M],
        real_estate_impairment=[np.nan, np.nan, np.nan, np.nan, 319.032 * M, 434.497 * M],
        gain_on_sale_real_estate=[np.nan, np.nan, np.nan, np.nan, 117.275 * M, 177.640 * M],
        pretax_income=[411.199 * M, 392.404 * M, 917.599 * M, 928.935 * M, 902.0 * M, 1155.129 * M],
        interest_expense=[309.336 * M, 323.644 * M, 465.223 * M, 730.423 * M, 970.009 * M, 1134.9 * M],
        shares_diluted_wavg=[345.415 * M, 414.770 * M, 612.181 * M, 693.024 * M, 865.842 * M, 911.015 * M],
        shares_outstanding=[361.3 * M, 591.3 * M, 660.3 * M, 752.5 * M, 891.5 * M, 934.0 * M],
        dividends_paid=[964.167 * M, 1169.026 * M, 1813.431 * M, 2111.793 * M, 2691.719 * M, 2920.895 * M],
        dps=[2.794, 2.833, 2.967, 3.051, 3.1255, 3.2170],
        total_debt=[np.nan, np.nan, np.nan, np.nan, np.nan, 29116.111 * M],
        cash=[np.nan, np.nan, np.nan, np.nan, 444.962 * M, 434.842 * M],
        total_assets=[20.74e9, 43.14e9, 49.67e9, 57.78e9, 68.84e9, 72.80e9],
        equity=[12e9, 25e9, 30e9, 33e9, 38e9, 39e9],
        operating_cash_flow=[1115.5 * M, 1322.2 * M, 2563.9 * M, 2958.8 * M, 3573.3 * M, 3994.8 * M],
        capex=[8.7 * M, 19.1 * M, 95.5 * M, 68.7 * M, 121.4 * M, 131.8 * M],
    )


def _tables(rows, sic=6798, industry="REIT - Retail", sector="Real Estate", mcap=55e9):
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sector=sector, industry=industry, sic_code=sic) for t in tickers])
    mkt = pd.DataFrame([dict(ticker=t, date=AS_OF, market_cap=mcap) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(mkt, "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


@pytest.fixture(scope="module")
def realty():
    cfg = load_config()
    t = _tables(realty_income_rows())
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    return cfg, t, mv, ms, prof


# ---- FFO and P/FFO ----------------------------------------------------------------------
def test_ffo_follows_nareit_and_matches_reported(realty):
    cfg, t, v, s, prof = realty
    assert prof["O"] == "reit"
    ffo_2025 = (1058.590 + 2524.200 + 434.497 - 177.640) * M
    assert v.at["O", "ffo_per_share"] == pytest.approx(ffo_2025 / (911.015 * M))
    assert v.at["O", "ffo_per_share"] == pytest.approx(4.25, rel=0.01)        # reported FFO/share
    assert ffo_2025 == pytest.approx(3860.323 * M, rel=0.01)                     # reported FFO
    assert v.at["O", "ffo_nareit_adjusted"] == 1
    assert v.at["O", "p_ffo_x"] == pytest.approx(55e9 / ffo_2025)


def test_ffo_payout_matches_reported_basis(realty):
    cfg, t, v, s, prof = realty
    ffo_2025 = (1058.590 + 2524.200 + 434.497 - 177.640) * M
    assert v.at["O", "ffo_payout_pct"] == pytest.approx(2920.895 * M / ffo_2025 * 100)
    assert v.at["O", "ffo_payout_pct"] == pytest.approx(3.2170 / 4.25 * 100, abs=1.0)  # reported $3.217 / $4.25


def test_ffo_without_impairment_data_falls_back_and_is_flagged():
    rows = realty_income_rows()
    for r in rows:
        r["real_estate_impairment"] = np.nan
        r["gain_on_sale_real_estate"] = np.nan
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.at["O", "ffo_per_share"] == pytest.approx((1058.590 + 2524.200) / 911.015)
    assert v.at["O", "ffo_nareit_adjusted"] == 0


# ---- dividends -------------------------------------------------------------------------------
def test_reit_dividend_history(realty):
    cfg, t, v, s, prof = realty
    assert v.at["O", "dividend_growth_streak_years"] == 5        # raised every year in the 6-year window
    assert v.at["O", "dividend_cut_5y"] == 0
    assert v.at["O", "dps_cagr_5y_pct"] == pytest.approx(((3.2170 / 2.794) ** (1 / 5) - 1) * 100)


# ---- leverage ----------------------------------------------------------------------------------
def test_leverage_uses_ebitdare(realty):
    cfg, t, v, s, prof = realty
    # operating income is not tagged -> EBIT = pretax + interest
    ebitda = (1155.129 + 1134.9 + 2524.200) * M
    ebitdare = ebitda + (434.497 - 177.640) * M
    net_debt = (29116.111 - 434.842) * M
    assert v.at["O", "net_debt_to_ebitdare_x"] == pytest.approx(net_debt / ebitdare)
    assert v.at["O", "net_debt_to_ebitdare_x"] == pytest.approx(5.4, abs=0.5)     # reported 5.4x (pro forma)
    assert v.at["O", "ebitdare_interest_coverage_x"] == pytest.approx(ebitdare / (1134.9 * M))
    assert s.at["O", "net_debt_to_ebitda_x"] == "na" and s.at["O", "interest_coverage_x"] == "na"


# ---- not-applicable metrics and substitutes ------------------------------------------------------
REIT_NA = ["eps_cagr_1y_pct", "eps_cagr_3y_pct", "eps_cagr_5y_pct", "pe_x", "peg_hist_x", "pe_vs_5y_median_pct",
           "payout_ratio_pct", "earnings_positive_share_5y_pct", "roe_pct", "roe_avg_common_pct",
           "fcf_conversion_3y_pct", "net_debt_to_ebitda_x", "interest_coverage_x", "gross_margin_pct",
           "current_ratio_x",
           # FCF: standardized REIT capex includes acquisitions/development -> FFO-based measures
           "fcf_margin_pct", "fcf_payout_3y_pct", "p_fcf_x", "fcf_yield_pct", "fcf_positive_share_5y_pct",
           "fcf_growth_scored_pct", "fcf_cagr_3y_pct"]


def test_reit_not_applicable_metrics(realty):
    cfg, t, v, s, prof = realty
    for m in REIT_NA:
        assert s.at["O", m] == "na", m
    for m in ["ffo_per_share", "p_ffo_x", "ffo_payout_pct", "net_debt_to_ebitdare_x", "revenue_cagr_3y_pct",
              "dividend_yield_pct", "ev_ebitda_x"]:
        assert s.at["O", m] == "ok", m


def test_reit_pillars_use_ffo_substitutes():
    cfg = load_config()
    for screen in ("growth", "income"):
        specs = pillar_specs_for_profile(cfg.screens[screen], cfg.profiles["reit"])
        used = {m["name"] for items in specs.values() for m in items}
        assert not used & set(REIT_NA), (screen, used & set(REIT_NA))
    g = {m["name"] for items in pillar_specs_for_profile(cfg.screens["growth"], cfg.profiles["reit"]).values()
         for m in items}
    assert {"ffo_ps_cagr_3y_pct", "net_debt_to_ebitdare_x", "ebitdare_interest_coverage_x"} <= g
    i = {m["name"] for items in pillar_specs_for_profile(cfg.screens["income"], cfg.profiles["reit"]).values()
         for m in items}
    # both screens score EPS growth through eps_growth_scored_pct, which REITs replace with FFO/share growth
    assert {"ffo_payout_pct", "p_ffo_x", "ffo_ps_cagr_3y_pct", "net_debt_to_ebitdare_x"} <= i


# ---- red flags -------------------------------------------------------------------------------------
def test_reit_flags_use_ffo_and_ebitdare():
    cfg = load_config()
    rows = realty_income_rows()
    for r in rows:                          # GAAP loss from a big impairment, but FFO stays positive
        if r["period_end"].year == 2025:
            r["net_income"] = -100 * M
            r["real_estate_impairment"] = 1600 * M
            r["pretax_income"] = -100 * M
    hl = [dict(r, ticker="HLV") for r in realty_income_rows()]
    for r in hl:                            # very high leverage
        if r["period_end"].year == 2025:
            r["total_debt"] = 60000 * M
    t = _tables(rows + hl)
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    uni = pd.Index(["HLV", "O"])
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles, uni)
    o_flags = set(fl[fl["ticker"] == "O"]["flag"])
    assert "negative_earnings" not in o_flags                # FFO positive despite the GAAP loss
    hlv = fl[(fl["ticker"] == "HLV") & (fl["flag"] == "high_leverage")]
    assert len(hlv) == 1 and "EBITDAre" in hlv.iloc[0]["detail"]


# ---- SEC parsing of REIT-style filings ------------------------------------------------------------
def test_sec_parser_reit_tags():
    from screener.providers.sec import parse_companyfacts

    def inst(v):
        return {"units": {"USD": [{"end": "2025-12-31", "val": v, "filed": "2026-02-25", "form": "10-K",
                                   "fy": 2025, "fp": "FY"}]}}

    def dur(v):
        return {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": v, "filed": "2026-02-25",
                                   "form": "10-K", "fy": 2025, "fp": "FY"}]}}

    facts = {"facts": {"us-gaap": {
        "Revenues": dur(5749.4), "NetIncomeLoss": dur(1058.6),
        "InterestExpenseOperating": dur(1134.9), "InterestPaidNet": dur(1072.5),
        "ImpairmentOfRealEstate": dur(471.3), "GainLossOnSaleOfProperties": dur(177.6),
        "NotesPayable": inst(25031.9), "LoansPayable": inst(1701.6), "CommercialPaper": inst(516.8),
        "SecuredDebt": inst(37.8),
    }}}
    r = parse_companyfacts(facts, "O").iloc[0]
    assert r["interest_expense"] == 1134.9                 # accrual expense, not cash interest paid
    assert r["total_debt"] == pytest.approx(25031.9 + 1701.6 + 37.8 + 516.8)
    assert r["real_estate_impairment"] == 471.3 and r["gain_on_sale_real_estate"] == 177.6


def test_ebit_fallback_when_operating_income_untagged():
    rows = fy_rows("HCO", [2024, 2025], revenue=[100.0, 110.0], pretax_income=[15.0, 18.0],
                   interest_expense=[5.0, 4.0], net_income=[12.0, 14.0])
    v, s = compute_metrics(_tables(rows, sic=3560, industry="Machinery", sector="Industrials"), AS_OF)
    assert v.at["HCO", "operating_margin_pct"] == pytest.approx(22 / 110 * 100)
    assert v.at["HCO", "interest_coverage_x"] == pytest.approx(22 / 4)


def test_reit_yield_trap_wording():
    from screener.providers.synthetic import AS_OF as SAS, build_synthetic_tables

    cfg = load_config()
    tb = build_synthetic_tables()
    v, s = compute_metrics(tb, pd.Timestamp(SAS))
    prof = assign_profiles(tb["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(v, prof, peer_group(tb["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(sorted(v.index[v.index.str.startswith("SYRT")])))
    trap = fl[(fl["ticker"] == "SYRTC") & (fl["flag"] == "yield_trap")].iloc[0]["detail"]
    assert "FFO/share" in trap and "EPS" not in trap


# ---- dividend-history data quality (found while validating REITs) ----------------------------------
def test_mis_tagged_dps_is_replaced_by_dividends_paid_per_share():
    """Prologis (PLD) SEC data: CommonStockDividendsPerShareDeclared holds 0.00-0.03 (not the common
    dividend, ~$4/share). Reported DPS more than 2x away from dividends paid / shares is discarded."""
    years = list(range(2019, 2026))
    rows = fy_rows("PLD", years,
                   dps=[0.00, 0.00, 0.00, 0.02, 0.00, 0.01, 0.03],
                   dividends_paid=[1345.66 * M, 1722.989 * M, 1872.861 * M, 2494.723 * M, 3228.589 * M,
                                   3570.48 * M, 3764.745 * M],
                   shares_diluted_wavg=[654.903 * M, 754.414 * M, 764.762 * M, 811.608 * M, 951.791 * M,
                                        953.59 * M, 956.832 * M],
                   shares_outstanding=[631.797 * M, 739.381 * M, 739.827 * M, 923.142 * M, 924.391 * M,
                                       926.283 * M, 929.153 * M],
                   net_income=[1.5e9, 1.5e9, 2.9e9, 3.4e9, 3.1e9, 3.7e9, 3.3e9],
                   revenue=[3.3e9, 4.4e9, 4.8e9, 6.0e9, 8.0e9, 8.2e9, 8.8e9],
                   total_assets=[40e9, 56e9, 58e9, 88e9, 93e9, 95e9, 98e9])
    v, s = compute_metrics(_tables(rows), AS_OF)
    assert v.at["PLD", "dividend_cut_5y"] == 0
    assert v.at["PLD", "dividend_paying_years"] == 7
    expected = (3764.745 / 956.832) / (1722.989 / 754.414)
    assert v.at["PLD", "dps_cagr_5y_pct"] == pytest.approx((expected ** (1 / 5) - 1) * 100)


def test_special_dividend_is_not_a_cut():
    """Public Storage (PSA) SEC DPS: $8.00 regular through 2021, $21.15 in 2022 (incl. a $13.15
    special), then $12.00. The 2022 spike must not make 2023 look like a dividend cut."""
    years = list(range(2019, 2026))
    rows = fy_rows("PSA", years, dps=[8.00, 8.00, 8.00, 21.15, 12.00, 12.00, 12.00],
                   shares_diluted_wavg=[174.53 * M, 174.642 * M, 175.568 * M, 176.28 * M, 176.143 * M,
                                        176.038 * M, 175.902 * M],
                   net_income=[1.5e9, 1.4e9, 1.9e9, 4.3e9, 2.1e9, 2.0e9, 1.7e9],
                   revenue=[2.86e9, 2.92e9, 3.42e9, 4.18e9, 4.52e9, 4.70e9, 4.82e9],
                   total_assets=[11e9, 12e9, 17e9, 18e9, 20e9, 19.8e9, 20e9])
    cfg = load_config()
    t = _tables(rows)
    v, s = compute_metrics(t, AS_OF)
    assert v.at["PSA", "special_dividends_detected"] == 1
    assert v.at["PSA", "dividend_cut_5y"] == 0
    assert v.at["PSA", "dividend_growth_streak_years"] == 0           # 12.00 -> 12.00: no raise in 2025
    # the 3-yr base year (FY2022) is the special-dividend year, so the 3-yr growth rate is left
    # missing rather than computed from a distorted base; the 5-yr rate uses regular dividends
    assert np.isnan(v.at["PSA", "dps_cagr_3y_pct"]) and s.at["PSA", "dps_cagr_3y_pct"] == "missing"
    assert v.at["PSA", "dps_cagr_5y_pct"] == pytest.approx(((12.0 / 8.0) ** (1 / 5) - 1) * 100)
    prof = assign_profiles(t["securities"], cfg.profiles)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(mv, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["PSA"]))
    assert "dividend_cut" not in set(fl["flag"])


def test_regular_dividend_cut_still_detected():
    years = list(range(2020, 2026))
    rows = fy_rows("CUT", years, dps=[2.0, 2.1, 2.2, 1.1, 1.1, 1.15], shares_diluted_wavg=[100 * M] * 6,
                   net_income=[300 * M] * 6, revenue=[1e9] * 6, total_assets=[5e9] * 6)
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.at["CUT", "dividend_cut_5y"] == 1 and v.at["CUT", "special_dividends_detected"] == 0


# ---- REIT dilution and leverage-change flags ---------------------------------------------------------
def _flags_for(rows):
    cfg = load_config()
    t = _tables(rows)
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(sorted(v.index)))
    return fl, v


def test_accretive_reit_equity_issuance_is_info_not_severe(realty):
    fl, v = _flags_for(realty_income_rows())
    d = fl[(fl["ticker"] == "O") & (fl["flag"] == "dilution")]
    assert v.at["O", "share_change_3y_cagr_pct"] > 8                 # would be "severe" for a normal company
    assert len(d) == 1 and d.iloc[0]["default_severity"] == "info" and "accretive" in d.iloc[0]["detail"]


def test_non_accretive_reit_dilution_stays_severe():
    rows = [dict(r, ticker="DLT") for r in realty_income_rows()]
    for r in rows:                        # same share issuance, but FFO is flat in dollars -> FFO/share falls
        r["net_income"] = 400 * M
        r["d_and_a"] = 1000 * M
        r["real_estate_impairment"] = np.nan
        r["gain_on_sale_real_estate"] = np.nan
    fl, v = _flags_for(rows)
    d = fl[(fl["ticker"] == "DLT") & (fl["flag"] == "dilution")]
    assert v.at["DLT", "ffo_ps_cagr_3y_pct"] < 0
    assert d.iloc[0]["default_severity"] == "severe"


def test_sec_parser_jv_gains_and_declared_dividends_fallback():
    from screener.providers.sec import parse_companyfacts

    def dur(v):
        return {"units": {"USD": [{"start": "2022-01-01", "end": "2022-12-31", "val": v, "filed": "2023-02-21",
                                   "form": "10-K", "fy": 2022, "fp": "FY"}]}}

    facts = {"facts": {"us-gaap": {
        "Revenues": dur(4182.2), "NetIncomeLoss": dur(4348.0),
        "GainsLossesOnSalesOfInvestmentRealEstate": dur(1.5),
        "EquityMethodInvestmentRealizedGainLossOnDisposal": dur(2128.9),   # PSA: sale of PS Business Parks
        "DividendsCommonStockCash": dur(3714.0),
    }}}
    r = parse_companyfacts(facts, "PSA").iloc[0]
    assert r["gain_on_sale_real_estate"] == pytest.approx(2130.4)
    assert r["dividends_paid"] == 3714.0


def test_ebit_fallback_excludes_property_gains():
    rows = fy_rows("GNS", [2024, 2025], revenue=[100.0, 100.0], pretax_income=[30.0, 90.0],
                   interest_expense=[5.0, 5.0], gain_on_sale_real_estate=[0.0, 60.0], net_income=[25.0, 85.0])
    v, _ = compute_metrics(_tables(rows, sic=3560, industry="Machinery", sector="Industrials"), AS_OF)
    assert v.at["GNS", "operating_margin_pct"] == pytest.approx(35.0)   # (90 + 5 - 60) / 100, not 95%


def test_dividends_paid_falls_back_to_dps_times_shares():
    """Public Storage stopped tagging common dividends paid after 2022; DPS is still tagged."""
    rows = realty_income_rows()
    for r in rows:
        r["dividends_paid"] = np.nan
    v, s = compute_metrics(_tables(rows), AS_OF)
    ffo_2025 = (1058.590 + 2524.200 + 434.497 - 177.640) * M
    assert v.at["O", "ffo_payout_pct"] == pytest.approx(3.2170 * 911.015 * M / ffo_2025 * 100)


def test_reit_dividend_coverage_is_measured_on_ffo():
    """Acquisition-heavy capex makes REIT FCF negative (Realty Income in Sharadar data): no
    negative_fcf / FCF-coverage flags; dividends above FFO are flagged instead."""
    import pandas as pd
    from screener.flags import detect_flags

    cfg = load_config()
    idx = pd.Index(["NETL", "OVER"], name="ticker")
    m = pd.DataFrame({"dividend_yield_pct": [5.5, 7.0], "fcf_payout_3y_pct": [900.0, 900.0],
                      "fcf_margin_pct": [-1.0, -1.0], "ffo_payout_pct": [75.0, 115.0],
                      "ffo_per_share": [4.0, 2.0]}, index=idx)
    prof = pd.Series("reit", index=idx)
    from screener.profiles import mask_not_applicable

    st = pd.DataFrame("ok", index=idx, columns=m.columns)
    m, st = mask_not_applicable(m, st, prof, cfg.profiles)
    f = detect_flags(m, prof, pd.Series("reit", index=idx), cfg.flags, cfg.profiles, idx, st)
    got = {(r.ticker, r.flag): r.detail for r in f.itertuples()}
    assert ("NETL", "negative_fcf") not in got and ("NETL", "weak_dividend_coverage") not in got
    assert got[("OVER", "weak_dividend_coverage")] == "dividends = 115% of FFO"
