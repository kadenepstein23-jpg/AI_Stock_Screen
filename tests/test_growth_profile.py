"""Growth / tech: secular-growth exit from the cyclical profile, scored growth blends, PEG guards,
EV/Sales, Rule of 40, near-zero and negative earnings, SBC vs dilution, and zero dividends.

NVDA inputs are its SEC company-facts values ($ millions; EPS and share counts as originally
reported, before the 2021 4:1 and 2024 10:1 splits - the engine adjusts them). FY2026 (ended
2026-01-25) reported: revenue $215,938M, operating income $130,387M, net income $120,067M,
diluted EPS $4.90, CFO $102,718M, capex $6,042M, SBC $6,386M, buybacks $40,086M.
Revenue, operating income, net income, EPS, shares and CFO are SEC values for every year; capex,
SBC, debt and cash are SEC values for FY2023-26. Equity, D&A and the earlier capex/SBC/debt/cash
figures are illustrative placeholders (no assertion depends on them).
"""
import copy

import numpy as np
import pandas as pd
import pytest

from screener import schema
from screener.config import load_config
from screener.flags import detect_flags
from screener.metrics.engine import compute_metrics
from screener.profiles import assign_profiles, mask_not_applicable, peer_group
from screener.providers.sec import parse_companyfacts
from screener.providers.synthetic import AS_OF as SYN_AS_OF, build_synthetic_tables
from screener.scoring import score_screen
from screener.universe import build_universe
from tests.conftest import fy_rows
from tests.test_cyclical_profile import NUE, _rows as nue_rows, _facts

M = 1e6
AS_OF = pd.Timestamp("2026-06-30")
NVDA_YEARS = list(range(2019, 2027))          # fiscal years ending late January
NVDA = dict(
    revenue=[11716, 10918, 16675, 26914, 26974, 60922, 130497, 215938],
    operating_income=[3804, 2846, 4532, 10041, 4224, 32972, 81453, 130387],
    net_income=[4141, 2796, 4332, 9752, 4368, 29760, 72880, 120067],
    eps_diluted=[6.63, 1.13, 1.73, 3.85, 0.17, 1.19, 2.94, 4.90],
    shares_diluted_wavg=[625, 2472, 2510, 2535, 25070, 24940, 24804, 24514],
    operating_cash_flow=[3743, 4761, 5822, 9108, 5641, 28090, 64089, 102718],
    capex=[600, 489, 1128, 976, 1833, 1069, 3236, 6042],
    sbc=[557, 844, 1397, 2004, 2709, 3549, 4737, 6386],
    d_and_a=[262, 381, 1098, 1174, 1544, 1508, 1864, 2400],
    total_debt=[1988, 1991, 6963, 10946, 10953, 9709, 8463, 8468],
    cash=[782, 10896, 847, 1990, 3389, 7280, 8589, 10605],
    equity=[9342, 12204, 16893, 26612, 22101, 42978, 79327, 157293],
)
PER_SHARE = {"eps_diluted"}


def nvda_rows(ticker="NVDA", data=NVDA):
    series = {k: [x if k in PER_SHARE else x * M for x in v] for k, v in data.items()}
    series["total_assets"] = [x * 1.4 for x in series["equity"]]
    rows = fy_rows(ticker, NVDA_YEARS, **series)
    for r in rows:                                   # fiscal year ends in late January
        r["period_end"] = pd.Timestamp(f"{r['period_end'].year}-01-28")
        r["filing_date"] = r["period_end"] + pd.Timedelta(days=30)
    return rows


def _tables(rows, sics, mcap=None):
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NASDAQ", security_type="common", share_class="single",
                              is_delisted=False, sic_code=sics.get(t, 7372)) for t in tickers])
    mkt = pd.DataFrame([dict(ticker=t, date=AS_OF, market_cap=(mcap or {}).get(t)) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(mkt, "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


def _metrics(rows, sics, mcap=None):
    cfg = load_config()
    t = _tables(rows, sics, mcap)
    v, s = compute_metrics(t, AS_OF)
    prof = assign_profiles(t["securities"], cfg.profiles, v)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    return cfg, t, mv, ms, prof


# ---- reported-figure checks ---------------------------------------------------------------------
def test_nvidia_metrics_match_reported_figures():
    _, _, v, s, _ = _metrics(nvda_rows(), {"NVDA": 3674})
    assert v.at["NVDA", "revenue_cagr_1y_pct"] == pytest.approx((215938 / 130497 - 1) * 100)      # +65.5%
    assert v.at["NVDA", "fcf_margin_pct"] == pytest.approx((102718 - 6042) / 215938 * 100)        # 44.8%
    assert v.at["NVDA", "sbc_pct_revenue"] == pytest.approx(6386 / 215938 * 100)                  # 3.0%
    assert v.at["NVDA", "operating_margin_pct"] == pytest.approx(130387 / 215938 * 100)
    assert v.at["NVDA", "eps_cagr_1y_pct"] == pytest.approx((4.90 / 2.94 - 1) * 100)
    assert v.at["NVDA", "splits_detected"] == 2                                                   # 4:1 and 10:1


# ---- secular-growth exit from the cyclical profile ------------------------------------------------
def test_semiconductor_with_secular_growth_leaves_cyclical_profile():
    rows = nvda_rows() + nue_rows("NUE", NUE)
    cfg, t, v, s, prof = _metrics(rows, {"NVDA": 3674, "NUE": 3312})
    assert assign_profiles(t["securities"], cfg.profiles)["NVDA"] == "cyclical"     # industry prior
    assert prof["NVDA"] == "default"                                               # own record wins
    assert "revenue trend" in prof.attrs["overrides"]["NVDA"]
    assert prof["NUE"] == "cyclical" and "NUE" not in prof.attrs["overrides"]
    # as a default-profile company its trailing P/E is scored and no peak flag fires
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["NVDA"]), s)
    assert "cyclical_peak" not in set(fl["flag"])


def test_secular_exit_needs_both_fast_growth_and_few_down_years():
    choppy = dict(NVDA)
    choppy["revenue"] = [11716, 15000, 13000, 26914, 22000, 60922, 50000, 215938]   # 3 down years in 5
    rows = nvda_rows("CHOP", choppy)
    _, _, v, _, prof = _metrics(rows, {"CHOP": 3674})
    assert v.at["CHOP", "revenue_trend_growth_pct"] > 15
    assert prof["CHOP"] == "cyclical"


# ---- scored growth: recent where trustworthy, robust otherwise --------------------------------------
def test_scored_eps_growth_blends_3y_and_trend_when_base_is_clean():
    eps = [2.98, 2.97, 3.28, 5.61, 6.11, 6.13, 6.08, 7.46]      # Apple FY2018-25 (3y 6.9%, trend 14.7%)
    rows = fy_rows("AAPL", list(range(2018, 2026)), eps_diluted=eps, revenue=[1e11] * 8, net_income=[1e10] * 8,
                   shares_diluted_wavg=[1.5e10] * 8)
    _, _, v, s, _ = _metrics(rows, {"AAPL": 3571})
    assert s.at["AAPL", "eps_cagr_3y_pct"] == "ok"
    expect = (v.at["AAPL", "eps_cagr_3y_pct"] + v.at["AAPL", "eps_trend_growth_pct"]) / 2
    assert v.at["AAPL", "eps_growth_scored_pct"] == pytest.approx(expect)
    assert 9 < v.at["AAPL", "eps_growth_scored_pct"] < 12


def test_scored_growth_uses_trend_alone_when_base_is_distorted():
    _, _, v, s, _ = _metrics(nvda_rows(), {"NVDA": 3674})
    assert s.at["NVDA", "eps_cagr_3y_pct"] == "nm"            # FY2023 trough base (0.17 vs 4.90)
    assert v.at["NVDA", "eps_growth_scored_pct"] == pytest.approx(v.at["NVDA", "eps_trend_growth_pct"])


def test_near_zero_base_makes_cagr_not_meaningful():
    eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.60, 1.10, 1.50]     # turned profitable; FY-3 base is 3% of latest
    rows = fy_rows("TURN", list(range(2018, 2026)), eps_diluted=eps, revenue=[1e10 * 1.3 ** i for i in range(8)],
                   net_income=[e * 1e9 for e in eps], shares_diluted_wavg=[1e9] * 8)
    _, _, v, s, _ = _metrics(rows, {"TURN": 7372})
    assert s.at["TURN", "eps_cagr_3y_pct"] == "nm" and not np.isnan(v.at["TURN", "eps_cagr_3y_pct"])
    assert s.at["TURN", "eps_growth_scored_pct"] in ("ok", "nm")


# ---- valuation: PEG, EV/Sales, Rule of 40 ---------------------------------------------------------
def test_peg_caps_hyper_growth_and_rejects_tiny_growth():
    _, _, v, s, _ = _metrics(nvda_rows(), {"NVDA": 3674}, mcap={"NVDA": 4.4e12})
    assert v.at["NVDA", "eps_growth_scored_pct"] > 40
    assert v.at["NVDA", "peg_hist_x"] == pytest.approx(v.at["NVDA", "pe_x"] / 40.0)
    slow = fy_rows("SLOW", list(range(2018, 2026)), eps_diluted=[5.0 * 1.01 ** i for i in range(8)],
                   revenue=[1e10] * 8, net_income=[5e8 * 1.01 ** i for i in range(8)], shares_diluted_wavg=[1e8] * 8)
    _, _, v2, s2, _ = _metrics(slow, {"SLOW": 2000}, mcap={"SLOW": 1e10})
    assert s2.at["SLOW", "peg_hist_x"] == "nm" and np.isnan(v2.at["SLOW", "peg_hist_x"])


def test_ev_sales_and_rule_of_40():
    mcap = 4.4e12                                      # illustrative market value
    _, _, v, _, _ = _metrics(nvda_rows(), {"NVDA": 3674}, mcap={"NVDA": mcap})
    ev = mcap + 8468 * M - 10605 * M
    assert v.at["NVDA", "enterprise_value_musd"] == pytest.approx(ev / M)
    assert v.at["NVDA", "ev_sales_x"] == pytest.approx(ev / (215938 * M))
    assert v.at["NVDA", "rule_of_40_pct"] == pytest.approx((215938 / 130497 - 1) * 100 + (102718 - 6042) / 215938 * 100)
    assert v.at["NVDA", "fcf_margin_ex_sbc_pct"] == pytest.approx((102718 - 6042 - 6386) / 215938 * 100)


def test_rule_of_40_is_reference_only():
    cfg = load_config()
    for spec in cfg.screens.values():
        names = {m["name"] for p in spec["pillars"].values() for m in p["metrics"]}
        assert "rule_of_40_pct" not in names and "fcf_margin_ex_sbc_pct" not in names
        assert not {"eps_cagr_3y_pct", "operating_margin_chg_3y_pp"} & names
    g = {m["name"] for m in cfg.screens["growth"]["pillars"]["operating_leverage"]["metrics"]}
    assert "operating_leverage_scored_pp" in g and "operating_margin_change_scored_pp" not in g   # no double count


# ---- negative and near-zero earnings --------------------------------------------------------------
def test_loss_making_company():
    eps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, -0.3]
    rows = fy_rows("LOSS", list(range(2018, 2026)), eps_diluted=eps, revenue=[1e10 * 1.2 ** i for i in range(8)],
                   net_income=[e * 1e9 for e in eps], shares_diluted_wavg=[1e9] * 8,
                   operating_income=[e * 1.3e9 for e in eps])
    cfg, t, v, s, prof = _metrics(rows, {"LOSS": 7372}, mcap={"LOSS": 2e10})
    for name in ("pe_x", "peg_hist_x", "eps_growth_scored_pct", "eps_trend_growth_pct"):
        assert s.at["LOSS", name] == "nm", name
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["LOSS"]), s)
    assert "negative_earnings" in set(fl["flag"])


def test_near_zero_earnings_pe_is_extreme_not_cheap():
    rows = fy_rows("THIN", list(range(2020, 2026)), eps_diluted=[1.0, 1.0, 1.0, 1.0, 1.0, 0.01],
                   revenue=[1e10] * 6, net_income=[1e9] * 5 + [1e7], shares_diluted_wavg=[1e9] * 6)
    _, _, v, s, _ = _metrics(rows, {"THIN": 2000}, mcap={"THIN": 1e10})
    assert v.at["THIN", "pe_x"] == pytest.approx(1000.0)        # ranks as the most expensive, not missing
    assert s.at["THIN", "eps_growth_scored_pct"] != "ok" or v.at["THIN", "eps_growth_scored_pct"] < 0


# ---- SBC and dilution are penalized once ------------------------------------------------------------
def _synth_scored(spec_name="growth"):
    cfg = load_config()
    t = build_synthetic_tables()
    v, s = compute_metrics(t, pd.Timestamp(SYN_AS_OF))
    uni, _ = build_universe(t["securities"], t["financials"], v, cfg.settings["universe"])
    prof = assign_profiles(t["securities"], cfg.profiles, v)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    peer = peer_group(t["securities"], prof, cfg.profiles)
    fl = detect_flags(v, prof, peer, cfg.flags, cfg.profiles, uni, s)
    return cfg, v, s, uni, prof, peer, fl


def test_sbc_flag_is_informational_when_dilution_is_already_penalized():
    cfg, v, s, uni, prof, peer, fl = _synth_scored()
    d = fl[fl["ticker"] == "SYDIL"].set_index("flag")
    assert d.at["dilution", "default_severity"] in ("moderate", "severe")
    assert d.at["high_sbc", "default_severity"] == "info" and bool(d.at["high_sbc", "downgraded"])
    # a screen that escalates high_sbc cannot re-escalate the downgraded row
    spec = copy.deepcopy(cfg.screens["growth"])
    spec.setdefault("flag_severity", {})["high_sbc"] = "severe"
    r = score_screen(spec, v, s, uni, prof, peer, fl, cfg.settings, cfg.profiles)
    ranked = r["ranked"].set_index("ticker")
    n_mod = (d["default_severity"] == "moderate").sum()
    assert ranked.at["SYDIL", "n_severe_flags"] == (d["default_severity"] == "severe").sum()
    assert ranked.at["SYDIL", "flag_penalty"] == pytest.approx(min(4 * n_mod, 20))


def test_sbc_flag_still_penalized_without_dilution():
    from screener.metrics.registry import METRICS
    v = pd.DataFrame({m: [np.nan] for m in METRICS}, index=["X"])
    v["sbc_pct_revenue"], v["share_change_3y_cagr_pct"] = 15.0, -1.0
    cfg = load_config()
    fl = detect_flags(v, pd.Series({"X": "default"}), pd.Series({"X": "Tech"}), cfg.flags, cfg.profiles,
                      pd.Index(["X"]))
    row = fl[fl["flag"] == "high_sbc"].iloc[0]
    assert row["default_severity"] == "moderate" and not row["downgraded"]


# ---- dividends: none vs missing ---------------------------------------------------------------------
def test_non_payer_records_zero_dividends():
    """Adobe files no dividend facts of any kind: dividends are 0, not missing."""
    f = _facts(duration={"Revenues": 23769 * M, "PaymentsForRepurchaseOfCommonStock": 11281 * M})
    row = parse_companyfacts(f, "ADBE").iloc[-1]
    assert row["dividends_paid"] == 0 and row["dps"] == 0
    # one dividend fact anywhere (even an unmapped tag) keeps a missing value missing
    f2 = _facts(duration={"Revenues": 1e10, "DividendsCash": 5e8})
    assert np.isnan(parse_companyfacts(f2, "X").iloc[-1]["dividends_paid"])


def test_zero_dividend_gives_zero_yield_and_a_shareholder_yield():
    rows = fy_rows("ADBE", list(range(2021, 2026)), revenue=[1.6e10, 1.76e10, 1.94e10, 2.15e10, 2.38e10],
                   net_income=[4.8e9, 4.76e9, 5.43e9, 5.56e9, 7.13e9], eps_diluted=[10.0, 10.1, 11.8, 12.4, 16.7],
                   shares_diluted_wavg=[4.8e8, 4.71e8, 4.59e8, 4.5e8, 4.27e8], dividends_paid=[0.0] * 5, dps=[0.0] * 5,
                   share_repurchases=[3.95e9, 6.55e9, 4.4e9, 9.5e9, 1.128e10],
                   net_common_issuance=[-3.95e9, -6.55e9, -4.4e9, -9.5e9, -1.128e10])
    _, _, v, s, _ = _metrics(rows, {"ADBE": 7372}, mcap={"ADBE": 1.5e11})
    assert v.at["ADBE", "dividend_yield_pct"] == 0 and s.at["ADBE", "dividend_yield_pct"] == "ok"
    assert s.at["ADBE", "shareholder_yield_pct"] == "ok" and v.at["ADBE", "shareholder_yield_pct"] > 0
