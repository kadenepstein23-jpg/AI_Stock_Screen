"""Energy / cyclicals: SEC parsing fixes (successor registrants, commercial paper, restricted cash),
starting-year checks on operating income / EBITDA / margins, normalized valuation and peak detection.

NUE and CVX inputs are the companies' SEC company-facts values ($ millions), FY2018-FY2025.
Reported FY2025 comparisons (Q4 2025 releases):
  CVX  EPS $6.63, CFO $33,939M, capex $17,347M, total debt $40,758M, cash $6,293M,
       dividends $12.8B, buybacks $12.1B (cash-flow line $12.2B incl. $146M excise tax)
  NUE  EPS $7.52, CFO $3,234M, capex $3,422M, dividends $512M, buybacks $700M
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
from screener.providers.sec import (SecProvider, merge_companyfacts, normalize_company_name,
                                    parse_companyfacts, predecessor_candidates, same_business)
from screener.providers.synthetic import AS_OF as SYN_AS_OF, build_synthetic_tables
from screener.scoring import pillar_specs_for_profile, score_screen
from screener.store import Store
from screener.universe import build_universe
from tests.conftest import fy_rows

M = 1e6
YEARS = list(range(2018, 2026))

NUE = dict(
    revenue=[25067, 22589, 20140, 36484, 41512, 34714, 30734, 32494],
    d_and_a=[631, 649, 702, 735, 827, 931, 1094, 1226],
    pretax_income=[3229, 1783, 836, 9201, 10245, 6273, 2902, 2568],
    interest_expense=[161, 157, 167, 163, 219, 246, 228, 170],
    net_income=[2351, 1264, 717, 6795, 7576, 4508, 2018, 1737],
    eps_diluted=[7.42, 4.14, 2.36, 23.16, 28.79, 18.00, 8.46, 7.52],
    shares_diluted_wavg=[317, 306, 303, 293, 263, 250, 238, 231],
    operating_cash_flow=[2394, 2809, 2697, 6231, 10072, 7112, 3979, 3234],
    capex=[983, 1477, 1543, 1622, 1948, 2214, 3173, 3422],
    total_debt=[4291, 4354, 5330, 5069, 6663, 6768, 5908, 7031],
    cash=[1399, 1535, 2640, 2365, 4281, 6383, 3558, 2260],
    equity=[9792, 10358, 10789, 14016, 18415, 20940, 20294, 20936],
    dividends_paid=[485, 492, 492, 483, 534, 515, 522, 512],
    dps=[1.54, 1.6025, 1.6125, 1.715, 2.01, 2.07, 2.17, 2.21],
)
PER_SHARE = {"eps_diluted", "dps"}
COUNTS = {"shares_diluted_wavg"}


def _rows(ticker, data, upto=2025):
    years = [y for y in YEARS if y <= upto]
    n = len(years)
    series = {}
    for k, v in data.items():
        scale = 1.0 if k in PER_SHARE else M
        series[k] = [x * scale for x in v[:n]]
    series["total_assets"] = [x * 2.0 for x in series["equity"]]
    return fy_rows(ticker, years, **series)


def _tables(rows, sic=3312, mcap=None, as_of=None):
    tickers = sorted({r["ticker"] for r in rows})
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sector="Basic Materials", industry="Steel", sic_code=sic)
                         for t in tickers])
    mkt = pd.DataFrame([dict(ticker=t, date=as_of, market_cap=(mcap or {}).get(t)) for t in tickers])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(mkt, "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


def _run(upto, mcap=None):
    as_of = pd.Timestamp(f"{upto + 1}-03-01")
    t = _tables(_rows("NUE", NUE, upto), mcap={"NUE": mcap} if mcap else None, as_of=as_of)
    cfg = load_config()
    v, s = compute_metrics(t, as_of)
    prof = assign_profiles(t["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    fl = detect_flags(v, prof, peer_group(t["securities"], prof, cfg.profiles), cfg.flags, cfg.profiles,
                      pd.Index(["NUE"]), s)
    return v.loc["NUE"], s.loc["NUE"], prof["NUE"], fl


# ---- SEC parsing ---------------------------------------------------------------------------------
def _fact(val, end, start=None, form="10-K", accn="0000093410-26-000010", filed="2026-02-20"):
    r = {"end": end, "val": val, "form": form, "accn": accn, "filed": filed, "fy": int(end[:4]), "fp": "FY"}
    if start:
        r["start"] = start
    return r


def _facts(instant=None, duration=None, entity="X"):
    us = {}
    for tag, val in (instant or {}).items():
        us[tag] = {"units": {"USD": [_fact(val, "2025-12-31")]}}
    for tag, val in (duration or {}).items():
        unit = "USD/shares" if tag.startswith("EarningsPerShare") else "USD"
        us[tag] = {"units": {unit: [_fact(val, "2025-12-31", "2025-01-01")]}}
    return {"entityName": entity, "facts": {"us-gaap": us}}


def test_commercial_paper_not_double_counted_with_short_term_borrowings():
    """Chevron: long-term debt $39,781M + short-term debt $977M = $40,758M reported. Its $4,642M of
    commercial paper is already inside short-term debt and must not be added again."""
    f = _facts(instant={"LongTermDebtAndCapitalLeaseObligations": 39781 * M, "ShortTermBorrowings": 977 * M,
                        "CommercialPaper": 4642 * M},
               duration={"Revenues": 184432 * M})
    assert parse_companyfacts(f, "CVX").iloc[-1]["total_debt"] == pytest.approx(40758 * M)
    # no short-term-borrowings line -> commercial paper IS the short-term debt
    f2 = _facts(instant={"LongTermDebt": 30000 * M, "CommercialPaper": 1500 * M}, duration={"Revenues": 1e10})
    assert parse_companyfacts(f2, "X").iloc[-1]["total_debt"] == pytest.approx(31500 * M)


def test_noncurrent_long_term_debt_plus_current_debt():
    """ExxonMobil: notes and loans payable (DebtCurrent) $9,296M + long-term debt $34,241M = $43,537M
    reported. LongTermDebtAndCapitalLeaseObligations is the NONCURRENT line, so current debt must be
    added; commercial paper ($3,059M) is already inside DebtCurrent."""
    f = _facts(instant={"LongTermDebtAndCapitalLeaseObligations": 34241 * M, "DebtCurrent": 9296 * M,
                        "CommercialPaper": 3059 * M, "LongTermDebtCurrent": 6200 * M},
               duration={"Revenues": 332238 * M})
    assert parse_companyfacts(f, "XOM").iloc[-1]["total_debt"] == pytest.approx(43537 * M)
    # no DebtCurrent: current maturities + commercial paper are added to the noncurrent line
    f2 = _facts(instant={"LongTermDebtAndCapitalLeaseObligations": 1000 * M, "LongTermDebtCurrent": 100 * M,
                         "CommercialPaper": 50 * M}, duration={"Revenues": 1e10})
    assert parse_companyfacts(f2, "X").iloc[-1]["total_debt"] == pytest.approx(1150 * M)


def test_cash_excludes_restricted_cash_when_only_the_total_is_tagged():
    """Chevron tags only cash + restricted cash ($7,285M); balance-sheet cash is $6,293M."""
    f = _facts(instant={"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 7285 * M,
                        "RestrictedCashCurrent": 174 * M, "RestrictedCashNoncurrent": 818 * M},
               duration={"Revenues": 184432 * M})
    assert parse_companyfacts(f, "CVX").iloc[-1]["cash"] == pytest.approx(6293 * M)
    f2 = _facts(instant={"CashAndCashEquivalentsAtCarryingValue": 500 * M,
                         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 900 * M},
                duration={"Revenues": 1e9})
    assert parse_companyfacts(f2, "X").iloc[-1]["cash"] == pytest.approx(500 * M)


# ---- successor registrants (XOM) -------------------------------------------------------------------
NEW_SUB = {"cik": "0002115436", "name": "ExxonMobil Holdings Corp", "formerNames": [],
           "addresses": {"business": {"street1": "22777 SPRINGWOODS VILLAGE PARKWAY", "zipCode": "77389"}},
           "filings": {"recent": {"form": ["8-K", "10-Q", "S-8 POS"],
                                  "accessionNumber": ["0002115436-26-000006", "0000034088-26-000093",
                                                      "0001193125-26-292536"]}}}
OLD_SUB = {"cik": "0000034088", "name": "EXXON MOBIL CORP", "formerNames": [],
           "addresses": {"business": {"street1": "22777 SPRINGWOODS VILLAGE PARKWAY", "zipCode": "77389-1425"}}}
AGENT_SUB = {"cik": "0001193125", "name": "DONNELLEY FINANCIAL", "addresses": {"business": {"street1": "x"}}}


def _xom_old_facts():
    """Predecessor history. 2025 revenue / NI / EPS are the reported totals; 2023-24 revenue figures
    are illustrative (only the merge mechanics are tested)."""
    us = {"Revenues": {"units": {"USD": []}}, "NetIncomeLoss": {"units": {"USD": []}},
          "EarningsPerShareDiluted": {"units": {"USD/shares": []}}}
    for y, rev, ni, eps in [(2023, 344582, 36010, 8.89), (2024, 339247, 33680, 7.84), (2025, 332238, 28844, 6.70)]:
        a = f"0000034088-{str(y + 1)[2:]}-000010"
        us["Revenues"]["units"]["USD"].append(_fact(rev * M, f"{y}-12-31", f"{y}-01-01", accn=a))
        us["NetIncomeLoss"]["units"]["USD"].append(_fact(ni * M, f"{y}-12-31", f"{y}-01-01", accn=a))
        us["EarningsPerShareDiluted"]["units"]["USD/shares"].append(_fact(eps, f"{y}-12-31", f"{y}-01-01", accn=a))
    return {"entityName": "EXXON MOBIL CORP", "facts": {"us-gaap": us}}


def _xom_new_facts():
    q = dict(form="10-Q", accn="0000034088-26-000093", filed="2026-08-03")
    return {"entityName": "ExxonMobil Holdings Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [dict(_fact(116017 * M, "2026-06-30", "2026-04-01", **q), fp="Q2")]}}}}}


def test_successor_registrant_detection():
    assert normalize_company_name("ExxonMobil Holdings Corp") == normalize_company_name("EXXON MOBIL CORP")
    cands = predecessor_candidates(NEW_SUB, _xom_new_facts(), 2115436)
    assert cands[0] == 34088 and 2115436 not in cands
    assert same_business(NEW_SUB, OLD_SUB)
    assert not same_business(NEW_SUB, AGENT_SUB)                   # a filing agent is never accepted


def test_merged_history_parses_fiscal_years():
    assert parse_companyfacts(_xom_new_facts(), "XOM").empty          # the bug: no FY rows at all
    fin = parse_companyfacts(merge_companyfacts(_xom_old_facts(), _xom_new_facts()), "XOM")
    assert list(fin["period_end"].dt.year) == [2023, 2024, 2025]
    assert fin.iloc[-1]["eps_diluted"] == pytest.approx(6.70)          # reported FY2025 GAAP EPS


def test_provider_attaches_predecessor(project, monkeypatch):
    prov = SecProvider(project, Store(project))
    docs = {"submissions/34088.json": OLD_SUB, "companyfacts/34088.json": _xom_old_facts(),
            "submissions/1193125.json": AGENT_SUB}

    def fake_cached(session, url, path, limiter, max_age_days=7):
        key = f"{path.parent.name}/{path.name}"
        if key not in docs:
            raise FileNotFoundError(key)
        return docs[key]

    monkeypatch.setattr(prov, "_cached_json", fake_cached)
    pred, facts = prov._attach_predecessor(None, project.data_dir / "cache" / "sec", None, 2115436,
                                           NEW_SUB, _xom_new_facts())
    assert pred == 34088
    assert len(parse_companyfacts(facts, "XOM")) == 3
    # an explicit override wins (and skips the name/address check)
    prov.settings["predecessor_ciks"] = {2115436: 34088}
    assert prov._attach_predecessor(None, project.data_dir, None, 2115436, NEW_SUB, _xom_new_facts())[0] == 34088


# ---- starting-year checks on operating income, EBITDA, margins --------------------------------------
def test_cyclical_trough_base_marks_opinc_ebitda_and_margin_changes_nm():
    v, s, prof, fl = _run(2025)
    assert prof == "cyclical"
    # FY2022 (the base of the 3y figures) was NUE's record year
    for name in ("opinc_cagr_3y_pct", "ebitda_cagr_3y_pct", "operating_margin_chg_3y_pp", "operating_leverage_3y_pp"):
        assert s[name] == "nm", name
        assert not np.isnan(v[name])                                  # value kept for reference
    assert v["operating_margin_chg_3y_pp"] < -15                       # -16.8pp peak-to-now
    assert abs(v["operating_margin_trend_3y_pp"]) < 3                  # through-cycle trend ~ flat
    # the margin flag reads the robust trend when the 3y change starts from an abnormal year
    assert "margin_deterioration" not in set(fl["flag"])


def test_utility_margin_base_check_nextera():
    """NextEra's FY2022 operating margin (17.7%, hedge losses) sits far below its other years."""
    from tests.test_utility_profile import nextera_rows, _tables as util_tables
    t = util_tables(nextera_rows())
    v, s = compute_metrics(t, pd.Timestamp("2026-06-30"))
    assert s.at["NEE", "operating_margin_chg_3y_pp"] == "nm"
    assert v.at["NEE", "op_margin_base_vs_trend_3y_pp"] < -5


def test_clean_series_not_marked_distorted():
    rows = fy_rows("STEADY", YEARS, revenue=[100e9 * 1.05 ** i for i in range(8)],
                   operating_income=[20e9 * 1.06 ** i for i in range(8)], d_and_a=[3e9] * 8,
                   net_income=[15e9 * 1.06 ** i for i in range(8)], eps_diluted=[3.0 * 1.06 ** i for i in range(8)],
                   shares_diluted_wavg=[5e9] * 8, equity=[50e9] * 8, total_assets=[100e9] * 8)
    v, s = compute_metrics(_tables(rows, sic=2000), pd.Timestamp("2026-03-01"))
    for name in ("opinc_cagr_3y_pct", "ebitda_cagr_3y_pct", "operating_margin_chg_3y_pp", "eps_cagr_3y_pct"):
        assert s.at["STEADY", name] == "ok", name
    assert v.at["STEADY", "opinc_trend_growth_pct"] == pytest.approx(6.0, abs=0.01)


# ---- boom / bust, normalized valuation, peak detection ---------------------------------------------
def test_nucor_boom_and_bust():
    peak_v, _, _, peak_fl = _run(2022)
    trough_v, _, _, trough_fl = _run(2025)
    # 2022: record earnings, flagged as a cyclical peak on both signals. (This fixture starts in 2018,
    # so the 2022 window has 5 years; with SEC history from 2015 the ratio is 2.2x.)
    assert peak_v["earnings_vs_normalized_x"] > 1.5
    assert peak_v["margin_vs_10y_avg_x"] > 1.5
    detail = peak_fl.set_index("flag").at["cyclical_peak", "detail"]
    assert "operating margin" in detail and "normalized" in detail
    # 2025: earnings below normal, no peak flag
    assert trough_v["earnings_vs_normalized_x"] < 0.75
    assert "cyclical_peak" not in set(trough_fl["flag"])
    # the 8-year EPS trend swings with the cycle phase -> not used for cyclicals
    assert peak_v["eps_trend_growth_pct"] > 40 and trough_v["eps_trend_growth_pct"] < 10
    # book value per share compounds through the cycle (scored for cyclicals instead): FY2020-25
    assert trough_v["bvps_cagr_5y_pct"] == pytest.approx(((20936 / 231) / (10789 / 303)) ** 0.2 * 100 - 100)


def test_peak_pe_looks_cheap_but_normalized_pe_does_not():
    mcap = 30e9          # the same market value in both years (illustrative, not a historical price)
    peak_v, _, _, _ = _run(2022, mcap=mcap)
    trough_v, _, _, _ = _run(2025, mcap=mcap)
    assert peak_v["pe_x"] < 5 and trough_v["pe_x"] > 15               # trailing P/E: "cheap" at the peak
    assert peak_v["pe_normalized_x"] == pytest.approx(peak_v["pe_x"] * peak_v["earnings_vs_normalized_x"])
    assert peak_v["pe_normalized_x"] > 1.5 * peak_v["pe_x"]
    assert trough_v["pe_normalized_x"] < trough_v["pe_x"]
    for v in (peak_v, trough_v):
        assert not np.isnan(v["ev_ebitda_normalized_x"]) and not np.isnan(v["fcf_yield_normalized_pct"])


def test_normalized_pe_does_not_penalize_a_steady_grower():
    rows = fy_rows("GROW", YEARS, revenue=[10e9 * 1.15 ** i for i in range(8)],
                   net_income=[1.5e9 * 1.15 ** i for i in range(8)], eps_diluted=[1.0 * 1.15 ** i for i in range(8)],
                   shares_diluted_wavg=[1.5e9] * 8, operating_income=[2e9 * 1.15 ** i for i in range(8)],
                   d_and_a=[0.5e9] * 8, equity=[10e9] * 8, total_assets=[20e9] * 8)
    t = _tables(rows, sic=3674, mcap={"GROW": 60e9}, as_of=pd.Timestamp("2026-03-01"))
    v, s = compute_metrics(t, pd.Timestamp("2026-03-01"))
    assert v.at["GROW", "pe_normalized_x"] == pytest.approx(v.at["GROW", "pe_x"], rel=1e-6)   # constant margin
    assert v.at["GROW", "earnings_vs_normalized_x"] == pytest.approx(1.0)


# ---- scoring -----------------------------------------------------------------------------------------
def test_cyclical_scoring_uses_normalized_valuation_and_book_value_growth():
    cfg = load_config()
    cyc = cfg.profiles["cyclical"]
    g = {m["name"]: m for items in pillar_specs_for_profile(cfg.screens["growth"], cyc).values() for m in items}
    i = {m["name"]: m for items in pillar_specs_for_profile(cfg.screens["income"], cyc).values() for m in items}
    for specs in (g, i):
        assert "pe_x" not in specs and "ev_ebitda_x" not in specs
        assert specs["pe_normalized_x"]["substituted_for"] == "pe_x"
        assert not {"peg_hist_x", "p_fcf_x", "pe_vs_5y_median_pct", "eps_growth_scored_pct"} & set(specs)
    assert g["bvps_cagr_5y_pct"]["weight"] == 0.75 and g["bvps_cagr_5y_pct"]["substituted_for"] == "eps_growth_scored_pct"
    assert not {"operating_leverage_scored_pp", "gross_margin_change_scored_pp", "ebitda_growth_scored_pct"} & set(g)
    # other profiles are untouched
    d = {m["name"] for items in pillar_specs_for_profile(cfg.screens["growth"], cfg.profiles["default"]).values()
         for m in items}
    assert {"pe_x", "ev_ebitda_x", "eps_growth_scored_pct", "operating_leverage_scored_pp"} <= d


def test_trailing_multiples_stay_visible_for_cyclicals():
    v, s, prof, _ = _run(2025, mcap=30e9)
    assert s["pe_x"] == "ok" and s["ev_ebitda_x"] == "ok"            # shown, just not scored


def test_growth_screen_sends_peak_cyclicals_to_review():
    cfg = load_config()
    t = build_synthetic_tables()
    v, s = compute_metrics(t, pd.Timestamp(SYN_AS_OF))
    uni, _ = build_universe(t["securities"], t["financials"], v, cfg.settings["universe"])
    prof = assign_profiles(t["securities"], cfg.profiles)
    v, s = mask_not_applicable(v, s, prof, cfg.profiles)
    peer = peer_group(t["securities"], prof, cfg.profiles)
    fl = detect_flags(v, prof, peer, cfg.flags, cfg.profiles, uni, s)
    spec = copy.deepcopy(cfg.screens["growth"])
    spec["gates"] = []                                     # synthetic cyclicals fail the revenue gate
    r = score_screen(spec, v, s, uni, prof, peer, fl, cfg.settings, cfg.profiles)
    review = r["needs_review"].set_index("ticker")
    assert "SYCYA" in review.index and "cyclical_peak" in review.at["SYCYA", "review_reason"]
    assert "SYCYA" not in set(r["ranked"]["ticker"])
    inc = score_screen(cfg.screens["income"], v, s, uni, prof, peer, fl, cfg.settings, cfg.profiles)
    assert {"SYCYC", "SYCYD"} <= set(inc["ranked"]["ticker"])       # mid-cycle cyclicals still rank


@pytest.mark.parametrize("sic,expected", [(1520, "cyclical"), (1531, "cyclical"), (3531, "cyclical"),
                                          (2911, "cyclical"), (3312, "cyclical"), (3674, "cyclical"),
                                          (4011, "default"), (2080, "default")])
def test_cyclical_sic_coverage(sic, expected):
    """Lennar files under SIC 1520 and Caterpillar under 3531; railroads and beverages are not cyclical."""
    cfg = load_config()
    secs = schema.conform(pd.DataFrame([dict(ticker="X", name="X", exchange="NYSE", security_type="common",
                                             share_class="single", is_delisted=False, sic_code=sic)]), "securities")
    assert assign_profiles(secs, cfg.profiles)["X"] == expected
