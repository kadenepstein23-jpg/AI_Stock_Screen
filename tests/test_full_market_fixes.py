"""Fixes found on the first full-market Sharadar run (2026-10-08). All numbers are invented."""
import pandas as pd

from screener import schema
from screener.config import load_config
from screener.flags import detect_flags
from screener.metrics.engine import compute_metrics
from screener.profiles import assign_profiles
from screener.store import Store
from tests.test_metrics import AS_OF, _tables, base_company
from tests.test_sharadar_sample import _secs


def test_pipelines_waste_and_equipment_makers_are_not_utilities():
    """SIC 4922-4924 (pipelines, LNG) and 4953 (refuse) sit inside the utility SIC range, and the
    'Utilities' sector holds equipment makers; none has a rate base."""
    cfg = load_config()
    secs = _secs([
        dict(ticker="KMI", sic_code=4922, sector="Energy", industry="Oil & Gas Midstream"),
        dict(ticker="AROC", sic_code=4922, sector="Energy", industry="Oil & Gas Equipment & Services"),
        dict(ticker="WM", sic_code=4953, sector="Industrials", industry="Waste Management"),
        dict(ticker="GEV", sic_code=3600, sector="Utilities", industry="Utilities - Renewable"),
        dict(ticker="VST", sic_code=4911, sector="Utilities", industry="Utilities - Independent Power Producers"),
        dict(ticker="ATO", sic_code=4923, sector="Utilities", industry="Utilities - Regulated Gas"),
        dict(ticker="DUK", sic_code=4931, sector="Utilities", industry="Utilities - Regulated Electric"),
    ])
    p = assign_profiles(secs, cfg.profiles)
    assert p["KMI"] == "cyclical" and p["AROC"] == "cyclical"        # energy, commodity-exposed
    assert p["WM"] == "default" and p["GEV"] == "default" and p["VST"] == "default"
    assert p["ATO"] == "utility" and p["DUK"] == "utility"


def test_former_reit_and_exchanges_keep_their_own_profile():
    cfg = load_config()
    secs = _secs([
        dict(ticker="CXW", sic_code=6798, sector="Industrials", industry="Security & Protection Services"),
        dict(ticker="O", sic_code=6798, sector="Real Estate", industry="REIT - Retail"),
        dict(ticker="CME", sic_code=6200, sector="Financial Services", industry="Financial Data & Stock Exchanges"),
        dict(ticker="CORZ", sic_code=6199, sector="Technology", industry="Software - Infrastructure"),
        dict(ticker="GS", sic_code=6211, sector="Financial Services", industry="Capital Markets"),
    ])
    vals = pd.DataFrame({"equity_to_assets_pct": [50.0, 50.0, 2.0, 10.0, 7.9]},
                        index=["CXW", "O", "CME", "CORZ", "GS"])
    p = assign_profiles(secs, cfg.profiles, vals)
    assert p["CXW"] == "default" and p["O"] == "reit"
    assert p["CME"] == "default" and p["CORZ"] == "default" and p["GS"] == "bank"


def _flags(rows_by_ticker):
    cfg = load_config()
    idx = pd.Index(list(rows_by_ticker), name="ticker")
    m = pd.DataFrame(rows_by_ticker).T.reindex(idx)
    from screener.metrics.registry import METRICS
    m = m.reindex(columns=sorted(set(METRICS) | set(m.columns))).apply(pd.to_numeric, errors="coerce")
    prof = pd.Series("default", index=idx)
    f = detect_flags(m, prof, pd.Series("Tech", index=idx), cfg.flags, cfg.profiles, idx)
    return {(r.ticker, r.flag): (r.default_severity, r.detail) for r in f.itertuples()}


def test_operating_loss_is_not_counted_twice_as_low_coverage():
    got = _flags({
        "CASHBURN": dict(net_margin_pct=-30.0, interest_coverage_x=-12.0, net_debt_musd=-800.0),
        "LEVLOSS": dict(net_margin_pct=-10.0, interest_coverage_x=-2.0, net_debt_musd=3_000.0),
        "THIN": dict(net_margin_pct=4.0, interest_coverage_x=1.2, net_debt_musd=3_000.0),
    })
    assert got[("CASHBURN", "negative_earnings")][0] == "moderate"
    assert got[("CASHBURN", "low_interest_coverage")][0] == "info"          # net cash: no extra penalty
    assert got[("LEVLOSS", "low_interest_coverage")][0] == "moderate"       # debt risk, not severe twice
    assert got[("THIN", "low_interest_coverage")][0] == "severe"            # profitable but 1.2x: unchanged


def test_dps_drop_not_seen_in_dividends_paid_is_not_a_cut():
    rows = base_company("ODD", dps=[2.29, 2.32, 2.39, 2.53, 1.92, 2.68],
                        dividends_paid=[22.9, 23.2, 23.9, 25.3, 26.1, 26.8])
    rows += base_company("CUT", dps=[2.0, 2.1, 2.2, 2.3, 1.2, 1.2], dividends_paid=[20, 21, 22, 23, 12, 12])
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["ODD", "dividend_cut_5y"] == 0
    assert v.loc["CUT", "dividend_cut_5y"] == 1


def test_sec_supplement_fills_reit_ffo_adjustments(project):
    """Sharadar has no impairment / gain fields; the SEC snapshot does. Build fills them for REITs."""
    from tests.conftest import fy_rows
    from screener.pipeline import build

    store = Store(project)
    years = list(range(2017, 2026))
    n = len(years)
    secs = schema.conform(pd.DataFrame([dict(ticker="RE", name="Re Inc", exchange="NYSE", security_type="common",
                                             share_class="single", is_delisted=False, sector="Real Estate",
                                             industry="REIT - Retail", sic_code=6798)]), "securities")
    base = dict(revenue=[1000.0] * n, net_income=[300.0] * n, d_and_a=[400.0] * n, shares_diluted_wavg=[100.0] * n,
                shares_outstanding=[100.0] * n, operating_cash_flow=[700.0] * n, equity=[5000.0] * n,
                total_assets=[9000.0] * n, total_debt=[3000.0] * n, cash=[100.0] * n, dividends_paid=[500.0] * n,
                dps=[5.0] * n)
    shar = schema.conform(pd.DataFrame(fy_rows("RE", years, **base)), "financials")
    sec_rows = pd.DataFrame(fy_rows("RE", years, real_estate_impairment=[0, 0, 0, 0, 0, 0, 50.0, 80.0, 90.0],
                                    gain_on_sale_real_estate=[10.0] * n))
    sec_rows["period_end"] = sec_rows["period_end"] + pd.Timedelta(days=-1)        # Dec 30 vs Dec 31
    snap = store.new_snapshot("sec", extra={"mode": "test"})
    snap.write("securities", secs)
    snap.write("financials", schema.conform(sec_rows, "financials"))
    store.activate(snap)
    snap = store.new_snapshot("sharadar", extra={"mode": "full"})
    snap.write("securities", secs)
    snap.write("financials", shar)
    snap.write("market", schema.conform(pd.DataFrame({"ticker": ["RE"], "date": [pd.Timestamp("2026-09-30")],
                                                      "market_cap": [8e3], "enterprise_value": [1.1e4]}), "market"))
    snap.write("prices", schema.conform(pd.DataFrame(columns=["ticker", "date"]), "prices"))
    store.activate(snap)
    out = build(project, "sharadar")
    v = out.read("metrics").set_index("ticker")
    assert out.manifest["supplement"]["values_filled"] == 2 * n
    assert v.at["RE", "ffo_nareit_adjusted"] == 1
    assert v.at["RE", "ffo_per_share"] == (300 + 400 + 90 - 10) / 100


def test_mortgage_reits_are_scored_like_lenders():
    cfg = load_config()
    secs = _secs([dict(ticker="NLY", sic_code=6798, sector="Real Estate", industry="REIT - Mortgage"),
                  dict(ticker="PLD", sic_code=6798, sector="Real Estate", industry="REIT - Industrial")])
    p = assign_profiles(secs, cfg.profiles)
    assert p["NLY"] == "bank" and p["PLD"] == "reit"


def test_reit_ffo_uses_the_fiscal_year_when_ttm_has_no_adjustments():
    """Adjustments exist for fiscal years only; a TTM loss from a new impairment must not become
    negative FFO (Sun Communities: TTM net income -$871M vs FY +$1,361M)."""
    from tests.test_reit_profile import _tables as reit_tables

    years = list(range(2018, 2026))
    n = len(years)
    rows = [dict(r, real_estate_impairment=10.0, gain_on_sale_real_estate=100.0) for r in
            __import__("tests.conftest", fromlist=["fy_rows"]).fy_rows(
                "SUN", years, revenue=[2000.0] * n, net_income=[500.0] * n, d_and_a=[500.0] * n,
                shares_diluted_wavg=[100.0] * n, shares_outstanding=[100.0] * n, dividends_paid=[400.0] * n,
                dps=[4.0] * n, equity=[6000.0] * n, total_assets=[15000.0] * n, total_debt=[7000.0] * n,
                cash=[100.0] * n, operating_cash_flow=[1000.0] * n, interest_expense=[300.0] * n,
                pretax_income=[520.0] * n)]
    rows.append(dict(rows[-1], period_type="TTM", period_end=pd.Timestamp("2026-06-30"),
                     filing_date=pd.Timestamp("2026-08-01"), net_income=-900.0,
                     real_estate_impairment=None, gain_on_sale_real_estate=None))
    v, _ = compute_metrics(reit_tables(rows), AS_OF)
    assert v.at["SUN", "ffo_per_share"] == (500 + 500 + 10 - 100) / 100      # FY basis, not -900 + ...
    assert v.at["SUN", "ffo_payout_pct"] == 400 / 910 * 100


def test_zero_dividends_paid_with_positive_dps_is_treated_as_missing():
    """Iron Mountain: Sharadar dividends paid is 0 every year while DPS is $2.00-$3.37. The zero
    contradicts the DPS, so the yield comes from DPS x shares. A company with DPS 0 and dividends 0
    still has no yield."""
    rows = base_company("ZEROTAG", dps=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5], dividends_paid=[0] * 6)
    rows += base_company("NODIV", dps=[0] * 6, dividends_paid=[0] * 6)
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["ZEROTAG", "dividend_yield_pct"] > 0
    assert v.loc["ZEROTAG", "dividend_cut_5y"] == 0
    assert not v.loc["NODIV", "dividend_yield_pct"] > 0
