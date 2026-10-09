"""Provider normalization tests using fixtures in each provider's raw format (no network).

The Sharadar fixtures below are hand-made rows in Sharadar's column layout with
invented numbers — not Sharadar data.
"""
import pandas as pd
import pytest

from screener.providers.sec import classify_sec_security, parse_companyfacts
from screener.providers.sharadar import (classify_category, detect_daily_scale, normalize_daily, normalize_sep,
                                         normalize_sf1, normalize_tickers, thin_daily)


# ---------------------------------------------------------------- Sharadar ----
def test_sharadar_categories():
    assert classify_category("Domestic Common Stock") == ("common", "single")
    assert classify_category("Domestic Common Stock Primary Class") == ("common", "primary")
    assert classify_category("Domestic Common Stock Secondary Class") == ("common", "secondary")
    assert classify_category("ADR Common Stock") == ("adr", "single")
    assert classify_category("Domestic Preferred Stock")[0] == "preferred"
    assert classify_category("Domestic Warrant")[0] == "warrant"
    assert classify_category("ETF")[0] == "etf"
    assert classify_category("CEF")[0] == "fund"


def test_sharadar_tickers_prefers_sf1_row():
    tk = pd.DataFrame([
        dict(table="SEP", ticker="ZZZ", name="Zed Inc", exchange="NYSE", isdelisted="N",
             category="Domestic Common Stock", siccode=3560, sector="Industrials", industry="Machinery"),
        dict(table="SF1", ticker="ZZZ", name="Zed Inc", exchange="NYSE", isdelisted="N",
             category="Domestic Common Stock", siccode=3560, sector="Industrials", industry="Machinery"),
        dict(table="SFP", ticker="FFF", name="Fund", exchange="NYSEARCA", isdelisted="N", category="ETF",
             siccode=None, sector=None, industry=None),
    ])
    out = normalize_tickers(tk)
    assert list(out["ticker"]) == ["ZZZ"]           # SFP (funds table) dropped
    assert out.iloc[0]["security_type"] == "common" and not out.iloc[0]["is_delisted"]


def test_sharadar_sf1_signs_dimensions_and_ttm():
    base = dict(ticker="ZZZ", revenue=1000.0, netinccmn=100.0, capex=-50.0, ncfdiv=-30.0, ncfcommon=-20.0,
                ncfo=180.0, fcf=130.0, epsdil=1.0, sharesbas=100.0, shareswadil=101.0, debt=200.0, cashneq=50.0)
    rows = []
    for y in (2023, 2024, 2025):
        rows.append(dict(base, dimension="MRY", reportperiod=f"{y}-12-31", datekey=f"{y + 1}-02-10",
                         calendardate=f"{y}-12-31"))
    rows.append(dict(base, dimension="MRT", reportperiod="2026-03-31", datekey="2026-05-01", calendardate="2026-03-31"))
    rows.append(dict(base, dimension="MRT", reportperiod="2026-06-30", datekey="2026-08-01", calendardate="2026-06-30"))
    rows.append(dict(base, dimension="ARQ", reportperiod="2026-06-30", datekey="2026-08-01", calendardate="2026-06-30"))
    out = normalize_sf1(pd.DataFrame(rows))
    assert (out["period_type"] == "FY").sum() == 3
    ttm = out[out["period_type"] == "TTM"]
    assert len(ttm) == 1 and ttm.iloc[0]["period_end"] == pd.Timestamp("2026-06-30")
    r = out.iloc[0]
    assert r["capex"] == 50.0 and r["dividends_paid"] == 30.0 and r["net_common_issuance"] == -20.0
    assert r["filing_date"] == pd.Timestamp("2024-02-10")


def test_sharadar_daily_scale_and_thinning():
    daily = pd.DataFrame({"ticker": ["ZZZ"] * 3, "date": ["2026-06-28", "2026-06-29", "2026-06-30"],
                          "marketcap": [1000.0, 1001.0, 1002.0], "ev": [1100.0, 1101.0, 1102.0]})
    sf1_mcap = pd.Series({"ZZZ": 1.002e9})
    scale = detect_daily_scale(daily, sf1_mcap)
    assert scale == 1e6
    m = normalize_daily(daily, scale)
    assert m["market_cap"].iloc[-1] == pytest.approx(1.002e9)
    long = pd.DataFrame({"ticker": "ZZZ", "date": pd.bdate_range("2020-01-01", "2026-06-30"), "marketcap": 1.0, "ev": 1.0})
    thin = thin_daily(long, pd.Timestamp("2026-06-30"))
    old = thin[thin["date"] < "2025-01-01"]
    assert len(old) == old["date"].dt.to_period("M").nunique()   # one row per month for old history


def test_sharadar_sep_dollar_volume_units():
    sep = pd.DataFrame({"ticker": ["ZZZ"], "date": ["2026-06-30"], "close": [50.0], "closeadj": [49.0],
                        "closeunadj": [100.0], "volume": [2000.0]})   # 2-for-1 split after this date
    p = normalize_sep(sep).iloc[0]
    assert p["close"] == 100.0 and p["close_adj"] == 49.0
    assert p["close"] * p["volume"] == pytest.approx(50.0 * 2000.0)     # dollar volume preserved


# --------------------------------------------------------------------- SEC ----
def _fact(start, end, val, filed, form="10-K"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(end[:4]), "fp": "FY"}
    if start:
        d["start"] = start
    return d


def companyfacts_fixture():
    usd = lambda rows: {"units": {"USD": rows}}  # noqa: E731
    return {"facts": {"us-gaap": {
        "Revenues": usd([_fact("2023-01-01", "2023-12-31", 900, "2024-02-15"),
                         _fact("2023-01-01", "2023-12-31", 950, "2025-02-15"),     # restated in next 10-K
                         _fact("2024-01-01", "2024-12-31", 1000, "2025-02-15"),
                         _fact("2024-10-01", "2024-12-31", 260, "2025-02-15"),     # quarterly -> ignored
                         _fact("2022-01-01", "2022-12-31", 800, "2023-03-01", form="10-Q")]),  # wrong form
        "NetIncomeLoss": usd([_fact("2023-01-01", "2023-12-31", 90, "2024-02-15"),
                              _fact("2024-01-01", "2024-12-31", 110, "2025-02-15")]),
        "OperatingIncomeLoss": usd([_fact("2024-01-01", "2024-12-31", 150, "2025-02-15")]),
        "DepreciationDepletionAndAmortization": usd([_fact("2024-01-01", "2024-12-31", 40, "2025-02-15")]),
        "NetCashProvidedByUsedInOperatingActivities": usd([_fact("2024-01-01", "2024-12-31", 170, "2025-02-15")]),
        "PaymentsToAcquirePropertyPlantAndEquipment": usd([_fact("2024-01-01", "2024-12-31", 30, "2025-02-15")]),
        "LongTermDebtNoncurrent": usd([_fact(None, "2024-12-31", 300, "2025-02-15")]),
        "LongTermDebtCurrent": usd([_fact(None, "2024-12-31", 20, "2025-02-15")]),
        "CommercialPaper": usd([_fact(None, "2024-12-31", 5, "2025-02-15")]),
        "StockholdersEquity": usd([_fact(None, "2024-12-31", 500, "2025-02-15")]),
        "EarningsPerShareDiluted": {"units": {"USD/shares": [_fact("2024-01-01", "2024-12-31", 1.1, "2025-02-15")]}},
    }}}


def test_sec_parse_companyfacts():
    out = parse_companyfacts(companyfacts_fixture(), "ZZZ").set_index("period_end")
    assert list(out.index) == [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    fy23, fy24 = out.loc["2023-12-31"], out.loc["2024-12-31"]
    assert fy23["revenue"] == 950                 # latest restatement wins
    assert fy23["filing_date"] == pd.Timestamp("2024-02-15")   # first-reported date kept
    assert fy24["revenue"] == 1000
    assert fy24["ebitda"] == 190 and fy24["free_cash_flow"] == 140
    assert fy24["total_debt"] == 325              # noncurrent + current + commercial paper
    assert fy24["eps_diluted"] == 1.1 and fy24["equity"] == 500
    assert pd.isna(fy24["sbc"])                   # not reported -> missing, not zero


def test_sec_security_heuristics():
    assert classify_sec_security("ABCW", "Abc Acquisition Corp Warrant", None) == "warrant"
    assert classify_sec_security("XYZ-PA", "Xyz Corp 5% Preferred", None) == "preferred"
    assert classify_sec_security("MSFT", "Microsoft Corp", 7372) == "common"
