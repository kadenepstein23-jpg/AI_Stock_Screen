"""Sharadar sample-mode refresh end to end, against a fake API in Sharadar's documented layout
(sharadar.com/docs, 2026-10-08). All numbers are invented - not Sharadar data.

Checks: every query passes `from` (the API defaults to one year), the fundamentals filing-date
column is `date`, daily marketcap/ev are USD millions, outflow signs, MRY/MRT handling, the public
test key is limited to AAPL, then screens, purge and the public-repo guard.
"""
from io import StringIO
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from screener.compliance import check_public, purge
from screener.pipeline import run
from screener.providers import sharadar as sh
from screener.store import Store


def _fake_tables(tickers):
    tk, sf1, daily, sep = [], [], [], []
    for i, t in enumerate(tickers):
        tk.append(dict(table="fundamentals", permaticker=100 + i, ticker=t, name=f"{t} Inc", exchange="NYSE",
                       isdelisted="N", category="Domestic Common Stock", siccode=3571 if t != "REIT" else 6798,
                       sector="Technology" if t != "REIT" else "Real Estate",
                       industry="Consumer Electronics" if t != "REIT" else "REIT - Industrial",
                       currency="USD", location="California; U.S.A"))
        tk.append(dict(tk[-1], table="stocks"))
        for y in range(2016, 2026):
            g = 1.08 ** (y - 2016)
            for dim in ("MRY", "ARY"):
                # documented: MR rows are dated by report period, AR rows by filing date
                sf1.append(dict(ticker=t, dimension=dim, calendardate=f"{y}-12-31",
                                date=f"{y}-12-31" if dim == "MRY" else f"{y + 1}-02-10",
                                reportperiod=f"{y}-12-31", fiscalperiod=f"{y}-12-31", lastupdated="2026-02-11",
                                revenue=10e9 * g, cor=6e9 * g, gp=4e9 * g, opinc=2e9 * g, ebit=2.1e9 * g,
                                ebitda=2.6e9 * g, depamor=0.5e9 * g, intexp=0.1e9, ebt=1.9e9 * g,
                                taxexp=0.4e9 * g, netinccmn=1.5e9 * g, epsdil=1.5 * g, shareswadil=1e9,
                                sharesbas=1e9, cashneq=2e9, assetsc=5e9, liabilitiesc=3e9, assets=20e9,
                                liabilities=10e9, debt=4e9, equity=10e9, intangibles=1e9, accoci=-0.1e9,
                                ncfo=2.4e9 * g, capex=-0.6e9 * g, fcf=1.8e9 * g, sbcomp=0.2e9 * g,
                                ncfdiv=-0.5e9 * g, ncfcommon=-0.3e9, dps=0.5 * g, marketcap=30e9))
                if t == "MSFT":
                    sf1[-1]["sbcomp"] = 0.0                     # Sharadar's "not reported" zero
        sf1.append(dict(sf1[-2], dimension="MRT", calendardate="2026-06-30", date="2026-06-30",
                        reportperiod="2026-06-30", revenue=11e9 * 1.08 ** 9, netinccmn=1.65e9 * 1.08 ** 9))
        sf1.append(dict(sf1[-1], dimension="ART", date="2026-08-01"))
        for d in pd.bdate_range("2014-01-01", "2026-09-30"):
            daily.append(dict(ticker=t, date=d.date().isoformat(), lastupdated="2026-10-01",
                              marketcap=30000.0, ev=32000.0, pe=15.0, pb=3.0, ps=2.0))   # USD millions
        for d in pd.bdate_range("2024-10-01", "2026-09-30"):
            adj = 30.0
            if t == "SPUN" and d < pd.Timestamp("2026-07-01"):
                adj = 30.0 * 0.6          # a 40% spin-off on 2026-07-01: earlier adjusted prices scaled down
            if t == "AAPL" and d < pd.Timestamp("2026-05-15"):
                adj = 30.0 * 0.99         # an ordinary 1% dividend
            sep.append(dict(ticker=t, date=d.date().isoformat(), open=30.0, high=31.0, low=29.0, close=30.0,
                            volume=5_000_000, closeadj=adj, closeunadj=30.0, lastupdated="2026-10-01"))
    return {"tickers": pd.DataFrame(tk), "fundamentals": pd.DataFrame(sf1),
            "daily": pd.DataFrame(daily), "stocks": pd.DataFrame(sep)}


class _Resp:
    def __init__(self, text):
        self.text, self.status_code = text, 200

    def raise_for_status(self):
        return None


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_get(session, url, params=None, **_):
        table = urlparse(url).path.rsplit("/", 1)[-1]
        calls.append((table, dict(params or {})))
        allowed = ["AAPL"] if params.get("api_key") == "test-api-key" else ["AAPL", "MSFT", "REIT", "SPUN"]
        wanted = [t for t in params.get("ticker", "").split(",") if t in allowed]
        df = _fake_tables(wanted)[table]
        if "from" in params and "date" in df:
            df = df[df["date"] >= params["from"]]
        off, lim = int(params.get("offset", params.get("skip", 0))), int(params.get("limit", 10000))
        return _Resp(df.iloc[off:off + lim].to_csv(index=False))

    monkeypatch.setattr(sh, "http_get", fake_get)
    return calls


def test_public_test_key_is_aapl_only_and_every_query_passes_from(project, fake_api, monkeypatch):
    monkeypatch.delenv("SHARADAR_SAMPLE_API_KEY", raising=False)
    snap = sh.SharadarProvider(project, Store(project)).refresh(sample=True)
    assert set(snap.read("securities")["ticker"]) == {"AAPL"}
    for table, params in fake_api:
        if table != "tickers":
            assert "from" in params, table
    assert {p["ticker"] for _, p in fake_api} == {"AAPL"}


def test_sample_refresh_maps_schema_units_and_dimensions(project, fake_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-account-key")
    store = Store(project)
    snap = sh.SharadarProvider(project, store).refresh(sample=True)
    sec = snap.read("securities").set_index("ticker")
    assert set(sec.index) == {"AAPL", "MSFT"}            # DJIA-30 list ∩ what the key may see
    assert sec.at["AAPL", "security_type"] == "common" and sec.at["AAPL", "sic_code"] == 3571
    fin = snap.read("financials")
    fy = fin[(fin["ticker"] == "AAPL") & (fin["period_type"] == "FY")].sort_values("period_end")
    assert len(fy) == 10                                                    # MRY only, ARY dropped
    assert (fin[fin["ticker"] == "AAPL"]["period_type"] == "TTM").sum() == 1
    last = fy.iloc[-1]
    assert last["filing_date"] == pd.Timestamp("2026-02-10")                # from the as-reported (ARY) row
    ttm = fin[(fin["ticker"] == "AAPL") & (fin["period_type"] == "TTM")].iloc[0]
    assert ttm["filing_date"] == pd.Timestamp("2026-08-01")                 # from the ART row
    assert last["capex"] > 0 and last["dividends_paid"] > 0                 # outflows made positive
    mkt = snap.read("market")
    assert mkt["market_cap"].iloc[-1] == pytest.approx(30e9)               # USD millions -> USD
    assert snap.manifest["daily_value_scale"] == 1e6
    msft = fin[(fin["ticker"] == "MSFT") & (fin["period_type"] == "FY")]
    assert msft["sbc"].isna().all()                                        # 0 -> missing
    px = snap.read("prices")
    assert px["close_split_adj"].notna().all()
    assert px["date"].min() >= pd.Timestamp("2024-10-01")


def test_sample_screens_purge_and_public_guard(project, fake_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-account-key")
    sh.SharadarProvider(project, Store(project)).refresh(sample=True, tickers=["AAPL", "MSFT", "REIT"])
    out, results, _ = run(project, "sharadar")
    assert {"growth", "income"} <= set(results)
    from screener.pipeline import ensure_metrics
    v = ensure_metrics(project, "sharadar").read("metrics").set_index("ticker")
    assert v.at["AAPL", "pe_x"] == pytest.approx(30e9 / (1.65e9 * 1.08 ** 9), rel=1e-6)   # TTM earnings
    assert v.at["AAPL", "dividend_yield_pct"] > 0 and not np.isnan(v.at["AAPL", "ev_sales_x"])
    assert check_public(project) == []
    rec = purge(project, "sharadar")
    assert rec["verified_clean"]
    assert not (project.data_dir / "store" / "sharadar").exists()
    assert not (project.data_dir / "raw" / "sharadar").exists()
    assert not (out / "full").exists() and (out / "retained").exists()


def test_tickers_table_names_old_and_new():
    rows = [dict(table=t, ticker="ZZZ", category="Domestic Common Stock") for t in ("stocks", "fundamentals")]
    assert len(sh.normalize_tickers(pd.DataFrame(rows))) == 1
    rows = [dict(table=t, ticker="ZZZ", category="Domestic Common Stock") for t in ("SEP", "SF1")]
    assert len(sh.normalize_tickers(pd.DataFrame(rows))) == 1


def test_outflow_sign_follows_the_data():
    assert sh._outflow_sign(pd.Series([-5.0, -3.0, 0.0])) == -1
    assert sh._outflow_sign(pd.Series([5.0, 3.0])) == 1
    sf1 = pd.DataFrame([dict(ticker="Z", dimension="MRY", reportperiod="2025-12-31", date="2026-02-01",
                             capex=50.0, ncfdiv=30.0, revenue=1.0)])
    r = sh.normalize_sf1(sf1).iloc[0]
    assert r["capex"] == 50.0 and r["dividends_paid"] == 30.0           # already positive: kept


def test_validate_against_sec_lists_largest_differences(project, fake_api, monkeypatch, capsys):
    from screener.cli import main

    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-account-key")
    store = Store(project)
    snap = sh.SharadarProvider(project, store).refresh(sample=True)
    ref = store.new_snapshot("sec")
    fin = snap.read("financials").copy()
    fin.loc[fin["ticker"] == "MSFT", "revenue"] *= 1.10          # a 10% revenue disagreement
    ref.write("securities", snap.read("securities"))
    ref.write("financials", fin)
    store.activate(ref)
    capsys.readouterr()
    assert main(["validate", "--primary", "sharadar", "--reference", "sec"]) == 0
    out = capsys.readouterr().out
    assert "match_rate_pct" in out and "Largest differences" in out
    line = next(x for x in out.splitlines() if "MSFT" in x and "revenue" in x)
    assert "9.09" in line                                        # |1 - 1/1.1| = 9.09%


def test_spin_off_is_detected_and_sent_to_review(project, fake_api, monkeypatch):
    """Honeywell-style case: after a spin-off the price reflects a smaller company while trailing
    earnings and dividends still include the spun-off business -> review, not a top Income rank."""
    from screener.pipeline import ensure_metrics

    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-account-key")
    sh.SharadarProvider(project, Store(project)).refresh(sample=True, tickers=["AAPL", "MSFT", "SPUN"])
    v = ensure_metrics(project, "sharadar").read("metrics").set_index("ticker")
    assert v.at["SPUN", "largest_distribution_1y_pct"] == pytest.approx(40.0, abs=0.01)
    assert v.at["AAPL", "largest_distribution_1y_pct"] == pytest.approx(1.0, abs=0.01)   # ordinary dividend
    out, results, _ = run(project, "sharadar")
    review = results["income"]["needs_review"]
    assert "SPUN" in set(review["ticker"]) and "AAPL" not in set(review["ticker"])
    reason = review.set_index("ticker").at["SPUN", "review_reason"]
    assert "recent_distribution" in reason


# ---- classification fixes found on the Sharadar sample ---------------------------------------------
def _secs(rows):
    from screener import schema
    return schema.conform(pd.DataFrame([dict(dict(exchange="NYSE", security_type="common", share_class="single",
                                                  is_delisted=False), **r) for r in rows]), "securities")


def test_managed_care_is_not_an_insurer():
    """UnitedHealth (SIC 6324, "Healthcare Plans") was scored as an insurer, which made its FCF and
    debt metrics n/a; managed care is a service business. Travelers stays an insurer."""
    from screener.config import load_config
    from screener.profiles import assign_profiles
    cfg = load_config()
    secs = _secs([dict(ticker="UNH", sic_code=6324, sector="Healthcare", industry="Healthcare Plans"),
                  dict(ticker="TRV", sic_code=6331, sector="Financial Services",
                       industry="Insurance - Property & Casualty")])
    prof = assign_profiles(secs, cfg.profiles)
    assert prof["UNH"] == "default" and prof["TRV"] == "insurance"


def test_bank_funded_financials_get_bank_treatment():
    """Goldman Sachs (SIC 6211, equity/assets ~8%) had net debt/EBITDA 33x and a severe leverage flag
    as a 'default' company. Financials funded like banks move to the bank profile; Visa does not."""
    from screener.config import load_config
    from screener.profiles import assign_profiles
    cfg = load_config()
    secs = _secs([dict(ticker="GS", sic_code=6211, sector="Financial Services", industry="Capital Markets"),
                  dict(ticker="AXP", sic_code=6199, sector="Financial Services", industry="Credit Services"),
                  dict(ticker="BLK", sic_code=6282, sector="Financial Services", industry="Asset Management"),
                  dict(ticker="V", sic_code=7389, sector="Financial Services", industry="Credit Services")])
    vals = pd.DataFrame({"equity_to_assets_pct": [7.9, 10.5, 35.0, 40.0]}, index=["GS", "AXP", "BLK", "V"])
    prof = assign_profiles(secs, cfg.profiles, vals)
    assert prof["GS"] == "bank" and prof["AXP"] == "bank"
    assert prof["BLK"] == "default" and prof["V"] == "default"
    assert "equity/assets" in prof.attrs["overrides"]["GS"]


def test_cli_survives_closed_pipe(monkeypatch):
    """`python -m screener ... | head` must exit quietly, not print a BrokenPipeError traceback."""
    from screener import cli

    def boom(cfg, args):
        raise BrokenPipeError()

    monkeypatch.setattr(cli, "cmd_status", boom, raising=False)
    monkeypatch.setattr(cli.os, "dup2", lambda *a: None)
    import argparse
    real = argparse.ArgumentParser.parse_args
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, argv=None: argparse.Namespace(verbose=False, fn=boom))
    assert cli.main([]) == 0
