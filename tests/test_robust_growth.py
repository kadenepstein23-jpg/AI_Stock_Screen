"""Robust EPS growth, starting-year distortion checks, generic utility note and snapshot naming.

EPS series are the companies' reported GAAP diluted EPS from SEC company facts (FY2018-FY2025):
  DUK  3.76 5.06 1.72 4.94 3.17 3.54 5.71 6.31  (2020 pipeline write-off; 2022-23 impairments)
       reported adjusted EPS 2020-2025: 5.12 5.24 5.27 5.56 5.90 6.31 (about 4%/yr)
  NEE  3.47 1.94 1.48 1.81 2.10 3.60 3.37 3.30  (split-adjusted; 2019-2022 hedge mark-to-market losses)
       reported adjusted EPS 2020 2.31 -> 2025 3.71 (about 10%/yr)
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener import schema
from screener.config import load_config
from screener.flags import detect_flags
from screener.metrics.engine import compute_metrics
from screener.metrics.robust import base_vs_trend, is_distorted, trend_growth
from screener.profiles import assign_profiles, peer_group
from screener.providers.synthetic import AS_OF as SYN_AS_OF, build_synthetic_tables
from screener.scoring import pillar_specs_for_profile, score_screen
from screener.store import Store
from screener.universe import build_universe
from tests.conftest import fy_rows

AS_OF = pd.Timestamp("2026-06-30")
YEARS = list(range(2018, 2026))
EPS = {
    "DUK": [3.76, 5.06, 1.72, 4.94, 3.17, 3.54, 5.71, 6.31],
    "NEE": [3.47, 1.94, 1.48, 1.81, 2.10, 3.60, 3.37, 3.30],
    "CLEAN": [5.00 * 1.06 ** i for i in range(8)],          # steady 6% compounder
    "SPIKE": [4.0, 4.2, 4.4, 4.6, 9.5, 5.1, 5.3, 5.6],      # one-off gain in the 3y base year (FY-3)
    "LOSS": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, -0.5],      # latest year is a loss
    "LOSSES": [2.0, -1.0, 2.2, -0.4, 2.4, 2.5, 2.6, 2.8],   # two loss years in the window
    "IPO": [np.nan, np.nan, np.nan, np.nan, 1.0, 1.2, 1.4, 1.6],
}


def frame(series):
    """Panel frame: index = ticker, columns = offsets 0 (latest) .. 7."""
    return pd.DataFrame({t: list(reversed(v)) for t, v in series.items()}).T


def _tables(names):
    rows = []
    for t in names:
        rows += fy_rows(t, YEARS, eps_diluted=EPS[t], revenue=[1e10] * 8, net_income=[1e9] * 8,
                        shares_diluted_wavg=[1e8] * 8)
    secs = pd.DataFrame([dict(ticker=t, name=t, exchange="NYSE", security_type="common", share_class="single",
                              is_delisted=False, sector="Utilities", industry="Utilities - Regulated Electric",
                              sic_code=4911) for t in names])
    return {"securities": schema.conform(secs, "securities"),
            "financials": schema.conform(pd.DataFrame(rows), "financials"),
            "market": schema.conform(pd.DataFrame(), "market"),
            "prices": schema.conform(pd.DataFrame(), "prices")}


@pytest.fixture(scope="module")
def metrics():
    return compute_metrics(_tables(list(EPS)), AS_OF)


# ---- robust trend ---------------------------------------------------------------------------------
def test_trend_recovers_exact_growth_on_clean_series():
    v, s = trend_growth(frame({"CLEAN": EPS["CLEAN"]}))
    assert s["CLEAN"] == "ok" and v["CLEAN"] == pytest.approx(6.0, abs=1e-9)


def test_trend_ignores_one_off_years():
    clean = trend_growth(frame({"CLEAN": EPS["CLEAN"]}))[0]["CLEAN"]
    bumped = list(EPS["CLEAN"])
    bumped[2] *= 0.3            # a write-off year
    bumped[5] *= 2.0            # and a one-off gain year
    v = trend_growth(frame({"X": bumped}))[0]["X"]
    assert abs(v - clean) < 1.0


def test_utility_trend_vs_inflated_plain_cagr(metrics):
    v, s = metrics
    # plain CAGRs as reported (kept for reference) ...
    assert v.at["DUK", "eps_cagr_3y_pct"] == pytest.approx(((6.31 / 3.17) ** (1 / 3) - 1) * 100)   # 25.8%
    assert v.at["DUK", "eps_cagr_5y_pct"] == pytest.approx(((6.31 / 1.72) ** (1 / 5) - 1) * 100)   # 29.7%
    assert v.at["NEE", "eps_cagr_3y_pct"] == pytest.approx(((3.30 / 2.10) ** (1 / 3) - 1) * 100)   # 16.3%
    # ... but the robust trend lands near the companies' adjusted-EPS growth (DUK ~4%, NEE ~10%)
    assert 5.0 < v.at["DUK", "eps_trend_growth_pct"] < 10.0
    assert 8.0 < v.at["NEE", "eps_trend_growth_pct"] < 13.0
    assert s.at["DUK", "eps_trend_growth_pct"] == "ok" and s.at["NEE", "eps_trend_growth_pct"] == "ok"


def test_trend_status_rules(metrics):
    v, s = metrics
    assert s.at["LOSS", "eps_trend_growth_pct"] == "nm"       # latest year is a loss
    assert s.at["LOSSES", "eps_trend_growth_pct"] == "nm"     # two loss years
    assert s.at["IPO", "eps_trend_growth_pct"] == "short"     # only 4 years of history
    assert np.isnan(v.at["LOSS", "eps_trend_growth_pct"])


def test_trend_needs_the_latest_year():
    v, s = trend_growth(frame({"X": EPS["CLEAN"][:-1] + [np.nan]}))
    assert s["X"] == "missing" and np.isnan(v["X"])


# ---- starting-year distortion check ---------------------------------------------------------------
def test_distorted_base_year_marks_plain_cagr_nm_but_keeps_value(metrics):
    v, s = metrics
    for t, k in [("DUK", 3), ("DUK", 5), ("NEE", 3), ("NEE", 5), ("SPIKE", 3)]:
        assert s.at[t, f"eps_cagr_{k}y_pct"] == "nm", (t, k)
        assert not np.isnan(v.at[t, f"eps_cagr_{k}y_pct"])                    # visible for reference
        r = v.at[t, f"eps_base_vs_trend_{k}y_x"]
        assert r < 1 / 1.5 or r > 1.5
    assert v.at["SPIKE", "eps_base_vs_trend_3y_x"] > 1.5                      # inflated base -> too low CAGR
    assert v.at["DUK", "eps_base_vs_trend_5y_x"] < 0.5                        # 2020 write-off year


def test_clean_base_year_left_ok(metrics):
    v, s = metrics
    assert s.at["CLEAN", "eps_cagr_3y_pct"] == "ok" and s.at["CLEAN", "eps_cagr_5y_pct"] == "ok"
    assert v.at["CLEAN", "eps_base_vs_trend_3y_x"] == pytest.approx(1.0)
    assert v.at["CLEAN", "eps_cagr_3y_pct"] == pytest.approx(6.0)


def test_base_vs_trend_needs_enough_other_years():
    f = frame({"X": [np.nan, np.nan, np.nan, np.nan, 1.0, 1.1, 1.2, 1.3]})
    assert np.isnan(base_vs_trend(f, base=3)["X"])          # only 3 other years
    assert not is_distorted(pd.Series([np.nan])).any()


def test_peg_uses_robust_growth():
    t = _tables(["DUK"])
    t["market"] = schema.conform(pd.DataFrame([dict(ticker="DUK", date=AS_OF, market_cap=95e9)]), "market")
    v, s = compute_metrics(t, AS_OF)
    # DUK's 3y CAGR is NM (distorted base), so the scored growth is the robust trend alone
    assert v.at["DUK", "eps_growth_scored_pct"] == pytest.approx(v.at["DUK", "eps_trend_growth_pct"])
    assert v.at["DUK", "peg_hist_x"] == pytest.approx(v.at["DUK", "pe_x"] / v.at["DUK", "eps_growth_scored_pct"])


# ---- scoring --------------------------------------------------------------------------------------
def test_no_screen_scores_plain_eps_cagr():
    cfg = load_config()
    for screen in cfg.screens.values():
        for prof in cfg.profiles.values():
            used = {m["name"] for items in pillar_specs_for_profile(screen, prof).values() for m in items}
            assert not used & {"eps_cagr_3y_pct", "eps_cagr_5y_pct"}
            replaced = (prof.get("not_applicable") or []) + list((prof.get("score_with") or {}))
            if "eps_growth_scored_pct" not in replaced:     # cyclicals score book-value growth instead
                assert "eps_growth_scored_pct" in used


def _score(t, screen="growth"):
    cfg = load_config()
    v, s = compute_metrics(t, pd.Timestamp(SYN_AS_OF))
    uni, _ = build_universe(t["securities"], t["financials"], v, cfg.settings["universe"])
    prof = assign_profiles(t["securities"], cfg.profiles)
    peer = peer_group(t["securities"], prof, cfg.profiles)
    fl = detect_flags(v, prof, peer, cfg.flags, cfg.profiles, uni)
    r = score_screen(cfg.screens[screen], v, s, uni, prof, peer, fl, cfg.settings, cfg.profiles)
    return v, s, r["ranked"].set_index("ticker")


def test_depressed_base_year_no_longer_lifts_the_score():
    """A one-year write-off three years ago used to inflate the scored 3y EPS CAGR."""
    base = build_synthetic_tables()
    v0, s0, r0 = _score(base)
    tkr = "SYCMD"
    fin = base["financials"].copy()
    fy = fin[(fin["ticker"] == tkr) & (fin["period_type"] == "FY")].sort_values("period_end")
    idx = fy.index[-4]                                   # FY-3
    fin.loc[idx, "eps_diluted"] = fin.loc[idx, "eps_diluted"] * 0.4
    hit = dict(base, financials=fin)
    v1, s1, r1 = _score(hit)
    assert v1.at[tkr, "eps_cagr_3y_pct"] > v0.at[tkr, "eps_cagr_3y_pct"] + 20   # plain CAGR balloons
    assert s1.at[tkr, "eps_cagr_3y_pct"] == "nm"
    assert abs(v1.at[tkr, "eps_trend_growth_pct"] - v0.at[tkr, "eps_trend_growth_pct"]) < 1.5
    # the scored growth falls back to the trend because the 3y CAGR's base is now distorted
    assert v1.at[tkr, "eps_growth_scored_pct"] == pytest.approx(v1.at[tkr, "eps_trend_growth_pct"])
    assert abs(r1.at[tkr, "composite_score"] - r0.at[tkr, "composite_score"]) < 1.0


# ---- generic utility note -------------------------------------------------------------------------
def test_utility_data_gap_note_is_generic():
    note = load_config().profiles["utility"]["data_gaps"]
    for company in ("NextEra", "NEE", "Duke", "DUK", "Southern", "Dominion"):
        assert company not in note
    assert "GAAP" in note and "adjusted" in note


# ---- snapshot naming ------------------------------------------------------------------------------
def test_snapshot_names_sort_in_creation_order_regardless_of_data_date(project):
    store = Store(project)
    a = store.new_snapshot("synthetic", as_of=date(2026, 10, 8))     # later data date, created first
    b = store.new_snapshot("synthetic", as_of=date(2026, 10, 7))     # earlier data date, created second
    assert a.snapshot_id < b.snapshot_id                              # name order == creation order
    assert sorted([b.snapshot_id, a.snapshot_id]) == store.list_snapshots("synthetic")
    for snap in (a, b):
        m = json.loads((snap.path / "manifest.json").read_text())
        created = pd.Timestamp(m["created_at"])
        assert created.tzinfo is not None and created.utcoffset().total_seconds() == 0
        assert snap.snapshot_id.startswith(created.strftime("%Y-%m-%d_%H%M%S"))   # named by UTC creation time
    assert b.as_of.date() == date(2026, 10, 7)                        # the data date still lives in the manifest


def test_snapshot_name_ignores_local_time_zone(project, monkeypatch):
    if not hasattr(__import__("time"), "tzset"):
        pytest.skip("tzset not available")
    import time
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        snap = Store(project).new_snapshot("synthetic", as_of=date(2026, 10, 7))
        created = pd.Timestamp(json.loads((snap.path / "manifest.json").read_text())["created_at"])
        assert snap.snapshot_id.startswith(created.tz_convert("UTC").strftime("%Y-%m-%d_%H%M%S"))
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
