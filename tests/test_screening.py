"""Universe, profiles, flags and scoring on the synthetic universe (each archetype tests a rule)."""
import numpy as np
import pandas as pd
import pytest

from screener.flags import detect_flags
from screener.metrics.engine import compute_metrics
from screener.profiles import assign_profiles, peer_group
from screener.providers.synthetic import AS_OF, build_synthetic_tables
from screener.scoring import _mid_rank_pct, pillar_specs_for_profile, score_screen, validate_screen
from screener.universe import build_universe


@pytest.fixture(scope="module")
def synth():
    from screener.config import load_config

    cfg = load_config()
    t = build_synthetic_tables()
    v, s = compute_metrics(t, pd.Timestamp(AS_OF))
    uni, exc = build_universe(t["securities"], t["financials"], v, cfg.settings["universe"])
    prof = assign_profiles(t["securities"], cfg.profiles)
    peer = peer_group(t["securities"], prof, cfg.profiles)
    flags = detect_flags(v, prof, peer, cfg.flags, cfg.profiles, uni)
    return dict(cfg=cfg, t=t, v=v, s=s, uni=uni, exc=exc.set_index("ticker"), prof=prof, peer=peer, flags=flags)


@pytest.mark.parametrize("ticker,reason", [
    ("SYETF", "security type not allowed"), ("SYPRA", "security type not allowed"),
    ("SYWTS", "security type not allowed"), ("SYADR", "security type not allowed"),
    ("SYSPC", "SPAC / blank check (SIC)"), ("SYCMA.B", "secondary share class"),
    ("SYSTL", "stale or missing filings"), ("SYPNY", "price below minimum or missing"),
    ("SYSML", "market cap below minimum or missing"), ("SYILQ", "illiquid (median $ volume) or missing"),
])
def test_universe_exclusions(synth, ticker, reason):
    assert ticker in synth["exc"].index
    assert reason in synth["exc"].at[ticker, "all_reasons"]


def test_operating_companies_are_included(synth):
    for t in ["SYCMA", "SYARA", "SYBKA", "SYRTA", "SYUTA", "SYCYA", "SYIPO"]:
        assert t in synth["uni"]


def test_profiles(synth):
    p = synth["prof"]
    assert p["SYBKA"] == "bank" and p["SYINA"] == "insurance" and p["SYRTA"] == "reit"
    assert p["SYUTA"] == "utility" and p["SYCYA"] == "cyclical" and p["SYCMA"] == "default"


def test_bank_pillars_drop_ev_metrics_and_substitute(synth):
    cfg = synth["cfg"]
    specs = pillar_specs_for_profile(cfg.screens["growth"], cfg.profiles["bank"])
    names = {m["name"] for items in specs.values() for m in items}
    assert "ev_ebitda_x" not in names and "net_debt_to_ebitda_x" not in names
    assert "ptbv_x" in names and "roa_pct" in names and "roe_avg_common_pct" in names
    assert "roe_pct" not in names
    reit = pillar_specs_for_profile(cfg.screens["income"], cfg.profiles["reit"])
    rn = {m["name"] for items in reit.values() for m in items}
    assert "p_ffo_x" in rn and "pe_x" not in rn and "ffo_payout_pct" in rn


def test_config_is_valid(synth):
    cfg = synth["cfg"]
    for spec in cfg.screens.values():
        assert validate_screen(spec, cfg.profiles) == []


def _flags_for(synth, t):
    f = synth["flags"]
    return set(f[f["ticker"] == t]["flag"])


def test_red_flags(synth):
    assert "yield_trap" in _flags_for(synth, "SYYTA")
    assert {"weak_dividend_coverage", "revenue_decline"} <= _flags_for(synth, "SYYTA")
    assert {"dilution", "high_sbc"} <= _flags_for(synth, "SYDIL")
    assert "negative_equity" in _flags_for(synth, "SYNEQ")
    assert "cyclical_peak" in _flags_for(synth, "SYCYA")
    assert "negative_earnings" in _flags_for(synth, "SYHGA")
    assert "high_leverage" in _flags_for(synth, "SYHLB")
    assert "short_history" in _flags_for(synth, "SYIPO")
    # banks/utilities: leverage & FCF flags disabled by profile
    assert not ({"high_leverage", "negative_fcf"} & _flags_for(synth, "SYBKA"))
    assert "negative_fcf" not in _flags_for(synth, "SYUTA")
    # healthy compounder: no warning flags
    assert not (_flags_for(synth, "SYCMA") - {"no_dividend"})


def test_mid_rank_percentiles():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert list(_mid_rank_pct(s, True)) == [0.125, 0.375, 0.625, 0.875]
    assert list(_mid_rank_pct(s, False)) == [0.875, 0.625, 0.375, 0.125]
    assert list(_mid_rank_pct(pd.Series([5.0]), True)) == [0.5]


def test_growth_screen(synth):
    d = synth
    r = score_screen(d["cfg"].screens["growth"], d["v"], d["s"], d["uni"], d["prof"], d["peer"], d["flags"],
                     d["cfg"].settings, d["cfg"].profiles)
    ranked = r["ranked"].set_index("ticker")
    top6 = set(r["ranked"]["ticker"].head(6))
    assert top6 == {"SYCMA", "SYCMB", "SYCMC", "SYCMD", "SYCME", "SYCMF"}   # compounders lead
    assert "SYARA" not in ranked.index            # fails the 8% revenue-growth gate
    assert ranked.at["SYDIL", "flag_penalty"] > 0  # dilution/SBC penalized
    assert "SYMIS" in set(r["needs_review"]["ticker"])  # patchy data -> needs review, not ranked
    assert ranked["composite_score"].between(0, 100).all()
    assert (ranked["rank"].diff().dropna() == 1).all()


def test_income_screen_excludes_yield_traps(synth):
    d = synth
    r = score_screen(d["cfg"].screens["income"], d["v"], d["s"], d["uni"], d["prof"], d["peer"], d["flags"],
                     d["cfg"].settings, d["cfg"].profiles)
    ranked = set(r["ranked"]["ticker"])
    review = r["needs_review"].set_index("ticker")
    assert "SYYTA" not in ranked and "yield_trap" in review.at["SYYTA", "review_reason"]
    assert "SYCMA" not in ranked                   # no dividend -> fails gate
    assert {"SYARA", "SYARB", "SYUTA", "SYRTA", "SYBKA"} <= ranked


def test_missing_metric_scored_neutral_low_and_reduces_coverage(synth):
    d = synth
    v, s = d["v"].copy(), d["s"].copy()
    s.loc["SYCMA", "roic_pct"] = "missing"
    v.loc["SYCMA", "roic_pct"] = np.nan
    r = score_screen(d["cfg"].screens["growth"], v, s, d["uni"], d["prof"], d["peer"], d["flags"],
                     d["cfg"].settings, d["cfg"].profiles)
    ms = r["metric_scores"]
    row = ms[(ms["ticker"] == "SYCMA") & (ms["metric"] == "roic_pct")].iloc[0]
    assert row["score"] == pytest.approx(0.35)
    cov = r["ranked"].set_index("ticker").at["SYCMA", "data_coverage_pct"]
    assert 90 < cov < 100
