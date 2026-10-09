"""Bank-specific handling: n/a labelling, suppressed debt figures, ROE on average common equity."""
import numpy as np
import pandas as pd
import pytest

from screener.cli import main
from screener.metrics.engine import compute_metrics
from screener.profiles import assign_profiles, mask_not_applicable
from screener.providers.synthetic import AS_OF, build_synthetic_tables
from screener.scoring import pillar_specs_for_profile
from tests.conftest import fy_rows
from tests.test_metrics import _tables

BANK_DEBT_METRICS = ["cash_musd", "total_debt_musd", "net_debt_musd", "debt_to_equity_x", "enterprise_value_musd"]


@pytest.fixture(scope="module")
def masked():
    from screener.config import load_config

    cfg = load_config()
    t = build_synthetic_tables()
    v, s = compute_metrics(t, pd.Timestamp(AS_OF))
    prof = assign_profiles(t["securities"], cfg.profiles)
    mv, ms = mask_not_applicable(v, s, prof, cfg.profiles)
    return cfg, v, s, mv, ms


# 1) non-applicable bank metrics are labelled "na", never "missing" -------------------
def test_bank_metrics_marked_not_applicable(masked):
    cfg, v, s, mv, ms = masked
    for m in ["gross_margin_pct", "operating_margin_pct", "ebitda_margin_pct", "roic_pct", "current_ratio_x",
              "interest_coverage_x", "fcf_margin_pct", "p_ffo_x", "ffo_payout_pct", "roe_pct"]:
        assert ms.at["SYBKA", m] == "na", m
        assert np.isnan(mv.at["SYBKA", m])
    # metrics that DO apply to banks are untouched
    for m in ["net_margin_pct", "roa_pct", "roe_avg_common_pct", "equity_to_assets_pct", "payout_ratio_pct"]:
        assert ms.at["SYBKA", m] == "ok", m
    # non-bank companies are unaffected
    from screener.metrics.registry import PROFILE_ONLY

    # (SYCMA pays no dividend, so its dividend-growth rates are n/a too)
    assert (ms.loc["SYCMA"].drop(list(PROFILE_ONLY) + ["dps_cagr_3y_pct", "dps_cagr_5y_pct"]) != "na").all()
    assert (ms.loc["SYCMA", ["dps_cagr_3y_pct", "dps_cagr_5y_pct"]] == "na").all()
    assert (ms.loc["SYCMA", list(PROFILE_ONLY)] == "na").all()      # REIT-only metrics
    assert ms.at["SYCMA", "cash_musd"] == "ok"


def test_company_view_labels_bank_na(project, capsys):
    assert main(["demo"]) == 0
    capsys.readouterr()
    assert main(["company", "SYBKA"]) == 0
    out = capsys.readouterr().out
    assert "Industry profile: bank" in out
    assert "[n/a for bank]" in out
    gross = next(line for line in out.splitlines() if line.strip().startswith("Gross margin "))
    assert "[n/a for bank]" in gross and "[missing]" not in gross
    assert main(["company", "SYCMA"]) == 0
    out = capsys.readouterr().out
    assert "[n/a for" not in out and "[n/a: REIT-only]" in out


# 2) misleading bank debt figures are neither shown nor used ---------------------------
def test_bank_debt_figures_suppressed(masked):
    cfg, v, s, mv, ms = masked
    assert v[BANK_DEBT_METRICS].loc["SYBKA"].notna().any()      # raw engine output had numbers...
    for m in BANK_DEBT_METRICS:
        assert np.isnan(mv.at["SYBKA", m]) and ms.at["SYBKA", m] == "na"   # ...the bank view does not


def test_company_view_hides_bank_debt(project, capsys):
    main(["demo"])
    capsys.readouterr()
    main(["company", "SYBKA"])
    lines = {ln.strip().split("  ")[0]: ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("    ")}
    for label in ["Cash ($M)", "Total debt ($M)", "Net debt ($M)", "Debt / equity"]:
        assert "—" in lines[label] and "[n/a for bank]" in lines[label], label


def test_bank_debt_not_used_in_scoring_or_outputs(project):
    from screener.pipeline import run

    from screener.providers.synthetic import SyntheticProvider
    from screener.store import Store

    SyntheticProvider(project, Store(project)).refresh()
    out, results, _ = run(project, "synthetic")
    for screen in ("growth", "income"):
        specs = pillar_specs_for_profile(project.screens[screen], project.profiles["bank"])
        used = {m["name"] for items in specs.values() for m in items}
        assert not used & set(BANK_DEBT_METRICS)
        ms = results[screen]["metric_scores"]
        assert not set(ms[ms["ticker"].str.startswith("SYBK")]["metric"]) & set(BANK_DEBT_METRICS)
    allm = pd.read_csv(out / "full" / "all_metrics.csv").set_index("ticker")
    assert allm.loc["SYBKA", BANK_DEBT_METRICS].isna().all()
    assert allm.loc["SYCMA", "cash_musd"] > 0


# 3) bank ROE uses average common equity, excluding preferred -------------------------
def _bank(ticker, equity, preferred, ni):
    years = [2023, 2024, 2025]
    return fy_rows(ticker, years, net_income=ni, equity=equity, preferred_equity=preferred,
                   total_assets=[e * 12 for e in equity], revenue=[n / 0.3 for n in ni])


def test_roe_avg_common_equity():
    rows = _bank("BNK", equity=[900.0, 1000.0, 1100.0], preferred=[100.0, 100.0, 100.0], ni=[80.0, 90.0, 100.0])
    rows += _bank("NOPF", equity=[900.0, 1000.0, 1100.0], preferred=[np.nan] * 3, ni=[80.0, 90.0, 100.0])
    v, st = compute_metrics(_tables(rows), pd.Timestamp("2026-06-30"))
    # (1100-100 + 1000-100) / 2 = 950
    assert v.at["BNK", "roe_avg_common_pct"] == pytest.approx(100 / 950 * 100)
    assert v.at["BNK", "roe_pct"] == pytest.approx(100 / 1100 * 100)          # old definition, unchanged
    # no preferred reported -> average total equity
    assert v.at["NOPF", "roe_avg_common_pct"] == pytest.approx(100 / 1050 * 100)


def test_roe_avg_common_falls_back_and_handles_negative_equity():
    one = fy_rows("ONE", [2025], net_income=[50.0], equity=[500.0], preferred_equity=[0.0], total_assets=[5000.0],
                  revenue=[150.0])
    neg = fy_rows("NEG", [2024, 2025], net_income=[10.0, 20.0], equity=[100.0, 50.0], preferred_equity=[120.0, 120.0],
                  total_assets=[1000.0, 1000.0], revenue=[60.0, 60.0])
    v, st = compute_metrics(_tables(one + neg), pd.Timestamp("2026-06-30"))
    assert v.at["ONE", "roe_avg_common_pct"] == pytest.approx(10.0)          # ending equity when no prior year
    assert st.at["NEG", "roe_avg_common_pct"] == "nm"                         # common equity <= 0


def test_sec_parser_reads_preferred_stock():
    from screener.providers.sec import parse_companyfacts
    from tests.test_providers import companyfacts_fixture

    facts = companyfacts_fixture()
    facts["facts"]["us-gaap"]["PreferredStockValue"] = {"units": {"USD": [
        {"end": "2024-12-31", "val": 40, "filed": "2025-02-15", "form": "10-K", "fy": 2024, "fp": "FY"}]}}
    out = parse_companyfacts(facts, "ZZZ").set_index("period_end")
    assert out.loc["2024-12-31", "preferred_equity"] == 40
