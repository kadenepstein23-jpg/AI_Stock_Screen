"""Methodology freeze + holdout protocol, on synthetic data."""
import json

import pytest

from screener import holdout as ho
from screener.providers import get_provider
from screener.store import Store


@pytest.fixture
def demo(project):
    get_provider("synthetic", project, Store(project)).refresh()
    return project


def test_freeze_detects_methodology_changes(demo):
    assert ho.freeze_status(demo)[0] == "not frozen"
    rec = ho.freeze(demo)
    assert ho.freeze_status(demo)[0] == "frozen"
    demo.screens["growth"]["pillars"]["growth"]["weight"] += 1          # a weight change
    st, _ = ho.freeze_status(demo)
    assert st == "modified"
    demo.screens["growth"]["pillars"]["growth"]["weight"] -= 1
    demo.settings["providers"]["sharadar"]["bulk_years"] = 5            # download settings don't count
    assert ho.freeze_status(demo)[0] == "frozen"
    assert len(rec["methodology_fingerprint"]) == 16


def test_holdout_excludes_development_set_and_is_drawn_once(demo):
    with pytest.raises(RuntimeError, match="freeze"):
        ho.draw_holdout(demo, "synthetic", n=12)
    ho.freeze(demo)
    known = set(Store(demo).current("synthetic").read("securities")["ticker"])
    dev = set().union(*ho.development_set(demo, known).values())
    r = ho.draw_holdout(demo, "synthetic", n=12)
    tickers, meta = ho.read_holdout(demo)
    assert meta["sha_ok"] and len(tickers) == r["n"] > 0
    assert not set(tickers) & dev
    with pytest.raises(RuntimeError, match="already exists"):
        ho.draw_holdout(demo, "synthetic", n=12)
    # editing the list breaks the checksum
    p = demo.root / "config" / ho.HOLDOUT_FILE
    p.write_text(p.read_text() + "ZZZZ\n")
    assert ho.read_holdout(demo)[1]["sha_ok"] is False


def test_evaluate_runs_preregistered_checks_and_refuses_a_changed_methodology(demo):
    ho.freeze(demo)
    ho.draw_holdout(demo, "synthetic", n=15)
    r = ho.evaluate(demo, "synthetic", audit_n=6)
    ids = {c["id"]: c["result"] for c in r["checks"]}
    assert {"H0", "H0b", "H1", "H2", "H3", "H3d", "H4-growth", "H4-income", "H5-growth", "H5-income", "H6"} <= set(ids)
    assert ids["H0"] == "PASS" and ids["H0b"] == "PASS"
    assert ids["H1"] == "N/A"                                           # no SEC snapshot in the demo
    assert (r["out"] / "holdout_audit_sheet.csv").exists()
    assert not (r["keep"] / "holdout_audit_sheet.csv").exists()       # raw values stay in the purgeable tier
    assert json.loads((r["out"] / "TIER.json").read_text())["tier"] == "full"
    demo.flags["dilution"]["threshold"] = 5.0
    with pytest.raises(RuntimeError, match="modified"):
        ho.evaluate(demo, "synthetic")
