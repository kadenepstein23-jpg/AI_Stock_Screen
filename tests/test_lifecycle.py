"""The subscribe -> ingest -> run -> purge -> re-subscribe lifecycle, and the public-repo guard.

A licensed provider is simulated by writing the synthetic tables into a
snapshot under the provider name 'sharadar' (no Sharadar data involved).
"""
import json

import pandas as pd
import pytest

from screener.compliance import check_public, export_portfolio, purge
from screener.pipeline import run
from screener.providers.synthetic import AS_OF, build_synthetic_tables
from screener.store import NoLocalData, Store


def fake_licensed_refresh(cfg, provider="sharadar"):
    store = Store(cfg)
    snap = store.new_snapshot(provider, as_of=AS_OF)
    (store.raw_dir(provider, snap.snapshot_id)).mkdir(parents=True, exist_ok=True)
    (store.raw_dir(provider, snap.snapshot_id) / "sf1.csv").write_text("ticker,datekey,ncfo\nX,2026-01-01,1\n")
    for name, df in build_synthetic_tables().items():
        snap.write(name, df)
    store.activate(snap)
    return snap


def test_full_lifecycle(project):
    cfg = project
    # 1) "subscribe" + refresh, run screens
    fake_licensed_refresh(cfg)
    out, results, snap = run(cfg, "sharadar")
    assert len(results["growth"]["ranked"]) > 0
    full, retained, public = out / "full", out / "retained", out / "public"
    assert json.loads((full / "TIER.json").read_text())["license_scope"] == "sharadar"
    # licensed raw values appear only in full/
    assert "cash_musd" in pd.read_csv(full / "all_metrics.csv").columns
    ret_cols = set(pd.read_csv(retained / "all_metrics_ratios.csv").columns)
    assert not ret_cols & {"cash_musd", "total_debt_musd", "market_cap_musd", "price"}
    assert "roic_pct" in ret_cols
    pub = pd.read_csv(public / "growth_top25.csv")
    assert set(pub.columns) <= {"rank", "ticker", "name", "sector", "profile", "composite_score",
                                "confidence_grade", "data_coverage_pct", "flags"} | {c for c in pub if c.startswith("pillar_")}
    assert "Sharadar" in (public / "README.md").read_text()

    # 2) purge after "cancelling"
    rec = purge(cfg, "sharadar")
    assert rec["verified_clean"]
    assert not (cfg.data_dir / "store" / "sharadar").exists()
    assert not (cfg.data_dir / "raw" / "sharadar").exists()
    assert not full.exists()
    assert retained.exists() and public.exists()           # derived outputs kept
    assert (cfg.root / rec["log"]).exists()

    # 3) the code still works; it simply asks for a refresh
    with pytest.raises(NoLocalData, match="refresh --provider sharadar"):
        run(cfg, "sharadar")
    # other providers are unaffected
    from screener.providers.synthetic import SyntheticProvider

    SyntheticProvider(cfg, Store(cfg)).refresh()
    run(cfg, "synthetic")

    # 4) "re-subscribe": refresh again and the same code runs immediately
    fake_licensed_refresh(cfg)
    out2, results2, _ = run(cfg, "sharadar")
    assert len(results2["growth"]["ranked"]) == len(results["growth"]["ranked"])


def test_check_public_guard(project):
    cfg = project
    assert check_public(cfg) == []
    (cfg.root / "leak.parquet").write_bytes(b"PAR1")
    (cfg.root / "notes").mkdir()
    (cfg.root / "notes" / "dump.csv").write_text("ticker,datekey,revenue,ncfo\nX,2026-01-01,1,2\n")
    (cfg.root / ".env").write_text("SHARADAR_API_KEY=abc\n")
    issues = {p["file"] for p in check_public(cfg)}
    assert "notes/dump.csv" in issues
    assert ".env" not in issues          # .env is git-ignored -> not publishable
    # a parquet anywhere is ignored by *.parquet in .gitignore, so remove that rule to prove detection
    gi = cfg.root / ".gitignore"
    gi.write_text(gi.read_text().replace("*.parquet\n", ""))
    assert "leak.parquet" in {p["file"] for p in check_public(cfg)}


def test_export_portfolio_is_public_safe(project):
    cfg = project
    fake_licensed_refresh(cfg)
    run(cfg, "sharadar")
    dest = export_portfolio(cfg)
    assert dest.parent.name == "portfolio"
    assert not [p for p in check_public(cfg) if p["file"].startswith("portfolio/")]
    assert not list(dest.rglob("all_metrics*.csv"))


def test_prune_orders_by_creation_time_and_never_deletes_current(project):
    """Regression: snapshot ids made on machines in different time zones (e.g. '..._232809' in UTC
    and '..._163037' local) don't sort chronologically; prune once deleted the snapshot it had
    just created."""
    import json
    import os
    from datetime import date

    store = Store(project)
    old = store.new_snapshot("synthetic", as_of=date(2026, 10, 7))
    new = store.new_snapshot("synthetic", as_of=date(2026, 10, 7))
    # make the OLDER snapshot's id sort last by name, as happened with the UTC/local mix
    old_named = old.path.parent / "2026-10-07_235959"
    os.rename(old.path, old_named)
    m = json.loads((old_named / "manifest.json").read_text())
    m["created_at"] = "2026-10-07T23:28:09+00:00"
    m["snapshot_id"] = old_named.name
    (old_named / "manifest.json").write_text(json.dumps(m))
    mn = json.loads((new.path / "manifest.json").read_text())
    mn["created_at"] = "2026-10-08T00:30:37+00:00"           # genuinely newer
    (new.path / "manifest.json").write_text(json.dumps(mn))
    store.activate(new)
    assert store.list_snapshots("synthetic") == [old_named.name, new.snapshot_id]
    removed = store.prune("synthetic", keep=1)
    assert removed == [old_named.name]
    assert new.path.exists() and store.current("synthetic").snapshot_id == new.snapshot_id
    # even keep=0 leaves the active snapshot alone
    assert store.prune("synthetic", keep=0) == [] and new.path.exists()
