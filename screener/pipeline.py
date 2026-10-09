"""Orchestration: build metrics from the local store, run screens, write outputs.

Nothing in this module touches the network. It works on whatever snapshot is
stored locally, so screens can be re-run (e.g. after changing weights) instantly
and without an active data subscription.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .flags import detect_flags
from .metrics.engine import compute_metrics
from .outputs import write_run
from .profiles import assign_profiles, mask_not_applicable, peer_group
from .scoring import score_screen, validate_screen
from .store import Snapshot, Store

log = logging.getLogger(__name__)


def load_tables(snap: Snapshot) -> Dict[str, pd.DataFrame]:
    return {t: snap.read(t) for t in ("securities", "financials", "market", "prices", "estimates")}


def build(cfg: Config, provider: str) -> Snapshot:
    """Compute metrics for the provider's current snapshot and store them alongside it."""
    snap = Store(cfg).current(provider)
    tables = load_tables(snap)
    from .supplement import apply_supplement

    tables["financials"], sup = apply_supplement(cfg, provider, tables)
    if sup:
        snap.manifest["supplement"] = sup
    values, status = compute_metrics(tables, snap.as_of, estimates=tables.get("estimates"))
    snap.write("metrics", values.reset_index())
    snap.write("metric_status", status.reset_index())
    snap.manifest["metrics_built_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    snap._save_manifest()
    return snap


def ensure_metrics(cfg: Config, provider: str) -> Snapshot:
    """Current snapshot with up-to-date metrics (rebuilt if missing or built by an older version)."""
    from .metrics.registry import METRICS

    snap = Store(cfg).current(provider)
    if not snap.has("metrics"):
        return build(cfg, provider)
    import pyarrow.parquet as pq

    cols = set(pq.read_schema(snap.path / "metrics.parquet").names)
    if not set(METRICS) <= cols:
        log.info("Metrics were built by an older version — rebuilding")
        return build(cfg, provider)
    return snap


def run(cfg: Config, provider: str, screens: Optional[List[str]] = None, run_id: Optional[str] = None):
    snap = ensure_metrics(cfg, provider)
    tables = load_tables(snap)
    values = snap.read("metrics").set_index("ticker")
    status = snap.read("metric_status").set_index("ticker")

    for name, spec in cfg.screens.items():
        problems = validate_screen(spec, cfg.profiles)
        if problems:
            raise ValueError(f"Config problems in screen '{name}':\n  " + "\n  ".join(problems))

    from .universe import build_universe

    profile = assign_profiles(tables["securities"], cfg.profiles, values)
    values, status = mask_not_applicable(values, status, profile, cfg.profiles)
    universe, excluded = build_universe(tables["securities"], tables["financials"], values,
                                        cfg.settings["universe"])
    peer = peer_group(tables["securities"], profile, cfg.profiles)
    flags = detect_flags(values, profile, peer, cfg.flags, cfg.profiles, universe, status)

    results = {}
    for name in (screens or list(cfg.screens)):
        results[name] = score_screen(cfg.screens[name], values, status, universe, profile, peer, flags,
                                     cfg.settings, cfg.profiles)
    out = write_run(cfg, snap, results, values, status, flags, excluded, universe, profile,
                    tables["securities"], run_id=run_id)
    return out, results, snap
