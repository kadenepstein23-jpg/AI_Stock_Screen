"""Local snapshot store.

Layout (all under data/, which is git-ignored):

    data/raw/<provider>/<snapshot_id>/...         untouched downloads
    data/store/<provider>/<snapshot_id>/*.parquet canonical tables + metrics
    data/store/<provider>/<snapshot_id>/manifest.json
    data/store/<provider>/CURRENT                 points at the active snapshot

A snapshot is immutable once written. `refresh` creates a new snapshot;
`purge` deletes every snapshot for a provider.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import schema
from .config import Config


class NoLocalData(RuntimeError):
    """Raised when a provider has no local snapshot (never downloaded, or purged)."""


@dataclass
class Snapshot:
    provider: str
    snapshot_id: str
    path: Path
    manifest: Dict

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest.get("as_of", self.snapshot_id[:10]))

    def has(self, table: str) -> bool:
        return (self.path / f"{table}.parquet").exists()

    def read(self, table: str) -> pd.DataFrame:
        p = self.path / f"{table}.parquet"
        if not p.exists():
            if table in schema.TABLES:
                return schema.conform(pd.DataFrame(), table)
            raise FileNotFoundError(p)
        return pd.read_parquet(p)

    def write(self, table: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path / f"{table}.parquet", index=False)
        self.manifest.setdefault("tables", {})[table] = int(len(df))
        self._save_manifest()

    def _save_manifest(self) -> None:
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---- locations ---------------------------------------------------------
    def provider_dir(self, provider: str) -> Path:
        return self.cfg.data_dir / "store" / provider

    def raw_dir(self, provider: str, snapshot_id: Optional[str] = None) -> Path:
        base = self.cfg.data_dir / "raw" / provider
        return base / snapshot_id if snapshot_id else base

    def cache_dir(self, provider: str) -> Path:
        return self.cfg.data_dir / "cache" / provider

    # ---- snapshots ---------------------------------------------------------
    def new_snapshot(self, provider: str, as_of: Optional[date] = None, extra: Optional[Dict] = None) -> Snapshot:
        as_of = as_of or date.today()
        now = datetime.now(timezone.utc)
        # The id is the UTC creation time, so names sort in true creation order on any machine and
        # in any time zone (the data date lives in the manifest's as_of). Ordering still uses the
        # timezone-aware created_at, which also covers snapshots named under the old scheme.
        snapshot_id = f"{now:%Y-%m-%d_%H%M%S}"
        if (self.provider_dir(provider) / snapshot_id).exists():
            snapshot_id += f"_{now:%f}"
        path = self.provider_dir(provider) / snapshot_id
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "provider": provider,
            "snapshot_id": snapshot_id,
            "as_of": as_of.isoformat(),
            "created_at": now.isoformat(timespec="microseconds"),   # UTC, timezone-aware
            "license_retention": self.cfg.retention(provider),
            "tables": {},
        }
        manifest.update(extra or {})
        snap = Snapshot(provider, snapshot_id, path, manifest)
        snap._save_manifest()
        return snap

    def activate(self, snap: Snapshot) -> None:
        (self.provider_dir(snap.provider) / "CURRENT").write_text(snap.snapshot_id)

    def list_snapshots(self, provider: str) -> List[str]:
        """Snapshot ids, oldest first, ordered by when they were created (not by name: ids made on
        machines in different time zones under the old naming scheme don't sort chronologically)."""
        d = self.provider_dir(provider)
        if not d.exists():
            return []
        snaps = [p for p in d.iterdir() if p.is_dir() and (p / "manifest.json").exists()]

        def created(p: Path) -> float:
            try:
                ts = datetime.fromisoformat(json.loads((p / "manifest.json").read_text()).get("created_at", ""))
                if ts.tzinfo is not None:
                    return ts.timestamp()
            except (ValueError, TypeError):
                pass
            # older manifests stored naive local time: use the first table's file time instead
            files = [f.stat().st_mtime for f in p.glob("securities.parquet")] or [p.stat().st_mtime]
            return min(files)

        return [p.name for p in sorted(snaps, key=lambda p: (created(p), p.name))]

    def current(self, provider: str) -> Snapshot:
        d = self.provider_dir(provider)
        pointer = d / "CURRENT"
        snap_id = pointer.read_text().strip() if pointer.exists() else None
        if not snap_id or not (d / snap_id / "manifest.json").exists():
            snaps = self.list_snapshots(provider)
            if not snaps:
                raise NoLocalData(
                    f"No local data for provider '{provider}'. "
                    f"Run: python -m screener refresh --provider {provider}"
                    + ("  (requires an active subscription)" if self.cfg.is_licensed(provider) else ""))
            snap_id = snaps[-1]
        path = d / snap_id
        manifest = json.loads((path / "manifest.json").read_text())
        return Snapshot(provider, snap_id, path, manifest)

    def prune(self, provider: str, keep: int = 2) -> List[str]:
        """Delete all but the newest `keep` snapshots (saves disk). The active (CURRENT) snapshot is
        never deleted."""
        snaps = self.list_snapshots(provider)
        pointer = self.provider_dir(provider) / "CURRENT"
        active = pointer.read_text().strip() if pointer.exists() else None
        removed = []
        for s in snaps[:-keep] if keep > 0 else snaps:
            if s == active:
                continue
            path = self.provider_dir(provider) / s
            shutil.rmtree(path, ignore_errors=True)
            shutil.rmtree(self.raw_dir(provider, s), ignore_errors=True)
            if not path.exists():          # report only what was actually deleted
                removed.append(s)
        return removed
