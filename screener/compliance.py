"""Data-license compliance: purge licensed data, guard the public repo, export portfolio files.

See docs/DATA_LICENSING.md for the full procedure.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .metrics.registry import raw_value_metrics
from .schema import RAW_VALUE_FIELDS

# Column names that only appear in Sharadar raw tables
SHARADAR_MARKERS = {"datekey", "reportperiod", "calendardate", "ncfo", "ncfdiv", "ncfcommon", "sharesbas",
                    "shareswadil", "closeunadj", "closeadj", "permaticker", "sbcomp", "netinccmn", "epsdil"}
BINARY_DATA_EXT = {".parquet", ".feather", ".arrow", ".zip", ".gz", ".bulk", ".pkl", ".pickle", ".h5", ".db",
                   ".sqlite", ".duckdb"}
TEXT_DATA_EXT = {".csv", ".tsv", ".json", ".txt", ".xlsx", ".xls"}
ALWAYS_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}


def _dir_stats(path: Path) -> Tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def purge_targets(cfg: Config, provider: str) -> List[Path]:
    targets = [cfg.data_dir / "raw" / provider, cfg.data_dir / "store" / provider, cfg.data_dir / "cache" / provider]
    # any output tier (runs, validation reports) marked as license-scoped to this provider
    if cfg.outputs_dir.exists():
        for tier in cfg.outputs_dir.rglob("TIER.json"):
            try:
                meta = json.loads(tier.read_text())
            except Exception:
                continue
            if meta.get("license_scope") == provider:
                targets.append(tier.parent)
    return [t for t in targets if t.exists()]


def purge(cfg: Config, provider: str, dry_run: bool = False) -> Dict:
    """Delete raw + normalized data and license-scoped outputs for `provider`. Keeps retained/public tiers."""
    targets = purge_targets(cfg, provider)
    record = {"provider": provider, "started_at": datetime.now().isoformat(timespec="seconds"),
              "dry_run": dry_run, "deleted": []}
    for t in targets:
        n, size = _dir_stats(t)
        record["deleted"].append({"path": str(t.relative_to(cfg.root)), "files": n, "bytes": size})
        if not dry_run:
            shutil.rmtree(t)
    remaining = [str(t.relative_to(cfg.root)) for t in purge_targets(cfg, provider)]
    record["remaining_after_purge"] = remaining
    record["verified_clean"] = not remaining if not dry_run else None
    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    record["kept"] = "retained/ and public/ output tiers (rankings, scores, ratios, summary statistics, charts)"
    if not dry_run:
        cfg.compliance_dir.mkdir(parents=True, exist_ok=True)
        log = cfg.compliance_dir / f"purge_{provider}_{datetime.now():%Y-%m-%d_%H%M%S}.json"
        log.write_text(json.dumps(record, indent=2))
        record["log"] = str(log.relative_to(cfg.root))
    return record


# ---------------------------------------------------------------------------
def _publishable_files(root: Path) -> List[Path]:
    """Files git would publish (tracked + untracked-not-ignored). Falls back to a .gitignore-aware walk."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, text=True, check=True)
        if out.stdout.strip() or (root / ".git").exists():
            return [root / line for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    patterns = []
    gi = root / ".gitignore"
    if gi.exists():
        patterns = [ln.strip() for ln in gi.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    def ignored(rel: str) -> bool:
        for pat in patterns:
            p = pat.rstrip("/")
            if pat.startswith("!"):
                continue
            if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p + "/*") or rel.startswith(p.lstrip("/") + "/") \
                    or fnmatch.fnmatch(os.path.basename(rel), p):
                return True
        return False

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [d for d in dirnames if d not in ALWAYS_SKIP_DIRS
                       and not ignored(os.path.normpath(os.path.join(rel_dir, d)).lstrip("./"))]
        for fn in filenames:
            rel = os.path.normpath(os.path.join(rel_dir, fn))
            rel = rel[2:] if rel.startswith("./") else rel
            if not ignored(rel):
                files.append(root / rel)
    return files


def check_public(cfg: Config) -> List[Dict[str, str]]:
    """Scan everything that would be published and report possible licensed raw data."""
    problems = []
    raw_cols = set(RAW_VALUE_FIELDS) | set(raw_value_metrics())
    licensed = {p for p in cfg.settings.get("providers", {}) if cfg.is_licensed(p)}
    for f in _publishable_files(cfg.root):
        if not f.is_file():
            continue
        rel = str(f.relative_to(cfg.root))
        ext = f.suffix.lower()
        if rel.startswith(("data/", "outputs/", "compliance/")):
            problems.append({"file": rel, "issue": "local data/outputs folder is not git-ignored"})
            continue
        if f.name == ".env":
            problems.append({"file": rel, "issue": ".env (API keys) would be published"})
            continue
        if ext in BINARY_DATA_EXT:
            problems.append({"file": rel, "issue": f"binary data file ({ext}) — could contain licensed data"})
            continue
        if ext not in TEXT_DATA_EXT:
            continue
        try:
            head = f.read_text(encoding="utf-8-sig", errors="ignore")[:20000]
        except Exception:
            continue
        first = head.splitlines()[0].lower() if head else ""
        cols = {c.strip().strip('"') for c in first.split(",")}
        if cols & SHARADAR_MARKERS:
            problems.append({"file": rel, "issue": "contains Sharadar raw column names"})
            continue
        # a sibling TIER.json / run_manifest.json tells us where the file came from
        prov = None
        for meta_name in ("TIER.json", "run_manifest.json"):
            meta = f.parent / meta_name
            if meta.exists():
                try:
                    prov = json.loads(meta.read_text()).get("provider")
                except Exception:
                    pass
        if prov in licensed and cols & raw_cols:
            problems.append({"file": rel, "issue": f"raw-value columns from licensed provider '{prov}'"})
        if prov is None and ext == ".csv" and len(cols & raw_cols) >= 3 and not rel.startswith("tests/"):
            problems.append({"file": rel, "issue": "CSV with raw financial columns and unknown provenance"})
    return problems


def export_portfolio(cfg: Config, run_id: Optional[str] = None) -> Path:
    """Copy a run's public tier into portfolio/<date>_<provider>/ (the folder meant for GitHub)."""
    runs = cfg.outputs_dir / "runs"
    if run_id is None:
        latest = cfg.outputs_dir / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("No runs yet. Run: python -m screener run")
        run_id = latest.read_text().strip()
    src = runs / run_id / "public"
    if not src.exists():
        raise FileNotFoundError(f"Run '{run_id}' has no public tier")
    manifest = json.loads((runs / run_id / "manifest.json").read_text())
    dest = cfg.root / "portfolio" / f"{manifest['data_as_of']}_{manifest['provider']}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("TIER.json"))
    return dest
