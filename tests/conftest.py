import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An isolated copy of the project config; data/ and outputs/ live under tmp_path."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    monkeypatch.setenv("SCREENER_HOME", str(tmp_path))
    from screener.config import load_config

    return load_config(tmp_path)


def fy_rows(ticker, years, **series):
    """Build canonical FY rows: series values are lists aligned with `years` (oldest first)."""
    rows = []
    for i, y in enumerate(years):
        r = {"ticker": ticker, "period_type": "FY", "period_end": pd.Timestamp(f"{y}-12-31"),
             "filing_date": pd.Timestamp(f"{y + 1}-02-15")}
        for k, v in series.items():
            r[k] = v[i]
        rows.append(r)
    return rows
