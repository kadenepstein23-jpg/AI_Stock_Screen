"""Cross-provider validation: compare canonical financials of two providers (e.g. Sharadar vs SEC).

The report contains values from both providers, so when either is licensed the
report is written as a license-scoped tier and is removed by `purge`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .store import Store

DEFAULT_FIELDS = ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted",
                  "shares_diluted_wavg", "cash", "total_debt", "equity", "total_assets",
                  "operating_cash_flow", "capex", "free_cash_flow", "sbc", "dividends_paid", "dps"]


def compare_providers(cfg: Config, primary: str, reference: str, tickers: Optional[List[str]] = None,
                      fields: Optional[List[str]] = None, tolerance_pct: float = 2.0) -> Path:
    store = Store(cfg)
    a = store.current(primary).read("financials")
    b = store.current(reference).read("financials")
    a, b = a[a["period_type"] == "FY"], b[b["period_type"] == "FY"]
    if tickers:
        tk = {t.upper() for t in tickers}
        a, b = a[a["ticker"].isin(tk)], b[b["ticker"].isin(tk)]
    fields = fields or DEFAULT_FIELDS
    a = a.sort_values("period_end")
    b = b.sort_values("period_end")
    m = pd.merge_asof(a, b, on="period_end", by="ticker", suffixes=("_a", "_b"),
                      tolerance=pd.Timedelta(days=10), direction="nearest")
    rows = []
    for f in fields:
        va, vb = m[f"{f}_a"], m[f"{f}_b"]
        both = va.notna() & vb.notna()
        denom = vb.abs().where(vb.abs() > 0)
        diff = ((va - vb).abs() / denom * 100).where(both)
        for t, pe, x, y, d in zip(m["ticker"], m["period_end"], va, vb, diff):
            if pd.isna(x) and pd.isna(y):
                continue
            rows.append(dict(ticker=t, period_end=pe.date(), field=f, primary=x, reference=y,
                             abs_diff_pct=d, within_tolerance=(d <= tolerance_pct) if not pd.isna(d) else None))
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("field")
               .agg(compared=("abs_diff_pct", lambda s: int(s.notna().sum())),
                    within_tol=("within_tolerance", lambda s: int((s == True).sum())),  # noqa: E712
                    median_diff_pct=("abs_diff_pct", "median"),
                    only_primary=("reference", lambda s: int(s.isna().sum())))
               .reset_index()) if not detail.empty else pd.DataFrame()
    if not summary.empty:
        summary["match_rate_pct"] = (summary["within_tol"] / summary["compared"].replace(0, np.nan) * 100).round(1)

    licensed = cfg.is_licensed(primary) or cfg.is_licensed(reference)
    scope = primary if cfg.is_licensed(primary) else (reference if licensed else None)
    out = cfg.outputs_dir / "validation" / f"{datetime.now():%Y-%m-%d_%H%M%S}_{primary}_vs_{reference}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "TIER.json").write_text(json.dumps({"tier": "full", "provider": scope or primary,
                                               "license_scope": scope, "contains_raw_values": True}, indent=2))
    detail.to_csv(out / "validation_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "validation_summary.csv", index=False, encoding="utf-8-sig")
    # the summary (counts, match rates, median % differences) holds no raw values -> retained copy
    keep = out.parent / f"{out.name}_summary"
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "TIER.json").write_text(json.dumps({"tier": "retained", "provider": scope or primary,
                                                "license_scope": None, "contains_raw_values": False}, indent=2))
    summary.to_csv(keep / "validation_summary.csv", index=False, encoding="utf-8-sig")
    return out
