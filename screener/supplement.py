"""Fill fields a licensed provider lacks from a public-domain provider, at build time.

Sharadar has no real-estate impairment or gain-on-sale fields, so REIT FFO falls back to
net income + D&A. SEC filings tag both. With `supplement` configured for the provider
(config/settings.yaml), `build` fills those fields for the configured profiles from the
current SEC snapshot, matching fiscal years by period end. Only empty cells are filled;
the merged rows live in the licensed snapshot and are purged with it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from .config import Config
from .store import NoLocalData, Store

log = logging.getLogger(__name__)


def apply_supplement(cfg: Config, provider: str, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """(financials with supplemented fields, summary). Returns the input unchanged when not configured
    or when the source provider has no local snapshot."""
    fin = tables["financials"]
    spec = (cfg.provider_settings(provider).get("supplement") or {})
    if not spec:
        return fin, {}
    src_name = spec.get("from", "sec")
    fields = [f for f in spec.get("fields", []) if f in fin.columns]
    try:
        src = Store(cfg).current(src_name).read("financials")
    except NoLocalData:
        return fin, {"source": src_name, "status": "no local snapshot", "companies": 0, "values_filled": 0}
    from .profiles import assign_profiles

    prof = assign_profiles(tables["securities"], cfg.profiles)
    want = set(prof.index[prof.isin(spec.get("profiles", []))]) if spec.get("profiles") else set(prof.index)
    listed = set(tables["securities"].loc[~tables["securities"]["is_delisted"].astype(bool), "ticker"])
    gap = pd.Timedelta(days=int(spec.get("max_period_gap_days", 10)))

    tgt = fin[(fin["period_type"] == "FY") & fin["ticker"].isin(want)].copy()
    s = src[(src["period_type"] == "FY") & src["ticker"].isin(want)][["ticker", "period_end"] + fields]
    s = s.dropna(subset=fields, how="all")
    if tgt.empty or s.empty:
        return fin, {"source": src_name, "status": "ok", "companies": 0, "values_filled": 0}
    tgt = tgt.reset_index().rename(columns={"index": "_row"})
    tgt["period_end"] = pd.to_datetime(tgt["period_end"])
    s = s.assign(period_end=pd.to_datetime(s["period_end"])).rename(columns={f: f"_s_{f}" for f in fields})
    m = pd.merge_asof(tgt.sort_values("period_end"), s.sort_values("period_end"), on="period_end",
                      by="ticker", direction="nearest", tolerance=gap)
    out = fin.copy()
    filled, companies = 0, set()
    for f in fields:
        sv = m.set_index("_row")[f"_s_{f}"]
        need = out.loc[sv.index, f].isna() & sv.notna()
        idx = need[need].index
        out.loc[idx, f] = sv.loc[idx].values
        filled += len(idx)
        companies |= set(out.loc[idx, "ticker"])
    summary = {"source": src_name, "status": "ok", "fields": fields, "companies": len(companies),
               "values_filled": int(filled), "eligible_companies": len(want & listed),
               "listed_companies_filled": len(companies & listed)}
    log.info("Supplement from %s: %d values for %d companies", src_name, filled, len(companies))
    return out, summary
