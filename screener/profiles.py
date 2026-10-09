"""Assign each security an industry profile (bank, insurance, reit, utility, cyclical, default)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _matches(rule: Dict[str, Any], sic, sector: str, industry: str) -> bool:
    if not rule:
        return True
    ind = (industry or "").lower()
    sec = (sector or "").lower()
    # exclusions first, so they also override a matching SIC range
    if any(p.lower() in ind for p in rule.get("exclude_industry_patterns", []) or []):
        return False
    for lo, hi in rule.get("exclude_sic_ranges", []) or []:
        if sic is not None and not pd.isna(sic) and lo <= int(sic) <= hi:
            return False
    # a SIC code can outlive the business it describes (CoreCivic still files as a REIT, SIC 6798,
    # after giving up REIT status): when the industry label is known it must agree with the SIC match
    need = [p.lower() for p in rule.get("sic_requires_industry_patterns", []) or []]
    for lo, hi in rule.get("sic_ranges", []) or []:
        if sic is not None and not pd.isna(sic) and lo <= int(sic) <= hi:
            if need and ind and not any(p in ind for p in need):
                break
            return True
    if any(p.lower() in ind for p in rule.get("industry_patterns", []) or []):
        return True
    if any(p.lower() in sec for p in rule.get("sector_patterns", []) or []):
        return True
    return False


def assign_profiles(securities: pd.DataFrame, profiles: Dict[str, Dict[str, Any]],
                    values: pd.DataFrame = None) -> pd.Series:
    """Industry profile per ticker. With `values` (computed metrics), a profile's `exit_rule` can move
    a company whose own record contradicts the industry label - e.g. a semiconductor company with
    years of uninterrupted, fast revenue growth is not treated as a cyclical. Moved tickers and the
    reason are recorded in `result.attrs["overrides"]`."""
    out = _assign_by_industry(securities, profiles)
    sic = pd.to_numeric(securities.set_index("ticker")["sic_code"], errors="coerce") \
        if "sic_code" in securities else pd.Series(dtype=float)
    ind = securities.set_index("ticker")["industry"].astype("string").fillna("") \
        if "industry" in securities else pd.Series(dtype="string")
    overrides = {}
    if values is not None:
        for name, cfg in profiles.items():
            rule = cfg.get("exit_rule")
            if not rule:
                continue
            for t in out.index[out == name]:
                if t not in values.index:
                    continue
                reason = _exit_reason(values.loc[t], rule, sic.get(t), str(ind.get(t, "") or ""))
                if reason:
                    out[t] = rule.get("to", "default")
                    overrides[t] = f"{name} industry, treated as {out[t]}: {reason}"
    out.attrs["overrides"] = overrides
    return out


def _exit_reason(row: pd.Series, rule: Dict[str, Any], sic=None, industry: str = ""):
    """All conditions must hold; returns a readable reason or None."""
    parts = []
    pats = [p.lower() for p in rule.get("industry_patterns", []) or []]
    if pats and not any(p in (industry or "").lower() for p in pats):
        return None
    ranges = rule.get("sic_ranges")
    if ranges:
        if sic is None or pd.isna(sic) or not any(lo <= int(sic) <= hi for lo, hi in ranges):
            return None
        parts.append(f"SIC {int(sic)}")
    for metric, cond in (rule.get("all") or {}).items():
        v = pd.to_numeric(row.get(metric), errors="coerce")
        if pd.isna(v):
            return None
        if "min" in cond and not v >= cond["min"]:
            return None
        if "max" in cond and not v <= cond["max"]:
            return None
        parts.append(f"{cond.get('label', metric)} {v:.1f}")
    return ", ".join(parts) if parts else None


def _assign_by_industry(securities: pd.DataFrame, profiles: Dict[str, Dict[str, Any]]) -> pd.Series:
    order = [k for k in profiles if k != "default"] + ["default"]
    out = {}
    for row in securities.itertuples(index=False):
        sector = row.sector if isinstance(row.sector, str) else ""
        industry = row.industry if isinstance(row.industry, str) else ""
        chosen = "default"
        for name in order:
            if name == "default":
                break
            if _matches(profiles[name].get("match", {}), row.sic_code, sector, industry):
                chosen = name
                break
        out[row.ticker] = chosen
    return pd.Series(out, name="profile")


def peer_group(securities: pd.DataFrame, profile: pd.Series, profiles: Dict[str, Dict[str, Any]]) -> pd.Series:
    """Peer group for relative ranking: special-industry profile, else sector."""
    sec = securities.set_index("ticker")
    sector = sec["sector"].astype("string").fillna("Unknown sector")
    special = profile.reindex(sec.index).fillna("default")
    is_special = ~special.isin(["default", "cyclical"])
    return pd.Series(special.where(is_special, sector), index=sec.index, name="peer_group")


def mask_not_applicable(values: pd.DataFrame, status: pd.DataFrame, profile: pd.Series,
                        profiles: Dict[str, Dict[str, Any]]):
    """Blank out metrics that do not apply to a company's industry profile.

    The value becomes NaN and the status becomes "na", so outputs and the company view show
    "n/a for <profile>" instead of a misleading number or a false "missing" data gap.
    """
    from .metrics.registry import PROFILE_ONLY

    values, status = values.copy(), status.copy()
    prof = profile.reindex(values.index).fillna("default")
    for metric, allowed in PROFILE_ONLY.items():
        if metric in values.columns:
            rows = prof.index[~prof.isin(allowed)]
            values.loc[rows, metric] = np.nan
            status.loc[rows, metric] = "na"
    for name, cfg in profiles.items():
        na = [m for m in (cfg.get("not_applicable") or []) if m in values.columns]
        rows = prof.index[prof == name]
        if not na or len(rows) == 0:
            continue
        values.loc[rows, na] = np.nan
        status.loc[rows, na] = "na"
    return values, status
