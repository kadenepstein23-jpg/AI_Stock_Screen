"""Multi-factor scoring.

Pipeline for one screen:
  1. gates           -> companies that qualify for this screen
  2. pillar specs    -> per industry profile: drop not-applicable metrics, apply substitutes/extras
  3. metric scores   -> percentile rank within peer group (blended with universe rank);
                        NM and missing values get configured scores, never silent zeros
  4. pillar scores   -> weighted average of metric scores; coverage tracked
  5. composite       -> weighted pillar average (0-100), minus moderate-flag penalties
  6. confidence      -> grade from data coverage; severe flags / grade below C -> needs review
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .metrics.registry import METRICS

OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


def validate_screen(spec: Dict[str, Any], profiles: Dict[str, Any]) -> List[str]:
    """Return a list of config problems (unknown metric names, bad directions)."""
    problems = []
    for g in spec.get("gates", []):
        if g["metric"] not in METRICS:
            problems.append(f"gate metric '{g['metric']}' is unknown")
    for pname, p in spec.get("pillars", {}).items():
        for mdef in p.get("metrics", []):
            if mdef["name"] not in METRICS:
                problems.append(f"pillar '{pname}': metric '{mdef['name']}' is unknown")
            if mdef.get("direction") not in ("higher", "lower"):
                problems.append(f"pillar '{pname}': metric '{mdef['name']}' needs direction higher|lower")
    for prof, pc in profiles.items():
        for name in pc.get("not_applicable", []) or []:
            if name not in METRICS:
                problems.append(f"profile '{prof}': not_applicable metric '{name}' is unknown")
        for n in ((pc.get("exit_rule") or {}).get("all") or {}):
            if n not in METRICS:
                problems.append(f"profile '{prof}': exit_rule metric '{n}' is unknown")
        to = (pc.get("exit_rule") or {}).get("to")
        if to and to not in profiles:
            problems.append(f"profile '{prof}': exit_rule target profile '{to}' is unknown")
        for key, n in (pc.get("flag_metrics") or {}).items():
            if n not in METRICS:
                problems.append(f"profile '{prof}': flag metric '{n}' ({key}) is unknown")
        for name in list(pc.get("not_scored") or []) + list((pc.get("metric_weights") or {}).keys()):
            if name not in METRICS:
                problems.append(f"profile '{prof}': metric '{name}' (not_scored/metric_weights) is unknown")
        for a, b in list((pc.get("substitutes") or {}).items()) + list((pc.get("score_with") or {}).items()):
            for n in (a, b):
                if n not in METRICS:
                    problems.append(f"profile '{prof}': substitute metric '{n}' is unknown")
    return problems


def pillar_specs_for_profile(spec: Dict[str, Any], prof_cfg: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Effective pillar metric lists for one industry profile."""
    na = set(prof_cfg.get("not_applicable", []) or [])
    subs = prof_cfg.get("substitutes", {}) or {}
    extras = prof_cfg.get("extra_metrics", {}) or {}
    # score_with: metric stays visible but is scored through another (e.g. P/E -> normalized P/E);
    # not_scored: visible, reference only; metric_weights: per-profile weight overrides
    score_with = prof_cfg.get("score_with", {}) or {}
    not_scored = set(prof_cfg.get("not_scored", []) or [])
    weights = prof_cfg.get("metric_weights", {}) or {}
    out: Dict[str, List[Dict]] = {}
    for pname, p in spec["pillars"].items():
        items: List[Dict] = []
        present = {m["name"] for m in p["metrics"]}
        for mdef in p["metrics"]:
            name = mdef["name"]
            if name in not_scored:
                continue
            if name in score_with and score_with[name] not in na:
                tgt = score_with[name]
                if tgt not in present and tgt not in {i["name"] for i in items}:
                    items.append(dict(mdef, name=tgt, substituted_for=name))
                continue
            if name in na:
                sub = subs.get(name)
                if sub and sub not in na and sub not in present and sub not in {i["name"] for i in items}:
                    items.append(dict(mdef, name=sub, substituted_for=name))
                continue
            items.append(dict(mdef))
        for extra in extras.get(pname, []) or []:
            if extra["name"] not in {i["name"] for i in items} and extra["name"] not in na:
                items.append(dict(extra))
        for it in items:
            if it["name"] in weights:
                it["weight"] = float(weights[it["name"]])
        out[pname] = items
    return out


def _mid_rank_pct(s: pd.Series, higher_better: bool) -> pd.Series:
    """Percentile in (0,1): best ~1, worst ~0, single member = 0.5."""
    s = s.dropna()
    if s.empty:
        return s
    r = s.rank(method="average", ascending=higher_better)
    return (r - 0.5) / len(s)


def apply_gates(values: pd.DataFrame, gates: List[Dict[str, Any]], tickers: pd.Index) -> Tuple[pd.Index, pd.DataFrame]:
    v = values.reindex(tickers)
    passed = pd.Series(True, index=tickers)
    failures = pd.Series("", index=tickers, dtype=object)
    for g in gates:
        col = pd.to_numeric(v[g["metric"]], errors="coerce")
        ok = OPS[g["op"]](col, g["value"]).fillna(False)
        failures[~ok] = failures[~ok] + f"{g['metric']} {g['op']} {g['value']}; "
        passed &= ok
    fail_df = pd.DataFrame({"ticker": failures[~passed].index, "failed_gates": failures[~passed].str.rstrip("; ").values})
    return passed[passed].index, fail_df


def score_screen(spec: Dict[str, Any], values: pd.DataFrame, status: pd.DataFrame, universe: pd.Index,
                 profile: pd.Series, peer: pd.Series, flags: pd.DataFrame, settings: Dict[str, Any],
                 profiles_cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    sc = settings["scoring"]
    eligible, gate_fail = apply_gates(values, spec.get("gates", []), universe)
    if len(eligible) == 0:
        empty = pd.DataFrame()
        return {"ranked": empty, "needs_review": empty, "gate_failures": gate_fail, "metric_scores": empty}

    prof = profile.reindex(eligible).fillna("default")
    peers = peer.reindex(eligible).fillna("Unknown")
    sector = peers.copy()
    specs_by_profile = {p: pillar_specs_for_profile(spec, profiles_cfg.get(p, {})) for p in prof.unique()}

    # every metric used by any profile in this screen
    used: Dict[str, str] = {}
    for specs in specs_by_profile.values():
        for items in specs.values():
            for it in items:
                used.setdefault(it["name"], it["direction"])

    # ---- metric percentile scores (peer-blended) -------------------------------------
    blend = float(sc.get("peer_blend", 0.7))
    min_group = int(sc.get("min_peer_group_size", 8))
    group_sizes = peers.value_counts()
    effective_peer = peers.where(peers.map(group_sizes) >= min_group, "__universe__")
    pct: Dict[str, pd.Series] = {}
    for name, direction in used.items():
        v = pd.to_numeric(values.reindex(eligible)[name], errors="coerce")
        st = status.reindex(eligible)[name]
        v = v.where(st == "ok")
        higher = direction == "higher"
        uni = _mid_rank_pct(v, higher).reindex(eligible)
        grp = pd.Series(np.nan, index=eligible)
        for gname, idx in effective_peer.groupby(effective_peer).groups.items():
            if gname == "__universe__":
                continue
            grp.loc[idx] = _mid_rank_pct(v.loc[idx], higher).reindex(idx)
        pct[name] = (blend * grp + (1 - blend) * uni).fillna(uni)

    miss_score = float(sc.get("missing_applicable_score", 0.35))
    nm_lower = float(sc.get("not_meaningful_score_lower", 0.0))
    nm_higher = float(sc.get("not_meaningful_score_higher", 0.25))

    # ---- per company aggregation -----------------------------------------------------
    rows = []
    metric_rows = []
    pillar_w = {p: float(cfg["weight"]) for p, cfg in spec["pillars"].items()}
    for t in eligible:
        p_specs = specs_by_profile[prof[t]]
        pillar_scores, pillar_cov = {}, {}
        missing_list = []
        for pname, items in p_specs.items():
            if not items:
                continue
            wsum = covered = acc = 0.0
            for it in items:
                name, w = it["name"], float(it.get("weight", 1.0))
                st = status.at[t, name] if name in status.columns else "missing"
                if st == "ok" and not pd.isna(pct[name].get(t)):
                    s = float(pct[name][t])
                    covered += w
                elif st == "nm":
                    s = float(it.get("nm_score", nm_higher if it["direction"] == "higher" else nm_lower))
                    covered += w
                else:
                    s = miss_score
                    missing_list.append(name)
                acc += w * s
                wsum += w
                metric_rows.append((t, pname, name, it["direction"], st, s, it.get("substituted_for")))
            pillar_scores[pname] = acc / wsum
            pillar_cov[pname] = covered / wsum
        tw = sum(pillar_w[p] for p in pillar_scores)
        composite = 100.0 * sum(pillar_w[p] * pillar_scores[p] for p in pillar_scores) / tw
        coverage = sum(pillar_w[p] * pillar_cov[p] for p in pillar_cov) / tw
        low_conf = [p for p, c in pillar_cov.items() if c < sc.get("min_pillar_coverage", 0.6)]
        row = dict(ticker=t, profile=prof[t], peer_group=sector[t], score_before_penalty=composite,
                   data_coverage_pct=coverage * 100, low_confidence_pillars=", ".join(low_conf),
                   missing_metrics=", ".join(sorted(set(missing_list))))
        for p in spec["pillars"]:
            row[f"pillar_{p}"] = pillar_scores[p] * 100 if p in pillar_scores else np.nan
        rows.append(row)
    res = pd.DataFrame(rows).set_index("ticker")

    # ---- flags with this screen's severities ------------------------------------------
    overrides = spec.get("flag_severity", {}) or {}
    f = flags[flags["ticker"].isin(eligible)].copy()
    # A screen override can change a flag's severity, but it never escalates a row the detector
    # deliberately downgraded to "info" (accretive REIT issuance, SBC already covered by dilution).
    down = f["downgraded"] if "downgraded" in f else pd.Series(False, index=f.index)
    f["severity"] = [ds if dg and overrides.get(fl) in ("moderate", "severe") else overrides.get(fl, ds)
                     for fl, ds, dg in zip(f["flag"], f["default_severity"], down.fillna(False))]
    f = f[f["severity"] != "ignore"]
    sev = f.groupby(["ticker", "severity"]).size().unstack(fill_value=0)
    for c in ("info", "moderate", "severe"):
        if c not in sev:
            sev[c] = 0
    sev = sev.reindex(res.index).fillna(0).astype(int)
    res["n_moderate_flags"] = sev["moderate"]
    res["n_severe_flags"] = sev["severe"]
    res["flags"] = f.groupby("ticker")["flag"].apply(lambda s: ", ".join(s)).reindex(res.index).fillna("")
    res["severe_flags"] = (f[f["severity"] == "severe"].groupby("ticker")["flag"]
                           .apply(lambda s: ", ".join(s)).reindex(res.index).fillna(""))
    penalty = (res["n_moderate_flags"] * float(sc.get("moderate_flag_penalty", 4.0))).clip(
        upper=float(sc.get("max_flag_penalty", 20.0)))
    res["flag_penalty"] = penalty
    res["composite_score"] = (res["score_before_penalty"] - penalty).clip(lower=0)

    grades = sc.get("confidence_grades", {"A": 0.9, "B": 0.75, "C": 0.6})
    cov = res["data_coverage_pct"] / 100

    def grade(c):
        for g, th in sorted(grades.items(), key=lambda kv: -kv[1]):
            if c >= th:
                return g
        return "D"

    res["confidence_grade"] = cov.map(grade)
    review_reason = pd.Series("", index=res.index, dtype=object)
    if sc.get("severe_flags_exclude", True):
        review_reason[res["n_severe_flags"] > 0] = "severe flag: " + res["severe_flags"]
    review_reason[(res["confidence_grade"] == "D") & (review_reason == "")] = "insufficient data coverage"
    res["review_reason"] = review_reason
    ranked = res[review_reason == ""].sort_values("composite_score", ascending=False).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["rank_in_peer_group"] = ranked.groupby("peer_group")["composite_score"].rank(ascending=False, method="min").astype(int)
    review = res[review_reason != ""].sort_values("score_before_penalty", ascending=False).copy()
    ms = pd.DataFrame(metric_rows, columns=["ticker", "pillar", "metric", "direction", "status", "score", "substituted_for"])
    return {"ranked": ranked.reset_index(), "needs_review": review.reset_index(),
            "gate_failures": gate_fail, "metric_scores": ms, "screen_flags": f}
