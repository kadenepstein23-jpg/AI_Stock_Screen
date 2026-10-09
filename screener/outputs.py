"""Run outputs, written in three tiers.

outputs/runs/<run_id>/
    manifest.json            what produced this run (provider, snapshot, config hash)
    full/      every column incl. raw figures (cash, debt, market cap, price).
               For licensed providers this tier is LICENSE-SCOPED and is deleted by `purge`.
    retained/  rankings, scores, pillar scores, ratios, growth rates, flags, summary
               statistics and charts — no raw figures. Kept after a purge
               (Sharadar license §10 allows keeping research outputs / summary statistics).
    public/    portfolio-safe: ranks, scores, grades and flag names for the top N,
               plus sector summary statistics, charts and attribution. Copy to
               portfolio/ with `python -m screener export-portfolio`.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .config import Config
from .metrics.registry import METRICS, raw_value_metrics

ID_COLS = ["ticker", "name", "sector", "industry", "profile", "peer_group"]
SCORE_COLS = ["rank", "composite_score", "score_before_penalty", "flag_penalty", "confidence_grade",
              "data_coverage_pct", "rank_in_peer_group", "n_moderate_flags", "n_severe_flags", "flags",
              "severe_flags", "low_confidence_pillars", "missing_metrics", "review_reason"]
CONTEXT_METRICS = ["market_cap_musd", "price", "dividend_yield_pct", "pe_x", "ev_ebitda_x", "p_fcf_x",
                   "net_debt_to_ebitda_x", "roic_pct", "revenue_cagr_3y_pct", "eps_trend_growth_pct",
                   "fcf_margin_pct", "operating_margin_pct", "latest_period_end", "fiscal_years_available"]

# chart styling (validated categorical palette, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _screen_metric_names(spec: Dict[str, Any], profiles: Dict[str, Any]) -> List[str]:
    names = []
    for p in spec["pillars"].values():
        names += [m["name"] for m in p["metrics"]]
    for prof in profiles.values():
        names += list((prof.get("substitutes") or {}).values())
        for extras in (prof.get("extra_metrics") or {}).values():
            names += [e["name"] for e in extras]
    seen, out = set(), []
    for n in CONTEXT_METRICS + names:
        if n not in seen and n in METRICS:
            seen.add(n)
            out.append(n)
    return out


def _to_csv(df: pd.DataFrame, path: Path, cfg: Config) -> None:
    out = df.copy()
    dec = int(cfg.settings["outputs"].get("float_decimals", 4))
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(dec)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding=cfg.settings["outputs"].get("csv_encoding", "utf-8-sig"))


def _tier_manifest(path: Path, tier: str, provider: str, licensed: bool, attribution: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "TIER.json").write_text(json.dumps({
        "tier": tier,
        "provider": provider,
        "license_scope": provider if (licensed and tier == "full") else None,
        "contains_raw_values": tier == "full",
        "attribution": attribution,
    }, indent=2))


def _charts(results: Dict[str, Dict[str, pd.DataFrame]], cfg: Config, dest: Path, funnel: Dict[str, int],
            attribution: str) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional
        return []
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    dest.mkdir(parents=True, exist_ok=True)
    made = []

    # 1) universe funnel (single series, ordinal)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, vals = list(funnel.keys()), list(funnel.values())
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:,}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Screening funnel: securities to ranked candidates", loc="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, attribution, color=INK2, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(dest / "universe_funnel.png", dpi=160)
    plt.close(fig)
    made.append("universe_funnel.png")

    for screen, res in results.items():
        ranked = res["ranked"]
        if ranked.empty:
            continue
        spec = cfg.screens[screen]
        pillars = list(spec["pillars"].keys())
        weights = np.array([float(spec["pillars"][p]["weight"]) for p in pillars])
        top = ranked.head(20).iloc[::-1]
        # weighted contribution of each pillar to the 0-100 score (before penalties)
        contrib = top[[f"pillar_{p}" for p in pillars]].fillna(0).values * weights / weights.sum()
        fig, ax = plt.subplots(figsize=(8.5, 6))
        left = np.zeros(len(top))
        for i, p in enumerate(pillars):
            ax.barh(np.arange(len(top)), contrib[:, i], left=left, height=0.62,
                    color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=2,
                    label=p.replace("_", " ").title())
            left += contrib[:, i]
        for yi, (sc_, pen) in enumerate(zip(top["composite_score"], top["flag_penalty"])):
            label = f"  {sc_:.1f}" + (f"  (after -{pen:.0f} flag penalty)" if pen > 0 else "")
            ax.text(left[yi], yi, label, va="center", color=INK, fontsize=8)
        ax.set_yticks(np.arange(len(top)), [f"{r}. {t}" for r, t in zip(top["rank"], top["ticker"])])
        ax.set_xlabel("Weighted pillar contribution (bar length = score before flag penalties; label = final score)")
        ax.set_xlim(0, max(left.max() * 1.35, 10))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=min(len(pillars), 6),
                  frameon=False, fontsize=8)
        ax.set_title(f"{spec.get('label', screen)}: top 20 by composite score", loc="left", color=INK, fontsize=11)
        fig.text(0.01, 0.005, attribution, color=INK2, fontsize=7)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(dest / f"{screen}_top20_pillars.png", dpi=160)
        plt.close(fig)
        made.append(f"{screen}_top20_pillars.png")

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(ranked["composite_score"], bins=np.arange(0, 101, 5), color=SERIES[0],
                edgecolor=SURFACE, linewidth=2)
        ax.set_xlabel("Composite score")
        ax.set_ylabel("Companies")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{spec.get('label', screen)}: score distribution ({len(ranked):,} ranked)",
                     loc="left", color=INK, fontsize=11)
        fig.text(0.01, 0.01, attribution, color=INK2, fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(dest / f"{screen}_score_distribution.png", dpi=160)
        plt.close(fig)
        made.append(f"{screen}_score_distribution.png")
    return made


def write_run(cfg: Config, snap, results: Dict[str, Dict[str, pd.DataFrame]], values: pd.DataFrame,
              status: pd.DataFrame, flags: pd.DataFrame, excluded: pd.DataFrame, universe: pd.Index,
              profile: pd.Series, securities: pd.DataFrame, run_id: Optional[str] = None) -> Path:
    provider = snap.provider
    licensed = cfg.is_licensed(provider)
    lic = cfg.provider_settings(provider).get("license", {})
    attribution = lic.get("attribution", "")
    run_id = run_id or f"{datetime.now():%Y-%m-%d_%H%M%S}_{provider}"
    root = cfg.outputs_dir / "runs" / run_id
    full, retained, public = root / "full", root / "retained", root / "public"
    for tier, path in (("full", full), ("retained", retained), ("public", public)):
        _tier_manifest(path, tier, provider, licensed, attribution)

    raw_cols = set(raw_value_metrics()) if licensed else set()
    sec = securities.drop_duplicates("ticker").set_index("ticker")[["name", "sector", "industry"]]

    def enrich(df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.merge(sec, left_on="ticker", right_index=True, how="left")
        mv = values.reindex(d["ticker"])[metric_names].reset_index(drop=True)
        d = pd.concat([d.reset_index(drop=True), mv], axis=1)
        pillar_cols = [c for c in d.columns if c.startswith("pillar_")]
        order = [c for c in ID_COLS + ["rank"] + SCORE_COLS + pillar_cols + metric_names if c in d.columns]
        return d[list(dict.fromkeys(order))]

    summary: Dict[str, Any] = {}
    for screen, res in results.items():
        spec = cfg.screens[screen]
        mnames = _screen_metric_names(spec, cfg.profiles)
        ranked = enrich(res["ranked"], mnames)
        review = enrich(res["needs_review"], mnames)
        _to_csv(ranked, full / f"{screen}_screen.csv", cfg)
        _to_csv(review, full / f"{screen}_needs_review.csv", cfg)
        _to_csv(res["metric_scores"], full / f"{screen}_metric_scores.csv", cfg)
        _to_csv(res["gate_failures"], full / f"{screen}_gate_failures.csv", cfg)
        keep = [c for c in ranked.columns if c not in raw_cols]
        _to_csv(ranked[keep], retained / f"{screen}_screen.csv", cfg)
        _to_csv(review[[c for c in review.columns if c not in raw_cols]] if not review.empty else review,
                retained / f"{screen}_needs_review.csv", cfg)
        _to_csv(res["metric_scores"], retained / f"{screen}_metric_scores.csv", cfg)
        pub_cols = [c for c in ["rank", "ticker", "name", "sector", "profile", "composite_score",
                                "confidence_grade", "data_coverage_pct", "flags"] if c in ranked.columns]
        pub_cols += [c for c in ranked.columns if c.startswith("pillar_")]
        top_n = int(cfg.settings["outputs"].get("public_top_n", 25))
        _to_csv(ranked[pub_cols].head(top_n) if not ranked.empty else ranked,
                public / f"{screen}_top{top_n}.csv", cfg)
        summary[screen] = {"gate_eligible": int(len(res["ranked"]) + len(res["needs_review"])),
                           "ranked": int(len(res["ranked"])), "needs_review": int(len(res["needs_review"]))}

    # all metrics (full) and ratio-only metrics (retained)
    allm = values.copy()
    allm.insert(0, "in_universe", allm.index.isin(universe))
    allm.insert(1, "profile", profile.reindex(allm.index))
    allm = allm.reset_index().merge(sec, left_on="ticker", right_index=True, how="left")
    front = ["ticker", "name", "sector", "industry", "profile", "in_universe"]
    allm = allm[front + [c for c in allm.columns if c not in front]]
    _to_csv(allm, full / "all_metrics.csv", cfg)
    _to_csv(status.reset_index(), full / "metric_status.csv", cfg)
    _to_csv(allm[[c for c in allm.columns if c not in raw_cols]], retained / "all_metrics_ratios.csv", cfg)
    fl = flags.merge(sec[["name"]], left_on="ticker", right_index=True, how="left")
    _to_csv(fl, full / "red_flags.csv", cfg)
    _to_csv(fl, retained / "red_flags.csv", cfg)
    _to_csv(excluded, full / "excluded.csv", cfg)
    _to_csv(excluded[["ticker", "reason"]], retained / "excluded.csv", cfg)

    # summary statistics by sector (medians of derived metrics; explicitly retainable)
    stat_cols = [c for c in ["revenue_cagr_3y_pct", "eps_trend_growth_pct", "operating_margin_pct", "fcf_margin_pct",
                             "roic_pct", "net_debt_to_ebitda_x", "pe_x", "ev_ebitda_x", "dividend_yield_pct",
                             "fcf_payout_3y_pct"] if c in values.columns]
    u = values.reindex(universe)[stat_cols].apply(pd.to_numeric, errors="coerce")
    u["sector"] = sec["sector"].reindex(universe).fillna("Unknown").values
    sector_summary = u.groupby("sector").median().round(2)
    sector_summary.insert(0, "companies", u.groupby("sector").size())
    sector_summary = sector_summary.reset_index()
    for tier_dir in (full, retained, public):
        _to_csv(sector_summary, tier_dir / "sector_summary.csv", cfg)

    funnel = {"Securities in dataset": int(len(sec)), "Investable universe": int(len(universe))}
    for screen, s in summary.items():
        funnel[f"{cfg.screens[screen].get('label', screen)}: ranked"] = s["ranked"]
    exclusion_counts = excluded["reason"].value_counts().to_dict() if not excluded.empty else {}
    uni_summary = {"securities": int(len(sec)), "investable_universe": int(len(universe)),
                   "excluded_by_first_reason": exclusion_counts, "screens": summary}
    for tier_dir in (full, retained, public):
        (tier_dir / "universe_summary.json").write_text(json.dumps(uni_summary, indent=2))

    if cfg.settings["outputs"].get("charts", True):
        made = _charts(results, cfg, retained / "charts", funnel, attribution)
        if made:
            shutil.copytree(retained / "charts", public / "charts", dirs_exist_ok=True)

    manifest = {
        "run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider, "snapshot_id": snap.snapshot_id, "data_as_of": str(snap.as_of.date()),
        "license_retention": cfg.retention(provider), "attribution": attribution,
        "config_fingerprint": cfg.fingerprint(), "screener_version": __version__,
        "universe": uni_summary,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for tier_dir in (retained, public):
        (tier_dir / "run_manifest.json").write_text(json.dumps(
            {k: v for k, v in manifest.items() if k != "snapshot_id"}, indent=2))
    readme = (f"# Screen results — {snap.as_of.date()}\n\n"
              f"Data as of **{snap.as_of.date()}** · provider: {provider} · run `{run_id}`\n\n"
              f"{attribution}" + (f" — {lic.get('attribution_url')}" if lic.get("attribution_url") else "") +
              "\n\nRanks, composite scores (0-100), pillar scores and red-flag names only. "
              "Scores are screening outputs, not investment recommendations.\n")
    (public / "README.md").write_text(readme)
    (cfg.outputs_dir / "LATEST").write_text(run_id)
    return root
