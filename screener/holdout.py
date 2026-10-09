"""Methodology freeze and holdout (anti-overfitting) validation.

Workflow (docs/HOLDOUT_PROTOCOL.md):
  1. `freeze`          records the methodology fingerprint (flags, profiles, screens, scoring and
                       universe rules). After this, only objective data/parsing/classification/
                       period-matching/calculation fixes are allowed; `run` and `holdout evaluate`
                       report whether the methodology still matches the freeze.
  2. `holdout draw`    builds the DEVELOPMENT set (every company used to develop or debug the
                       screener) and draws a stratified random HOLDOUT from the rest of the
                       investable universe. Written once to config/holdout_tickers.txt.
  3. `holdout evaluate` runs the pre-registered checks on the holdout, comparing it with the
                       development set and the rest of the universe. Nothing is tuned here.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import Config

FREEZE_FILE = "METHODOLOGY_FREEZE.json"
HOLDOUT_FILE = "holdout_tickers.txt"
DEV_FILE = "development_tickers.txt"

# words in docs/tests that look like tickers but are financial acronyms (over-inclusion is safe:
# a company wrongly put in the development set is only left out of the holdout)
_STOP = {"FFO", "AFFO", "EPS", "SEC", "FCF", "ROE", "ROA", "ROIC", "CFO", "TTM", "FY", "USD", "NAREIT", "REIT",
         "REITS", "EBIT", "EBITDA", "GAAP", "DPS", "PEG", "CAGR", "ETF", "ETFS", "ADR", "ADRS", "SIC", "NM", "NA",
         "OK", "CSV", "API", "AI", "US", "CEO", "IPO", "PE", "PB", "EV", "SBC", "MRY", "MRT", "ARY", "ART", "MRQ",
         "DJIA", "NYSE", "NASDAQ", "TODO", "NOTE", "PASS", "FAIL", "WARN", "JSON", "YAML", "URL", "HTTP", "HTTPS",
         "CP", "LP", "TBV", "BVPS", "AOCI", "OCI", "CET", "NIM", "PP", "YOY", "QOQ", "MD", "PNG", "PDF", "README",
         "MIT", "CLI", "OTC", "BATS", "TXSE", "NYSEMKT", "NYSEARCA", "UTC", "IRS", "LLC", "INC", "CORP", "THE",
         "AND", "FOR", "NOT", "ALL", "ANY", "ONE", "TWO", "NEW", "OLD", "MAX", "MIN", "KEY", "SUM", "AVG", "STD"}


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------
def methodology_fingerprint(cfg: Config) -> str:
    """Hash of everything that decides WHICH companies rank where: flags, industry profiles, screens,
    scoring and universe rules (not provider/download settings)."""
    blob = json.dumps({"flags": cfg.flags, "profiles": cfg.profiles, "screens": cfg.screens,
                       "scoring": cfg.settings.get("scoring"), "universe": cfg.settings.get("universe")},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def freeze(cfg: Config, note: str = "") -> Dict[str, Any]:
    path = cfg.root / "config" / FREEZE_FILE
    rec = {"methodology_fingerprint": methodology_fingerprint(cfg),
           "frozen_at": datetime.now().isoformat(timespec="seconds"), "note": note,
           "rules": "After the freeze: no changes to weights, thresholds, profile rules or company-specific "
                    "exceptions. Only objective data, parsing, classification, period-matching or calculation "
                    "fixes. Any methodology change requires a new freeze AND a new, unused holdout."}
    if path.exists():
        rec["previous"] = json.loads(path.read_text())
    path.write_text(json.dumps(rec, indent=2))
    return rec


def freeze_status(cfg: Config) -> Tuple[str, Optional[Dict[str, Any]]]:
    """('frozen', rec) | ('modified', rec) | ('not frozen', None)"""
    path = cfg.root / "config" / FREEZE_FILE
    if not path.exists():
        return "not frozen", None
    rec = json.loads(path.read_text())
    return ("frozen" if rec.get("methodology_fingerprint") == methodology_fingerprint(cfg) else "modified"), rec


# ---------------------------------------------------------------------------
# development set and holdout draw
# ---------------------------------------------------------------------------
def _read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        out += [t.strip().upper() for t in re.split(r"[,\s]+", ln) if t.strip()]
    return out


def development_set(cfg: Config, known: Set[str]) -> Dict[str, Set[str]]:
    """Every company used to develop or debug the screener, by source. `known` = tickers in the data
    (a word in a document only counts if it is a real ticker)."""
    from .diagnostics import DEFAULT_SPOT_CHECKS
    from .providers.sharadar import DJIA_SAMPLE

    src: Dict[str, Set[str]] = {
        "explicit list (config/development_tickers.txt)": set(_read_list(cfg.root / "config" / DEV_FILE)),
        "validation tickers": set(_read_list(cfg.root / "config" / "validation_tickers.txt")),
        "Sharadar free sample (DJIA 30)": set(DJIA_SAMPLE),
        "universe spot checks": set(DEFAULT_SPOT_CHECKS),
    }
    pat = re.compile(r"(?<![A-Za-z0-9_$])([A-Z]{2,5}(?:\.[A-Z])?)(?![a-z0-9_])")
    named: Set[str] = set()
    for folder, glob in (("docs", "*.md"), ("tests", "*.py"), ("config", "*.yaml"), ("screener", "*.py")):
        for p in (cfg.root / folder).rglob(glob):
            if p.name in (HOLDOUT_FILE,):
                continue
            named |= {m for m in pat.findall(p.read_text(errors="ignore")) if m not in _STOP}
    src["named in docs, tests, config or code"] = named & known
    return src


def draw_holdout(cfg: Config, provider: str, n: int = 150, seed: int = 20261008,
                 force: bool = False) -> Dict[str, Any]:
    path = cfg.root / "config" / HOLDOUT_FILE
    if path.exists() and not force:
        raise RuntimeError(f"{path.relative_to(cfg.root)} already exists. A holdout is drawn once; evaluating "
                           "and then redrawing would defeat its purpose. Use --force only for a NEW freeze.")
    status, rec = freeze_status(cfg)
    if status != "frozen":
        raise RuntimeError(f"Methodology is {status}: run `python -m screener freeze` first.")
    ctx = _context(cfg, provider)
    universe, profile, values = ctx["universe"], ctx["profile"], ctx["values"]
    known = set(ctx["tables"]["securities"]["ticker"])
    dev_src = development_set(cfg, known)
    dev = set().union(*dev_src.values())
    pool = pd.Index([t for t in universe if t not in dev])
    mc = pd.to_numeric(values["market_cap_musd"].reindex(pool), errors="coerce")
    size = pd.cut(mc, [0, 2_000, 10_000, float("inf")], labels=["small", "mid", "large"]).astype(str)
    strata = profile.reindex(pool).fillna("default") + "|" + size
    rng = np.random.default_rng(seed)
    counts = strata.value_counts()
    alloc = (counts / counts.sum() * n).round().astype(int).clip(lower=1)
    # every special profile gets at least 3 companies (if it has them) so its rules are tested at all
    for prof in profile.reindex(pool).unique():
        cells = [c for c in counts.index if c.startswith(f"{prof}|")]
        short = 3 - alloc[cells].sum()
        if short > 0 and cells:
            big = max(cells, key=lambda c: counts[c])
            alloc[big] = min(counts[big], alloc[big] + short)
    picks: List[str] = []
    for cell, k in alloc.items():
        members = sorted(pool[strata.values == cell])
        k = min(k, len(members))
        picks += list(rng.choice(members, size=k, replace=False))
    picks = sorted(str(p) for p in picks)
    body = "\n".join(picks)
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    header = [
        "# HOLDOUT SET - companies NOT used to develop or debug the screener.",
        "# Do not inspect, debug or tune on these names before `python -m screener holdout evaluate` has run.",
        f"# drawn_at: {datetime.now().isoformat(timespec='seconds')}",
        f"# provider_snapshot: {ctx['snap'].snapshot_id}",
        f"# methodology_fingerprint: {rec['methodology_fingerprint']}",
        f"# seed: {seed}",
        f"# size: {len(picks)}",
        f"# list_sha256: {digest}",
        f"# pool: {len(pool)} investable companies after removing {len(dev & set(universe))} development-set names",
        "# stratified by industry profile x size (small <$2B, mid $2-10B, large >$10B); >=3 per special profile",
    ]
    path.write_text("\n".join(header) + "\n" + body + "\n")
    dev_path = cfg.root / "config" / "development_set_resolved.txt"
    lines = ["# Resolved development set at holdout draw (union of the sources below). Generated - do not edit."]
    for k, v in dev_src.items():
        lines.append(f"# {k}: {len(v)}")
    dev_path.write_text("\n".join(lines) + "\n" + "\n".join(sorted(dev)) + "\n")
    by = (profile.reindex(picks).value_counts().to_dict(), size.reindex(picks).value_counts().to_dict())
    return {"path": path, "n": len(picks), "pool": len(pool), "dev_in_universe": len(dev & set(universe)),
            "dev_sources": {k: len(v) for k, v in dev_src.items()}, "by_profile": by[0], "by_size": by[1],
            "sha": digest}


def read_holdout(cfg: Config) -> Tuple[List[str], Dict[str, str]]:
    path = cfg.root / "config" / HOLDOUT_FILE
    if not path.exists():
        raise RuntimeError("No holdout drawn yet: python -m screener holdout draw")
    meta, tickers = {}, []
    for ln in path.read_text().splitlines():
        if ln.startswith("#"):
            if ":" in ln:
                k, v = ln[1:].split(":", 1)
                meta[k.strip()] = v.strip()
        elif ln.strip():
            tickers.append(ln.strip().upper())
    meta["sha_ok"] = hashlib.sha256("\n".join(tickers).encode()).hexdigest()[:16] == meta.get("list_sha256")
    return tickers, meta


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _context(cfg: Config, provider: str) -> Dict[str, Any]:
    from .flags import detect_flags
    from .pipeline import ensure_metrics, load_tables
    from .profiles import assign_profiles, mask_not_applicable, peer_group
    from .universe import build_universe

    snap = ensure_metrics(cfg, provider)
    tables = load_tables(snap)
    values = snap.read("metrics").set_index("ticker")
    status = snap.read("metric_status").set_index("ticker")
    profile = assign_profiles(tables["securities"], cfg.profiles, values)
    values, status = mask_not_applicable(values, status, profile, cfg.profiles)
    universe, excluded = build_universe(tables["securities"], tables["financials"], values, cfg.settings["universe"])
    peer = peer_group(tables["securities"], profile, cfg.profiles)
    flags = detect_flags(values, profile, peer, cfg.flags, cfg.profiles, universe, status)
    return dict(snap=snap, tables=tables, values=values, status=status, profile=profile, universe=universe,
                peer=peer, flags=flags)


def _score(cfg: Config, ctx: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    from .scoring import score_screen

    return score_screen(spec, ctx["values"], ctx["status"], ctx["universe"], ctx["profile"], ctx["peer"],
                        ctx["flags"], cfg.settings, cfg.profiles)


def _z(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 == 0 or n2 == 0 or np.isnan(p1) or np.isnan(p2):
        return float("nan")
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float((p1 - p2) / se) if se > 0 else 0.0


# objective industry-label consistency (classification audit)
_PROFILE_LABELS = {
    "reit": ["reit"],
    "utility": ["utilities"],
    "bank": ["bank", "capital markets", "credit services", "mortgage", "asset management", "savings", "thrift"],
    "insurance": ["insurance"],
}


def evaluate(cfg: Config, provider: str, reference: str = "sec", audit_n: int = 20,
             allow_modified: bool = False) -> Dict[str, Any]:
    status, rec = freeze_status(cfg)
    if status != "frozen" and not allow_modified:
        raise RuntimeError(f"Methodology is {status} relative to the freeze: the holdout must be evaluated "
                           "on the frozen methodology (re-freeze AND draw a new holdout after a change).")
    hold, meta = read_holdout(cfg)
    ctx = _context(cfg, provider)
    universe, profile, values, flags = ctx["universe"], ctx["profile"], ctx["values"], ctx["flags"]
    known = set(ctx["tables"]["securities"]["ticker"])
    dev = set().union(*development_set(cfg, known).values())
    U = set(universe)
    H = [t for t in hold if t in U]
    D = sorted(dev & U)
    R = sorted(U - dev - set(hold))                       # rest of the universe (neither set)
    checks: List[Dict[str, Any]] = []

    def check(cid, name, ok, detail, kind="criterion"):
        res = ("PASS" if ok else "FAIL") if ok is not None else ("TODO" if kind == "manual" else "N/A")
        checks.append(dict(id=cid, check=name, result=res, detail=detail, kind=kind))

    check("H0", "holdout list intact and still investable", meta["sha_ok"] and len(H) >= 0.9 * len(hold),
          f"list checksum {'matches' if meta['sha_ok'] else 'DOES NOT MATCH (list edited)'}; "
          f"{len(H)} of {len(hold)} holdout names are in the current universe")
    leaked = sorted(set(hold) & dev)
    check("H0b", "holdout not contaminated (no holdout name used in development since the draw)", not leaked,
          ", ".join(leaked) or "none of the holdout names appear in docs, tests, config, code or the development list")
    dev = dev - set(hold)

    # ---- H1 data accuracy vs SEC (objective) -----------------------------------------
    core = ["revenue", "net_income", "operating_cash_flow", "total_assets", "equity", "dividends_paid",
            "eps_diluted", "shares_diluted_wavg"]
    acc = {}
    try:
        from .store import Store

        a = Store(cfg).current(provider).read("financials")
        b = Store(cfg).current(reference).read("financials")

        def match_rates(tk):
            aa = a[(a["period_type"] == "FY") & a["ticker"].isin(tk)].sort_values("period_end")
            bb = b[(b["period_type"] == "FY") & b["ticker"].isin(tk)].sort_values("period_end")
            if aa.empty or bb.empty:
                return {}, 0
            m = pd.merge_asof(aa, bb, on="period_end", by="ticker", suffixes=("_a", "_b"),
                              tolerance=pd.Timedelta(days=10), direction="nearest")
            out = {}
            for f in core:
                va, vb = m[f"{f}_a"], m[f"{f}_b"]
                both = va.notna() & vb.notna() & (vb.abs() > 0)
                if both.sum() == 0:
                    continue
                out[f] = float(((va - vb).abs() / vb.abs() * 100 <= 2.0)[both].mean() * 100)
            return out, int(m.loc[m[[f"{c}_b" for c in core]].notna().any(axis=1), "ticker"].nunique())

        acc["holdout"], nh = match_rates(H)
        acc["development"], nd = match_rates(D)
        if nh < 0.5 * len(H):
            check("H1", "data accuracy vs SEC (holdout vs development set)", None,
                  f"SEC data for only {nh} of {len(H)} holdout companies: refresh it first "
                  f"(refresh --provider sec --tickers-file config/holdout_tickers.txt ...)")
        else:
            gaps = {f: acc["holdout"][f] - acc["development"].get(f, np.nan) for f in acc["holdout"]}
            worst = min(gaps, key=lambda f: gaps[f] if not np.isnan(gaps[f]) else 1e9)
            ok = all((np.isnan(g) or g >= -10.0) for g in gaps.values()) and \
                min(acc["holdout"].values()) >= 80.0
            check("H1", "data accuracy vs SEC (holdout vs development set)", ok,
                  f"{nh} holdout companies with SEC data; match within 2%: " +
                  ", ".join(f"{f} {acc['holdout'][f]:.0f}% (dev {acc['development'].get(f, float('nan')):.0f}%)"
                            for f in acc["holdout"]) + f"; largest shortfall {worst} {gaps[worst]:+.0f} pp")
    except Exception as e:  # no reference snapshot
        check("H1", "data accuracy vs SEC (holdout vs development set)", None, f"not run: {e}")

    # ---- H2 classification consistency (objective) ------------------------------------
    sec = ctx["tables"]["securities"].drop_duplicates("ticker").set_index("ticker")
    ind = sec["industry"].astype("string").fillna("").str.lower()

    def misclassified(tk):
        bad = []
        for t in tk:
            p, i = profile.get(t, "default"), ind.get(t, "")
            if p in _PROFILE_LABELS and not any(w in i for w in _PROFILE_LABELS[p]):
                bad.append(f"{t} ({p}: '{i}')")
            for q, words in _PROFILE_LABELS.items():   # labelled special industry but default profile
                if p == "default" and q in ("reit", "insurance") and any(w in i for w in words):
                    bad.append(f"{t} (default: '{i}')")
        return bad

    bh, bd = misclassified(H), misclassified(D)
    check("H2", "profile matches industry label", len(bh) <= 0.03 * max(len(H), 1),
          f"holdout {len(bh)}/{len(H)} inconsistent ({', '.join(bh[:6]) or 'none'}); development {len(bd)}/{len(D)}")

    # ---- H3 rule-firing parity: holdout vs development vs rest ------------------------
    fl = flags.copy()
    rows = []
    for flag in sorted(fl["flag"].unique()):
        has = set(fl.loc[fl["flag"] == flag, "ticker"])
        r = {g: (len(has & set(s)) / len(s) if len(s) else np.nan) for g, s in (("holdout", H), ("development", D),
                                                                               ("rest", R))}
        rows.append(dict(flag=flag, holdout_pct=r["holdout"] * 100, development_pct=r["development"] * 100,
                         rest_pct=r["rest"] * 100,
                         z_holdout_vs_rest=_z(r["holdout"], len(H), r["rest"], len(R)),
                         z_development_vs_rest=_z(r["development"], len(D), r["rest"], len(R))))
    parity = pd.DataFrame(rows)
    off_h = parity[parity["z_holdout_vs_rest"].abs() > 3]
    off_d = parity[parity["z_development_vs_rest"].abs() > 3]
    check("H3", "red-flag rates: holdout looks like the rest of the universe", off_h.empty,
          "no flag differs by more than 3 standard errors" if off_h.empty else
          "; ".join(f"{r.flag} {r.holdout_pct:.0f}% vs {r.rest_pct:.0f}%" for r in off_h.itertuples()))
    check("H3d", "red-flag rates: development set vs rest (overfitting signal)", off_d.empty,
          "no flag differs by more than 3 standard errors" if off_d.empty else
          "; ".join(f"{r.flag} {r.development_pct:.0f}% vs {r.rest_pct:.0f}%" for r in off_d.itertuples()),
          kind="diagnostic")

    # ---- H4 screen outcomes + H5 robustness -------------------------------------------
    outcome_rows, robust_rows = [], []
    base_results = {}
    for name, spec in cfg.screens.items():
        res = _score(cfg, ctx, spec)
        base_results[name] = res
        ranked = res["ranked"].set_index("ticker") if not res["ranked"].empty else pd.DataFrame()
        review = set(res["needs_review"]["ticker"]) if not res["needs_review"].empty else set()
        for g, s in (("holdout", H), ("development", D), ("rest", R)):
            s = set(s)
            rk = ranked.reindex(sorted(s & set(ranked.index)))
            outcome_rows.append(dict(screen=name, group=g, n=len(s), ranked_pct=len(rk) / len(s) * 100 if s else np.nan,
                                     review_pct=len(s & review) / len(s) * 100 if s else np.nan,
                                     median_score=float(rk["composite_score"].median()) if len(rk) else np.nan,
                                     grade_A_pct=float((rk["confidence_grade"] == "A").mean() * 100) if len(rk) else np.nan))
        base = ranked["composite_score"]
        for pname in spec["pillars"]:
            for mult in (0.8, 1.2):
                sp = json.loads(json.dumps(spec))
                sp["pillars"][pname]["weight"] = sp["pillars"][pname]["weight"] * mult
                alt = _score(cfg, ctx, sp)["ranked"]
                alt = alt.set_index("ticker")["composite_score"] if not alt.empty else pd.Series(dtype=float)
                for g, s in (("holdout", H), ("development", D)):
                    idx = sorted(set(s) & set(base.index) & set(alt.index))
                    if len(idx) < 5:
                        continue
                    rho = pd.Series(base.reindex(idx).values).rank().corr(pd.Series(alt.reindex(idx).values).rank())
                    q = max(1, len(idx) // 5)
                    top_b = set(base.reindex(idx).sort_values(ascending=False).index[:q])
                    top_a = set(alt.reindex(idx).sort_values(ascending=False).index[:q])
                    robust_rows.append(dict(screen=name, group=g, pillar=pname, weight_x=mult, n=len(idx),
                                            spearman=rho, top_quintile_overlap_pct=len(top_b & top_a) / q * 100))
    outcomes = pd.DataFrame(outcome_rows)
    robust = pd.DataFrame(robust_rows)
    for name in cfg.screens:
        o = outcomes[outcomes["screen"] == name].set_index("group")
        dz = abs(o.at["holdout", "review_pct"] - o.at["rest", "review_pct"])
        check(f"H4-{name}", f"{name}: holdout ranked / review shares look like the rest", dz <= 10.0,
              f"needs review: holdout {o.at['holdout', 'review_pct']:.0f}%, rest {o.at['rest', 'review_pct']:.0f}%, "
              f"development {o.at['development', 'review_pct']:.0f}%; median score holdout "
              f"{o.at['holdout', 'median_score']:.1f} vs rest {o.at['rest', 'median_score']:.1f}")
        rh = robust[(robust["screen"] == name) & (robust["group"] == "holdout")]
        if rh.empty:
            check(f"H5-{name}", f"{name}: ranking robust to +/-20% pillar weights", None, "too few ranked holdout names")
            continue
        ok = rh["spearman"].median() >= 0.95 and rh["spearman"].min() >= 0.90 and \
            rh["top_quintile_overlap_pct"].median() >= 80
        check(f"H5-{name}", f"{name}: ranking robust to +/-20% pillar weights", ok,
              f"holdout Spearman median {rh['spearman'].median():.3f} (min {rh['spearman'].min():.3f}), "
              f"top-quintile overlap median {rh['top_quintile_overlap_pct'].median():.0f}% "
              f"(min {rh['top_quintile_overlap_pct'].min():.0f}%) over {len(rh)} perturbations")

    # ---- H6 manual audit sheet (filled in by hand) ------------------------------------
    rng = np.random.default_rng(int(meta.get("seed", "1")) + 1)
    pick: List[str] = []
    for name, res in base_results.items():
        top = [t for t in res["ranked"]["ticker"] if t in set(H)][:5] if not res["ranked"].empty else []
        pick += top
    rev = sorted(set(H) & set().union(*[set(r["needs_review"]["ticker"]) for r in base_results.values()
                                        if not r["needs_review"].empty]))
    pick += [str(x) for x in rng.choice(rev, size=min(5, len(rev)), replace=False)] if rev else []
    rest_h = sorted(set(H) - set(pick))
    pick += [str(x) for x in rng.choice(rest_h, size=min(max(0, audit_n - len(set(pick))), len(rest_h)),
                                        replace=False)]
    pick = list(dict.fromkeys(pick))[:audit_n]
    fy = ctx["tables"]["financials"]
    fy = fy[fy["period_type"] == "FY"].sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")
    fl_by = (flags["flag"] + " (" + flags["default_severity"] + ")").groupby(flags["ticker"]).agg("; ".join) \
        if not flags.empty else pd.Series(dtype=str)
    audit = pd.DataFrame({
        "ticker": pick,
        "name": sec["name"].reindex(pick).values,
        "industry": sec["industry"].reindex(pick).values,
        "profile": profile.reindex(pick).values,
        "latest_fy_end": fy["period_end"].reindex(pick).dt.date.values,
        "revenue_musd": (fy["revenue"].reindex(pick) / 1e6).round(1).values,
        "net_income_musd": (fy["net_income"].reindex(pick) / 1e6).round(1).values,
        "operating_cash_flow_musd": (fy["operating_cash_flow"].reindex(pick) / 1e6).round(1).values,
        "total_debt_musd": (fy["total_debt"].reindex(pick) / 1e6).round(1).values,
        "dps": fy["dps"].reindex(pick).values,
        "flags": fl_by.reindex(pick).fillna("").values,
        **{f"{n}_rank": [int(r) if not pd.isna(r) else None for r in
                         (base_results[n]["ranked"].set_index("ticker")["rank"].reindex(pick)
                          if not base_results[n]["ranked"].empty else pd.Series(np.nan, index=pick))]
           for n in base_results},
        "check_profile_correct (Y/N)": "", "check_financials_match_10K (Y/N)": "",
        "check_flags_factually_correct (Y/N)": "", "notes": "",
    })
    check("H6", "manual audit of holdout companies", None,
          f"{len(audit)} companies in holdout_audit_sheet.csv: check each against its latest 10-K "
          f"(criterion: >= 90% Y on every column; any N must be an objective error to be fixable)", kind="manual")

    # ---- write ---------------------------------------------------------------------------
    stamp = f"{datetime.now():%Y-%m-%d_%H%M%S}_{provider}"
    out = cfg.outputs_dir / "holdout" / stamp
    keep = cfg.outputs_dir / "holdout" / f"{stamp}_summary"
    licensed = cfg.is_licensed(provider)
    for p, tier in ((out, "full"), (keep, "retained")):
        p.mkdir(parents=True, exist_ok=True)
        (p / "TIER.json").write_text(json.dumps({"tier": tier, "provider": provider,
                                                 "license_scope": provider if (licensed and tier == "full") else None,
                                                 "contains_raw_values": tier == "full"}, indent=2))
    audit.to_csv(out / "holdout_audit_sheet.csv", index=False, encoding="utf-8-sig")   # raw values -> full tier
    chk = pd.DataFrame(checks)
    for p in (out, keep):                                # rates, correlations, pass/fail: no raw values
        chk.to_csv(p / "holdout_checks.csv", index=False, encoding="utf-8-sig")
        parity.round(2).to_csv(p / "flag_rate_parity.csv", index=False, encoding="utf-8-sig")
        outcomes.round(2).to_csv(p / "screen_outcomes_by_group.csv", index=False, encoding="utf-8-sig")
        robust.round(4).to_csv(p / "weight_robustness.csv", index=False, encoding="utf-8-sig")
    summary = {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "provider": provider,
               "snapshot": ctx["snap"].snapshot_id, "methodology_fingerprint": methodology_fingerprint(cfg),
               "freeze_status": status, "holdout_size": len(hold), "holdout_in_universe": len(H),
               "development_in_universe": len(D), "rest_of_universe": len(R),
               "holdout_meta": {k: v for k, v in meta.items() if k != "sha_ok"},
               "accuracy_match_rates": acc,
               "results": {c["id"]: c["result"] for c in checks}}
    for p in (out, keep):
        (p / "holdout_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"checks": checks, "out": out, "keep": keep, "summary": summary}
