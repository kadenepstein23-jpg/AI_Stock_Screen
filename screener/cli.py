"""Command-line interface:  python -m screener <command> [options]

  status            what data is stored locally, how old it is, license rules
  check-key         confirm the paid Sharadar key works (small API calls, no download)
  refresh           download data from a provider (with check-key, the only commands that use the network)
  build             compute metrics from the local snapshot
  universe          validate the downloaded universe (coverage, history, exclusions) before screening
  run               run the screens on local data and write outputs
  company TICKER    show one company's metrics, flags and scores
  validate          compare two providers' financials (e.g. sharadar vs sec)
  purge             delete a provider's licensed data (keeps retained/public outputs)
  freeze            freeze the methodology (weights, thresholds, profiles); --verify checks it is unchanged
  holdout draw|evaluate   anti-overfitting holdout set: draw once, evaluate before any further tuning
  check-public      scan the repo for anything that must not be published
  export-portfolio  copy a run's public tier into portfolio/ (the GitHub-safe folder)
  docs              regenerate docs/DATA_DICTIONARY.md
  demo              synthetic data end to end (refresh + build + run)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import load_config
from .providers.base import ApiError
from .store import NoLocalData, Store


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def cmd_status(cfg, args) -> int:
    store = Store(cfg)
    _p(f"Project: {cfg.root}")
    _p(f"Active provider: {cfg.active_provider}\n")
    for prov in cfg.settings.get("providers", {}):
        snaps = store.list_snapshots(prov)
        lic = cfg.provider_settings(prov).get("license", {})
        _p(f"[{prov}]  license: {lic.get('retention')}")
        if not snaps:
            _p("   no local data" + ("  (refresh after subscribing)" if cfg.is_licensed(prov) else ""))
            continue
        cur = store.current(prov)
        age = (pd.Timestamp.now().normalize() - cur.as_of).days
        _p(f"   current snapshot: {cur.snapshot_id}   data as of {cur.as_of.date()} ({age} days old)")
        _p("   tables: " + ", ".join(f"{k}={v:,}" for k, v in cur.manifest.get("tables", {}).items()))
        _p(f"   snapshots stored: {len(snaps)}")
        if lic.get("retention") == "delete_on_cancel":
            _p(f"   NOTE: purge within {lic.get('purge_deadline_days', 30)} days of cancelling: "
               f"python -m screener purge --provider {prov}")
    latest = cfg.outputs_dir / "LATEST"
    if latest.exists():
        _p(f"\nLatest run: {latest.read_text().strip()}")
    return 0


def cmd_refresh(cfg, args) -> int:
    from .providers import get_provider

    prov = args.provider or cfg.active_provider
    tickers = _tickers(args)
    p = get_provider(prov, cfg, Store(cfg))
    snap = p.refresh(tickers=tickers, sample=args.sample, bulk=args.bulk, prices_file=args.prices_file)
    _p(f"Stored snapshot {snap.snapshot_id} for '{prov}' (data as of {snap.as_of.date()}).")
    _p("Tables: " + ", ".join(f"{k}={v:,}" for k, v in snap.manifest.get("tables", {}).items()))
    if args.build:
        from .pipeline import build

        build(cfg, prov)
        _p("Metrics built.")
    if args.keep is not None:
        removed = Store(cfg).prune(prov, keep=args.keep)
        if removed:
            _p(f"Pruned old snapshots: {', '.join(removed)}")
    return 0


def cmd_build(cfg, args) -> int:
    from .pipeline import build

    prov = args.provider or cfg.active_provider
    snap = build(cfg, prov)
    _p(f"Metrics built for {prov} snapshot {snap.snapshot_id}: {snap.manifest['tables'].get('metrics', 0):,} companies.")
    sup = snap.manifest.get("supplement")
    if sup:
        _p(f"Supplement from {sup.get('source')}: {sup.get('status')}; {sup.get('values_filled', 0):,} values filled "
           f"for {sup.get('listed_companies_filled', sup.get('companies', 0))} of {sup.get('eligible_companies', '?')} "
           f"listed companies in the profile")
    return 0


def cmd_run(cfg, args) -> int:
    from .pipeline import run

    prov = args.provider or cfg.active_provider
    screens = args.screen if args.screen and "all" not in args.screen else None
    out, results, snap = run(cfg, prov, screens=screens)
    _p(f"Data: {prov} snapshot {snap.snapshot_id} (as of {snap.as_of.date()})")
    for name, res in results.items():
        ranked = res["ranked"]
        _p(f"\n{cfg.screens[name].get('label', name)}: {len(ranked)} ranked, {len(res['needs_review'])} need review")
        if not ranked.empty:
            cols = ["rank", "ticker", "profile", "composite_score", "confidence_grade", "flags"]
            _p(ranked[cols].head(args.top).round(1).to_string(index=False))
    from .holdout import freeze_status

    fst, frec = freeze_status(cfg)
    _p(f"\nMethodology: {fst}" + (f" (fingerprint {frec['methodology_fingerprint']})" if frec else "")
       + ("  <- CHANGED since the freeze: results are not comparable to the holdout evaluation" if fst == "modified" else ""))
    _p(f"Outputs: {out.relative_to(cfg.root)}/  (full/, retained/, public/)")
    if cfg.is_licensed(prov):
        _p("License: full/ contains raw licensed values and is deleted by `purge`; retained/ and public/ are kept.")
    return 0


def cmd_company(cfg, args) -> int:
    from .metrics.registry import METRICS, PROFILE_ONLY

    from .pipeline import ensure_metrics
    from .profiles import assign_profiles, mask_not_applicable

    prov = args.provider or cfg.active_provider
    snap = ensure_metrics(cfg, prov)
    v = snap.read("metrics").set_index("ticker")
    s = snap.read("metric_status").set_index("ticker")
    t = args.ticker.upper()
    if t not in v.index:
        _p(f"{t} not found in {prov} snapshot {snap.snapshot_id}")
        return 1
    profile = assign_profiles(snap.read("securities"), cfg.profiles, v)
    v, s = mask_not_applicable(v, s, profile, cfg.profiles)
    prof = profile.get(t, "default")
    _p(f"{t} — {prov} data as of {snap.as_of.date()}")
    _p(f"Industry profile: {prof} ({cfg.profiles.get(prof, {}).get('label', '')})")
    if t in profile.attrs.get("overrides", {}):
        _p(f"Profile note: {profile.attrs['overrides'][t]}")
    gaps = cfg.profiles.get(prof, {}).get("data_gaps")
    if gaps:
        _p(f"Not in standardized data: {gaps}")
    # plain CAGRs shown for reference; scoring uses the robust trend
    # plain start/end figures shown for reference; scoring uses the robust / normalized measure
    pc = cfg.profiles.get(prof, {})
    REFERENCE_ONLY = {"eps_cagr_3y_pct": "EPS growth (scored)", "eps_cagr_5y_pct": "EPS growth (scored)",
                      "eps_trend_growth_pct": "EPS growth (scored)",
                      "ebitda_cagr_3y_pct": "EBITDA growth (scored)", "ebitda_trend_growth_pct": "EBITDA growth (scored)",
                      "opinc_cagr_3y_pct": None, "opinc_trend_growth_pct": None,
                      "operating_leverage_3y_pp": "operating leverage (scored)",
                      "operating_leverage_trend_pp": "operating leverage (scored)",
                      "operating_margin_chg_3y_pp": "op-margin change (scored)",
                      "operating_margin_trend_3y_pp": "op-margin change (scored)",
                      "gross_margin_chg_3y_pp": "gross-margin change (scored)",
                      "gross_margin_trend_3y_pp": "gross-margin change (scored)",
                      "fcf_cagr_3y_pct": "FCF growth (scored)", "fcf_cagr_5y_pct": "FCF growth (scored)",
                      "fcf_trend_growth_pct": "FCF growth (scored)",
                      "rule_of_40_pct": None, "fcf_margin_ex_sbc_pct": None}
    for a, b in (pc.get("score_with") or {}).items():
        REFERENCE_ONLY[a] = METRICS[b].label if b in METRICS else b
    for a in pc.get("not_scored") or []:
        REFERENCE_ONLY[a] = None
    cat = None
    for name, mdef in METRICS.items():
        if mdef.category != cat:
            cat = mdef.category
            _p(f"\n  {cat.upper().replace('_', ' ')}")
        val = v.at[t, name] if name in v.columns else None
        st = s.at[t, name] if name in s.columns else "missing"
        shown = "—" if val is None or (isinstance(val, float) and pd.isna(val)) else (
            f"{val:,.2f}" if isinstance(val, (int, float)) else str(val))
        if st == "na" and name in PROFILE_ONLY and prof not in PROFILE_ONLY[name]:
            tag = f"[n/a: {'/'.join(PROFILE_ONLY[name]).upper()}-only]"
        else:
            tag = "" if st == "ok" else (f"[n/a for {prof}]" if st == "na" else f"[{st}]")
        if st == "na" and name.startswith("dps_cagr") and v.at[t, "dividend_paying_years"] == 0:
            tag = "[n/a: no dividend]"
        if st == "nm" and shown != "—":
            tag = "[nm: distorted base year — reference only]"
        elif name in REFERENCE_ONLY and st == "ok":
            tag = (f"[reference only — scoring uses {REFERENCE_ONLY[name]}]" if REFERENCE_ONLY[name]
                   else (f"[reference only — not scored for {prof}]" if name in (pc.get("not_scored") or [])
                         else "[reference only]"))
        _p(f"    {mdef.label:<36} {shown:>14}   {tag}")
    latest = cfg.outputs_dir / "LATEST"
    if latest.exists():
        run_dir = cfg.outputs_dir / "runs" / latest.read_text().strip() / "retained"
        for f in sorted(run_dir.glob("*_screen.csv")):
            df = pd.read_csv(f)
            hit = df[df["ticker"] == t]
            if not hit.empty:
                r = hit.iloc[0]
                _p(f"\n  {f.stem}: rank {int(r['rank'])} of {len(df)}, score {r['composite_score']:.1f}, "
                   f"grade {r['confidence_grade']}, flags: {r.get('flags') if isinstance(r.get('flags'), str) else 'none'}")
        for f in sorted(run_dir.glob("*_needs_review.csv")):
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            hit = df[df["ticker"] == t] if "ticker" in df else df.iloc[0:0]
            if not hit.empty:
                _p(f"\n  {f.stem.replace('_needs_review', '')}: NOT RANKED — {hit.iloc[0]['review_reason']}")
        rf = run_dir / "red_flags.csv"
        if rf.exists():
            flags = pd.read_csv(rf)
            for _, r in flags[flags["ticker"] == t].iterrows():
                _p(f"  flag {r['flag']} ({r['default_severity']}): {r['detail']}")
    return 0


def cmd_validate(cfg, args) -> int:
    from .validate import compare_providers

    out = compare_providers(cfg, args.primary, args.reference, tickers=_tickers(args), tolerance_pct=args.tolerance)
    summary = pd.read_csv(out / "validation_summary.csv")
    _p(summary.to_string(index=False) if not summary.empty else "No overlapping company-years to compare.")
    detail = pd.read_csv(out / "validation_detail.csv")
    if not detail.empty:
        latest = detail.sort_values("period_end").groupby(["ticker", "field"]).tail(1)
        bad = latest[latest["within_tolerance"] == False].sort_values("abs_diff_pct", ascending=False)  # noqa: E712
        if len(bad):
            _p(f"\nLargest differences, latest fiscal year ({len(bad)} outside {args.tolerance}%; top {args.show}):")
            cols = ["ticker", "period_end", "field", "primary", "reference", "abs_diff_pct"]
            _p(bad[cols].head(args.show).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    _p(f"\nReport: {out.relative_to(cfg.root)}")
    return 0


def cmd_purge(cfg, args) -> int:
    from .compliance import purge, purge_targets

    prov = args.provider
    targets = purge_targets(cfg, prov)
    if not targets:
        _p(f"Nothing to purge for '{prov}'.")
        return 0
    _p(f"The following will be PERMANENTLY deleted for '{prov}':")
    for t in targets:
        _p(f"   {t.relative_to(cfg.root)}")
    _p("Kept: retained/ and public/ output tiers, portfolio/, docs, code.")
    if args.dry_run:
        _p("(dry run — nothing deleted)")
        return 0
    if not args.yes:
        ans = input("Type the provider name to confirm: ").strip()
        if ans != prov:
            _p("Cancelled.")
            return 1
    rec = purge(cfg, prov)
    _p(f"Purged {len(rec['deleted'])} locations. Verified clean: {rec['verified_clean']}. Log: {rec['log']}")
    return 0


def cmd_check_key(cfg, args) -> int:
    from .providers import get_provider

    prov = args.provider or "sharadar"
    p = get_provider(prov, cfg, Store(cfg))
    if not hasattr(p, "check_key"):
        _p(f"check-key is only needed for licensed providers (got '{prov}').")
        return 0
    rows = p.check_key()
    for name, ok, detail in rows:
        _p(f"   {'PASS' if ok else 'FAIL'}  {name:<44} {detail}")
    bad = [r for r in rows if not r[1]]
    _p("\nKey OK: ready for the full bulk download." if not bad else
       f"\n{len(bad)} check(s) failed: fix these before downloading.")
    return 0 if not bad else 2


def cmd_universe(cfg, args) -> int:
    from .diagnostics import universe_report

    prov = args.provider or cfg.active_provider
    lines, checks = universe_report(cfg, prov, spot=_tickers(args))
    for ln in lines:
        _p(ln)
    _p("\nCHECKS")
    for name, status, detail in checks:
        _p(f"   {status:<4}  {name:<48} {detail}")
    fails = sum(1 for c in checks if c[1] == "FAIL")
    warns = sum(1 for c in checks if c[1] == "WARN")
    _p(f"\n{'Universe OK' if not fails else 'Universe NOT ready'}: {fails} failed, {warns} warnings.")
    return 0 if not fails else 2


def cmd_freeze(cfg, args) -> int:
    from .holdout import freeze, freeze_status

    if args.verify:
        st, rec = freeze_status(cfg)
        _p(f"Methodology: {st}" + (f" (fingerprint {rec['methodology_fingerprint']}, frozen {rec['frozen_at']})"
                                  if rec else ""))
        return 0 if st == "frozen" else 2
    rec = freeze(cfg, note=args.note or "")
    _p(f"Methodology frozen: fingerprint {rec['methodology_fingerprint']} ({rec['frozen_at']}).")
    _p("From now on only objective data, parsing, classification, period-matching or calculation fixes.")
    return 0


def cmd_holdout(cfg, args) -> int:
    from . import holdout as ho

    prov = args.provider or cfg.active_provider
    if args.action == "draw":
        r = ho.draw_holdout(cfg, prov, n=args.n, seed=args.seed, force=args.force)
        _p(f"Holdout: {r['n']} companies drawn from {r['pool']:,} (after removing {r['dev_in_universe']} "
           f"development-set names from the universe). Written to {r['path'].relative_to(cfg.root)}")
        _p("Development set sources: " + "; ".join(f"{k} {v}" for k, v in r["dev_sources"].items()))
        _p("By profile: " + ", ".join(f"{k} {v}" for k, v in r["by_profile"].items()))
        _p("By size: " + ", ".join(f"{k} {v}" for k, v in r["by_size"].items()))
        _p(f"List checksum {r['sha']}. Do not look at, debug or tune on these names before `holdout evaluate`.")
        return 0
    r = ho.evaluate(cfg, prov, audit_n=args.audit_n, allow_modified=args.allow_modified)
    for c in r["checks"]:
        _p(f"   {c['result']:<4} {c['id']:<10} {c['check']}")
        _p(f"        {c['detail']}")
    crit = [c for c in r["checks"] if c["kind"] == "criterion"]
    fails = [c for c in crit if c["result"] == "FAIL"]
    na = [c for c in crit if c["result"] == "N/A"]
    _p(f"\nCriteria: {len(crit) - len(fails) - len(na)} passed, {len(fails)} failed, {len(na)} not run. "
       f"Manual audit pending: {r['out'].relative_to(cfg.root)}/holdout_audit_sheet.csv")
    _p(f"Summary (no licensed values, kept after purge): {r['keep'].relative_to(cfg.root)}")
    return 0 if not fails else 2


def cmd_check_public(cfg, args) -> int:
    from .compliance import check_public

    problems = check_public(cfg)
    if not problems:
        _p("OK — nothing that looks like licensed raw data, API keys or local data folders would be published.")
        return 0
    _p("NOT SAFE TO PUBLISH:")
    for p in problems:
        _p(f"   {p['file']}: {p['issue']}")
    return 2


def cmd_export_portfolio(cfg, args) -> int:
    from .compliance import check_public, export_portfolio

    dest = export_portfolio(cfg, args.run)
    _p(f"Copied public outputs to {dest.relative_to(cfg.root)}/")
    problems = [p for p in check_public(cfg) if p["file"].startswith("portfolio/")]
    if problems:
        _p("WARNING: check-public found problems in portfolio/:")
        for p in problems:
            _p(f"   {p['file']}: {p['issue']}")
        return 2
    return 0


def cmd_docs(cfg, args) -> int:
    from .metrics.registry import METRICS

    lines = ["# Data dictionary", "",
             "Generated from `screener/metrics/registry.py` (`python -m screener docs`). "
             "Units: pct = percent (12.5 = 12.5%), pp = percentage points, x = multiple, $M = millions of USD.",
             "", "**Kind**: `raw_value` = absolute figure from the data provider (excluded from retained/public "
             "outputs of licensed providers); `derived` = ratio, growth rate, count or score computed by this "
             "project; `meta` = data-coverage information.", ""]
    cat = None
    for m in METRICS.values():
        if m.category != cat:
            cat = m.category
            lines += ["", f"## {cat.replace('_', ' ').title()}", "", "| Column | Label | Unit | Kind | Definition |",
                      "|---|---|---|---|---|"]
        lines.append(f"| `{m.name}` | {m.label} | {m.unit} | {m.kind} | {m.description} |")
    lines += ["", "## Status codes (metric_status.csv)", "",
              "| Code | Meaning |", "|---|---|",
              "| ok | computed |", "| missing | an input is absent |",
              "| nm | not meaningful (e.g. negative denominator) |",
              "| short | not enough fiscal-year history for this look-back |"]
    path = cfg.root / "docs" / "DATA_DICTIONARY.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    _p(f"Wrote {path.relative_to(cfg.root)}")
    return 0


def cmd_demo(cfg, args) -> int:
    args.provider = "synthetic"
    args.sample = args.bulk = False
    args.prices_file = None
    args.tickers = args.tickers_file = None
    args.build = True
    args.keep = 1
    cmd_refresh(cfg, args)
    args.screen = None
    return cmd_run(cfg, args)


def _tickers_from(spec: str) -> List[str]:
    """'sharadar:reit' -> the investable tickers of that provider's snapshot in those profiles
    (comma-separated profiles; 'sharadar' alone = the whole investable universe)."""
    from .config import load_config
    from .pipeline import ensure_metrics, load_tables
    from .profiles import assign_profiles
    from .universe import build_universe

    cfg = load_config()
    prov, _, profs = spec.partition(":")
    snap = ensure_metrics(cfg, prov)
    t = load_tables(snap)
    values = snap.read("metrics").set_index("ticker")
    universe, _ = build_universe(t["securities"], t["financials"], values, cfg.settings["universe"])
    if not profs:
        return list(universe)
    prof = assign_profiles(t["securities"], cfg.profiles, values)
    keep = {p.strip() for p in profs.split(",") if p.strip()}
    return [x for x in universe if prof.get(x) in keep]


def _tickers(args) -> Optional[List[str]]:
    out: List[str] = []
    if getattr(args, "tickers", None):
        out += [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    files = getattr(args, "tickers_file", None) or []
    for f in ([files] if isinstance(files, str) else files):
        out += [ln.split("#", 1)[0].strip().upper() for ln in Path(f).read_text().splitlines()
                if ln.split("#", 1)[0].strip()]
    if getattr(args, "tickers_from", None):
        found = _tickers_from(args.tickers_from)
        _p(f"{len(found)} tickers from {args.tickers_from}")
        out += found
    out = list(dict.fromkeys(out))          # de-duplicate, keep order
    return out or None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m screener", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("refresh")
    sp.add_argument("--provider")
    sp.add_argument("--tickers", help="comma-separated tickers")
    sp.add_argument("--tickers-file", action="append", help="file with one ticker per line (repeatable)")
    sp.add_argument("--tickers-from", help="PROVIDER:PROFILE, e.g. sharadar:reit = that snapshot's investable REITs")
    sp.add_argument("--sample", action="store_true", help="Sharadar public sample (DJIA 30), no subscription")
    sp.add_argument("--bulk", action="store_true", help="SEC: full-market bulk download")
    sp.add_argument("--prices-file", help="SEC: optional CSV of ticker,date,close,volume")
    sp.add_argument("--no-build", dest="build", action="store_false", help="skip computing metrics")
    sp.add_argument("--keep", type=int, default=None, help="keep only the newest N snapshots")
    sp.set_defaults(fn=cmd_refresh, build=True)

    sp = sub.add_parser("build")
    sp.add_argument("--provider")
    sp.set_defaults(fn=cmd_build)

    sp = sub.add_parser("run")
    sp.add_argument("--provider")
    sp.add_argument("--screen", action="append", help="growth | income | all (repeatable)")
    sp.add_argument("--top", type=int, default=15)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("company")
    sp.add_argument("ticker")
    sp.add_argument("--provider")
    sp.set_defaults(fn=cmd_company)

    sp = sub.add_parser("validate")
    sp.add_argument("--primary", default="sharadar")
    sp.add_argument("--reference", default="sec")
    sp.add_argument("--tickers")
    sp.add_argument("--tickers-file", action="append")
    sp.add_argument("--tolerance", type=float, default=2.0, help="percent difference treated as a match")
    sp.add_argument("--show", type=int, default=25, help="how many of the largest differences to list")
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("purge")
    sp.add_argument("--provider", required=True)
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_purge)

    sp = sub.add_parser("check-key", help="confirm the paid API key works (no download)")
    sp.add_argument("--provider", default="sharadar")
    sp.set_defaults(fn=cmd_check_key)

    sp = sub.add_parser("universe", help="validate the downloaded universe before screening")
    sp.add_argument("--provider")
    sp.add_argument("--tickers", help="extra comma-separated tickers to spot-check")
    sp.add_argument("--tickers-file", action="append")
    sp.set_defaults(fn=cmd_universe)

    sp = sub.add_parser("freeze", help="freeze the methodology (or --verify it is unchanged)")
    sp.add_argument("--verify", action="store_true")
    sp.add_argument("--note")
    sp.set_defaults(fn=cmd_freeze)

    sp = sub.add_parser("holdout", help="draw / evaluate the anti-overfitting holdout set")
    sp.add_argument("action", choices=["draw", "evaluate"])
    sp.add_argument("--provider")
    sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=20261008)
    sp.add_argument("--force", action="store_true", help="redraw (only after a NEW freeze)")
    sp.add_argument("--audit-n", type=int, default=20)
    sp.add_argument("--allow-modified", action="store_true", help="evaluate even if the methodology changed")
    sp.set_defaults(fn=cmd_holdout)

    sp = sub.add_parser("check-public")
    sp.set_defaults(fn=cmd_check_public)

    sp = sub.add_parser("export-portfolio")
    sp.add_argument("--run", help="run id (default: latest)")
    sp.set_defaults(fn=cmd_export_portfolio)

    sp = sub.add_parser("docs")
    sp.set_defaults(fn=cmd_docs)

    sp = sub.add_parser("demo")
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    try:
        return int(args.fn(cfg, args) or 0)
    except NoLocalData as e:
        _p(str(e))
        return 3
    except ApiError as e:   # messages are already key-free; no traceback
        _p(f"API error: {e}")
        return 4
    except BrokenPipeError:
        # Output piped into `head`/`grep -m` that closed early: not an error.
        # Point stdout at devnull so the interpreter's final flush is silent.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
