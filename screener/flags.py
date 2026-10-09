"""Red-flag detection. Returns a long table: one row per (ticker, flag)."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import pandas as pd


# leverage metrics where a LOWER value means MORE leverage
LOWER_IS_WORSE_LEVERAGE = {"cfo_to_debt_pct"}


def _effective_flag_cfg(flags_cfg: Dict[str, Any], profile_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(flags_cfg)
    for name, over in (profile_cfg.get("flag_thresholds") or {}).items():
        if name in cfg:
            cfg[name].update(over)
    for name in profile_cfg.get("disabled_flags", []) or []:
        if name in cfg:
            cfg[name]["severity"] = "ignore"
    return cfg


def detect_flags(metrics: pd.DataFrame, profile: pd.Series, peer: pd.Series,
                 flags_cfg: Dict[str, Any], profiles_cfg: Dict[str, Any],
                 universe: pd.Index, status: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Detect flags for every ticker in `universe`. Severity here is the DEFAULT severity;
    screens apply their own overrides in scoring. `status` (optional) lets flags skip start/end
    comparisons whose starting year was marked distorted (status nm with a value)."""
    m = metrics.reindex(universe)
    stat = status.reindex(universe) if status is not None else None

    def distorted(t, c) -> bool:
        return (stat is not None and c in stat.columns and stat.at[t, c] == "nm"
                and not pd.isna(pd.to_numeric(m.at[t, c], errors="coerce")))
    num = lambda c: pd.to_numeric(m[c], errors="coerce")  # noqa: E731

    # peer median yield (payers only) for the yield-trap test
    dy = num("dividend_yield_pct")
    peer_u = peer.reindex(universe)
    peer_med = dy.where(dy > 0).groupby(peer_u).transform("median")

    rows: List[Dict[str, Any]] = []
    for t in universe:
        prof = profile.get(t, "default")
        pcfg = profiles_cfg.get(prof, {})
        cfg = _effective_flag_cfg(flags_cfg, pcfg)
        cyclical = bool(pcfg.get("cyclical"))
        fm = pcfg.get("flag_metrics") or {}
        lev_metric = fm.get("leverage", "net_debt_to_ebitda_x")
        cov_metric = fm.get("coverage", "interest_coverage_x")
        earn_metric = fm.get("earnings", "net_margin_pct")
        chg_metric = fm.get("leverage_change", "net_debt_to_ebitda_chg_3y_x")
        dil_offset = fm.get("dilution_offset")
        divcov_metric = fm.get("dividend_coverage", "fcf_payout_3y_pct")
        r = m.loc[t]

        def g(c):
            v = pd.to_numeric(r.get(c), errors="coerce")
            return None if pd.isna(v) else float(v)

        def add(name: str, detail: str, severity: str = None) -> None:
            sev = severity or cfg[name]["severity"]
            if cfg[name]["severity"] == "ignore":
                return
            rows.append(dict(ticker=t, flag=name, default_severity=sev, detail=detail,
                             description=cfg[name].get("description", "").strip(),
                             downgraded=bool(severity == "info" and cfg[name]["severity"] != "info")))

        ni_margin = g("net_margin_pct")
        earn = g(earn_metric)
        if earn is not None and earn < 0:
            add("negative_earnings", "negative FFO" if earn_metric == "ffo_per_share" else f"net margin {earn:.1f}%")
        rg = g("revenue_cagr_3y_pct")
        if rg is not None and rg < cfg["revenue_decline"]["threshold"]:
            add("revenue_decline", f"3y revenue CAGR {rg:.1f}%")
        # a 3-yr change that starts from a distorted year (e.g. a cyclical peak or a write-off year)
        # is replaced by the robust 8-yr margin trend expressed per 3 years
        if distorted(t, "operating_margin_chg_3y_pp"):
            omc = g("operating_margin_trend_3y_pp")
            omc_txt = "operating margin trend {:+.1f} pp per 3y (3y change starts from an abnormal year)"
        else:
            omc = g("operating_margin_chg_3y_pp")
            omc_txt = "operating margin {:+.1f} pp over 3y"
        if omc is not None and omc <= cfg["margin_deterioration"]["threshold"]:
            add("margin_deterioration", omc_txt.format(omc))
        fm = g("fcf_margin_pct")
        if fm is not None and fm < 0:
            add("negative_fcf", f"FCF margin {fm:.1f}%")
        conv = g("fcf_conversion_3y_pct")
        if conv is not None and conv < cfg["poor_fcf_conversion"]["threshold"]:
            add("poor_fcf_conversion", f"3y FCF/net income {conv:.0f}%")
        nd = g(lev_metric)
        lev_label = {"net_debt_to_ebitdare_x": "net debt/EBITDAre", "debt_to_capital_pct": "debt/capital",
                     "cfo_to_debt_pct": "CFO/debt"}.get(lev_metric, "net debt/EBITDA")
        lower_is_worse = lev_metric in LOWER_IS_WORSE_LEVERAGE

        def lev_fmt(x):
            return f"{x:.0f}%" if lev_metric.endswith("_pct") else f"{x:.1f}x"
        hl = cfg["high_leverage"]
        if nd is not None:
            if lower_is_worse:      # e.g. cash flow / debt: a LOW value means high leverage
                breach = nd < hl["threshold"]
                severe = nd < hl.get("severe_threshold", -1e9)
            else:
                breach = nd > hl["threshold"]
                severe = nd > hl.get("severe_threshold", 1e9)
            if breach:
                add("high_leverage", f"{lev_label} {lev_fmt(nd)}", "severe" if severe else None)
        ndc = g(chg_metric)
        rl = cfg["rising_leverage"]
        level = g("net_debt_to_ebitda_x") if lower_is_worse else nd
        level_label = "net debt/EBITDA" if lower_is_worse else lev_label
        if ndc is not None and level is not None and ndc > rl["threshold"] and level > rl.get("min_level", 0):
            add("rising_leverage", f"{level_label} {ndc:+.1f}x over 3y to {level:.1f}x")
        cov = g(cov_metric)
        ic = cfg["low_interest_coverage"]
        if cov is not None and cov < ic["threshold"]:
            label = f"{'EBITDAre' if cov_metric == 'ebitdare_interest_coverage_x' else 'EBIT'}/interest {cov:.1f}x"
            loss = cov < 0 and earn is not None and earn < 0
            netd = g("net_debt_musd")
            if loss and netd is not None and netd <= 0:
                # an operating loss with net cash: the loss is already the negative_earnings flag and there
                # is no debt burden to cover - counting it twice would send every cash-rich loss-maker to review
                add("low_interest_coverage", f"{label}: operating loss, net cash (carried by negative_earnings)",
                    "info")
            elif loss:
                # operating loss AND net debt: a real debt risk, but the loss itself is already flagged, so the
                # coverage flag adds a moderate penalty rather than escalating to severe on the same loss
                add("low_interest_coverage", f"{label}: operating loss with net debt")
            else:
                add("low_interest_coverage", label, "severe" if cov < ic.get("severe_threshold", -1e9) else None)
        dil = g("share_change_3y_cagr_pct")
        dc = cfg["dilution"]
        if dil is not None and dil > dc["threshold"]:
            sev = "severe" if dil > dc.get("severe_threshold", 1e9) else None
            detail = f"share count +{dil:.1f}%/yr (3y)"
            off = g(dil_offset) if dil_offset else None
            if off is not None and off > 0:
                # e.g. a REIT issuing equity to buy properties while FFO/share keeps rising
                sev, detail = "info", detail + f"; accretive (FFO/share {off:+.1f}%/yr)"
            add("dilution", detail, sev)
            dilution_penalized = sev != "info"
        else:
            dilution_penalized = False
        sbc = g("sbc_pct_revenue")
        if sbc is not None and sbc > cfg["high_sbc"]["threshold"]:
            # stock pay is usually WHY the share count grows: when the dilution flag already carries a
            # penalty, the SBC flag is kept for information only, so the same issuance isn't penalized twice
            if dilution_penalized:
                add("high_sbc", f"SBC {sbc:.1f}% of revenue (penalty carried by the dilution flag)", "info")
            else:
                add("high_sbc", f"SBC {sbc:.1f}% of revenue")

        yld = g("dividend_yield_pct") or 0.0
        pays = yld > 0
        fp3 = g("fcf_payout_3y_pct")
        dc = g(divcov_metric)
        if pays and dc is not None and dc > cfg["weak_dividend_coverage"]["threshold"]:
            add("weak_dividend_coverage", f"dividends = {dc:.0f}% of FFO" if divcov_metric == "ffo_payout_pct"
                else f"3y dividends = {dc:.0f}% of FCF")
        pr = g("payout_ratio_pct")
        if pays and ((pr is not None and pr > cfg["earnings_payout_over_100"]["threshold"]) or
                     (pr is None and ni_margin is not None and ni_margin < 0)):
            add("earnings_payout_over_100", "dividends exceed net income" if pr is None else f"payout {pr:.0f}%")

        # yield trap: unusually high yield + a deterioration signal
        yt = cfg["yield_trap"]
        pm = peer_med.get(t)
        own = g("dividend_yield_5y_median_pct")
        high_vs_peer = pays and pm is not None and not pd.isna(pm) and yld >= yt["yield_vs_peer_median"] * pm
        high_vs_own = pays and own is not None and own > 0 and yld >= yt["yield_vs_own_5y_median"] * own
        if high_vs_peer or high_vs_own:
            signals = []
            eg = g("eps_trend_growth_pct")
            if prof == "reit":
                eg = g("ffo_ps_cagr_3y_pct")
            if (eg is not None and eg < 0) or (earn is not None and earn < 0):
                signals.append("falling FFO/share" if prof == "reit" else "falling EPS")
            ret = g("return_1y_pct")
            if ret is not None and ret < -20:
                signals.append(f"price {ret:.0f}% in 1y")
            if fp3 is not None and fp3 > 100 and prof not in ("utility", "bank"):
                signals.append("dividend not covered by FCF")
            if prof == "reit":
                fo = g("ffo_payout_pct")
                if fo is not None and fo > 90:
                    signals.append(f"FFO payout {fo:.0f}%")
            if signals:
                basis = (f"{yld:.1f}% yield vs peer median {pm:.1f}%" if high_vs_peer
                         else f"{yld:.1f}% yield vs own 5y median {own:.1f}%")
                add("yield_trap", f"{basis}; " + ", ".join(signals))

        cut = g("dividend_cut_5y")
        if cut == 1.0:
            add("dividend_cut", "DPS cut >10% in last 5 FYs")
        eq_assets = g("equity_to_assets_pct")
        if eq_assets is not None and eq_assets < 0:
            add("negative_equity", f"equity/assets {eq_assets:.0f}%")
        pe = g("pe_x")
        evs = g("ev_sales_x")
        ev_c = cfg["extreme_valuation"]
        parts = []
        if pe is not None and pe > ev_c["pe_threshold"] and prof != "reit":
            parts.append(f"P/E {pe:.0f}x")
        if evs is not None and evs > ev_c["ev_sales_threshold"] and prof not in ("bank", "insurance"):
            parts.append(f"EV/sales {evs:.0f}x")
        if parts:
            add("extreme_valuation", ", ".join(parts))
        # peak-cycle earnings: margin far above its 10-yr average OR net income far above normalized
        mv = g("margin_vs_10y_avg_x")
        evn = g("earnings_vs_normalized_x")
        if cyclical:
            peak = []
            if mv is not None and mv > cfg["cyclical_peak"]["threshold"]:
                peak.append(f"operating margin {mv:.1f}x its 10y average")
            if evn is not None and evn > cfg["cyclical_peak"].get("earnings_threshold", 1.5):
                peak.append(f"net income {evn:.1f}x normalized (8y-average margin x current revenue)")
            if peak:
                add("cyclical_peak", "; ".join(peak))
        dist = g("largest_distribution_1y_pct")
        if dist is not None and dist > cfg["recent_distribution"]["threshold"]:
            ago = g("largest_distribution_days_ago")
            add("recent_distribution", f"{dist:.0f}% of the share price distributed in one day"
                + (f", {ago:.0f} days ago" if ago is not None else "")
                + " (spin-off or special dividend): trailing figures still include it")
        fy = g("fiscal_years_available")
        if fy is not None and fy < cfg["short_history"]["threshold"]:
            add("short_history", f"{fy:.0f} fiscal years available")
        if not pays:
            add("no_dividend", "no dividend")
    return pd.DataFrame(rows, columns=["ticker", "flag", "default_severity", "detail", "description", "downgraded"])
