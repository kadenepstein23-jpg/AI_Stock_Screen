"""Metric engine: canonical tables -> one row of metrics per ticker.

Every metric is returned together with a STATUS code so missing data is never
silently treated as a bad (or good) value:

    ok       value computed
    missing  an input the metric needs is absent
    nm       not meaningful (e.g. P/E with negative earnings, CAGR from a negative base)
    short    not enough fiscal-year history for this look-back

Conventions
-----------
* Fiscal years are aligned per company: offset 0 = latest FY, 1 = the year before, ...
  (offsets come from period-end dates, so a missing year leaves a gap instead of
  shifting the series).
* "Current" values use the latest TTM row when it is newer than the latest FY,
  otherwise the latest FY. Multi-year trends always use fiscal years.
* Share counts, diluted EPS and dividends per share are split-adjusted to today's
  share basis before any multi-year comparison (see splits.py).
* Percentages are expressed in percent (12.5 = 12.5%), multiples in x, money in $M.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .registry import METRICS
from .robust import (base_residual_linear, base_vs_trend, cycle_average, is_distorted, linear_trend,
                     trend_growth)
from .splits import split_adjust_panel

MAX_OFFSET = 10
NEAR_ZERO_BASE = 0.05     # CAGR base below 5% of the latest value -> not meaningful
PEG_MIN_GROWTH = 3.0      # %/yr; slower growth -> PEG not meaningful
PEG_GROWTH_CAP = 40.0     # %/yr; faster trailing growth is capped in the PEG denominator
DPS_SHARE_JUMP = 0.25     # share-count change (e.g. a stock-paid merger) that makes dividends paid / shares unreliable
DPS_CALC_TOLERANCE = 0.015  # +/- noise allowed in streaks when DPS is calculated
FIELDS = ["revenue", "cost_of_revenue", "gross_profit", "operating_income", "ebit", "ebitda", "d_and_a",
          "interest_expense", "real_estate_impairment", "gain_on_sale_real_estate", "fx_gain_loss", "pretax_income", "income_tax", "net_income", "eps_diluted",
          "shares_diluted_wavg", "shares_outstanding", "cash", "current_assets", "current_liabilities",
          "total_assets", "total_liabilities", "total_debt", "equity", "preferred_equity", "aoci", "intangibles",
          "operating_cash_flow", "capex", "free_cash_flow", "sbc", "dividends_paid",
          "share_repurchases", "net_common_issuance", "dps"]


class _Out:
    """Collects metric values + status codes."""

    def __init__(self, index: pd.Index):
        self.index = index
        self.values: Dict[str, pd.Series] = {}
        self.status: Dict[str, pd.Series] = {}

    def put(self, name: str, value: pd.Series, status: Optional[pd.Series] = None) -> None:
        if name not in METRICS:
            raise KeyError(f"Metric '{name}' is not in the registry")
        value = pd.Series(value, index=self.index, dtype="float64")
        value = value.where(np.isfinite(value))
        if status is None:
            status = pd.Series(np.where(value.notna(), "ok", "missing"), index=self.index)
        else:
            status = pd.Series(status, index=self.index).astype(object)
            status = status.where(~(value.notna() & status.isin(["missing"])), "ok")
            status = status.where(~(value.isna() & (status == "ok")), "missing")
        self.values[name] = value
        self.status[name] = status

    def frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.values, index=self.index), pd.DataFrame(self.status, index=self.index)


def _panel(fin: pd.DataFrame, tickers: pd.Index) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Wide fiscal-year panel: panel[field] -> DataFrame(index=ticker, columns=0..MAX_OFFSET)."""
    fy = fin[fin["period_type"] == "FY"].dropna(subset=["period_end"]).copy()
    fy = fy.sort_values(["ticker", "period_end", "filing_date"])
    fy = fy.drop_duplicates(["ticker", "period_end"], keep="last")
    latest = fy.groupby("ticker")["period_end"].transform("max")
    fy["k"] = ((latest - fy["period_end"]).dt.days / 365.25).round().astype(int)
    fy = fy[(fy["k"] >= 0) & (fy["k"] <= MAX_OFFSET)].drop_duplicates(["ticker", "k"], keep="last")
    cols = list(range(MAX_OFFSET + 1))
    panel = {}
    for f in FIELDS + ["period_end", "filing_date"]:
        w = fy.pivot(index="ticker", columns="k", values=f).reindex(index=tickers, columns=cols)
        panel[f] = w
    return panel, fy


def _cagr(end: pd.Series, start: pd.Series, years: int, short: pd.Series) -> Tuple[pd.Series, pd.Series]:
    valid = (end > 0) & (start > 0)
    val = ((end / start) ** (1.0 / years) - 1.0) * 100.0
    val = val.where(valid)
    status = np.select([valid, end.notna() & start.notna(), short], ["ok", "nm", "short"], "missing")
    return val, pd.Series(status, index=end.index)


def _safe_div(a: pd.Series, b: pd.Series, require_pos_denominator: bool = True) -> Tuple[pd.Series, pd.Series]:
    both = a.notna() & b.notna()
    good = both & ((b > 0) if require_pos_denominator else (b != 0))
    val = (a / b).where(good)
    status = np.select([good, both], ["ok", "nm"], "missing")
    return val, pd.Series(status, index=a.index)


def compute_metrics(tables: Dict[str, pd.DataFrame], as_of: pd.Timestamp,
                    estimates: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (values, status) DataFrames indexed by ticker."""
    sec = tables["securities"]
    fin = tables["financials"].copy()
    # Dividends paid reported as exactly zero next to a positive reported DPS for the same period
    # contradict each other. This is a provider mapping gap; Iron Mountain shows 0 in every year
    # from 2016 while paying $2.00-$3.37 per share. Treat the zero as missing so the DPS x shares
    # fill below applies. A real zero (no DPS, or DPS of 0) is unchanged.
    if {"dividends_paid", "dps"} <= set(fin.columns):
        fin.loc[(fin["dividends_paid"] == 0) & (fin["dps"] > 0), "dividends_paid"] = np.nan
    mkt =tables.get("market", pd.DataFrame(columns=["ticker", "date", "market_cap", "enterprise_value"]))
    px = tables.get("prices", pd.DataFrame(columns=["ticker", "date", "close", "close_adj", "volume"]))
    as_of = pd.Timestamp(as_of)

    tickers = pd.Index(sorted(set(sec["ticker"]).union(fin["ticker"])), name="ticker")
    P, fy = _panel(fin, tickers)
    P, n_splits = split_adjust_panel(P)   # put multi-year share counts / EPS / DPS on today's share basis
    o = _Out(tickers)

    def F(field: str, k: int) -> pd.Series:
        return P[field][k]

    n_years = P["period_end"].notna().sum(axis=1)

    def short(k: int) -> pd.Series:
        return P["period_end"][k].isna() & (n_years > 0)

    # Operating income is often untagged (common for REITs and holding companies). Fall back to
    # EBIT = pretax income + interest expense, excluding property-sale gains and impairments
    # (non-operating items that otherwise swing margins), when pretax and interest are reported.
    def _ebit_fallback(src):
        return (src["pretax_income"] + src["interest_expense"].abs()
                + src["real_estate_impairment"].fillna(0) - src["gain_on_sale_real_estate"].fillna(0)
                # currency remeasurement is not operating (Public Storage 2025: a $216M FX loss on euro
                # notes). Equity-affiliate income stays in: filers that tag no operating-income line
                # (oil majors) treat affiliates as operations, and many don't tag it separately.
                - src["fx_gain_loss"].fillna(0))

    P["operating_income"] = P["operating_income"].fillna(_ebit_fallback(P))
    # Dividends paid not tagged (some filers switch to company-specific tags) -> reported DPS x
    # diluted shares, so payout ratios and yields stay available.
    _sh = P["shares_diluted_wavg"].fillna(P["shares_outstanding"])
    raw_dividends_paid = P["dividends_paid"].copy()       # before the DPS x shares fill (for DPS checks)
    P["dividends_paid"] = P["dividends_paid"].fillna(P["dps"] * _sh)

    # ---- current (TTM if newer than latest FY, else FY0) -----------------------------
    ttm = fin[fin["period_type"] == "TTM"].sort_values("period_end").groupby("ticker").tail(1)
    ttm = ttm.set_index("ticker").reindex(tickers)
    use_ttm = ttm["period_end"].notna() & (ttm["period_end"] > P["period_end"][0].fillna(pd.Timestamp.min))
    cur = {}
    for f in FIELDS:
        cur[f] = ttm[f].where(use_ttm).fillna(F(f, 0)) if f in ttm else F(f, 0)
    cur["operating_income"] = cur["operating_income"].fillna(_ebit_fallback(cur))
    cur["dividends_paid"] = cur["dividends_paid"].fillna(
        cur["dps"] * cur["shares_diluted_wavg"].fillna(cur["shares_outstanding"]))
    cur_end = ttm["period_end"].where(use_ttm).fillna(P["period_end"][0])
    cur_filed = ttm["filing_date"].where(use_ttm).fillna(P["filing_date"][0])

    # derived per-period helpers --------------------------------------------------------
    def gp(k):  # gross profit with fallback
        return P["gross_profit"][k].fillna(P["revenue"][k] - P["cost_of_revenue"][k])

    def ebitda(k):
        return P["ebitda"][k].fillna(P["operating_income"][k] + P["d_and_a"][k])

    def fcf(k):
        return P["free_cash_flow"][k].fillna(P["operating_cash_flow"][k] - P["capex"][k])

    def shares(k):
        return P["shares_diluted_wavg"][k].fillna(P["shares_outstanding"][k])

    def dps_src(k):
        """(DPS, calculated?) for fiscal year offset k."""
        # Reported DPS is checked against dividends paid / shares (both split-adjusted). A value
        # more than 2x off (e.g. a preferred-stock or per-unit fact under the common DPS tag) is
        # discarded and the calculated figure from reported totals is used instead.
        rep = P["dps"][k]
        implied = (P["dividends_paid"][k] / shares(k)).where(shares(k) > 0)
        ratio = rep / implied
        invalid = rep.notna() & (implied > 0) & ((rep <= 0) | (ratio > 2) | (ratio < 0.5))
        # a one-year outlier: reported DPS is >10% off dividends paid / shares while the company's
        # reported and calculated figures agree within 3% in the years on both sides (P&G FY2017:
        # 3.00 tagged vs 2.64 paid / shares; 2.66 and 2.79 either side)
        def _ratio(j):
            if j < 0 or j > MAX_OFFSET:
                return pd.Series(np.nan, index=tickers)
            # only REPORTED dividends paid count as evidence (a DPS x shares fill would agree with
            # the DPS tag by construction)
            imp_j = (raw_dividends_paid[j] / shares(j)).where(shares(j) > 0)
            return P["dps"][j] / imp_j

        steady = ((_ratio(k - 1) - 1).abs() < 0.03) & ((_ratio(k + 1) - 1).abs() < 0.03)
        invalid |= rep.notna() & (implied > 0) & ((ratio - 1).abs() > 0.10) & steady
        rep = rep.mask(invalid)
        # dividends paid / weighted shares is unreliable in a year the share count jumps (a merger
        # paid in stock: Chubb 2016) - the payments and the average share count cover different
        # share bases. Such a calculated value is dropped (and may be bridged below).
        if k < MAX_OFFSET:
            jump = (shares(k) / shares(k + 1) - 1).abs() > DPS_SHARE_JUMP
            implied = implied.mask(jump & rep.isna())
        return rep.fillna(implied), rep.isna() & implied.notna()

    def dps(k):
        return dps_src(k)[0]

    cur_gp = cur["gross_profit"].fillna(cur["revenue"] - cur["cost_of_revenue"])
    cur_ebitda = cur["ebitda"].fillna(cur["operating_income"] + cur["d_and_a"])
    cur_ebit = cur["ebit"].fillna(cur["operating_income"])
    cur_fcf = cur["free_cash_flow"].fillna(cur["operating_cash_flow"] - cur["capex"])
    cur_div = cur["dividends_paid"]

    # ---- market data -------------------------------------------------------------------
    m = mkt.dropna(subset=["date"])
    m = m[m["date"] <= as_of].sort_values("date")
    m_last = m.groupby("ticker").tail(1).set_index("ticker").reindex(tickers)
    mcap = m_last["market_cap"]
    ev_provider = m_last["enterprise_value"]
    ev = ev_provider.fillna(mcap + cur["total_debt"].fillna(0) - cur["cash"].fillna(0))
    ev = ev.where(mcap.notna())

    p = px.dropna(subset=["date"])
    p = p[p["date"] <= as_of].sort_values(["ticker", "date"])
    price = p.groupby("ticker")["close"].last().reindex(tickers)
    dv = (p["close"] * p["volume"]).groupby(p["ticker"]).apply(lambda s: s.tail(60).median())
    adv = dv.reindex(tickers)
    adj_now = p.groupby("ticker")["close_adj"].last().reindex(tickers)
    p_year = p[p["date"] <= as_of - pd.Timedelta(days=365)]
    adj_then = p_year.groupby("ticker")["close_adj"].last().reindex(tickers)
    # tolerate a few missing days at the start of the window
    first_date = p.groupby("ticker")["date"].min().reindex(tickers)
    has_year = first_date <= as_of - pd.Timedelta(days=358)
    adj_then = adj_then.where(adj_then.notna(), p[p["date"] <= as_of - pd.Timedelta(days=358)]
                              .groupby("ticker")["close_adj"].last().reindex(tickers))
    if price.isna().all() and mcap.notna().any():
        price = mcap / cur["shares_outstanding"]

    o.put("price", price)
    o.put("market_cap_musd", mcap / 1e6)
    o.put("enterprise_value_musd", ev / 1e6)
    o.put("median_dollar_volume_60d_musd", adv / 1e6)

    # Large one-day distributions (spin-offs, big special dividends). The split-adjusted close and the
    # fully adjusted close (splits + dividends + spin-offs) move together except on distribution days,
    # when their ratio steps down by the distributed fraction of the price. A big distribution since
    # the start of the trailing-twelve-month window means trailing earnings, dividends and multiples
    # still include businesses or cash that are no longer in the stock (Honeywell 2026).
    dist_pct = pd.Series(np.nan, index=tickers)
    dist_days = pd.Series(np.nan, index=tickers)
    if "close_split_adj" in p and p["close_split_adj"].notna().any():
        q = p.dropna(subset=["close_split_adj", "close_adj"]).copy()
        q = q[(q["close_adj"] > 0) & (q["close_split_adj"] > 0)]
        q["r"] = q["close_split_adj"] / q["close_adj"]
        q["d"] = (1 - q["r"] / q.groupby("ticker")["r"].shift(1)) * 100
        start = (cur_end - pd.Timedelta(days=365)).reindex(q["ticker"]).values
        q = q[q["date"].values >= start]
        if not q.empty:
            idx = q.groupby("ticker")["d"].idxmax().dropna()
            top = q.loc[idx].set_index("ticker")
            dist_pct = top["d"].reindex(tickers)
            dist_days = (as_of - top["date"]).dt.days.astype(float).reindex(tickers)
    o.put("largest_distribution_1y_pct", dist_pct)
    o.put("largest_distribution_days_ago", dist_days)
    o.put("return_1y_pct", ((adj_now / adj_then - 1) * 100).where(has_year))

    # market values at each fiscal-year end (for valuation-vs-history)
    def mcap_at_fy(k: int) -> pd.Series:
        ends = P["period_end"][k].dropna()
        if ends.empty or m.empty:
            return pd.Series(np.nan, index=tickers)
        left = pd.DataFrame({"ticker": ends.index, "date": ends.values}).sort_values("date")
        right = m[["ticker", "date", "market_cap", "enterprise_value"]].sort_values("date")
        right = right.astype({"ticker": object})
        left = left.astype({"ticker": object})
        j = pd.merge_asof(left, right, on="date", by="ticker", direction="backward",
                          tolerance=pd.Timedelta(days=10))
        return j.set_index("ticker")["market_cap"].reindex(tickers)

    hist_mcap = {k: mcap_at_fy(k) for k in range(0, 6)}

    # ---- growth ------------------------------------------------------------------------
    for metric, field_fn, k in [
        ("revenue_cagr_1y_pct", lambda k: P["revenue"][k], 1),
        ("revenue_cagr_3y_pct", lambda k: P["revenue"][k], 3),
        ("revenue_cagr_5y_pct", lambda k: P["revenue"][k], 5),
        ("eps_cagr_1y_pct", lambda k: P["eps_diluted"][k], 1),
        ("eps_cagr_3y_pct", lambda k: P["eps_diluted"][k], 3),
        ("eps_cagr_5y_pct", lambda k: P["eps_diluted"][k], 5),
        ("fcf_cagr_3y_pct", fcf, 3),
        ("fcf_cagr_5y_pct", fcf, 5),
        ("ebitda_cagr_3y_pct", ebitda, 3),
        ("opinc_cagr_3y_pct", lambda k: P["operating_income"][k], 3),
    ]:
        v, s = _cagr(field_fn(0), field_fn(k), k, short(k))
        o.put(metric, v, s)
    # Robust EPS growth (scored) and starting-year distortion check on the plain CAGRs (reference).
    o.put("eps_trend_growth_pct", *trend_growth(P["eps_diluted"]))
    for k in (3, 5):
        ratio = base_vs_trend(P["eps_diluted"], base=k)
        o.put(f"eps_base_vs_trend_{k}y_x", ratio)
        name = f"eps_cagr_{k}y_pct"
        o.status[name] = o.status[name].where(~(is_distorted(ratio) & (o.status[name] == "ok")), "nm")
    o.put("rule_of_40_pct", o.values["revenue_cagr_1y_pct"] + (cur_fcf / cur["revenue"] * 100).where(cur["revenue"] > 0),
          np.where(o.status["revenue_cagr_1y_pct"] == "short", "short", "missing"))
    o.put("revenue_accel_pp", o.values["revenue_cagr_1y_pct"] - o.values["revenue_cagr_3y_pct"],
          np.where(o.status["revenue_cagr_3y_pct"] == "short", "short", "missing"))

    # ---- profitability (current) ---------------------------------------------------------
    rev = cur["revenue"]
    rev_ok = rev > 0

    def margin(num):
        v = (num / rev * 100).where(rev_ok)
        s = np.select([v.notna(), num.notna() & rev.notna()], ["ok", "nm"], "missing")
        return v, s

    for name, num in [("gross_margin_pct", cur_gp), ("operating_margin_pct", cur["operating_income"]),
                      ("ebitda_margin_pct", cur_ebitda), ("net_margin_pct", cur["net_income"]),
                      ("fcf_margin_pct", cur_fcf), ("sbc_pct_revenue", cur["sbc"])]:
        o.put(name, *margin(num))
    # reference only: FCF treats stock pay as free; this shows FCF if SBC were paid in cash
    o.put("fcf_margin_ex_sbc_pct", *margin(cur_fcf - cur["sbc"]))

    v, s = _safe_div(cur["net_income"], cur["equity"])
    o.put("roe_pct", v * 100, s)

    # ROE on AVERAGE COMMON equity (preferred stock excluded where reported) — the way banks
    # report ROE. Average of current and prior fiscal-year-end; ending value if prior is missing.
    common_cur = cur["equity"] - cur["preferred_equity"].fillna(0)
    common_prev = P["equity"][1] - P["preferred_equity"][1].fillna(0)
    avg_common = ((common_cur + common_prev) / 2).where(common_prev.notna() & (common_prev > 0), common_cur)
    v, s = _safe_div(cur["net_income"], avg_common)
    o.put("roe_avg_common_pct", v * 100, s)
    v, s = _safe_div(cur["net_income"], cur["total_assets"])
    o.put("roa_pct", v * 100, s)

    pretax, tax = cur["pretax_income"], cur["income_tax"]
    eff = (tax / pretax).where(pretax > 0)
    tax_rate = eff.where((eff >= 0) & (eff <= 0.5), 0.21).fillna(0.21)
    nopat = cur_ebit * (1 - tax_rate)
    ic_cur = cur["total_debt"].fillna(0) + cur["equity"] - cur["cash"].fillna(0)
    ic_prev = P["total_debt"][1].fillna(0) + P["equity"][1] - P["cash"][1].fillna(0)
    ic = ((ic_cur + ic_prev) / 2).where((ic_cur > 0) & (ic_prev > 0), ic_cur)
    v, s = _safe_div(nopat, ic)
    o.put("roic_pct", v * 100, s)

    # ---- margin trends (fiscal years) ----------------------------------------------------
    def fy_margin(num_fn, k):
        r = P["revenue"][k]
        return (num_fn(k) / r * 100).where(r > 0)

    def chg(num_fn, k):
        a, b = fy_margin(num_fn, 0), fy_margin(num_fn, k)
        st = np.select([a.notna() & b.notna(), short(k)], ["ok", "short"], "missing")
        return a - b, st

    o.put("gross_margin_chg_3y_pp", *chg(gp, 3))
    o.put("operating_margin_chg_3y_pp", *chg(lambda k: P["operating_income"][k], 3))
    o.put("operating_margin_chg_5y_pp", *chg(lambda k: P["operating_income"][k], 5))
    o.put("fcf_margin_chg_3y_pp", *chg(fcf, 3))
    ol = o.values["opinc_cagr_3y_pct"] - o.values["revenue_cagr_3y_pct"]
    o.put("operating_leverage_3y_pp", ol, o.status["opinc_cagr_3y_pct"].where(ol.isna(), "ok"))

    # ---- robust trends + starting-year checks: operating income, EBITDA, margins ---------------
    # Same idea as for EPS: the trend through up to 8 fiscal years is what gets scored; the plain
    # start/end figures stay visible and are marked NM when their starting year is clearly abnormal.
    def frame_of(fn) -> pd.DataFrame:
        return pd.DataFrame({k: fn(k) for k in range(MAX_OFFSET + 1)})

    rev_f = frame_of(lambda k: P["revenue"][k])
    opinc_f = frame_of(lambda k: P["operating_income"][k])
    ebitda_f = frame_of(ebitda)
    o.put("revenue_trend_growth_pct", *trend_growth(rev_f))
    o.put("opinc_trend_growth_pct", *trend_growth(opinc_f))
    o.put("ebitda_trend_growth_pct", *trend_growth(ebitda_f))

    def mark_nm(name: str, distorted: pd.Series) -> None:
        o.status[name] = o.status[name].where(~(distorted.fillna(False) & (o.status[name] == "ok")), "nm")

    for name, f in [("opinc_cagr_3y_pct", opinc_f), ("ebitda_cagr_3y_pct", ebitda_f)]:
        mark_nm(name, is_distorted(base_vs_trend(f, base=3)))
    mark_nm("operating_leverage_3y_pp", o.status["opinc_cagr_3y_pct"] == "nm")
    lt = o.values["opinc_trend_growth_pct"] - o.values["revenue_trend_growth_pct"]
    st_o, st_r = o.status["opinc_trend_growth_pct"], o.status["revenue_trend_growth_pct"]
    o.put("operating_leverage_trend_pp", lt,
          np.select([lt.notna(), (st_o == "nm") | (st_r == "nm"), (st_o == "short") | (st_r == "short")],
                    ["ok", "nm", "short"], "missing"))

    opm_f = frame_of(lambda k: fy_margin(lambda j: P["operating_income"][j], k))
    gm_f = frame_of(lambda k: fy_margin(gp, k))
    fcfm_f = frame_of(lambda k: fy_margin(fcf, k))
    for name, f in [("operating_margin_trend_3y_pp", opm_f), ("gross_margin_trend_3y_pp", gm_f)]:
        slope, st = linear_trend(f)
        o.put(name, slope * 3, st)          # pp per year x 3, comparable with the 3-yr change
    for name, f, k in [("gross_margin_chg_3y_pp", gm_f, 3), ("operating_margin_chg_3y_pp", opm_f, 3),
                       ("operating_margin_chg_5y_pp", opm_f, 5), ("fcf_margin_chg_3y_pp", fcfm_f, 3)]:
        resid, tol = base_residual_linear(f, base=k)
        if name == "operating_margin_chg_3y_pp":
            o.put("op_margin_base_vs_trend_3y_pp", resid)
        mark_nm(name, resid.abs() > tol)

    # A starting value that is a tiny fraction of the latest one (a business just turning profitable)
    # makes any compound rate meaningless (0.10 -> 2.50 EPS = 192%/yr). Base below 5% of latest -> NM.
    fcf_f = frame_of(fcf)
    for name, f, k in [("eps_cagr_3y_pct", P["eps_diluted"], 3), ("eps_cagr_5y_pct", P["eps_diluted"], 5),
                       ("opinc_cagr_3y_pct", opinc_f, 3), ("ebitda_cagr_3y_pct", ebitda_f, 3),
                       ("fcf_cagr_3y_pct", fcf_f, 3), ("fcf_cagr_5y_pct", fcf_f, 5)]:
        mark_nm(name, (f[k] > 0) & (f[0] > 0) & (f[k] < NEAR_ZERO_BASE * f[0]))
    # Free cash flow gets the same starting-year check and robust trend as EPS. FCF is often
    # negative in some years: the log trend uses positive years and is NM with 2+ negative years.
    o.put("fcf_trend_growth_pct", *trend_growth(fcf_f))
    for k in (3, 5):
        mark_nm(f"fcf_cagr_{k}y_pct", is_distorted(base_vs_trend(fcf_f, base=k)))
    mark_nm("operating_leverage_3y_pp", o.status["opinc_cagr_3y_pct"] == "nm")

    # ---- scored growth / margin measures: recent where trustworthy, robust otherwise ----------
    # The 8-yr trends lag a company whose growth has recently changed (Apple's EPS trend is 14.7%/yr
    # while its 3-yr CAGR is 6.9%). When the plain 3-yr figure is trustworthy (status ok: its base
    # year is not distorted or near zero) the scored value is the average of the two; otherwise the
    # robust trend alone.
    def blend(name: str, plain: str, robust: str) -> None:
        p_ok, r_ok = o.status[plain] == "ok", o.status[robust] == "ok"
        val = ((o.values[plain] + o.values[robust]) / 2).where(p_ok & r_ok, o.values[robust].where(r_ok))
        st = o.status[robust].where(~r_ok, "ok")
        o.put(name, val, st)

    blend("eps_growth_scored_pct", "eps_cagr_3y_pct", "eps_trend_growth_pct")
    blend("ebitda_growth_scored_pct", "ebitda_cagr_3y_pct", "ebitda_trend_growth_pct")
    blend("fcf_growth_scored_pct", "fcf_cagr_3y_pct", "fcf_trend_growth_pct")
    blend("operating_leverage_scored_pp", "operating_leverage_3y_pp", "operating_leverage_trend_pp")
    blend("operating_margin_change_scored_pp", "operating_margin_chg_3y_pp", "operating_margin_trend_3y_pp")
    blend("gross_margin_change_scored_pp", "gross_margin_chg_3y_pp", "gross_margin_trend_3y_pp")

    # ---- balance sheet -------------------------------------------------------------------
    debt, cash = cur["total_debt"], cur["cash"]
    net_debt = debt.fillna(0) - cash.fillna(0)
    net_debt = net_debt.where(debt.notna() | cash.notna())
    o.put("cash_musd", cash / 1e6)
    o.put("total_debt_musd", debt / 1e6)
    o.put("net_debt_musd", net_debt / 1e6)

    def nd_ebitda(nd, e):
        good = e > 0
        val = (nd / e).where(good)
        st = np.select([good & nd.notna(), nd.notna() & e.notna()], ["ok", "nm"], "missing")
        return val, pd.Series(st, index=tickers)

    v, s = nd_ebitda(net_debt, cur_ebitda)
    o.put("net_debt_to_ebitda_x", v, s)
    nd0 = P["total_debt"][0].fillna(0) - P["cash"][0].fillna(0)
    nd3 = P["total_debt"][3].fillna(0) - P["cash"][3].fillna(0)
    a, sa = nd_ebitda(nd0.where(P["total_debt"][0].notna()), ebitda(0))
    b, sb = nd_ebitda(nd3.where(P["total_debt"][3].notna()), ebitda(3))
    st = np.select([(sa == "ok") & (sb == "ok"), short(3), (sa == "nm") | (sb == "nm")], ["ok", "short", "nm"], "missing")
    o.put("net_debt_to_ebitda_chg_3y_x", a - b, st)

    v, s = _safe_div(debt, cur["equity"])
    o.put("debt_to_equity_x", v, s)
    v, s = _safe_div(cur["equity"], cur["total_assets"], require_pos_denominator=True)
    o.put("equity_to_assets_pct", v * 100, s)
    v, s = _safe_div(cur["operating_cash_flow"], debt)
    o.put("cfo_to_debt_pct", v * 100, s)
    v, s = _safe_div(debt, debt + cur["equity"])
    o.put("debt_to_capital_pct", (v * 100).where(cur["equity"] > 0), np.where(cur["equity"] <= 0, "nm", s))

    # ---- book value per share (key for insurers and banks) -----------------------------
    # BVPS = common equity (total equity - preferred) / shares outstanding (split-adjusted).
    # The ex-AOCI version removes accumulated other comprehensive income (mostly unrealized
    # gains/losses on the bond portfolio), which swings with interest rates, not underwriting.
    def common_eq(k):
        return P["equity"][k] - P["preferred_equity"][k].fillna(0)

    def bvps(k, ex_aoci=False):
        eq = common_eq(k) - (P["aoci"][k] if ex_aoci else 0)
        so = P["shares_outstanding"][k].fillna(P["shares_diluted_wavg"][k])
        return (eq / so).where(so > 0)

    o.put("bvps", bvps(0), np.where(bvps(0).notna(), "ok", "missing"))
    o.put("bvps_ex_aoci", bvps(0, True))
    for name, k, ex in [("bvps_growth_1y_pct", 1, False), ("bvps_ex_aoci_growth_1y_pct", 1, True),
                        ("bvps_cagr_3y_pct", 3, False), ("bvps_cagr_5y_pct", 5, False),
                        ("bvps_ex_aoci_cagr_3y_pct", 3, True), ("bvps_ex_aoci_cagr_5y_pct", 5, True)]:
        v, s = _cagr(bvps(0, ex), bvps(k, ex), k, short(k))
        o.put(name, v, s)
    v, s = _safe_div(cur["current_assets"], cur["current_liabilities"])
    o.put("current_ratio_x", v, s)

    intexp = cur["interest_expense"].abs()
    cov = (cur_ebit / intexp).where(intexp > 0).clip(upper=100)
    no_debt = (intexp.isna() | (intexp == 0)) & (debt.fillna(0) <= 0.02 * cur["total_assets"])
    cov = cov.where(~(cov.isna() & no_debt & cur_ebit.notna()), 100.0)
    o.put("interest_coverage_x", cov)

    # ---- valuation -----------------------------------------------------------------------
    ni = cur["net_income"]
    v, s = _safe_div(mcap, ni)
    o.put("pe_x", v, s)
    v, s = _safe_div(ev, cur_ebitda)
    o.put("ev_ebitda_x", v, s)
    v, s = _safe_div(ev, rev)
    o.put("ev_sales_x", v, s)
    v, s = _safe_div(mcap, cur_fcf)
    o.put("p_fcf_x", v, s)
    v, s = _safe_div(cur_fcf, mcap)
    o.put("fcf_yield_pct", v * 100, s)
    # Normalized ("through-the-cycle") valuation. Trailing multiples look cheapest exactly when a
    # cyclical's earnings peak. Normalized earnings = the AVERAGE margin over up to 8 fiscal years
    # (losses included) x CURRENT revenue: the average margin removes the boom/bust in profitability,
    # and using today's revenue keeps a growing company from being penalized for its smaller past.
    # (A trend line through EPS was tested and rejected: a boom at the end of the window reads as
    # growth, so it called NUE's 2023 near-peak earnings "below normal".)
    def avg_margin(num_fn):
        mf = frame_of(lambda k: fy_margin(num_fn, k))
        return cycle_average(mf)

    rev0 = P["revenue"][0]
    nm_net, nm_net_st = avg_margin(lambda k: P["net_income"][k])
    nm_ebitda, nm_ebitda_st = avg_margin(ebitda)
    nm_fcf, nm_fcf_st = avg_margin(fcf)
    norm_ni = nm_net / 100 * rev0
    sh_now = cur["shares_diluted_wavg"].fillna(cur["shares_outstanding"])

    def carry(v, s, base_st):
        s = pd.Series(s, index=tickers).astype(object)
        return v, s.where(~((s == "missing") & base_st.isin(["short", "nm"])), base_st)

    o.put("net_margin_cycle_avg_pct", nm_net, nm_net_st)
    o.put("eps_normalized", *carry((norm_ni / sh_now).where(sh_now > 0), np.where(norm_ni.notna(), "ok", "missing"),
                                   nm_net_st))
    v, s = _safe_div(mcap, norm_ni)
    o.put("pe_normalized_x", *carry(v, s, nm_net_st))
    v, s = _safe_div(ev, nm_ebitda / 100 * rev0)
    o.put("ev_ebitda_normalized_x", *carry(v, s, nm_ebitda_st))
    v, s = _safe_div(nm_fcf / 100 * rev0, mcap)
    o.put("fcf_yield_normalized_pct", *carry(v * 100, s, nm_fcf_st))
    v, s = _safe_div(P["net_income"][0], norm_ni)
    o.put("earnings_vs_normalized_x", *carry(v, s, nm_net_st))
    # price-to-book uses COMMON equity: preferred stock belongs to preferred holders, not to the
    # common shares the market cap prices
    cur_common = cur["equity"] - cur["preferred_equity"].fillna(0)
    v, s = _safe_div(mcap, cur_common)
    o.put("pb_x", v, s)
    v, s = _safe_div(mcap, cur_common - cur["intangibles"].fillna(0))
    o.put("ptbv_x", v, s)
    # ---- REIT measures ----------------------------------------------------------------
    # FFO (NAREIT): net income to common + real-estate D&A + impairments - gains on property sales.
    # Without impairment/gain data FFO falls back to net income + D&A (flagged by ffo_nareit_adjusted).
    def ffo_of(ni_, da_, imp_, gain_):
        return ni_ + da_ + imp_.fillna(0) - gain_.fillna(0)

    # Impairments and gains exist only for fiscal years (FY tags, or the SEC supplement for Sharadar). When
    # the trailing-twelve-month row has none, FFO and EBITDAre use the latest fiscal year throughout, so net
    # income and its adjustments cover the same period: a TTM loss from a new impairment (Sun Communities
    # TTM net income -$871M vs FY +$1,361M) would otherwise be read as negative FFO.
    _adj = ("real_estate_impairment", "gain_on_sale_real_estate")
    ttm_has_adj = pd.concat([ttm[c].notna() for c in _adj if c in ttm], axis=1).any(axis=1) \
        if all(c in ttm for c in _adj) else pd.Series(False, index=tickers)
    fy_has_adj = F(_adj[0], 0).notna() | F(_adj[1], 0).notna()
    fy_basis = use_ttm & ~ttm_has_adj & fy_has_adj
    re = {f: cur[f].where(~fy_basis, P[f][0]) for f in
          ("net_income", "d_and_a", "real_estate_impairment", "gain_on_sale_real_estate", "shares_diluted_wavg",
           "shares_outstanding", "pretax_income", "interest_expense", "dividends_paid")}
    re_ebitda = cur_ebitda.where(~fy_basis, ebitda(0))
    re_div = re["dividends_paid"]
    ffo = ffo_of(re["net_income"], re["d_and_a"], re["real_estate_impairment"], re["gain_on_sale_real_estate"])
    v, s = _safe_div(mcap, ffo)
    o.put("p_ffo_x", v, s)
    o.put("ffo_nareit_adjusted",
          (cur["real_estate_impairment"].notna() | cur["gain_on_sale_real_estate"].notna()).astype(float)
          .where(ffo.notna()))

    def ffo_ps(k):
        f = ffo_of(P["net_income"][k], P["d_and_a"][k], P["real_estate_impairment"][k],
                   P["gain_on_sale_real_estate"][k])
        return (f / shares(k)).where(shares(k) > 0)

    o.put("ffo_per_share", (ffo / re["shares_diluted_wavg"].fillna(re["shares_outstanding"])))
    for name, k in [("ffo_ps_cagr_3y_pct", 3), ("ffo_ps_cagr_5y_pct", 5)]:
        v, s = _cagr(ffo_ps(0), ffo_ps(k), k, short(k))
        o.put(name, v, s)
    # EBITDAre (NAREIT): net income + interest + taxes + D&A + real-estate impairments - gains on
    # property sales = pretax + interest + D&A + impairments - gains. Falls back to EBITDA + impairments
    # - gains when pretax or interest is missing.
    def ebitdare_of(src, ebitda_):
        adj = src["real_estate_impairment"].fillna(0) - src["gain_on_sale_real_estate"].fillna(0)
        nareit = src["pretax_income"] + src["interest_expense"].abs() + src["d_and_a"] + adj
        return nareit.fillna(ebitda_ + adj)

    ebitdare = ebitdare_of(re, re_ebitda)
    v, s = nd_ebitda(net_debt, ebitdare)
    o.put("net_debt_to_ebitdare_x", v, s)

    def ebitdare_fy(k):
        return ebitdare_of({f: P[f][k] for f in ("real_estate_impairment", "gain_on_sale_real_estate",
                                                 "pretax_income", "interest_expense", "d_and_a")}, ebitda(k))

    a, sa = nd_ebitda(nd0.where(P["total_debt"][0].notna()), ebitdare_fy(0))
    b, sb = nd_ebitda(nd3.where(P["total_debt"][3].notna()), ebitdare_fy(3))
    st = np.select([(sa == "ok") & (sb == "ok"), short(3), (sa == "nm") | (sb == "nm")], ["ok", "short", "nm"], "missing")
    o.put("net_debt_to_ebitdare_chg_3y_x", a - b, st)
    re_int = re["interest_expense"].abs()
    cov_re = (ebitdare / re_int).where(re_int > 0).clip(upper=100)
    cov_re = cov_re.where(~(cov_re.isna() & no_debt & ebitdare.notna()), 100.0)
    o.put("ebitdare_interest_coverage_x", cov_re)

    # PEG on the scored EPS growth. Growth below 3%/yr makes the ratio explode (P/E 20 / 0.5 = 40)
    # -> NM; growth above 40%/yr is capped, because trailing hyper-growth is not a sustainable rate.
    eps_g = o.values["eps_growth_scored_pct"]
    pe = o.values["pe_x"]
    peg = (pe / eps_g.clip(upper=PEG_GROWTH_CAP)).where((eps_g >= PEG_MIN_GROWTH) & pe.notna())
    st = np.select([peg.notna(), (o.status["pe_x"] == "nm") | (eps_g < PEG_MIN_GROWTH) |
                    (o.status["eps_growth_scored_pct"] == "nm")], ["ok", "nm"], "missing")
    o.put("peg_hist_x", peg, st)

    def vs_history(cur_val: pd.Series, hist_fn) -> Tuple[pd.Series, np.ndarray]:
        hist = pd.concat([hist_fn(k) for k in range(1, 6)], axis=1)
        hist = hist.where(hist > 0)
        med = hist.median(axis=1).where(hist.notna().sum(axis=1) >= 3)
        val = (cur_val / med - 1) * 100
        st = np.select([val.notna(), cur_val.isna() & mcap.notna()], ["ok", "nm"], "missing")
        return val, st

    o.put("pe_vs_5y_median_pct", *vs_history(pe, lambda k: hist_mcap[k] / P["net_income"][k].where(P["net_income"][k] > 0)))
    ev_hist = {k: hist_mcap[k] + P["total_debt"][k].fillna(0) - P["cash"][k].fillna(0) for k in range(1, 6)}
    o.put("ev_ebitda_vs_5y_median_pct",
          *vs_history(o.values["ev_ebitda_x"], lambda k: ev_hist[k] / ebitda(k).where(ebitda(k) > 0)))

    div_yield = (cur_div / mcap * 100).where(mcap > 0)
    alt = (cur["dps"] / price * 100).where(price > 0)
    div_yield = div_yield.fillna(alt)
    o.put("dividend_yield_pct", div_yield)
    yk = pd.concat([(P["dividends_paid"][k] / hist_mcap[k] * 100) for k in range(1, 6)], axis=1)
    yk = yk.where(yk > 0)
    o.put("dividend_yield_5y_median_pct", yk.median(axis=1).where(yk.notna().sum(axis=1) >= 3))

    if estimates is not None and not estimates.empty:
        e = estimates.dropna(subset=["fiscal_period_end"]).copy()
        e = e.merge(pd.DataFrame({"ticker": tickers, "fy0": P["period_end"][0].values}), on="ticker")
        e = e[e["fiscal_period_end"] > e["fy0"]].sort_values("fiscal_period_end").groupby("ticker").head(1)
        eps_next = e.set_index("ticker")["eps_estimate"].reindex(tickers)
        v, s = _safe_div(price, eps_next)
        o.put("forward_pe_x", v, s)
    else:
        o.put("forward_pe_x", pd.Series(np.nan, index=tickers))

    # ---- capital allocation --------------------------------------------------------------
    v, s = _cagr(shares(0), shares(1), 1, short(1))
    o.put("share_change_1y_pct", v, s)
    v, s = _cagr(shares(0), shares(3), 3, short(3))
    o.put("share_change_3y_cagr_pct", v, s)

    gross_bb = cur["share_repurchases"]
    net_iss = cur["net_common_issuance"]
    bb = gross_bb.fillna((-net_iss).clip(lower=0))
    o.put("buyback_yield_pct", (bb / mcap * 100).where(mcap > 0))
    nby = (-net_iss / mcap * 100).where(mcap > 0)
    o.put("net_buyback_yield_pct", nby)
    o.put("shareholder_yield_pct", div_yield + nby.fillna(0).where(div_yield.notna()))

    # Special dividends: a year whose DPS is >1.4x BOTH neighbouring years is treated as containing a
    # one-off special payment. It still counts as a paying year, but growth, streak and cut tests use
    # the regular-dividend series (that year skipped), so a special doesn't look like a later "cut".
    _src = [dps_src(k) for k in range(MAX_OFFSET + 1)]
    D = pd.concat([x[0] for x in _src], axis=1)
    D.columns = list(range(MAX_OFFSET + 1))
    CALC = pd.concat([x[1] for x in _src], axis=1)
    CALC.columns = D.columns
    # bridge ONE unknown year between two known years when the dividend rose across it
    # (geometric midpoint, marked as calculated)
    for k in range(1, MAX_OFFSET):
        gap = D[k].isna() & (D[k - 1] > 0) & (D[k + 1] > 0) & (D[k - 1] > D[k + 1]) & (P["period_end"][k].notna())
        D.loc[gap, k] = np.sqrt(D.loc[gap, k - 1] * D.loc[gap, k + 1])
        CALC.loc[gap, k] = True
    spike = pd.DataFrame(False, index=tickers, columns=D.columns)
    for k in range(1, MAX_OFFSET):
        nb = pd.concat([D[k - 1], D[k + 1]], axis=1).max(axis=1)
        spike[k] = (D[k - 1] > 0) & (D[k + 1] > 0) & (D[k] > 1.4 * nb)
    R = D.mask(spike)                                   # regular dividends
    older = R.shift(-1, axis=1).bfill(axis=1)          # nearest older regular value

    never_paid = (D.fillna(0) == 0).all(axis=1) & D[0].notna()
    for name, k in [("dps_cagr_3y_pct", 3), ("dps_cagr_5y_pct", 5)]:
        v, s = _cagr(R[0], R[k], k, short(k))
        o.put(name, v, s)
        o.status[name] = o.status[name].where(~never_paid, "na")      # a non-payer has no dividend growth

    paying = pd.Series(0, index=tickers, dtype=float)
    streak = pd.Series(0, index=tickers, dtype=float)
    alive_pay = pd.Series(True, index=tickers)
    alive_grow = pd.Series(True, index=tickers)
    for k in range(0, MAX_OFFSET + 1):
        alive_pay &= D[k] > 0
        paying += alive_pay.astype(float)
        if k < MAX_OFFSET:
            # when either year's DPS is calculated from dividends paid / shares, timing noise of a
            # percent or two is not evidence of a cut: within DPS_CALC_TOLERANCE counts as continuing
            calc_pair = CALC[k] | CALC[k + 1]
            grew = (R[k] > 0) & ((R[k] > older[k] * 1.001)
                                 | (calc_pair & (R[k] >= older[k] * (1 - DPS_CALC_TOLERANCE))))
            alive_grow &= spike[k] | grew
            streak += (alive_grow & ~spike[k]).astype(float)
    has_div_data = D[0].notna()
    o.put("dividend_paying_years", paying.where(has_div_data))
    o.put("dividend_growth_streak_years", streak.where(has_div_data))
    # A DPS drop must also show in dividends paid per share when both years have it: a DPS figure out of
    # line with the cash actually paid (seen in Sharadar data for one REIT: DPS -24%
    # while dividends paid per share rose 3%) is not a cut.
    pps = {k: (P["dividends_paid"][k] / shares(k)).where(shares(k) > 0) for k in range(0, 6)}
    cut = pd.Series(0.0, index=tickers)
    for k in range(0, 5):
        now, prev = R[k], older[k]
        paid_held = pps[k].notna() & (pps[k + 1] > 0) & (pps[k] >= 0.97 * pps[k + 1])
        cut = cut.where(~(~spike[k] & now.notna() & (prev > 0) & (now < 0.9 * prev) & ~paid_held), 1.0)
    o.put("dividend_cut_5y", cut.where(has_div_data))
    o.put("special_dividends_detected", spike.iloc[:, :6].sum(axis=1).astype(float).where(has_div_data))

    def payout(div, base):
        both = div.notna() & base.notna()
        zero = both & (div <= 0)
        good = both & (base > 0)
        val = (div / base * 100).where(good)
        val = val.where(~zero, 0.0)
        st = np.select([val.notna(), both], ["ok", "nm"], "missing")
        return val, st

    o.put("payout_ratio_pct", *payout(cur_div, ni))
    o.put("fcf_payout_ratio_pct", *payout(cur_div, cur_fcf))
    div3 = sum(P["dividends_paid"][k] for k in range(3))
    fcf3 = sum(fcf(k) for k in range(3))
    o.put("fcf_payout_3y_pct", *payout(div3, fcf3))
    o.put("ffo_payout_pct", *payout(re_div, ffo))

    # ---- quality / consistency -----------------------------------------------------------
    def share_positive(fn, need=3):
        vals = pd.concat([fn(k) for k in range(5)], axis=1)
        cnt = vals.notna().sum(axis=1)
        v = ((vals > 0).sum(axis=1) / cnt * 100).where(cnt >= need)
        st = np.where(v.notna(), "ok", np.where((n_years > 0) & (n_years < need), "short", "missing"))
        return v, st

    o.put("earnings_positive_share_5y_pct", *share_positive(lambda k: P["net_income"][k]))
    o.put("fcf_positive_share_5y_pct", *share_positive(fcf))
    om = pd.concat([fy_margin(lambda k: P["operating_income"][k], k) for k in range(5)], axis=1)
    sd = om.std(axis=1, ddof=1).where(om.notna().sum(axis=1) >= 4)
    o.put("operating_margin_stdev_5y_pp", sd, np.where(sd.notna(), "ok", np.where(n_years < 4, "short", "missing")))
    yoy = pd.concat([(P["revenue"][k] < P["revenue"][k + 1]).where(P["revenue"][k].notna() & P["revenue"][k + 1].notna())
                     for k in range(5)], axis=1)
    dec = yoy.sum(axis=1).where(yoy.notna().sum(axis=1) >= 3)
    o.put("revenue_decline_years_5y", dec, np.where(dec.notna(), "ok", np.where(n_years < 4, "short", "missing")))
    ni3 = sum(P["net_income"][k] for k in range(3))
    v, s = _safe_div(fcf3, ni3)
    o.put("fcf_conversion_3y_pct", v * 100, s)
    om10 = pd.concat([fy_margin(lambda k: P["operating_income"][k], k) for k in range(MAX_OFFSET)], axis=1)
    avg10 = om10.mean(axis=1).where(om10.notna().sum(axis=1) >= 5)
    v, s = _safe_div(om10[0], avg10)
    s = np.where((avg10.isna()) & (n_years < 5), "short", s)
    o.put("margin_vs_10y_avg_x", v, s)

    # ---- data / meta ---------------------------------------------------------------------
    o.put("fiscal_years_available", n_years.astype(float))
    o.put("latest_period_end", pd.Series(np.nan, index=tickers))  # filled as dates below
    o.put("latest_filing_date", pd.Series(np.nan, index=tickers))
    age = (as_of - cur_filed).dt.days
    o.put("filing_age_days", age.astype(float))
    o.put("uses_ttm", use_ttm.astype(float))
    o.put("splits_detected", n_splits.where(n_years > 0))

    values, status = o.frames()
    values["latest_period_end"] = cur_end.dt.date
    values["latest_filing_date"] = cur_filed.dt.date
    status.loc[cur_end.notna(), "latest_period_end"] = "ok"
    status.loc[cur_filed.notna(), "latest_filing_date"] = "ok"
    return values, status
