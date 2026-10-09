"""Single source of truth for every metric the screener computes.

kind:
  raw_value -> an absolute figure taken (or trivially summed) from licensed data,
               e.g. cash in $M. Excluded from retained/public outputs for licensed providers.
  derived   -> a ratio, growth rate, margin, count or score computed by this project.
  meta      -> dates/counts describing data coverage.

The registry also generates docs/DATA_DICTIONARY.md (python -m screener docs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MetricDef:
    name: str
    label: str
    unit: str          # pct | pp | x | musd | usd | count | flag | date | days
    kind: str          # raw_value | derived | meta
    category: str
    description: str


_M = [
    # ---- market --------------------------------------------------------------------------------
    ("price", "Share price", "usd", "raw_value", "market", "Latest unadjusted close."),
    ("market_cap_musd", "Market cap ($M)", "musd", "raw_value", "market", "Latest market capitalization."),
    ("enterprise_value_musd", "Enterprise value ($M)", "musd", "raw_value", "market",
     "Provider EV, else market cap + total debt - cash."),
    ("median_dollar_volume_60d_musd", "Median $ volume 60d ($M)", "musd", "raw_value", "market",
     "Median daily close x volume over the last 60 trading days."),
    ("return_1y_pct", "1-yr price return", "pct", "derived", "market", "Total-return-adjusted price change over ~1 year."),
    ("largest_distribution_1y_pct", "Largest one-day distribution (% of price)", "pct", "derived", "market",
     "Biggest one-day step between the split-adjusted and the fully adjusted close since the start of the "
     "trailing-twelve-month window: the share of the price paid out in one go (spin-off, special or large "
     "dividend). Needs both price series (Sharadar)."),
    ("largest_distribution_days_ago", "Days since that distribution", "count", "derived", "market",
     "Calendar days from that distribution to the data date."),
    # ---- growth --------------------------------------------------------------------------------
    ("revenue_cagr_1y_pct", "Revenue growth 1y", "pct", "derived", "growth", "Latest FY vs prior FY."),
    ("revenue_cagr_3y_pct", "Revenue CAGR 3y", "pct", "derived", "growth", "Compound annual growth, FY0 vs FY-3."),
    ("revenue_cagr_5y_pct", "Revenue CAGR 5y", "pct", "derived", "growth", "Compound annual growth, FY0 vs FY-5."),
    ("eps_cagr_1y_pct", "Diluted EPS growth 1y", "pct", "derived", "growth", "NM when either EPS is <= 0."),
    ("eps_cagr_3y_pct", "Diluted EPS CAGR 3y", "pct", "derived", "growth",
     "Reference only (not scored). NM when either EPS is <= 0, or when the starting year (FY-3) is "
     "distorted: more than 1.5x above or below the robust trend of the other years (value kept)."),
    ("eps_cagr_5y_pct", "Diluted EPS CAGR 5y", "pct", "derived", "growth",
     "Reference only (not scored). NM when either EPS is <= 0 or the starting year (FY-5) is distorted."),
    ("eps_trend_growth_pct", "Diluted EPS trend growth (robust)", "pct", "derived", "growth",
     "Annualized Theil-Sen slope of log diluted EPS over up to 8 fiscal years (FY0..FY-7): the median of "
     "all pairwise growth rates, so one or two write-off, impairment or peak years barely move it. Needs "
     "the latest year and 5+ positive years; NM if the latest year is a loss or 2+ years are losses."),
    ("eps_growth_scored_pct", "EPS growth (scored)", "pct", "derived", "growth",
     "Average of the 3y EPS CAGR and the robust 8y EPS trend when the 3y figure is trustworthy (status ok); "
     "the robust trend alone when the 3y base year is distorted or near zero. Recent where reliable, "
     "robust otherwise."),
    ("eps_base_vs_trend_3y_x", "EPS FY-3 vs trend of other years", "x", "derived", "growth",
     "Starting-year diluted EPS of the 3y CAGR / EPS implied by a robust trend through the other years "
     "(up to FY0..FY-7). Outside 0.67-1.5 = distorted base year."),
    ("eps_base_vs_trend_5y_x", "EPS FY-5 vs trend of other years", "x", "derived", "growth",
     "As above for the 5y CAGR's starting year (FY-5)."),
    ("fcf_cagr_3y_pct", "FCF CAGR 3y", "pct", "derived", "growth",
     "Free cash flow. Reference only. NM when either end is <= 0, the FY-3 base is distorted (>1.5x off the "
     "trend of the other years) or near zero (< 5% of the latest)."),
    ("fcf_cagr_5y_pct", "FCF CAGR 5y", "pct", "derived", "growth", "As above, FY-5 base."),
    ("fcf_trend_growth_pct", "FCF trend growth (robust)", "pct", "derived", "growth",
     "Theil-Sen trend of log FCF over up to 8 FY (positive years; NM with 2+ negative years or a negative latest year)."),
    ("fcf_growth_scored_pct", "FCF growth (scored)", "pct", "derived", "growth",
     "Same blend as EPS growth (scored): 3y FCF CAGR and robust trend, or the trend alone."),
    ("ebitda_cagr_3y_pct", "EBITDA CAGR 3y", "pct", "derived", "growth",
     "Reference only (not scored). NM when either end is <= 0 or the FY-3 starting year is distorted "
     "(more than 1.5x off the robust trend of the other years)."),
    ("opinc_cagr_3y_pct", "Operating income CAGR 3y", "pct", "derived", "growth",
     "Reference only. NM when either end is <= 0 or the FY-3 starting year is distorted."),
    ("revenue_trend_growth_pct", "Revenue trend growth (robust)", "pct", "derived", "growth",
     "Theil-Sen trend of log revenue over up to 8 FY (same method as EPS trend growth)."),
    ("opinc_trend_growth_pct", "Operating income trend growth (robust)", "pct", "derived", "growth",
     "Theil-Sen trend of log operating income (EBIT fallback) over up to 8 FY."),
    ("ebitda_trend_growth_pct", "EBITDA trend growth (robust)", "pct", "derived", "growth",
     "Theil-Sen trend of log EBITDA over up to 8 FY."),
    ("ebitda_growth_scored_pct", "EBITDA growth (scored)", "pct", "derived", "growth",
     "Same blend as EPS growth (scored): 3y EBITDA CAGR and robust trend, or the trend alone."),
    ("revenue_accel_pp", "Revenue acceleration", "pp", "derived", "growth", "1-yr growth minus 3-yr CAGR."),
    ("rule_of_40_pct", "Rule of 40 (revenue growth + FCF margin)", "pct", "derived", "growth",
     "1-yr revenue growth + FCF margin. Reference only: both parts are already scored separately."),
    # ---- profitability -------------------------------------------------------------------------
    ("gross_margin_pct", "Gross margin", "pct", "derived", "profitability", "Gross profit / revenue (current)."),
    ("operating_margin_pct", "Operating margin", "pct", "derived", "profitability",
     "Operating income / revenue. When operating income is not reported, EBIT (pretax income + interest)."),
    ("ebitda_margin_pct", "EBITDA margin", "pct", "derived", "profitability", "EBITDA / revenue."),
    ("net_margin_pct", "Net margin", "pct", "derived", "profitability", "Net income to common / revenue."),
    ("fcf_margin_pct", "FCF margin", "pct", "derived", "profitability", "(Operating cash flow - capex) / revenue."),
    ("fcf_margin_ex_sbc_pct", "FCF margin after SBC", "pct", "derived", "profitability",
     "(FCF - stock-based compensation) / revenue. Reference only: SBC is penalized once, in capital allocation."),
    ("roe_pct", "ROE", "pct", "derived", "profitability", "Net income / equity; NM when equity <= 0."),
    ("roe_avg_common_pct", "ROE (avg common equity)", "pct", "derived", "profitability",
     "Net income to common / average of current and prior year-end common equity (total equity minus "
     "preferred stock where reported). The bank-reporting convention; used for banks."),
    ("roa_pct", "ROA", "pct", "derived", "profitability", "Net income / total assets."),
    ("roic_pct", "ROIC", "pct", "derived", "profitability",
     "EBIT x (1 - tax rate) / average invested capital (debt + equity - cash). Tax rate = effective, "
     "bounded 0-50%, else 21%."),
    # ---- trends --------------------------------------------------------------------------------
    ("gross_margin_chg_3y_pp", "Gross margin change 3y", "pp", "derived", "trend",
     "FY0 minus FY-3, percentage points. Reference only; NM when the FY-3 margin is distorted: more than "
     "max(3pp, 3x the typical deviation of the other years) away from their robust trend."),
    ("operating_margin_chg_3y_pp", "Operating margin change 3y", "pp", "derived", "trend",
     "FY0 minus FY-3. Reference only; NM when the FY-3 margin is distorted (see gross margin change)."),
    ("operating_margin_chg_5y_pp", "Operating margin change 5y", "pp", "derived", "trend",
     "FY0 minus FY-5. NM when the FY-5 margin is distorted."),
    ("fcf_margin_chg_3y_pp", "FCF margin change 3y", "pp", "derived", "trend",
     "FY0 minus FY-3. NM when the FY-3 margin is distorted."),
    ("op_margin_base_vs_trend_3y_pp", "Op margin FY-3 vs trend of other years", "pp", "derived", "trend",
     "FY-3 operating margin minus the robust trend of the other years (pp). Large = distorted base year."),
    ("operating_margin_trend_3y_pp", "Operating margin trend (robust, per 3y)", "pp", "derived", "trend",
     "Theil-Sen slope of operating margin over up to 8 FY, x3 so it reads like a 3-yr change."),
    ("gross_margin_trend_3y_pp", "Gross margin trend (robust, per 3y)", "pp", "derived", "trend",
     "Theil-Sen slope of gross margin over up to 8 FY, x3."),
    ("operating_leverage_3y_pp", "Operating leverage 3y", "pp", "derived", "trend",
     "Operating income CAGR minus revenue CAGR (3y). Reference only; NM when the op-income base year is distorted."),
    ("operating_leverage_trend_pp", "Operating leverage (robust)", "pp", "derived", "trend",
     "Operating income trend growth minus revenue trend growth (both robust, up to 8 FY)."),
    ("operating_leverage_scored_pp", "Operating leverage (scored)", "pp", "derived", "trend",
     "Blend of operating leverage 3y and the robust version (or robust alone when the 3y base is distorted)."),
    ("operating_margin_change_scored_pp", "Operating margin change (scored)", "pp", "derived", "trend",
     "Blend of the 3y operating-margin change and the robust trend per 3y (or the trend alone)."),
    ("gross_margin_change_scored_pp", "Gross margin change (scored)", "pp", "derived", "trend",
     "Blend of the 3y gross-margin change and the robust trend per 3y (or the trend alone)."),
    ("net_debt_to_ebitda_chg_3y_x", "Net debt/EBITDA change 3y", "x", "derived", "trend", "FY0 minus FY-3."),
    # ---- financial strength --------------------------------------------------------------------
    ("cash_musd", "Cash ($M)", "musd", "raw_value", "balance_sheet", "Cash and equivalents."),
    ("total_debt_musd", "Total debt ($M)", "musd", "raw_value", "balance_sheet", "Short + long-term debt."),
    ("net_debt_musd", "Net debt ($M)", "musd", "raw_value", "balance_sheet", "Total debt - cash."),
    ("net_debt_to_ebitda_x", "Net debt / EBITDA", "x", "derived", "balance_sheet",
     "Negative = net cash. NM when EBITDA <= 0."),
    ("debt_to_equity_x", "Debt / equity", "x", "derived", "balance_sheet", "NM when equity <= 0."),
    ("equity_to_assets_pct", "Equity / assets", "pct", "derived", "balance_sheet", "Capital ratio (key for banks)."),
    ("current_ratio_x", "Current ratio", "x", "derived", "balance_sheet", "Current assets / current liabilities."),
    ("interest_coverage_x", "Interest coverage", "x", "derived", "balance_sheet",
     "EBIT / interest expense, capped at 100x (100 = no meaningful debt)."),
    ("cfo_to_debt_pct", "Operating cash flow / debt", "pct", "derived", "balance_sheet",
     "Proxy for the rating agencies' FFO/debt (utilities): operating cash flow (including working-capital "
     "swings) / total debt. Higher = less levered."),
    ("debt_to_capital_pct", "Debt / capital", "pct", "derived", "balance_sheet",
     "Total debt / (total debt + shareholders' equity); the standard insurer leverage ratio. NM when equity <= 0."),
    # ---- book value ----------------------------------------------------------------------------
    ("bvps", "Book value per share", "usd", "derived", "book_value",
     "(Total equity - preferred stock) / shares outstanding (split-adjusted)."),
    ("bvps_ex_aoci", "Book value/share ex-AOCI", "usd", "derived", "book_value",
     "As above, excluding accumulated other comprehensive income (interest-rate-driven unrealized "
     "gains/losses). Companies define 'adjusted' book value differently; some exclude only unrealized "
     "investment gains/losses."),
    ("bvps_growth_1y_pct", "Book value/share growth 1y", "pct", "derived", "book_value", "Latest FY vs prior FY."),
    ("bvps_ex_aoci_growth_1y_pct", "BVPS ex-AOCI growth 1y", "pct", "derived", "book_value", "Latest FY vs prior FY."),
    ("bvps_cagr_3y_pct", "Book value/share CAGR 3y", "pct", "derived", "book_value", "NM when either end is <= 0."),
    ("bvps_cagr_5y_pct", "Book value/share CAGR 5y", "pct", "derived", "book_value", "NM when either end is <= 0."),
    ("bvps_ex_aoci_cagr_3y_pct", "BVPS ex-AOCI CAGR 3y", "pct", "derived", "book_value",
     "Book value growth without interest-rate swings in the bond portfolio."),
    ("bvps_ex_aoci_cagr_5y_pct", "BVPS ex-AOCI CAGR 5y", "pct", "derived", "book_value", "As above, 5 years."),
    # ---- valuation -----------------------------------------------------------------------------
    ("pe_x", "P/E (trailing)", "x", "derived", "valuation", "Market cap / net income; NM when earnings <= 0."),
    ("forward_pe_x", "Forward P/E", "x", "derived", "valuation", "Requires the optional estimates module."),
    ("ev_ebitda_x", "EV / EBITDA", "x", "derived", "valuation", "NM when EBITDA <= 0."),
    ("ev_sales_x", "EV / sales", "x", "derived", "valuation", "Enterprise value / revenue."),
    ("p_fcf_x", "Price / FCF", "x", "derived", "valuation", "NM when FCF <= 0."),
    ("fcf_yield_pct", "FCF yield", "pct", "derived", "valuation", "FCF / market cap (negative allowed)."),
    ("net_margin_cycle_avg_pct", "Net margin, 8y average", "pct", "derived", "valuation",
     "Average net margin over up to 8 fiscal years (min 5), losses included."),
    ("eps_normalized", "Normalized EPS", "usd", "derived", "valuation",
     "8y-average net margin x latest revenue / current diluted shares: through-the-cycle earnings that "
     "do not penalize a company for having been smaller in the past."),
    ("earnings_vs_normalized_x", "Earnings vs normalized", "x", "derived", "valuation",
     "Latest FY net income / normalized net income. Well above 1 in a cyclical = earnings near a peak; "
     "well below 1 = near a trough."),
    ("pe_normalized_x", "P/E on normalized EPS", "x", "derived", "valuation",
     "Market cap / normalized net income (8y-average net margin x latest revenue). Scored instead of P/E for cyclicals."),
    ("ev_ebitda_normalized_x", "EV / normalized EBITDA", "x", "derived", "valuation",
     "EV / (8y-average EBITDA margin x latest revenue). Scored instead of EV/EBITDA for cyclicals."),
    ("fcf_yield_normalized_pct", "FCF yield on normalized FCF", "pct", "derived", "valuation",
     "(8y-average FCF margin x latest revenue) / market cap. Scored instead of FCF yield for cyclicals."),
    ("pb_x", "Price / book", "x", "derived", "valuation",
     "Market cap / common equity (preferred stock excluded). NM when common equity <= 0."),
    ("ptbv_x", "Price / tangible book", "x", "derived", "valuation",
     "Market cap / (common equity - goodwill - intangibles). NM when tangible equity <= 0."),
    ("peg_hist_x", "PEG (historical)", "x", "derived", "valuation",
     "Trailing P/E / EPS growth (scored), with growth capped at 40%/yr; NM below 3%/yr growth. Uses history, not estimates."),
    ("pe_vs_5y_median_pct", "P/E vs own 5y median", "pct", "derived", "valuation",
     "Current P/E relative to the median FY-end P/E of the prior 5 years (needs 3+)."),
    ("ev_ebitda_vs_5y_median_pct", "EV/EBITDA vs own 5y median", "pct", "derived", "valuation", "As above."),
    ("p_ffo_x", "Price / FFO (REIT)", "x", "derived", "valuation",
     "Market cap / FFO. FFO (NAREIT) = net income to common + D&A + real-estate impairments - gains on "
     "property sales; net income + D&A when impairment/gain data are unavailable."),
    ("dividend_yield_pct", "Dividend yield", "pct", "derived", "valuation", "Trailing dividends paid / market cap."),
    ("dividend_yield_5y_median_pct", "Dividend yield, own 5y median", "pct", "derived", "valuation",
     "Median FY-end yield over the prior 5 years (yield-trap check)."),
    # ---- capital allocation --------------------------------------------------------------------
    ("share_change_1y_pct", "Share count change 1y", "pct", "derived", "capital_allocation", "Diluted weighted shares."),
    ("share_change_3y_cagr_pct", "Share count CAGR 3y", "pct", "derived", "capital_allocation",
     "Positive = dilution, negative = net buybacks."),
    ("sbc_pct_revenue", "SBC / revenue", "pct", "derived", "capital_allocation", "Stock-based compensation intensity."),
    ("buyback_yield_pct", "Buyback yield", "pct", "derived", "capital_allocation",
     "Gross repurchases / market cap (net repurchases when gross is unavailable)."),
    ("net_buyback_yield_pct", "Net buyback yield", "pct", "derived", "capital_allocation",
     "(Repurchases - issuance) / market cap."),
    ("shareholder_yield_pct", "Shareholder yield", "pct", "derived", "capital_allocation",
     "Dividend yield + net buyback yield."),
    ("dps_cagr_3y_pct", "Dividend/share CAGR 3y", "pct", "derived", "capital_allocation", "NM when not paying 3 yrs ago."),
    ("dps_cagr_5y_pct", "Dividend/share CAGR 5y", "pct", "derived", "capital_allocation", "NM when not paying 5 yrs ago."),
    ("dividend_paying_years", "Consecutive dividend years", "count", "derived", "capital_allocation", "From the latest FY back."),
    ("dividend_growth_streak_years", "Dividend growth streak", "count", "derived", "capital_allocation",
     "Consecutive FYs with a higher regular dividend per share (special-dividend years skipped)."),
    ("dividend_cut_5y", "Dividend cut in 5y", "flag", "derived", "capital_allocation",
     "1 if regular DPS fell >10% in any of the last 5 FYs (special dividends excluded)."),
    ("special_dividends_detected", "Special-dividend years (5y)", "count", "derived", "capital_allocation",
     "FYs whose DPS was >1.4x both neighbouring years; excluded from dividend growth, streak and cut tests."),
    ("payout_ratio_pct", "Payout ratio (earnings)", "pct", "derived", "capital_allocation",
     "Dividends / net income; NM when net income <= 0 while paying."),
    ("fcf_payout_ratio_pct", "Payout ratio (FCF)", "pct", "derived", "capital_allocation", "Dividends / FCF (current)."),
    ("fcf_payout_3y_pct", "Payout ratio (FCF, 3y)", "pct", "derived", "capital_allocation",
     "3-yr dividends / 3-yr FCF; smooths lumpy cash flows."),
    ("ffo_payout_pct", "Payout ratio (FFO)", "pct", "derived", "capital_allocation", "Dividends paid / FFO (REITs)."),
    # ---- REIT-specific -------------------------------------------------------------------------
    ("ffo_per_share", "FFO per share (REIT)", "usd", "derived", "reit",
     "FFO / diluted weighted-average shares (split-adjusted)."),
    ("ffo_ps_cagr_3y_pct", "FFO/share CAGR 3y (REIT)", "pct", "derived", "reit",
     "Replaces EPS growth for REITs. NM when either end is <= 0."),
    ("ffo_ps_cagr_5y_pct", "FFO/share CAGR 5y (REIT)", "pct", "derived", "reit", "As above, 5 years."),
    ("net_debt_to_ebitdare_x", "Net debt / EBITDAre (REIT)", "x", "derived", "reit",
     "EBITDAre (NAREIT) = EBITDA + real-estate impairments - gains on property sales. Replaces net "
     "debt / EBITDA for REITs. Trailing figure; REITs often report an annualized pro forma version."),
    ("net_debt_to_ebitdare_chg_3y_x", "Net debt/EBITDAre change 3y (REIT)", "x", "derived", "reit",
     "FY0 minus FY-3. Replaces the EBITDA-based leverage change for REITs."),
    ("ebitdare_interest_coverage_x", "EBITDAre / interest (REIT)", "x", "derived", "reit",
     "Replaces EBIT interest coverage for REITs, whose EBIT is depressed by depreciation. Capped at 100x."),
    ("ffo_nareit_adjusted", "FFO adjusted for gains/impairments", "flag", "meta", "reit",
     "1 = impairment/gain data were available (NAREIT-style FFO); 0 = FFO is net income + D&A only."),
    # ---- quality / risk ------------------------------------------------------------------------
    ("earnings_positive_share_5y_pct", "Profitable years (5y)", "pct", "derived", "quality",
     "Share of the last 5 FYs (min 3) with positive net income."),
    ("fcf_positive_share_5y_pct", "FCF-positive years (5y)", "pct", "derived", "quality", "As above, for FCF."),
    ("operating_margin_stdev_5y_pp", "Operating margin volatility (5y)", "pp", "derived", "quality",
     "Standard deviation of operating margin over the last 5 FYs (min 4)."),
    ("revenue_decline_years_5y", "Revenue decline years (5y)", "count", "derived", "quality",
     "Number of year-over-year revenue declines in the last 5 years."),
    ("fcf_conversion_3y_pct", "FCF conversion (3y)", "pct", "derived", "quality",
     "3-yr FCF / 3-yr net income; NM when net income sum <= 0."),
    ("margin_vs_10y_avg_x", "Op margin vs 10y average", "x", "derived", "quality",
     "Current operating margin / average of up to 10 FYs (min 5). >1.5 in cyclicals = possible peak."),
    # ---- data / meta ---------------------------------------------------------------------------
    ("fiscal_years_available", "Fiscal years of data", "count", "meta", "data", "Annual periods available."),
    ("latest_period_end", "Latest fiscal period end", "date", "meta", "data", "Most recent FY (or TTM) period end."),
    ("latest_filing_date", "Latest filing date", "date", "meta", "data", "Filing date of the latest period."),
    ("filing_age_days", "Days since latest filing", "days", "meta", "data", "Staleness check."),
    ("uses_ttm", "Current values are TTM", "flag", "meta", "data", "1 = trailing-twelve-month data used for current values."),
    ("splits_detected", "Stock splits adjusted", "count", "meta", "data",
     "Splits detected in the share-count history; older share counts, EPS and DPS were rescaled to today's basis."),
]

METRICS: Dict[str, MetricDef] = {m[0]: MetricDef(*m) for m in _M}

# Metrics that only apply to certain industry profiles; every other profile shows them as n/a.
PROFILE_ONLY: Dict[str, tuple] = {
    name: ("reit",) for name in ("p_ffo_x", "ffo_payout_pct", "ffo_per_share", "ffo_ps_cagr_3y_pct",
                                 "ffo_ps_cagr_5y_pct", "net_debt_to_ebitdare_x",
                                 "ebitdare_interest_coverage_x", "ffo_nareit_adjusted",
                                 "net_debt_to_ebitdare_chg_3y_x")
}


def names(kind: str = None) -> List[str]:
    return [m.name for m in METRICS.values() if kind is None or m.kind == kind]


def raw_value_metrics() -> List[str]:
    return names("raw_value")
