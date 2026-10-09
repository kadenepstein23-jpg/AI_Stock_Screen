"""SEC EDGAR provider (free, public-domain fundamentals).

Uses the SEC's XBRL "company facts" API. Good for:
  * building and testing the pipeline on real companies with no subscription;
  * spot-checking licensed data against the primary source (see `validate`);
  * public test fixtures (SEC data is public domain).

Limitations (documented, by design):
  * no prices or market caps — valuation metrics are missing unless you pass
    --prices-file (a CSV with ticker,date,close[,volume]);
  * XBRL tags vary by company; the tag map below covers the common cases and
    leaves a field missing rather than guessing;
  * security-type classification is heuristic (SEC does not label ETFs,
    preferreds or warrants), so SEC mode is not the production universe.

SEC fair-access rules: max 10 requests/second and a User-Agent with a name and
email. Set SEC_USER_AGENT in .env, e.g.  SEC_USER_AGENT="Jane Doe jane@example.com"
"""
from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import schema
from ..store import Snapshot
from .base import Provider, RateLimiter, http_get

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
BULK_FACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
MIN_FY_BEFORE_PREDECESSOR_SEARCH = 3   # fewer annual rows than this -> look for a predecessor CIK

# canonical field -> ordered list of us-gaap tags (first available per period wins)
DURATION_TAGS: Dict[str, List[str]] = {
    # all revenue tags (used to find fiscal years); the revenue VALUE is chosen by _pick_revenue
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense", "TotalRevenuesAndOtherIncome"],
    # total / consolidated revenue line of the income statement. Net revenues (after interest expense)
    # first: it is the headline figure for broker-dealers and banks that tag it
    "revenue_total": ["RevenuesNetOfInterestExpense", "Revenues", "TotalRevenuesAndOtherIncome"],
    # revenue from contracts with customers (ASC 606) can be only PART of revenue: it excludes lease /
    # rental income, alternative-revenue programs (utilities), interest and principal transactions
    "revenue_partial": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
    # accrual interest expense first; InterestPaidNet (cash paid) is only a last resort
    "interest_expense": ["InterestExpense", "InterestExpenseOperating", "InterestExpenseNonoperating",
                         "InterestExpenseDebt", "InterestAndDebtExpense", "InterestPaidNet"],
    # REIT FFO inputs (NAREIT definition adds back real-estate impairments, removes gains on sale)
    "real_estate_impairment": ["ImpairmentOfRealEstate", "ImpairmentOfRealEstateHeldForUse",
                               "AssetImpairmentCharges"],
    "gain_on_sale_real_estate": ["GainLossOnSaleOfProperties", "GainsLossesOnSalesOfInvestmentRealEstate",
                                 "GainLossOnSaleOfRealEstateInvestmentProperty",
                                 "GainLossOnSaleOfPropertyPlantEquipment",
                                 # further standard concepts REITs use for property sales / remeasurements
                                 "GainLossOnDispositionOfAssets1", "GainLossOnDispositionOfAssets",
                                 "GainsLossesOnSalesOfAssets",
                                 "DisposalGroupNotDiscontinuedOperationGainLossOnDisposal",
                                 "BusinessCombinationStepAcquisitionEquityInterestInAcquireeRemeasurementGain"],
    # gains on selling interests in unconsolidated entities (also excluded from NAREIT FFO)
    "gain_on_sale_jv": ["EquityMethodInvestmentRealizedGainLossOnDisposal",
                        "GainLossOnSaleOfEquityMethodInvestments"],
    "fx_gain_loss": ["ForeignCurrencyTransactionGainLossBeforeTax"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends", "PaymentsOfDividends",
                       "DividendsCommonStockCash",
                       "DividendsCommonStock"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "share_issuance": ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised"],
}
PER_SHARE_TAGS: Dict[str, List[str]] = {
    # basic EPS only fills years with no diluted figure at all (Lennar tagged only basic for FY2025);
    # basic is >= diluted, typically by under 1%
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "dps": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
    # each DPS tag on its own, so the one consistent with dividends paid can be chosen (see _pick_dps)
    "dps_declared": ["CommonStockDividendsPerShareDeclared"],
    "dps_paid": ["CommonStockDividendsPerShareCashPaid"],
}
SHARE_DURATION_TAGS: Dict[str, List[str]] = {
    "shares_diluted_wavg": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                            "WeightedAverageNumberOfSharesOutstandingBasic"],
}
INSTANT_TAGS: Dict[str, List[str]] = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    # Some filers (e.g. Chevron) tag only the cash-flow-statement total, which includes restricted
    # cash; restricted amounts are subtracted when tagged so cash matches the balance sheet.
    "cash_incl_restricted": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "restricted_cash_total": ["RestrictedCashAndRestrictedCashEquivalents", "RestrictedCash"],
    "restricted_cash_current": ["RestrictedCashCurrent", "RestrictedCashAndCashEquivalentsAtCarryingValue"],
    "restricted_cash_noncurrent": ["RestrictedCashNoncurrent", "RestrictedCashAndCashEquivalentsNoncurrent"],
    "cash_only": ["Cash"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "preferred_equity": ["PreferredStockValue", "PreferredStockValueOutstanding",
                         "PreferredStockIncludingAdditionalPaidInCapitalNetOfDiscount"],
    "goodwill": ["Goodwill"],
    "other_intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    # a single all-in debt total, when the filer tags one (e.g. Travelers); used instead of summing parts
    "debt_total_combined": ["DebtLongtermAndShorttermCombinedAmount"],
    "aoci": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
    # LongTermDebt includes current maturities; LongTermDebtAndCapitalLeaseObligations is the
    # NONCURRENT balance-sheet line (taxonomy definition), kept separate so current debt can be added
    "lt_debt_total": ["LongTermDebt"],
    "lt_debt_and_leases_noncurrent": ["LongTermDebtAndCapitalLeaseObligations"],
    # its matching current line (includes current finance-lease obligations) first
    "lt_debt_and_leases_current": ["LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtCurrent"],
    # total current debt (notes and loans payable incl. current maturities), e.g. ExxonMobil
    "debt_current_total": ["DebtCurrent"],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"],
    "lt_debt_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "short_term_borrowings": ["ShortTermBorrowings"],
    "commercial_paper": ["CommercialPaper"],
    # debt components, used only when no total long-term-debt tag exists (common for REITs,
    # which report notes, term loans, mortgages and the revolver as separate lines)
    "debt_notes": ["SeniorNotes", "NotesPayable", "SeniorLongTermNotes", "UnsecuredDebt"],
    "debt_loans": ["LoansPayable", "LongTermLoansPayable"],
    "debt_secured": ["SecuredDebt", "SecuredLongTermDebt"],
    "debt_revolver": ["LongTermLineOfCredit", "LineOfCredit"],
    "debt_convertible": ["ConvertibleNotesPayable", "ConvertibleDebtNoncurrent"],
}
SHARE_INSTANT_TAGS: Dict[str, List[str]] = {
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}


# ---------------------------------------------------------------------------
# Pure parsing (unit-tested with fixtures; no network)
# ---------------------------------------------------------------------------
def _facts_frame(facts: Dict[str, Any], tag: str, unit: str) -> pd.DataFrame:
    node = facts.get("us-gaap", {}).get(tag) or facts.get("dei", {}).get(tag)
    if not node:
        return pd.DataFrame()
    rows = node.get("units", {}).get(unit, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df.get("form", pd.Series(dtype=str)).isin(ANNUAL_FORMS)] if "form" in df else df.iloc[0:0]
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
    return df


def _annual_duration(facts: Dict[str, Any], tags: Iterable[str], unit: str) -> pd.DataFrame:
    """Return DataFrame[end, val, first_filed] of ~12-month values, restated to latest filing."""
    out = None
    for tag in tags:
        df = _facts_frame(facts, tag, unit)
        if df.empty or "start" not in df:
            continue
        days = (df["end"] - df["start"]).dt.days
        df = df[(days >= 330) & (days <= 400)]
        if df.empty:
            continue
        df = df.sort_values("filed")
        g = df.groupby("end")
        s = pd.DataFrame({"val": g["val"].last(), "first_filed": g["filed"].first()})
        out = s if out is None else out.combine_first(s)
    return out if out is not None else pd.DataFrame(columns=["val", "first_filed"])


def _instant(facts: Dict[str, Any], tags: Iterable[str], unit: str) -> pd.Series:
    out = None
    for tag in tags:
        df = _facts_frame(facts, tag, unit)
        if df.empty:
            continue
        if "start" in df:
            df = df[df["start"].isna()]
        df = df.sort_values("filed")
        s = df.groupby("end")["val"].last()
        out = s if out is None else out.combine_first(s)
    return out if out is not None else pd.Series(dtype=float)


def _has_common_dividend_facts(facts: Dict[str, Any]) -> bool:
    for tag in (facts.get("us-gaap") or {}):
        t = tag.lower()
        if "dividend" in t and "preferred" not in t and "receiv" not in t and "income" not in t:
            return True
    return False


def _pick_revenue(total: float, partial: float) -> float:
    """Total revenue when the filer tags it, else the contract-revenue line. A 'total' smaller than the
    contract-revenue figure is not the consolidated total (a sub-line tagged with the generic concept),
    so the larger figure is used then. Target Hospitality FY2025: contract revenue tag $274.7M vs total
    $320.6M; Evergy: $5,705.1M vs total operating revenues $5,961.6M; Piper Sandler: $1,834.1M vs net
    revenues $1,899.4M."""
    if np.isnan(total):
        return partial
    if np.isnan(partial):
        return total
    return total if total >= 0.98 * partial else max(total, partial)


def _pick_dps(declared: float, paid: float, dividends: float, shares: float) -> float:
    """Dividend per share. Declared is preferred, but some filers put a different figure under the
    declared tag (Prologis FY2025: $0.03 'declared' vs $4.04 paid, $3.8B paid on ~0.93B shares).
    Declared is kept when within 2x of dividends paid / shares (timing differences); otherwise the
    figure closer to that is used, and one more than 3x away is discarded (missing beats wrong)."""
    cands = [v for v in (declared, paid) if not np.isnan(v)]
    if not cands:
        return np.nan
    implied = dividends / shares if (not np.isnan(dividends) and not np.isnan(shares) and shares > 0
                                     and dividends > 0) else np.nan
    if np.isnan(implied):
        return cands[0]
    pos = [v for v in cands if v > 0]
    if not pos:
        return cands[0]
    if not np.isnan(declared) and declared > 0 and abs(np.log(declared / implied)) <= np.log(2.0):
        return declared                    # declared vs paid differ only by timing: keep declared
    best = min(pos, key=lambda v: abs(np.log(v / implied)))
    return best if abs(np.log(best / implied)) <= np.log(3.0) else np.nan


def parse_companyfacts(facts_json: Dict[str, Any], ticker: str, max_years: int = 12) -> pd.DataFrame:
    """Convert one company's SEC company-facts JSON into canonical FY rows."""
    facts = facts_json.get("facts", {})
    dur = {k: _annual_duration(facts, v, "USD") for k, v in DURATION_TAGS.items()}
    dur.update({k: _annual_duration(facts, v, "USD/shares") for k, v in PER_SHARE_TAGS.items()})
    dur.update({k: _annual_duration(facts, v, "shares") for k, v in SHARE_DURATION_TAGS.items()})

    anchor = None
    for key in ("revenue", "net_income", "operating_cash_flow"):
        if not dur[key].empty:
            anchor = dur[key] if anchor is None else anchor.combine_first(dur[key])
    if anchor is None or anchor.empty:
        return schema.conform(pd.DataFrame(), "financials")
    ends = sorted(anchor.index)[-max_years:]

    inst = {k: _instant(facts, v, "USD") for k, v in INSTANT_TAGS.items()}
    inst.update({k: _instant(facts, v, "shares") for k, v in SHARE_INSTANT_TAGS.items()})

    def dval(key, end):
        df = dur.get(key)
        return float(df.at[end, "val"]) if df is not None and end in df.index else np.nan

    def ival(key, end):
        s = inst.get(key)
        return float(s.loc[end]) if s is not None and end in s.index else np.nan

    def nz(x):
        return 0.0 if np.isnan(x) else x

    rows = []
    for end in ends:
        filed = anchor.at[end, "first_filed"]
        lt_total = ival("lt_debt_total", end)
        noncurrent_only = False
        if np.isnan(lt_total):
            lt_total = ival("lt_debt_and_leases_noncurrent", end)
            noncurrent_only = not np.isnan(lt_total)
        if np.isnan(lt_total):
            nonc, cur = ival("lt_debt_noncurrent", end), ival("lt_debt_current", end)
            lt_total = np.nan if np.isnan(nonc) and np.isnan(cur) else nz(nonc) + nz(cur)
        if np.isnan(lt_total):
            comps = [ival(k, end) for k in ("debt_notes", "debt_loans", "debt_secured", "debt_revolver",
                                            "debt_convertible")]
            lt_total = np.nan if all(np.isnan(c) for c in comps) else sum(nz(c) for c in comps)
        stb, cp = ival("short_term_borrowings", end), ival("commercial_paper", end)
        # Commercial paper is normally a component of short-term borrowings (Chevron's $4.6B of CP sits
        # inside its $1.0B net short-term debt line after reclassification), so it is added only when
        # no short-term-borrowings total is tagged.
        parts = [lt_total, stb] if not np.isnan(stb) else [lt_total, cp]
        # Only the noncurrent long-term line is tagged and there is no short-term-borrowings line:
        # current debt comes from DebtCurrent (ExxonMobil: $9,296M notes and loans payable + $34,241M
        # long-term = $43,537M reported; adding only commercial paper gave $37,300M).
        dcur = ival("debt_current_total", end)
        if noncurrent_only and np.isnan(stb):
            parts = [lt_total, dcur] if not np.isnan(dcur) else [lt_total, ival("lt_debt_and_leases_current", end), cp]
        elif noncurrent_only and np.isnan(dcur):
            # noncurrent long-term line + short-term borrowings, and no current-debt total: current
            # maturities are a separate balance-sheet line (Nucor: $6,909M + $122M + $90M = $7,121M).
            # When a DebtCurrent total IS tagged next to short-term borrowings (Chevron), the
            # short-term line already includes the current maturities and nothing is added.
            parts = [lt_total, stb, ival("lt_debt_and_leases_current", end)]
        debt = np.nan if all(np.isnan(p) for p in parts) else sum(nz(p) for p in parts)
        # Filers use the "combined" total inconsistently: Travelers' includes everything, Honeywell's
        # leaves out commercial paper. Take whichever of (combined, sum of parts) is larger: the larger
        # figure is the more complete one, and a part already inside the combined total can't be
        # double-counted because the two are never added together.
        combined = ival("debt_total_combined", end)
        if not np.isnan(combined):
            debt = combined if np.isnan(debt) else max(debt, combined)
        cash = ival("cash", end)
        if np.isnan(cash):
            total = ival("cash_incl_restricted", end)
            if not np.isnan(total):
                restricted = ival("restricted_cash_total", end)
                if np.isnan(restricted):
                    rc, rn = ival("restricted_cash_current", end), ival("restricted_cash_noncurrent", end)
                    restricted = np.nan if np.isnan(rc) and np.isnan(rn) else nz(rc) + nz(rn)
                cash = total - restricted if not np.isnan(restricted) and restricted < total else total
            else:
                cash = ival("cash_only", end)
        gw, oi = ival("goodwill", end), ival("other_intangibles", end)
        intang = np.nan if np.isnan(gw) and np.isnan(oi) else nz(gw) + nz(oi)
        rev = _pick_revenue(dval("revenue_total", end), dval("revenue_partial", end))
        if np.isnan(rev):
            rev = dval("revenue", end)
        cor, gp = dval("cost_of_revenue", end), dval("gross_profit", end)
        if np.isnan(gp) and not np.isnan(rev) and not np.isnan(cor):
            gp = rev - cor
        opinc, da = dval("operating_income", end), dval("d_and_a", end)
        ocf, capex = dval("operating_cash_flow", end), dval("capex", end)
        rep, iss = dval("share_repurchases", end), dval("share_issuance", end)
        net_iss = np.nan if np.isnan(rep) and np.isnan(iss) else nz(iss) - nz(rep)
        rows.append(dict(
            ticker=ticker, period_type="FY", period_end=end, filing_date=filed,
            revenue=rev, cost_of_revenue=cor, gross_profit=gp, operating_income=opinc,
            ebit=opinc, ebitda=(opinc + da) if not (np.isnan(opinc) or np.isnan(da)) else np.nan,
            d_and_a=da, interest_expense=dval("interest_expense", end),
            real_estate_impairment=dval("real_estate_impairment", end),
            gain_on_sale_real_estate=(np.nan if np.isnan(dval("gain_on_sale_real_estate", end))
                                      and np.isnan(dval("gain_on_sale_jv", end))
                                      else nz(dval("gain_on_sale_real_estate", end)) + nz(dval("gain_on_sale_jv", end))),
            fx_gain_loss=dval("fx_gain_loss", end),
            pretax_income=dval("pretax_income", end), income_tax=dval("income_tax", end),
            net_income=dval("net_income", end), eps_diluted=dval("eps_diluted", end),
            shares_diluted_wavg=dval("shares_diluted_wavg", end),
            shares_outstanding=ival("shares_outstanding", end),
            cash=cash, current_assets=ival("current_assets", end),
            current_liabilities=ival("current_liabilities", end),
            total_assets=ival("total_assets", end), total_liabilities=ival("total_liabilities", end),
            total_debt=debt, equity=ival("equity", end), preferred_equity=ival("preferred_equity", end),
            aoci=ival("aoci", end),
            intangibles=intang,
            operating_cash_flow=ocf, capex=capex,
            free_cash_flow=(ocf - capex) if not (np.isnan(ocf) or np.isnan(capex)) else np.nan,
            sbc=dval("sbc", end), dividends_paid=dval("dividends_paid", end),
            share_repurchases=rep, net_common_issuance=net_iss,
            dps=_pick_dps(dval("dps_declared", end), dval("dps_paid", end), dval("dividends_paid", end),
                          dval("shares_diluted_wavg", end)),
        ))
    df = pd.DataFrame(rows)
    # A filer with NO common-dividend facts of any kind, in any year, pays no dividend (e.g. Adobe):
    # record 0 rather than missing, so dividend yield, shareholder yield and payout read 0 instead of
    # being treated as data gaps. One dividend fact anywhere (even under a tag not mapped here) keeps
    # the field missing, because then a payment may exist under an unrecognized tag.
    if not _has_common_dividend_facts(facts):
        df["dividends_paid"] = df["dividends_paid"].fillna(0.0)
        df["dps"] = df["dps"].fillna(0.0)
    # shares outstanding fallback: diluted weighted average
    df["shares_outstanding"] = df["shares_outstanding"].fillna(df["shares_diluted_wavg"])
    return schema.conform(df, "financials")


# ---------------------------------------------------------------------------
# Successor registrants (holding-company reorganizations, re-domiciles)
# ---------------------------------------------------------------------------
# When a company inserts a new holding company (ExxonMobil Holdings Corp, CIK 2115436, became the
# XOM registrant in July 2026), the ticker moves to a NEW CIK whose company facts start with its
# first 10-Q, while the 10-year history stays under the predecessor's CIK. The predecessor is found
# from the accession numbers of the successor's own periodic reports (the first 10 digits are the
# CIK of the entity that submitted the filing - here 0000034088, the old Exxon Mobil Corp), and it
# is only accepted when it is clearly the same business (same name after removing corporate words,
# or same business address). config: providers.sec.predecessor_ciks {successor: predecessor} overrides.
PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"}
_NAME_STOPWORDS = {"corp", "corporation", "inc", "incorporated", "co", "company", "holdings", "holding",
                   "group", "ltd", "limited", "plc", "the", "llc", "lp", "new", "de", "nv", "sa"}


def normalize_company_name(name: str) -> str:
    return "".join(t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if t not in _NAME_STOPWORDS)


def predecessor_candidates(sub: Dict[str, Any], facts_json: Dict[str, Any], own_cik: int) -> List[int]:
    """CIKs (most frequent first) that submitted this registrant's periodic reports or facts."""
    seen: Counter = Counter()
    recent = (sub.get("filings") or {}).get("recent") or {}
    for form, acc in zip(recent.get("form", []), recent.get("accessionNumber", [])):
        if form in PERIODIC_FORMS and acc[:10].isdigit():
            seen[int(acc[:10])] += 1
    for taxonomy in (facts_json.get("facts") or {}).values():
        for node in taxonomy.values():
            for rows in (node.get("units") or {}).values():
                for r in rows:
                    if r.get("form") in PERIODIC_FORMS and str(r.get("accn", ""))[:10].isdigit():
                        seen[int(r["accn"][:10])] += 1
    seen.pop(int(own_cik), None)
    return [c for c, _ in seen.most_common()]


def same_business(sub_a: Dict[str, Any], sub_b: Dict[str, Any]) -> bool:
    na, nb = normalize_company_name(sub_a.get("name", "")), normalize_company_name(sub_b.get("name", ""))
    if na and na == nb:
        return True
    for prev in sub_b.get("formerNames") or []:
        if na and normalize_company_name(prev.get("name", "")) == na:
            return True
    ba = ((sub_a.get("addresses") or {}).get("business") or {})
    bb = ((sub_b.get("addresses") or {}).get("business") or {})
    key = lambda b: (str(b.get("street1") or "").strip().lower(), str(b.get("zipCode") or "")[:5])
    return bool(ba.get("street1")) and key(ba) == key(bb)


def merge_companyfacts(predecessor: Dict[str, Any], successor: Dict[str, Any]) -> Dict[str, Any]:
    """Union of two company-facts documents; where both report a period, the later filing wins."""
    out: Dict[str, Any] = {"entityName": successor.get("entityName"), "facts": {}}
    for doc in (predecessor, successor):
        for tax, tags in (doc.get("facts") or {}).items():
            for tag, node in tags.items():
                dst = out["facts"].setdefault(tax, {}).setdefault(tag, {"units": {}})
                for unit, rows in (node.get("units") or {}).items():
                    dst["units"].setdefault(unit, []).extend(rows)
    return out


def classify_sec_security(ticker: str, name: str, sic: Optional[int]) -> str:
    t, n = ticker.upper(), (name or "").lower()
    if "-P" in t or " preferred" in n or " pfd" in n or "depositary shares" in n:
        return "preferred"
    if t.endswith(("-WT", ".WS", "-WS")) or " warrant" in n or (len(t) == 5 and t.endswith("W")):
        return "warrant"
    if t.endswith(("-U", ".U", "-UN")) or (len(t) == 5 and t.endswith("U") and "unit" in n):
        return "unit"
    if t.endswith(("-R", ".R", "-RT")) or " rights" in n:
        return "right"
    if " etf" in n or " fund" in n or (sic in (6221,)):
        return "etf"
    if " adr" in n or "ads" in n.split():
        return "adr"
    return "common"


EXCHANGE_MAP = {"Nasdaq": "NASDAQ", "NYSE": "NYSE", "NYSE American": "NYSEMKT", "NYSE Arca": "NYSEARCA",
                "CBOE": "BATS", "OTC": "OTC"}



# ---------------------------------------------------------------------------
# Company-extension gains (filing-level XBRL)
# ---------------------------------------------------------------------------
# The companyfacts API carries only standard (us-gaap/srt/dei) concepts. Many REITs report property-sale
# and remeasurement gains under their own extension concepts (Simon Property FY2025:
# "spg:AcquisitionOfControllingInterestSaleOrDisposalOf..." $2,887M and "spg:GainLossOnDisposalExchangeOr..."
# -$86M), so NAREIT FFO could not remove them. These helpers read the 10-K's XBRL instance and pick out
# such extension facts by name. Used only to fill years with no standard gain fact.
EXT_GAIN_WORDS = ("Sale", "Sold", "Disposal", "Disposition", "Dispose", "ControllingInterest",
                  "Remeasurement", "Revaluation", "EvaluationOfEquity", "ExchangeOfEquity", "ExchangeOrEvaluation")
EXT_GAIN_EXCLUDE = ("IncomeLoss", "ProfitLoss", "NetIncome", "Before", "Including", "Attributable",
                    "Unrealized", "Extinguishment", "Debt", "Derivative", "Hedge", "Currency", "Foreign",
                    "Tax", "Marketable", "Securities", "Comprehensive", "PerShare", "Proceeds", "Payment",
                    "Revenue", "Cost", "Expense", "Commission", "Receivable", "Deferred", "Discontinued",
                    "Held", "NoncontrollingInterestHolders", "Distribution")


def extension_gain_sign(local_name: str) -> Optional[int]:
    """+1 when a positive value of this extension concept is a gain on selling / remeasuring real-estate
    interests, -1 when a positive value is a loss (loss-first names), None when it is not such a line."""
    n = local_name
    if not any(w in n for w in EXT_GAIN_WORDS) or any(w in n for w in EXT_GAIN_EXCLUDE):
        return None
    if not any(w in n for w in ("Gain", "Loss", "Net", "Impairment", "Recovery")):
        return None
    gi, li = n.find("Gain"), n.find("Loss")
    if li != -1 and (gi == -1 or li < gi):
        return -1
    return 1


def income_statement_concepts(filing_summary_xml: bytes) -> List[str]:
    """FilingSummary.xml -> R-page file names of the income statement(s) (not parenthetical)."""
    import re

    out = []
    for rep in re.findall(rb"<Report[^>]*>(.*?)</Report>", filing_summary_xml, flags=re.S):
        def field(name):
            m = re.search(rb"<" + name + rb">(.*?)</" + name + rb">", rep, flags=re.S)
            return m.group(1).decode(errors="ignore").strip() if m else ""
        names = (field(b"ShortName") + " " + field(b"LongName")).lower()
        if field(b"MenuCategory").lower() != "statements" or "parenthetical" in names:
            continue
        if re.search(r"(income|operations|earnings)", names) and not re.search(
                r"(balance|financial position|cash flow|equity|stockholders|shareholders|partners. capital)", names):
            html = field(b"HtmlFileName")
            if html:
                out.append(html)
    return out


def concepts_on_page(r_html: bytes) -> set:
    """Concept local names shown on an XBRL viewer R page ('defref_<prefix>_<Name>' anchors)."""
    import re

    return {m.decode() for m in re.findall(rb"defref_[A-Za-z0-9-]+_([A-Za-z0-9]+)", r_html)}


def parse_extension_gains(xml_bytes: bytes, standard_prefixes=("us-gaap", "dei", "srt", "ifrs-full"),
                          allowed: Optional[set] = None) -> Dict[str, float]:
    """XBRL instance -> {period_end 'YYYY-MM-DD': total extension gain} for ~annual, undimensioned USD facts.
    `allowed` = the concept names shown on the income statement: only those lines count (a note table
    or a cash-flow line tagged with another extension concept would otherwise be added a second time).
    Two concepts with the same value in the same period are counted once."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    ns_prefix = {}
    contexts = {}
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str) or not tag.endswith("}context"):
            continue
        has_segment = any(isinstance(c.tag, str) and (c.tag.endswith("}segment") or c.tag.endswith("}scenario"))
                          for c in el.iter())
        start = end = None
        for c in el.iter():
            if isinstance(c.tag, str) and c.tag.endswith("}startDate"):
                start = (c.text or "").strip()
            elif isinstance(c.tag, str) and c.tag.endswith("}endDate"):
                end = (c.text or "").strip()
        contexts[el.get("id")] = (start, end, has_segment)
    # namespace URI -> prefix as declared on the root (ElementTree drops prefixes)
    import re

    for m in re.finditer(rb'xmlns:([A-Za-z0-9_-]+)="([^"]+)"', xml_bytes[:200000]):
        ns_prefix[m.group(2).decode()] = m.group(1).decode()
    seen: Dict[tuple, float] = {}
    for el in root:
        tag = el.tag
        if not isinstance(tag, str) or not tag.startswith("{") or el.get("contextRef") is None:
            continue
        uri, local = tag[1:].split("}", 1)
        prefix = ns_prefix.get(uri, "")
        if prefix in standard_prefixes or "xbrl.org" in uri or "fasb.org" in uri:
            continue
        unit = (el.get("unitRef") or "").lower()
        if "usd" not in unit or "share" in unit:
            continue
        if allowed is not None and local not in allowed:
            continue
        sign = extension_gain_sign(local)
        if sign is None:
            continue
        start, end, seg = contexts.get(el.get("contextRef"), (None, None, True))
        if seg or not start or not end:
            continue
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if not 330 <= days <= 400:
            continue
        try:
            val = float(el.text) * sign
        except (TypeError, ValueError):
            continue
        seen[(end, local)] = val
    out: Dict[str, float] = {}
    for end in sorted({e for e, _ in seen}):
        vals = []
        for (e, _), v in seen.items():
            if e == end and not any(abs(v - w) < 1 for w in vals):
                vals.append(v)
        out[end] = float(sum(vals))
    return out


def fill_extension_gains(fin: pd.DataFrame, ext: Dict[str, float], max_gap_days: int = 10) -> pd.DataFrame:
    """Fill FY rows whose standard gain fact is missing with the extension total for the same period end."""
    if fin.empty or not ext:
        return fin
    fin = fin.copy()
    ends = pd.to_datetime(pd.Series(list(ext.keys())))
    vals = pd.Series(list(ext.values()))
    for i, r in fin.iterrows():
        if r["period_type"] != "FY" or not pd.isna(r.get("gain_on_sale_real_estate")):
            continue
        gap = (ends - pd.Timestamp(r["period_end"])).abs()
        if len(gap) and gap.min() <= pd.Timedelta(days=max_gap_days):
            fin.at[i, "gain_on_sale_real_estate"] = float(vals[gap.idxmin()])
    return fin

# ---------------------------------------------------------------------------
# Provider (network)
# ---------------------------------------------------------------------------
class SecProvider(Provider):
    name = "sec"

    def _session(self):
        import requests

        ua = os.getenv(self.settings.get("user_agent_env", "SEC_USER_AGENT"))
        if not ua:
            raise RuntimeError('SEC requires a User-Agent with your name and email. Add to .env:\n'
                               '  SEC_USER_AGENT="Your Name your@email.com"')
        s = requests.Session()
        s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        return s

    def _cached_json(self, session, url: str, path: Path, limiter: RateLimiter, max_age_days: int = 7):
        import time

        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            return json.loads(path.read_text())
        r = http_get(session, url, limiter=limiter)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return r.json()

    def _attach_predecessor(self, session, cache: Path, limiter, cik: int, sub: Dict[str, Any],
                            facts: Dict[str, Any]):
        """Return (predecessor_cik, merged facts) for a new registrant, or (None, facts)."""
        overrides = {int(k): int(v) for k, v in (self.settings.get("predecessor_ciks") or {}).items()}
        cands = [overrides[cik]] if cik in overrides else predecessor_candidates(sub, facts, cik)[:3]
        for cand in cands:
            try:
                psub = self._cached_json(session, SUBMISSIONS_URL.format(cik=cand),
                                         cache / "submissions" / f"{cand}.json", limiter)
                if cik not in overrides and not same_business(sub, psub):
                    continue
                pfacts = self._cached_json(session, FACTS_URL.format(cik=cand),
                                           cache / "companyfacts" / f"{cand}.json", limiter)
            except Exception as e:  # a filing agent or a CIK without facts: try the next one
                log.debug("predecessor candidate %s for CIK %s rejected: %s", cand, cik, e)
                continue
            return cand, merge_companyfacts(pfacts, facts)
        return None, facts

    def _with_extension_gains(self, session, cache: Path, limiter, cik: int, sub: Dict[str, Any],
                              fin: pd.DataFrame) -> pd.DataFrame:
        """Fill missing REIT gain facts from company-extension concepts in the latest 10-K instances."""
        fy = fin[fin["period_type"] == "FY"].sort_values("period_end").tail(6)
        if fy.empty or fy["gain_on_sale_real_estate"].notna().all():
            return fin
        rec = sub.get("filings", {}).get("recent", {})
        n_filings = int(self.settings.get("extension_gain_filings", 4))
        accs = [a for a, f in zip(rec.get("accessionNumber", []), rec.get("form", [])) if f == "10-K"][:n_filings]
        ext: Dict[str, float] = {}
        for acc in accs:                                   # newest first: a later filing's restated value wins
            path = cache / "extension_gains_v2" / f"{acc}.json"
            if path.exists():
                got = json.loads(path.read_text())
            else:
                try:
                    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
                    idx = http_get(session, f"{base}/index.json", limiter=limiter).json()
                    names = [it["name"] for it in idx.get("directory", {}).get("item", [])]
                    inst = [x for x in names if x.endswith("_htm.xml")] or \
                        [x for x in names if x.endswith(".xml") and not any(k in x for k in
                         ("_cal", "_def", "_lab", "_pre", "FilingSummary", "MetaLinks")) and not x.startswith("R")]
                    if not inst or "FilingSummary.xml" not in names:
                        continue
                    pages = income_statement_concepts(
                        http_get(session, f"{base}/FilingSummary.xml", limiter=limiter).content)
                    allowed = set()
                    for pg in pages:
                        allowed |= concepts_on_page(http_get(session, f"{base}/{pg}", limiter=limiter).content)
                    if not allowed:
                        continue                   # cannot tell which lines are on the income statement
                    got = parse_extension_gains(http_get(session, f"{base}/{inst[0]}", limiter=limiter,
                                                         timeout=120).content, allowed=allowed)
                except Exception as e:
                    log.debug("extension gains for CIK %s %s failed: %s", cik, acc, e)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(got))
            for end, v in got.items():
                ext.setdefault(end, v)
        return fill_extension_gains(fin, ext)

    def refresh(self, tickers: Optional[List[str]] = None, bulk: bool = False,
                prices_file: Optional[str] = None, **_: Any) -> Snapshot:
        session = self._session()
        limiter = RateLimiter(self.settings.get("max_requests_per_second", 8))
        cache = self.store.cache_dir(self.name)
        listing = self._cached_json(session, TICKERS_URL, cache / "company_tickers_exchange.json", limiter, 1)
        lst = pd.DataFrame(listing["data"], columns=listing["fields"])
        lst["ticker"] = lst["ticker"].astype(str).str.upper()
        if tickers:
            wanted = {t.upper() for t in tickers}
            missing = wanted - set(lst["ticker"])
            if missing:
                log.warning("Not found in SEC ticker list: %s", ", ".join(sorted(missing)))
            lst = lst[lst["ticker"].isin(wanted)]
        elif not bulk:
            raise ValueError("SEC mode needs --tickers (development) or --bulk (full market, ~1.5 GB download).")
        lst = lst.copy()
        lst["share_class"] = np.where(lst.duplicated("cik", keep="first"), "secondary", "primary")
        multi = lst.groupby("cik")["ticker"].transform("count") == 1
        lst.loc[multi, "share_class"] = "single"

        bulk_zip = None
        if bulk:
            zpath = cache / "companyfacts.zip"
            if not zpath.exists():
                log.info("Downloading SEC bulk company facts (large)…")
                r = http_get(session, BULK_FACTS_URL, limiter=limiter, stream=True, timeout=600)
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            bulk_zip = zipfile.ZipFile(zpath)

        secs, fins = [], []
        for i, row in enumerate(lst.itertuples(index=False), 1):
            cik = int(row.cik)
            try:
                sub = self._cached_json(session, SUBMISSIONS_URL.format(cik=cik),
                                        cache / "submissions" / f"{cik}.json", limiter)
                if bulk_zip is not None:
                    name = f"CIK{cik:010d}.json"
                    if name not in bulk_zip.namelist():
                        continue
                    facts = json.loads(bulk_zip.read(name))
                else:
                    facts = self._cached_json(session, FACTS_URL.format(cik=cik),
                                              cache / "companyfacts" / f"{cik}.json", limiter)
            except Exception as e:  # keep going; report at the end
                log.warning("[%d/%d] %s failed: %s", i, len(lst), row.ticker, e)
                continue
            predecessor = None
            fin = parse_companyfacts(facts, row.ticker)
            if len(fin) < MIN_FY_BEFORE_PREDECESSOR_SEARCH and bulk_zip is None:
                predecessor, facts = self._attach_predecessor(session, cache, limiter, cik, sub, facts)
                if predecessor:
                    fin = parse_companyfacts(facts, row.ticker)
                    log.info("%s: new registrant CIK %d; history merged from predecessor CIK %d",
                             row.ticker, cik, predecessor)
            sic = int(sub["sic"]) if str(sub.get("sic", "")).isdigit() else None
            if sic in set(self.settings.get("extension_gain_sic", [6798])) and bulk_zip is None:
                fin = self._with_extension_gains(session, cache, limiter, cik, sub, fin)
            secs.append(dict(
                ticker=row.ticker, name=row.name, exchange=EXCHANGE_MAP.get(row.exchange, row.exchange),
                security_type=classify_sec_security(row.ticker, row.name, sic),
                share_class=row.share_class, is_delisted=False,
                sector=None, industry=sub.get("sicDescription"), sic_code=sic, cik=cik,
                location=(sub.get("addresses", {}).get("business", {}) or {}).get("stateOrCountry"),
                currency="USD", source_category=f"sec:{sub.get('entityType', '')}"
                + (f";predecessor_cik={predecessor}" if predecessor else "")))
            fins.append(fin)
            log.info("[%d/%d] %s ok", i, len(lst), row.ticker)

        snap = self.store.new_snapshot(self.name, extra={"source": "SEC EDGAR XBRL companyfacts",
                                                         "tickers_requested": len(lst)})
        snap.write("securities", schema.conform(pd.DataFrame(secs), "securities"))
        snap.write("financials", schema.conform(pd.concat(fins, ignore_index=True) if fins
                                                else pd.DataFrame(), "financials"))
        if prices_file:
            px = pd.read_csv(prices_file, parse_dates=["date"])
            px["ticker"] = px["ticker"].str.upper()
            if "close_adj" not in px:
                px["close_adj"] = px["close"]
            snap.write("prices", schema.conform(px, "prices"))
            fin = snap.read("financials")
            latest_shares = (fin.sort_values("period_end").groupby("ticker")["shares_outstanding"].last())
            last_px = px.sort_values("date").groupby("ticker").last()
            mkt = pd.DataFrame({"ticker": last_px.index, "date": last_px["date"].values,
                                "market_cap": last_px["close"].values *
                                latest_shares.reindex(last_px.index).values})
            snap.write("market", schema.conform(mkt, "market"))
        self.store.activate(snap)
        return snap
