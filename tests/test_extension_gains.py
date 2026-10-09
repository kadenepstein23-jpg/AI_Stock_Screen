"""REIT gains tagged with company-extension XBRL concepts (not in the SEC companyfacts API).
Simon Property FY2025 10-K: 'Gain (loss) on acquisition of controlling interest, sale or disposal of ...
and impairment, net' $2,887,460K and '(Loss) gain due to disposal, exchange, or revaluation of equity
interests, net' -$86,119K, both company extensions; NAREIT FFO excludes both."""
import numpy as np
import pandas as pd
import pytest

from screener.providers.sec import extension_gain_sign, fill_extension_gains, parse_companyfacts, parse_extension_gains
from tests.test_cyclical_profile import _facts

M = 1e6

INSTANCE = b"""<?xml version="1.0" encoding="utf-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:us-gaap="http://fasb.org/us-gaap/2025"
  xmlns:spg="http://www.simon.com/20251231" xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
  <xbrli:context id="FY25"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="FY24"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="Q4"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2025-10-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="FY25_LP"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier>
    <xbrli:segment><xbrldi:explicitMember dimension="dei:LegalEntityAxis">spg:LPMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
    <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
  <spg:AcquisitionOfControllingInterestSaleOrDisposalOfOrRecoveryOnAssetsAndInterestsInUnconsolidatedEntitiesAndImpairmentNet contextRef="FY25" unitRef="usd" decimals="-3">2887460000</spg:AcquisitionOfControllingInterestSaleOrDisposalOfOrRecoveryOnAssetsAndInterestsInUnconsolidatedEntitiesAndImpairmentNet>
  <spg:AcquisitionOfControllingInterestSaleOrDisposalOfOrRecoveryOnAssetsAndInterestsInUnconsolidatedEntitiesAndImpairmentNet contextRef="FY24" unitRef="usd" decimals="-3">-75818000</spg:AcquisitionOfControllingInterestSaleOrDisposalOfOrRecoveryOnAssetsAndInterestsInUnconsolidatedEntitiesAndImpairmentNet>
  <spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet contextRef="FY25" unitRef="usd" decimals="-3">-86119000</spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet>
  <spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet contextRef="FY24" unitRef="usd" decimals="-3">451172000</spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet>
  <spg:CashFlowAdjustmentGainOnAcquisitionOfControllingInterestSaleOrDisposal contextRef="FY25" unitRef="usd" decimals="-3">2887460000</spg:CashFlowAdjustmentGainOnAcquisitionOfControllingInterestSaleOrDisposal>
  <spg:UnrealizedGainsLossesInFVEquityInstrumentsAndDerivativeInstrumentNet contextRef="FY25" unitRef="usd" decimals="-3">-106082000</spg:UnrealizedGainsLossesInFVEquityInstrumentsAndDerivativeInstrumentNet>
  <spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet contextRef="Q4" unitRef="usd" decimals="-3">5000000</spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet>
  <spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet contextRef="FY25_LP" unitRef="usd" decimals="-3">999000000</spg:GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet>
  <spg:ProceedsFromSaleOfInterestsInUnconsolidatedEntities contextRef="FY25" unitRef="usd" decimals="-3">300000000</spg:ProceedsFromSaleOfInterestsInUnconsolidatedEntities>
  <us-gaap:GainLossOnSaleOfProperties contextRef="FY25" unitRef="usd" decimals="-3">7000000</us-gaap:GainLossOnSaleOfProperties>
</xbrli:xbrl>"""


def test_extension_gain_names():
    assert extension_gain_sign("GainLossOnDisposalExchangeOrEvaluationOfEquityInterestsNet") == 1
    assert extension_gain_sign("AcquisitionOfControllingInterestSaleOrDisposalOfOrRecoveryOnAssetsAndInterestsIn"
                               "UnconsolidatedEntitiesAndImpairmentNet") == 1
    assert extension_gain_sign("LossOnSaleOfRealEstate") == -1
    for not_a_gain in ("UnrealizedGainsLossesInFVEquityInstrumentsAndDerivativeInstrumentNet",
                       "ProceedsFromSaleOfInterestsInUnconsolidatedEntities", "GainLossOnExtinguishmentOfDebtNet",
                       "LeaseIncome", "CostOfSales"):
        assert extension_gain_sign(not_a_gain) is None, not_a_gain


def test_parse_extension_gains_simon_fy2025():
    got = parse_extension_gains(INSTANCE)
    # FY2025: 2,887,460 - 86,119 (the cash-flow duplicate of 2,887,460 counted once; unrealized, quarterly,
    # dimensioned, proceeds and standard us-gaap facts ignored)
    assert got["2025-12-31"] == pytest.approx((2887.46 - 86.119) * M)
    assert got["2024-12-31"] == pytest.approx((-75.818 + 451.172) * M)
    assert set(got) == {"2024-12-31", "2025-12-31"}


def test_fill_only_years_without_a_standard_gain():
    fin = pd.DataFrame({"period_type": ["FY", "FY", "TTM"],
                        "period_end": pd.to_datetime(["2024-12-31", "2025-12-31", "2026-06-30"]),
                        "gain_on_sale_real_estate": [12.0 * M, np.nan, np.nan]})
    out = fill_extension_gains(fin, parse_extension_gains(INSTANCE))
    assert out.loc[0, "gain_on_sale_real_estate"] == 12.0 * M            # standard fact kept
    assert out.loc[1, "gain_on_sale_real_estate"] == pytest.approx(2801.341 * M)
    assert np.isnan(out.loc[2, "gain_on_sale_real_estate"])              # TTM untouched


def test_more_standard_gain_concepts_are_read():
    f = _facts(duration={"Revenues": 1000 * M, "NetIncomeLoss": 300 * M,
                         "GainLossOnDispositionOfAssets1": 45 * M})
    assert parse_companyfacts(f, "R").iloc[-1]["gain_on_sale_real_estate"] == pytest.approx(45 * M)


# ---- Welltower FY2025: one income-statement gain line ($1,449.0M); a pre-tax subtotal whose name also
# mentions dispositions, and a note-table concept, must not be added to it ------------------------------
WELL = b"""<?xml version="1.0" encoding="utf-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:well="http://www.welltower.com/20251231">
  <xbrli:context id="c-1"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
  <well:GainLossOnDispositionOfRealEstateAndAssetAcquisitionsOfControllingInterestsNet contextRef="c-1" unitRef="usd">1449043000</well:GainLossOnDispositionOfRealEstateAndAssetAcquisitionsOfControllingInterestsNet>
  <well:IncomeLossFromContinuingOperationsPreTaxBookBeforeEquityMethodAndGainLossOnRealEstateDispositions contextRef="c-1" unitRef="usd">-480025000</well:IncomeLossFromContinuingOperationsPreTaxBookBeforeEquityMethodAndGainLossOnRealEstateDispositions>
  <well:RealEstateDispositionsGrossSalesPriceGainLossNoteTable contextRef="c-1" unitRef="usd">2100000000</well:RealEstateDispositionsGrossSalesPriceGainLossNoteTable>
</xbrli:xbrl>"""

SUMMARY = b"""<FilingSummary><MyReports>
 <Report instance="well-20251231.htm"><HtmlFileName>R2.htm</HtmlFileName><LongName>0000002 - Statement - CONSOLIDATED BALANCE SHEETS</LongName><ShortName>CONSOLIDATED BALANCE SHEETS</ShortName><MenuCategory>Statements</MenuCategory></Report>
 <Report instance="well-20251231.htm"><HtmlFileName>R4.htm</HtmlFileName><LongName>0000004 - Statement - CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME</LongName><ShortName>CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME</ShortName><MenuCategory>Statements</MenuCategory></Report>
 <Report instance="well-20251231.htm"><HtmlFileName>R5.htm</HtmlFileName><LongName>0000005 - Statement - CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME (Parenthetical)</LongName><ShortName>STATEMENTS OF INCOME (Parenthetical)</ShortName><MenuCategory>Statements</MenuCategory></Report>
 <Report instance="well-20251231.htm"><HtmlFileName>R8.htm</HtmlFileName><LongName>0000008 - Statement - CONSOLIDATED STATEMENTS OF CASH FLOWS</LongName><ShortName>CONSOLIDATED STATEMENTS OF CASH FLOWS</ShortName><MenuCategory>Statements</MenuCategory></Report>
 <Report instance="well-20251231.htm"><HtmlFileName>R60.htm</HtmlFileName><LongName>0000060 - Disclosure - Dispositions (Details)</LongName><ShortName>Dispositions (Details)</ShortName><MenuCategory>Details</MenuCategory></Report>
</MyReports></FilingSummary>"""

R4 = b"""<html><body><table>
<tr><td><a onclick="top.Show.showAR( this, 'defref_us-gaap_Revenues', window );">Total revenues</a></td></tr>
<tr><td><a onclick="top.Show.showAR( this, 'defref_well_IncomeLossFromContinuingOperationsPreTaxBookBeforeEquityMethodAndGainLossOnRealEstateDispositions', window );">Income before</a></td></tr>
<tr><td><a onclick="top.Show.showAR( this, 'defref_well_GainLossOnDispositionOfRealEstateAndAssetAcquisitionsOfControllingInterestsNet', window );">Gain on dispositions</a></td></tr>
</table></body></html>"""


def test_only_income_statement_gain_lines_count():
    from screener.providers.sec import concepts_on_page, income_statement_concepts

    assert income_statement_concepts(SUMMARY) == ["R4.htm"]          # not balance sheet, parenthetical, cash flow, notes
    allowed = concepts_on_page(R4)
    assert "GainLossOnDispositionOfRealEstateAndAssetAcquisitionsOfControllingInterestsNet" in allowed
    got = parse_extension_gains(WELL, allowed=allowed)
    assert got == {"2025-12-31": pytest.approx(1449.043 * M)}
    # without the statement whitelist the note-table concept would have been added
    assert parse_extension_gains(WELL)["2025-12-31"] == pytest.approx((1449.043 + 2100) * M)
