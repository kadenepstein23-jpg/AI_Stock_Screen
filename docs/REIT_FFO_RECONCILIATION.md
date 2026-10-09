# REIT FFO reconciliation: screener vs company-reported (FY2025)

Final check before the engine-development phase closed (2026-10-09). Run `2026-10-09_003743_sharadar`
(methodology fingerprint `fc2159212b4d7966`, unchanged). This compares screener FFO/share with company-reported
NAREIT FFO/share for 33 REITs from the research set and the largest REITs by market cap. Every reported figure
comes from the company's own FY2025 earnings release or 8-K Ex. 99.1. Prologis and American Tower report FFO only
as a dollar total, so their per-share figures are calculated (total ÷ diluted shares).

**Screener definition** (see [methodology §5](SCREENING_METHODOLOGY.md)): NAREIT FFO = net income to common + total
D&A + real-estate impairments − gains on property sales. It uses consolidated figures with no joint-venture or
noncontrolling-interest adjustments, on the latest fiscal year (or trailing twelve months when that row has
adjustments, or when none are available).

**Result.** 26 of 33 are within 10%: 17 within 5% and 9 between 5% and 10%. The median absolute gap is 4.7%.
Seven exceed 10%, and every one reconciles line by line to a known limitation of the standardized definition.
None is a parsing or data error in the FFO calculation. The pass did turn up two other issues:
1. An objective **dividend data error** (Iron Mountain). Fixed.
2. A stale **vendor REIT label** (Uniti). Documented; it can't be fixed under the freeze.

Both are described at the end.

## Gaps above 10%

$ millions unless shown per share. Classification: **(2)** = expected limitation of the standardized
methodology.

| Ticker | Screener | Reported | Gap | Bridge from screener to company figure | Class |
|---|---|---|---|---|---|
| EQIX | $35.50 | $27.19 | +30.6% | Screener $3,484 = NI $1,350 + total D&A $2,066 + impairment $68. Company: − non-real-estate depreciation $568, − other amortization and accretion $216, − impairment $68 (Equinix leaves impairments in FFO), + JV $36 = **$2,668** (exact) | (2) total D&A, not real-estate D&A |
| AMT | $9.97 | ~$8.88 calc | +12.3% | Screener ≈ $4,674. Company starts from total NI (+$99 noncontrolling-interest share), uses real-estate D&A $1,899.6 (vs total $2,041.6: −$142), and applies $(450.7) of unconsolidated/noncontrolling-interest adjustments (mainly the partners' share of D&A in data-centre and European JVs) = **$4,160.9** | (2) no NCI adjustment; total D&A |
| SPG | $10.34 | $12.34 | −16.2% | Company adds its **$811.7 share of JV/Klépierre/TRG D&A** ($2.15 per unit). $12.34 − $2.15 = $10.19 vs screener $10.34; the remainder is D&A-definition and NCI items. Per-share bases agree (common NI ÷ common shares ≈ OP FFO ÷ shares + units) | (2) no JV depreciation |
| PLD | $5.23 | ~$5.94 calc | −12.0% | Screener $5,004. Company: + $257.7 development and land gains (Prologis keeps these in NAREIT FFO; the standard gain tag includes them), + $551.0 share of unconsolidated co-investment ventures, − $47.8 NCI, − $87.2 D&A definition = **$5,677.7** vs $5,679.5 reported | (2) no JV depreciation; gain scope |
| CCI | $2.62 | $4.04 | −35.1% | Screener $1,145 = NI $444 + D&A $690 + write-downs $11. Company adds back the **$659 loss from discontinued operations** (fiber/small-cells sale) and uses real-estate D&A $650 (−$40) = **$1,764** (exact) | (2) no discontinued-operations adjustment |
| IRM | $5.10 (TTM 6/2026); $4.04 FY2025 | $1.96 | +106% (FY) | No SEC gain/impairment fact was found, so the screener uses TTM NI + D&A. FY2025: NI $144.6 + D&A $1,057.2 = $1,201.8. Company uses **real-estate depreciation $421.6 out of $1,024.4 total D&A** (−$635.6 incl. definition gap), +$7.7 NI, −$3.3 gain, +$7.4 lease intangibles, +$6.3 JV = **$584.2** (exact). Iron Mountain's own Normalized FFO is $3.63 | (2) total D&A; most of Iron Mountain's D&A is racking, equipment and customer intangibles |
| TRNO | $2.48 | $2.77 | −10.5% | Sharadar D&A (cash-flow line) is $92.3; Terreno's real-estate D&A is **$121.4** (SEC-tagged $121.6), +$29.1 → $2.76 | (2) vendor D&A field definition |

## Within 10% (no action)

| Ticker | Screener | Reported | Gap | | Ticker | Screener | Reported | Gap |
|---|---|---|---|---|---|---|---|---|
| CBL | $7.39 | $6.74 | +9.6% | | KRG | $2.01 | $2.10 | −4.3% |
| AMH | $1.92 | $1.79 | +7.3% | | VTR | $3.39 | $3.50 | −3.1% |
| SKT | $2.49 | $2.33 | +6.9% | | CTRE | $1.88 | $1.82 | +3.3% |
| DLR | $6.46 | $6.96 | −7.2% | | HST | $2.07 | $2.03 | +2.0% |
| WELL | $2.49 | $2.68 | −7.1% | | MAA | $8.49 | $8.32 | +2.0% |
| REG | $4.92 | $4.64 | +6.0% | | PSA | $15.58 | $15.81 | −1.5% |
| PECO | $2.69 | $2.54 | +5.9% | | GLPI | $3.91 | $3.86 | +1.3% |
| FR | $2.79 | $2.96 | −5.7% | | INVH | $1.82 | $1.80 | +1.1% |
| KIM | $1.67 | $1.76 | −5.1% | | BRX | $2.27 | $2.25 | +0.9% |
| RHP | $8.30 | $7.93 | +4.7% | | UDR | $2.45 | $2.43 | +0.8% |
| KRC | $4.39 | $4.20 | +4.5% | | O | $4.27 | $4.25 | +0.5% |
| EPR | $4.97 | $4.96 | +0.2% | | EGP | $8.98 | $8.98 | 0.0% |
| OHI | $2.95 | $2.95 | 0.0% | | EPRT | $2.05 | $2.05 | 0.0% |

- **CBL (+9.6%)** also reconciles. Its gain line ($104.0M) includes JV gains, against $74.2M tagged in the
  SEC data. It adds $13.0M of JV D&A and removes $1.6M of NCI D&A.
- **Welltower** is close on NAREIT FFO. That figure is depressed by $1,497.4M of special incentive-plan
  compensation; Welltower's headline Normalized FFO is $5.29, about twice its NAREIT figure. The screen
  measures NAREIT FFO only.

## What the limitations mean for the screens

Total D&A overstates FFO for owners with large non-real-estate asset bases: data centres (EQIX, DLR) and Iron
Mountain. Missing JV depreciation understates it for JV-heavy owners (SPG, PLD). Neither moves a REIT into or
out of the top 20 on either screen:
- **Overstated FFO:** EQIX ranks Income #185 and Growth #319; IRM ranks Income #443 and Growth #445.
- **Understated FFO:** SPG ranks Income #171; PLD ranks Income #378.

Fixing them needs the real-estate share of D&A and proportionate JV data, which standardized feeds don't carry.
That would be a methodology revision (a new freeze and a new holdout), not a parsing fix.

## Other issues found during the pass

1. **Iron Mountain dividends: objective data error, fixed.** Sharadar's dividends-paid field is 0 in every
   year from 2016 to TTM 6/2026, while reported DPS is $2.00–$3.37. The zero blocked the existing DPS × shares
   fill, so the dividend yield was 0% and IRM failed the Income yield gate (≥ 1.5%).
   - **Rule (general, no company exception):** dividends paid of exactly 0 next to a positive DPS for the same
     period is treated as missing, so the existing fill applies. One regression test; 195 tests pass;
     fingerprint unchanged.
   - **Scope:** 4 investable companies (IRM, DMLP, FLOC, CEG).
     - IRM: 2.78% yield, now ranked Income #443 of 504.
     - Dorchester Minerals: 9.29% yield, but a 157% earnings payout sends it to needs-review.
     - Flowco (1.26%) and Constellation Energy (0.46%) still fail the yield gate.
   - **Effect on the rest of the screen:** Growth is unchanged. The Income top 20 is the same set; KRG and
     KFY swap #3 and #4 because percentiles moved slightly.
   - **Holdout:** IRM is a holdout company. It is now inspected and named in code, so it is contaminated in
     the same way as TH, EVRG and PIPR. The canonical pre-inspection result stands (see
     [VALIDATION_REPORT](VALIDATION_REPORT.md)).
2. **Uniti Group (UNIT): stale vendor REIT label. Documented, not fixed.**
   - Uniti's FY2025 10-K says that after the August 1, 2025 Windstream merger the company "does not qualify to
     be a REIT", and it no longer reports FFO.
   - Sharadar still labels it "REIT - Specialty" (SEC SIC 4813, telephone communications), so it gets the REIT
     profile. Its FFO/share of $7.13 includes a $1,683.9M one-time settlement gain, which gives a P/FFO of 1.0x.
   - SIC can't separate it from real REITs: 13 genuine REITs in the universe also carry non-6798 codes (INVH,
     TRNO, hotel REITs on 7011). Excluding UNIT would need a company-specific override (prohibited under the
     freeze) or a REIT-status data field.
   - **Impact:** Growth #38 and an Income gate failure. It is not in any top-20 list or the research set.
     Treat its rank as unreliable.

Sources: the FY2025 releases and 8-K Ex. 99.1 exhibits of each company (Equinix 8-K 2026-02-11; American Tower
FY2025 release; Simon Ex. 99.1 0001104659-26-009209; Prologis FY2025 release; Crown Castle FY2025 release; Iron
Mountain Q4 2025 release; Terreno Q4 2025 update; CBL Ex. 99.1 0001193125-26-050190; Uniti Ex. 99.1 and 10-K
0001628280-26-013196). Screener values come from licensed Sharadar data and are shown here only as derived
per-share summaries.
