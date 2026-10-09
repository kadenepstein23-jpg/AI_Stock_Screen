# Validation report — frozen methodology `fc2159212b4d7966`

Data: Sharadar Core US Equities Bundle (10-year), snapshot `2026-10-08_192527`, as of 2026-10-07;
SEC EDGAR company facts as the independent reference. Protocol and pre-registered criteria:
[HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md). Validation closed 2026-10-08; engine development closed 2026-10-09.

**Verdict.** On a 152-company holdout drawn after the freeze and never inspected beforehand, the frozen
methodology passed all 9 pre-registered automatic criteria, and the 20-company manual 10-K audit
passed 20/20 in every column (criterion: >= 90%). One objective error was found during the audit — in
the SEC *comparison* adapter, not in the screener's data or rules — and fixed without changing the
methodology fingerprint. A pass shows the rules behave on unseen companies as they do on the companies
used to build them; it is not evidence that the rankings predict returns.

## Timeline

| # | Event | Result |
|---|---|---|
| 1 | Methodology frozen (`freeze`, 2026-10-08 14:04) | fingerprint **`fc2159212b4d7966`** (red-flag rules, industry profiles, screens and weights, scoring and universe settings) |
| 2 | Holdout drawn (seed 20261008): 152 companies from 2,010 investable names after removing the 203-name development set; checksum `89500a4fafc15b9e` | stratified by profile x size: default 97, cyclical 23, bank 17, REIT 6, insurance 5, utility 4 |
| 3 | **Automated evaluation of the untouched holdout** (`outputs/holdout/2026-10-08_140744_sharadar_summary`) | **9 passed, 0 failed** (H3d diagnostic flagged, see below) — this is the canonical automated result |
| 4 | H6 manual audit began (20 companies: top 5 per screen, 5 in needs-review, the rest random) | |
| 5 | Objective issue found: the SEC adapter preferred the ASC 606 "revenue from contracts with customers" tag, which can be only part of revenue (Target Hospitality $274.7M vs 10-K total $320.6M; Evergy $5,705.1M vs $5,961.6M; Piper Sandler $1,834.1M vs net revenues $1,899.4M). Sharadar matched the 10-K in all three | parsing fix: prefer the total line (net revenues, then Revenues, then total revenues and other income) when it is at least 98% of the contract figure; 5 regression tests |
| 6 | Fix verified | methodology fingerprint unchanged (`fc2159212b4d7966`); holdout revenue match vs SEC **89% → 95%**; validation set 90.1% → 93.9%; all SEC companies 85.5% → 96.1% of company-years |
| 7 | Re-evaluation (`2026-10-08_162653_sharadar_summary`) | 8 passed, **1 failed: H0b**, correctly — TH, EVRG and PIPR had been inspected in H6 and are now named in the parser regression tests. The original holdout was **not** redrawn and the contamination was not removed |
| 8 | H6 manual audit completed | **20/20 Y** on profile, financials and flags (100% in each column) |
| 9 | Research stage: SEC parser read REIT gains tagged under company-extension concepts (Simon Property $2,887.5M Taubman remeasurement gain) | SEC parsing fix; fingerprint unchanged; run `2026-10-08_171854` |
| 10 | Final REIT FFO reconciliation, 33 REITs vs company-reported FY2025 NAREIT FFO ([REIT_FFO_RECONCILIATION.md](REIT_FFO_RECONCILIATION.md)) | 26/33 within 10%, median gap 4.7%. Every gap above 10% reconciles to a documented standardized-definition limitation; no FFO parsing error |
| 11 | Same pass: objective data error — Iron Mountain (a **holdout** company) had dividends paid of 0 next to DPS of $2.00–$3.37 in every year (this was among the canonical run's H1 dividends-paid mismatches, but was not inspected then) | data-normalization fix (zero dividends paid with positive DPS = missing); 4 investable companies affected; fingerprint unchanged. IRM is now contaminated like TH, EVRG and PIPR. The canonical step-3 result is unchanged and the holdout was not redrawn. Final frozen run: `2026-10-09_003743` |

The development set grew from 203 to 207 names in the universe at step 7: TH, EVRG and PIPR, plus
Cloudflare (NET) because the word "NET" appears in a test comment (not inspected, not in the holdout).

## Automated criteria (canonical run, step 3)

| ID | Check | Result |
|---|---|---|
| H0 | holdout intact, still investable | PASS — checksum matches; 152/152 in the universe |
| H0b | no contamination | PASS (at step 3; FAIL by design at step 7) |
| H1 | data accuracy vs SEC | PASS — match within 2%: revenue 89% (95% after the parser fix), net income 96%, CFO 99%, assets 100%, equity 99%, dividends paid 93%, EPS 93%, shares 92%; no field more than 4 pp below the development set |
| H2 | profile vs industry label | PASS — 1/152 (Willis Towers Watson, an insurance broker, in default: brokers are excluded from the insurance profile by design) |
| H3 | red-flag rates, holdout vs rest | PASS — no flag differs by more than 3 standard errors |
| H4 | needs-review share vs rest | PASS — Growth 15% vs 14%; Income 11% vs 13% |
| H5 | +/-20% pillar-weight robustness | PASS — Spearman median 0.998 / 0.996, min 0.996 / 0.989; top-quintile overlap median 100%, min 88% / 80% |
| H3d | development set vs rest (diagnostic) | flagged — development names have fewer loss-makers and SBC flags and more yield traps. Reweighting the rest to the development set's profile x size mix shrinks the gaps (low coverage 15% vs 19%, negative earnings 9% vs 12%); the remainder fits how development names were chosen (large caps, top-ranked names, problem cases picked for debugging). Not evidence of rules behaving differently on unseen names — H3 on the random holdout passed |

## H6 manual audit (20 companies, latest 10-K and 10-Qs)

Profile 20/20, financials 20/20, flags 20/20 (criterion >= 90%). Per-company results:
`outputs/holdout/2026-10-08_140744_sharadar_summary/h6_manual_audit_results.csv`.

Companies: TH, EVRG, PIPR, WNC, HWM, OTIS, INNV, PTC, NXT, BKNG, AWR, PRI, IBOC, EPR, ARLP, DCTH, NOW,
UNM, UFPT, GPOR. Net income and operating cash flow matched the filings for all 20 (largest gap 1.5%);
revenue matched for all 20 once total revenue was compared (the four SEC-side gaps — TH, EVRG, PIPR,
AWR — were the parser issue above; IBOC bank revenue within 2% of net interest + noninterest income).

## Methodology weaknesses logged (not fixed: the rules worked as specified)

1. **Cyclical-peak period.** It compares the latest fiscal year with the fiscal-year history, while most
   flags use the trailing twelve months. A one-off fiscal-year gain can fire it while the company is
   currently loss-making (Wabash: Q1 2025 operating income +$314.6M after a $450M 2024 legal charge;
   TTM loss).
2. **Predecessor history.** The 10-year history can include a pre-separation predecessor (Howmet 2016–2019
   as Arconic Inc.: $12–13B revenue, 4–8% margins), inflating "margin vs 10-year average".
3. **Consecutive special dividends.** Two years of large specials are not recognised as specials, so the
   later fall reads as a dividend cut (Piper Sandler FY2023, info severity).
4. **One-off gains in trailing figures.** TTM net income can include a one-off quarter (PTC: $590.7M net
   income on $774.3M revenue), flattering TTM margin and P/E.
5. **Classification judgment.** Industry codes can disagree with business labels (Howmet: SIC 3350 metals,
   labelled Aerospace & Defense; scored as cyclical — defensible, not an objective error).

Any change addressing these is a methodology revision: new freeze and a new, never-inspected holdout.

## Limits of this validation

- The industry-profile fixes before the freeze used universe-wide industry labels, so the holdout tests
  the scoring rules more independently than the classification rules.
- Thin special profiles in the holdout (REIT 6, insurance 5, utility 4) make results for those profiles
  indicative only.
- No return/backtest test: that needs point-in-time data including delisted companies.
- The development set is not random, so H3d cannot cleanly separate selection from overfitting.

## Files

- **Published (in this repository):** [`validation/holdout_2026-10-08/`](../validation/holdout_2026-10-08/), holding the holdout summary, criterion results and manual audit verdicts and rates. The paths below are local only (git-ignored).
- Canonical automated result: `outputs/holdout/2026-10-08_140744_sharadar_summary/` (+ `H6_ADDENDUM_AND_CANONICAL_RESULT.json`, `h6_manual_audit_results.csv`)
- Post-fix rerun: `outputs/holdout/2026-10-08_162653_sharadar_summary/`
- Final frozen screen run: `outputs/runs/2026-10-09_003743_sharadar/` (public tier exported to `portfolio/`)
- REIT reconciliation: [REIT_FFO_RECONCILIATION.md](REIT_FFO_RECONCILIATION.md)
- Freeze record: `config/METHODOLOGY_FREEZE.json`; holdout list: `config/holdout_tickers.txt`; development set at draw: `config/development_set_resolved.txt`
- Audit sheets with provider values (deleted at purge): `outputs/holdout/2026-10-08_140744_sharadar/`
