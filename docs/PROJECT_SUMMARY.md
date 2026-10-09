# Project summary: résumé and interview reference

Engine development closed on 2026-10-09. This page summarizes the frozen state. Screen results are research inputs,
not investment recommendations.

## Final frozen output set

| Item | Location |
|---|---|
| Methodology fingerprint | `fc2159212b4d7966` (`config/METHODOLOGY_FREEZE.json`) |
| Final screen run | `outputs/runs/2026-10-09_003743_sharadar/`. Only `public/` and `retained/` are kept after the purge |
| GitHub-safe outputs | `portfolio/2026-10-07_sharadar/` (Growth and Income top 25, sector summary, universe funnel, charts); `check-public` OK |
| Canonical holdout result | `outputs/holdout/2026-10-08_140744_sharadar_summary/`: 9/9 automatic criteria, H6 manual audit 20/20 |
| Validation narrative | [VALIDATION_REPORT.md](VALIDATION_REPORT.md), [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) |
| Methodology | [SCREENING_METHODOLOGY.md](SCREENING_METHODOLOGY.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md) |
| REIT accuracy check | [REIT_FFO_RECONCILIATION.md](REIT_FFO_RECONCILIATION.md) |
| Licensing | [DATA_LICENSING.md](DATA_LICENSING.md). Sharadar raw data is never committed and is purged after cancelling |

## Key numbers

- **Universe:** 21,024 securities; 2,213 investable U.S. operating companies with 10 years of fundamentals. Growth ranked 534; Income/Value ranked 504.
- **Model:** 70+ metrics and six industry profiles (default, cyclical, bank, insurance, REIT, utility). Scores are peer-relative pillars with red-flag penalties and confidence grades.
- **Tests:** 195 automated tests, covering the providers, metrics, profiles, flags, scoring, holdout, licensing guards and parser regressions.
- **Data accuracy:** Sharadar matches SEC EDGAR within 2% on 100% of operating cash flow and total assets and 99% of net income for 30 validation companies over 10 years. Revenue matched for 96.1% of company-years after the SEC parser fix.
- **Anti-overfitting:**
  - Freeze before testing, then a 152-company holdout drawn after the freeze.
  - 9/9 pre-registered criteria passed.
  - ±20% weight perturbation gives a Spearman median of 0.998 (Growth) and 0.996 (Income).
  - Manual 10-K audit: 20/20.
- **REIT FFO:** 26 of 33 within 10% of company-reported NAREIT FFO (median gap 4.7%). Every larger gap is bridged line by line.

## Résumé bullets

- Built a full-market U.S. equity screener in Python (pandas): 2,213 companies, 10 years of fundamentals, 70+ metrics. Industry-specific scoring for banks, insurers, REITs, utilities and cyclicals, and two ranked screens (Growth, Income/Value).
- Designed an anti-overfitting protocol:
  - Froze the methodology with a config fingerprint, then validated on a 152-company holdout drawn after the freeze.
  - Passed 9/9 pre-registered criteria and a 20/20 manual 10-K audit.
- Reconciled licensed vendor data (Sharadar) against SEC EDGAR XBRL and fixed 10+ objective parsing and data errors without changing the frozen rules. Errors included:
  - Partial revenue tags.
  - REIT gains hidden in company-extension XBRL concepts (one REIT's FFO was overstated 93%).
  - Zero-dividend vendor records.
- Engineered license-compliant data handling with three output tiers (licensed / retained / public), an automated publish check and a purge workflow, so the public repo never contains licensed raw data. 195 automated tests.

## Interview talking points

1. **Why freeze and use a holdout?** Every rule tuned while looking at rankings risks overfitting to familiar names.
   - I froze weights, thresholds and profiles. After that I allowed only objective data, parsing or calculation fixes, and each one had to leave the fingerprint unchanged.
   - I then tested on companies never looked at. When later fixes touched holdout names (Target Hospitality, Evergy, Piper Sandler, Iron Mountain), I documented the contamination instead of redrawing the holdout.
2. **A bug I found by checking against primary sources.** Simon Property screened #3 on Income with FFO/share of $19.98 against $12.34 reported.
   - Cause: a $2.89B Taubman remeasurement gain tagged under a company-specific XBRL concept that the SEC API doesn't expose.
   - Fix: parse the 10-K XBRL instance, restricted to concepts on the income statement. The first pass over-captured Welltower; the whitelist fixed that, and a regression test now covers it.
   - Result: Simon fell to #179.
3. **Knowing the limits of standardized data.** The screen's FFO uses total D&A and no JV share. That is why Equinix reads +31% and Simon −16% against company figures. I reconciled each gap line by line and documented it instead of adding company-specific patches.
4. **A high score isn't a recommendation.** The research write-ups separate screen output from thesis. Examples:
   - Edison International ranks highly partly on one-off wildfire recoveries.
   - Uniti is still labelled a REIT by the vendor after it stopped qualifying, so its rank is unreliable.
5. **Engineering choices.**
   - A provider-neutral schema: synthetic, SEC and Sharadar adapters feed one engine.
   - All thresholds live in YAML.
   - Missing-data status codes (ok / missing / nm / short) instead of silent zeros.
   - Licensing is enforced in code.

## Logged weaknesses (would need a new freeze and a new holdout)

- The cyclical-peak flag reads the fiscal year while other flags read TTM.
- Predecessor history (Howmet).
- Consecutive special dividends read as a cut.
- One-off gains in TTM figures (PTC, EIX).
- REIT FFO lacks real-estate-only D&A, JV share and discontinued-operations adjustments.
- Stale vendor REIT labels (Uniti).
- No return backtest: that needs point-in-time data including delisted companies.
