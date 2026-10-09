# Methodology freeze and holdout validation (anti-overfitting)

Results of the first run (closed 2026-10-08): [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

The screener was developed and debugged on a known set of companies: the SEC validation list, the
Sharadar free sample (DJIA 30), the universe spot checks, every company named in the docs, tests,
config or code, and every company inspected while debugging the full-market run
(`config/development_tickers.txt`). Rules tuned on those names may look right on them and wrong
elsewhere. The holdout tests the frozen methodology on companies nobody looked at.

## 1. Freeze (done once, before the holdout is drawn)

`python -m screener freeze` records `config/METHODOLOGY_FREEZE.json`: a fingerprint of the red-flag
rules, industry profiles, screen definitions (pillars, weights, gates) and the scoring and universe
settings. Every `run` prints whether the methodology still matches it.

**After the freeze:**

- no changes to scoring weights or thresholds;
- no company-specific exceptions;
- no profile-rule changes because a ranking looks unintuitive;
- allowed: fixes for **objective** data, parsing, classification, period-matching or calculation
  errors (a value that disagrees with the filing, a company in an industry profile its own industry
  label contradicts, a fiscal year matched to the wrong period, a formula that computes the wrong thing).
  Such a fix changes code, not the fingerprinted config; if it must change config (e.g. a profile
  match rule that misreads an industry label), it is logged with the evidence and the methodology is
  re-frozen, and the holdout is re-evaluated as-is (not redrawn).

Any deliberate methodology change (weights, thresholds, new rules) requires a new freeze **and a new,
never-inspected holdout**. The current holdout cannot be reused to tune on.

## 2. Draw (once)

`python -m screener holdout draw --provider sharadar` (150 companies, seed 20261008):

- pool = current investable universe minus the development set;
- stratified by industry profile x size (small < $2B, mid $2–10B, large > $10B), proportional, with at
  least 3 companies per special profile (bank, insurance, REIT, utility, cyclical) so their rules are tested;
- written to `config/holdout_tickers.txt` with the snapshot id, methodology fingerprint, seed and a
  checksum of the list; `config/development_set_resolved.txt` records the development set it excluded;
- the command refuses to redraw (`--force` only after a new freeze).

**Do not open, debug or discuss individual holdout companies before the evaluation has run.**
`holdout evaluate` fails check H0b if a holdout name later appears in the docs, tests, config, code
or development list.

## 3. Evaluate (before any further ranking or scoring change)

Refresh the free SEC data for the holdout first (keep the validation names and REITs in the same
refresh, because the SEC snapshot is replaced and the REIT FFO supplement reads it):

```
python -m screener refresh --provider sec --tickers-file config/validation_tickers.txt --tickers-file config/holdout_tickers.txt --tickers-from sharadar:reit
python -m screener build --provider sharadar
python -m screener holdout evaluate --provider sharadar
```

Pre-registered criteria (decided before the holdout was drawn; not changed after seeing results):

| ID | Check | Pass when | Type |
|---|---|---|---|
| H0 | holdout list intact, still investable | checksum matches; >= 90% still in the universe | criterion |
| H0b | no contamination | no holdout name in docs/tests/config/code/development list | criterion |
| H1 | data accuracy vs SEC filings | each core field (revenue, net income, CFO, assets, equity, dividends paid, EPS, shares) matches SEC within 2% for >= 80% of holdout company-years, and no field more than 10 pp below the development set | criterion |
| H2 | classification | <= 3% of holdout companies in a special profile their industry label contradicts (or a REIT/insurer label left in default) | criterion |
| H3 | red-flag rates | no flag's holdout rate differs from the rest of the universe by more than 3 standard errors | criterion |
| H3d | development set vs rest | same test on the development set: a gap here means the rules behave differently on names they were tuned on | diagnostic |
| H4 | screen outcomes | needs-review share of the holdout within 10 pp of the rest of the universe, per screen | criterion |
| H5 | weight robustness | each pillar weight x0.8 and x1.2: holdout rank correlation (Spearman) median >= 0.95, min >= 0.90; top-quintile overlap median >= 80% | criterion |
| H6 | manual audit | 20 holdout companies (top 5 per screen, 5 in needs-review, the rest random) checked against the latest 10-K: profile correct, financials match, flags factually correct; >= 90% Y in every column | manual |

**Reading the results.** A failure is first diagnosed: an objective error (wrong value, wrong period,
wrong classification, wrong formula) may be fixed under the freeze rules; anything else is recorded as
a methodology weakness and **not** fixed by tuning until a deliberate, documented methodology revision
(new freeze, new holdout). A pass does not show the screener picks good investments - only that its
rules behave on unseen companies as they do on the companies used to build it.

**Limits.** The industry-profile fixes made on 2026-10-08 used industry labels across the whole
universe (counts by profile and industry), not individual development names, so the holdout tests the
scoring rules more independently than the classification rules. There is no performance (return)
test: rankings are of current fundamentals, and a backtest would need point-in-time data and the
delisted companies, which is separate work.

## Outputs

`outputs/holdout/<timestamp>_<provider>/` (license-scoped, deleted by `purge`): the audit sheet with
values to check. `<timestamp>_<provider>_summary/` (kept): pass/fail table, flag-rate parity,
screen outcomes by group, weight-robustness table, summary JSON - rates and correlations only.
