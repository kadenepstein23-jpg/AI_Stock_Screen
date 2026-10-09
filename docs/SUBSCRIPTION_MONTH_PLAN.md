# Subscription-month runbook (Sharadar, ~$49 / month, 10-year bundle)

The goal is to finish every free task **before** paying, then use the paid month only
for full-market ingestion, validation, tuning and final outputs.

## Phase 0: before subscribing (free)

| Step | Command / action | Done when |
|---|---|---|
| 0.1 | `python -m screener demo` and `python -m pytest -q` | Demo runs and all tests pass |
| 0.2 | Add `SEC_USER_AGENT="Your Name your@email.com"` to `.env` | |
| 0.3 | `python -m screener refresh --provider sec --tickers-file config/validation_tickers.txt` | SEC snapshot stored for about 25 companies |
| 0.4 | `python -m screener company AAPL --provider sec`, and spot-check 3–4 companies against their 10-Ks | Revenue, net income, FCF and debt match the filings |
| 0.5 | `python -m screener refresh --provider sharadar --sample` — free: the public `test-api-key` returns AAPL only; a free Sharadar account (no card) gives a key for the DJIA-30 sample (`SHARADAR_SAMPLE_API_KEY` in .env). Then `validate`, `run --provider sharadar`, `check-public`, `purge --provider sharadar` | The adapter parses Sharadar's real column layout; values agree with SEC |
| 0.6 | `python -m screener validate --primary sharadar --reference sec` | Most fields match within 2% (fix tag mappings if not) |
| 0.7 | `python -m screener purge --provider sharadar` | Sample data removed (it's Sharadar data too) |
| 0.8 | Review `docs/SCREENING_METHODOLOGY.md` and adjust weights and thresholds in `config/` | The methodology is settled before paying |

## Phase 1: week 1 of the subscription

| Step | Command | Done when |
|---|---|---|
| 1.1 | Add `SHARADAR_API_KEY` to `.env`, then `python -m screener check-key` | Every row PASS: the key is not the sample key, XOM (outside the free sample) returns ~10 fiscal years, all four bulk routes answer 302, enough disk space. No data is downloaded |
| 1.2 | `python -m screener -v refresh --provider sharadar` | Full bulk download (tickers snapshot, fundamentals 10y, daily 10y, prices 5y zips; only `price_history_days` of prices kept), then metrics built |
| 1.3 | `python -m screener universe --provider sharadar` | No FAIL: whole market incl. delisted names, universe ~1,500–4,000, history depth matches the 10-year tier, every universe company has a recent price, reference companies present with the expected profile |
| 1.4 | `python -m screener validate --primary sharadar --reference sec --tickers-file config/validation_tickers.txt` (after `refresh --provider sec --tickers-file ...`) | Same definition differences as the sample; no new systematic gaps |
| 1.4b | `python -m screener refresh --provider sec --tickers-file config/validation_tickers.txt --tickers-from sharadar:reit` then `python -m screener build --provider sharadar` | SEC impairments/gains fill REIT FFO (build prints how many values were filled); the SEC snapshot also covers the validation set |
| 1.5 | Hand-check 10–15 companies across profiles (bank, REIT, utility, cyclical, mega-cap tech, small-cap) against their 10-K/10-Q | Values and profiles agree |
| 1.6 | `python -m screener run --provider sharadar`; review `needs_review`, `excluded.csv` and the flag distribution | Thresholds that misfire are tuned before final screens |

Errors never print the API key (messages are redacted); `401` = key rejected, `403 Exceeds free tier` = the request ran as free tier.

## Phase 1b: freeze and holdout (before any further ranking or scoring change)

See [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md). `freeze` → `holdout draw` → SEC refresh incl. the holdout → `build` → `holdout evaluate` → manual audit sheet. After the freeze only objective data/parsing/classification/period-matching/calculation fixes.

## Phase 2: weeks 2–3

- Run sensitivity tests on the weights: copy a screen YAML (e.g. `growth_quality_tilt.yaml`), re-run, and compare the top 25.
- Write up 3–5 top candidates per screen using the stock write-up format in the project instructions.
- Produce the final run, then `python -m screener export-portfolio`. Screenshot the charts for your résumé or GitHub README.

## Phase 3: before cancelling (week 4)

Follow the cancellation checklist in [DATA_LICENSING.md](DATA_LICENSING.md#cancellation-checklist).

## Re-subscribing later

```bash
# add SHARADAR_API_KEY back to .env
python -m screener refresh --provider sharadar
python -m screener run --provider sharadar
# ... then purge again after cancelling
```

No code changes are needed. The screener runs as soon as a snapshot exists.
