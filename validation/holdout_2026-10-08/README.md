# Holdout validation: 2026-10-08 (canonical result)

These are the published results of the anti-overfitting test described in
[docs/HOLDOUT_PROTOCOL.md](../../docs/HOLDOUT_PROTOCOL.md) and [docs/VALIDATION_REPORT.md](../../docs/VALIDATION_REPORT.md).
The methodology was frozen at fingerprint `fc2159212b4d7966`, and the holdout of 152 companies was then drawn
(seed 20261008, list checksum `89500a4fafc15b9e`; the list is in [config/holdout_tickers.txt](../../config/holdout_tickers.txt)).
The holdout was evaluated **before any holdout company was inspected**.

| File | Contents |
|---|---|
| `holdout_summary.json` | Run metadata, holdout draw, per-criterion PASS/FAIL, and SEC match rates for holdout vs development set (% of company-years within 2%) |
| `holdout_criteria_results.csv` | Each pre-registered criterion (H0–H5) and the H3d diagnostic, with its result and detail |
| `manual_10k_audit_verdicts.csv` | H6: Y/N verdicts per company on profile, financials vs the 10-K, and flags |
| `manual_10k_audit_summary.json` | H6 pass rates (100% / 100% / 100%; criterion ≥ 90%) and the canonical-result note |

**Result:** 9 of 9 automatic criteria passed. The manual 10-K audit was 20/20 in every column. `H6: TODO` in
`holdout_summary.json` is how the automated run recorded the audit before it was done; the audit files above
complete it. The H3d diagnostic compares the hand-picked development set with the rest of the universe, and it is
explained in the validation report.

These files hold only results, rates and verdicts. They contain no licensed price or fundamental values.
