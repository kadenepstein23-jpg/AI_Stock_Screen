"""Split-adjusted per-share history (regression test for the Apple 2020 4-for-1 split bug).

The Apple figures below are SEC EDGAR company-facts values (public domain) exactly as
the SEC provider stored them: FY2017 and earlier are on the PRE-split basis for EPS,
DPS and diluted shares; FY2018 shares outstanding are also pre-split while FY2018
EPS/DPS/diluted shares are restated — the mixed state that caused the bug.
"""
import numpy as np
import pandas as pd
import pytest

from screener.metrics.engine import compute_metrics
from screener.metrics.splits import detect_splits, snap_split_factor
from tests.conftest import fy_rows
from tests.test_metrics import _tables

AS_OF = pd.Timestamp("2026-06-30")


def apple_rows():
    years = list(range(2014, 2026))
    ends = ["2014-09-27", "2015-09-26", "2016-09-24", "2017-09-30", "2018-09-29", "2019-09-28",
            "2020-09-26", "2021-09-25", "2022-09-24", "2023-09-30", "2024-09-28", "2025-09-27"]
    rows = fy_rows(
        "AAPL", years,
        shares_diluted_wavg=[6.122663e9, 5.793069e9, 5.500281e9, 5.251692e9, 20.00044e9, 18.59565e9,
                             17.52821e9, 16.86492e9, 16.32582e9, 15.81255e9, 15.40810e9, 15.00470e9],
        shares_outstanding=[5.866161e9, 5.578753e9, 5.336166e9, 5.126201e9, 4.754986e9, 17.77294e9,
                            16.97676e9, 16.42679e9, 15.94342e9, 15.55006e9, 15.11679e9, 14.77326e9],
        eps_diluted=[6.45, 9.22, 8.31, 9.21, 2.98, 2.97, 3.28, 5.61, 6.11, 6.13, 6.08, 7.46],
        dps=[1.82, 1.98, 2.18, 2.40, 0.68, 0.75, 0.795, 0.85, 0.90, 0.94, 0.98, 1.02],
        dividends_paid=[11.031e9, 11.561e9, 11.965e9, 12.563e9, 13.712e9, 14.119e9, 14.081e9, 14.467e9,
                        14.841e9, 15.025e9, 15.234e9, 15.421e9],
        net_income=[39.510e9, 53.394e9, 45.687e9, 48.351e9, 59.531e9, 55.256e9, 57.411e9, 94.680e9,
                    99.803e9, 96.995e9, 93.736e9, 112.010e9],
        revenue=[182.795e9, 233.715e9, 215.639e9, 229.234e9, 265.595e9, 260.174e9, 274.515e9, 365.817e9,
                 394.328e9, 383.285e9, 391.035e9, 416.161e9],
        total_assets=[231.839e9, 290.345e9, 321.686e9, 375.319e9, 365.725e9, 338.516e9, 323.888e9,
                      351.002e9, 352.755e9, 352.583e9, 364.980e9, 359.241e9],
        equity=[100e9] * 12,
    )
    for r, e in zip(rows, ends):
        r["period_end"] = pd.Timestamp(e)
    return rows


def test_snap_split_factor():
    assert snap_split_factor(3.81) == 4          # 4:1 split net of a year of buybacks
    assert snap_split_factor(19.6) == 20
    assert snap_split_factor(0.1) == pytest.approx(0.1)   # 1-for-10 reverse split
    assert snap_split_factor(1.5) == 1.5
    assert snap_split_factor(1.07) is None        # ordinary dilution
    assert snap_split_factor(2.45) is None


def test_apple_dividend_streak_and_history_are_split_adjusted():
    v, st = compute_metrics(_tables(apple_rows(), mcap=3.8e12), AS_OF)
    r = v.loc["AAPL"]
    assert r["splits_detected"] == 1
    # Apple raised its dividend every fiscal year in the window (FY2015..FY2025 = 11 years)
    assert r["dividend_paying_years"] == 11
    assert r["dividend_growth_streak_years"] == 10      # 10 year-over-year increases inside the window
    assert r["dividend_cut_5y"] == 0
    assert r["dps_cagr_5y_pct"] == pytest.approx(((1.02 / 0.795) ** (1 / 5) - 1) * 100)
    assert r["eps_cagr_5y_pct"] == pytest.approx(((7.46 / 3.28) ** (1 / 5) - 1) * 100)
    assert r["share_change_1y_pct"] == pytest.approx((15.00470 / 15.40810 - 1) * 100)


def test_apple_individual_years_rescaled_only_where_needed():
    from screener.metrics.engine import _panel
    from screener.metrics.splits import split_adjust_panel
    from screener import schema

    fin = schema.conform(pd.DataFrame(apple_rows()), "financials")
    P, _ = _panel(fin, pd.Index(["AAPL"]))
    P, n = split_adjust_panel(P)
    # offsets: 0 = FY2025 ... 7 = FY2018, 8 = FY2017
    assert P["dps"].loc["AAPL", 8] == pytest.approx(0.60)        # 2.40 / 4 (pre-split value)
    assert P["dps"].loc["AAPL", 7] == pytest.approx(0.68)        # already restated -> untouched
    assert P["eps_diluted"].loc["AAPL", 8] == pytest.approx(9.21 / 4)
    assert P["eps_diluted"].loc["AAPL", 7] == pytest.approx(2.98)
    assert P["shares_outstanding"].loc["AAPL", 7] == pytest.approx(4.754986e9 * 4)
    assert P["shares_diluted_wavg"].loc["AAPL", 7] == pytest.approx(20.00044e9)  # restated -> untouched
    # every DPS value now rises year over year
    dps = P["dps"].loc["AAPL"].dropna().to_numpy()
    assert (dps[:-1] > dps[1:]).all()


def _company(ticker, shares, eps, dps, assets, ni=None, div=None):
    n = len(shares)
    years = list(range(2026 - n, 2026))
    ni = ni or [e * s for e, s in zip(eps, shares)]
    div = div or [d * s for d, s in zip(dps, shares)]
    return fy_rows(ticker, years, shares_diluted_wavg=shares, shares_outstanding=shares, eps_diluted=eps,
                   dps=dps, net_income=ni, dividends_paid=div, total_assets=assets,
                   revenue=[a * 0.5 for a in assets], equity=[a * 0.4 for a in assets])


def test_stock_funded_acquisition_is_not_a_split():
    # shares double AND assets more than double -> merger, keep as real dilution
    rows = _company("MRG", shares=[100e6, 100e6, 200e6, 200e6], eps=[2.0, 2.1, 1.6, 1.8],
                    dps=[1.0, 1.05, 1.1, 1.15], assets=[1e9, 1.05e9, 2.3e9, 2.4e9])
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["MRG", "splits_detected"] == 0
    assert v.loc["MRG", "share_change_3y_cagr_pct"] == pytest.approx((2 ** (1 / 3) - 1) * 100)


def test_two_splits_with_partial_restatement_and_fast_growth():
    # 4:1 split, then a 10:1 split in a year when assets grew 60% (NVIDIA-like).
    # Per-share history is restated for the older split but not the newer one in some years.
    true_eps = [0.10, 0.12, 0.15, 0.20, 0.30, 0.50]          # today's basis
    shares_today = 24e9
    shares = [shares_today / 40, shares_today / 40, shares_today / 10, shares_today / 10,
              shares_today, shares_today]                    # pre-4:1, pre-4:1, post-4:1, post-4:1, post-10:1 ...
    ni = [e * shares_today for e in true_eps]
    reported_eps = [true_eps[0] * 40, true_eps[1] * 40, true_eps[2] * 10, true_eps[3] * 10,
                    true_eps[4], true_eps[5]]
    rows = _company("NVX", shares=shares, eps=reported_eps, dps=[0.0] * 6,
                    assets=[10e9, 11e9, 12e9, 13e9, 21e9, 30e9], ni=ni, div=[0.0] * 6)
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["NVX", "splits_detected"] == 2
    assert v.loc["NVX", "eps_cagr_5y_pct"] == pytest.approx(((0.50 / 0.10) ** (1 / 5) - 1) * 100)
    assert v.loc["NVX", "share_change_3y_cagr_pct"] == pytest.approx(0.0, abs=1e-9)


def test_reverse_split():
    rows = _company("REV", shares=[500e6, 500e6, 50e6, 50e6], eps=[0.10, 0.11, 1.2, 1.3],
                    dps=[0.0] * 4, assets=[1e9, 1e9, 1.05e9, 1.1e9])
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["REV", "splits_detected"] == 1
    assert v.loc["REV", "eps_cagr_3y_pct"] == pytest.approx(((1.3 / 1.0) ** (1 / 3) - 1) * 100)


def test_no_split_leaves_history_untouched():
    shares = np.array([100.0, 99.0, 97.0, 96.0])
    cum, events = detect_splits(shares, np.ones(4), np.ones(4))
    assert events == [] and (cum == 1).all()


def test_stock_merger_with_lagging_weighted_shares_is_not_a_split():
    """Realty Income (O), SEC company-facts values: the all-stock VEREIT merger closed Nov 2021.
    Weighted-average diluted shares jumped ~1.48x a year later (2021 -> 2022), which looks like a
    3:2 split on its own; shares outstanding jumped 1.64x in 2021. Neither is a split."""
    years = list(range(2017, 2026))
    rows = fy_rows(
        "O", years,
        shares_diluted_wavg=[273.94e6, 289.92e6, 316.16e6, 345.42e6, 414.77e6, 612.18e6, 693.02e6, 863.79e6, 908.33e6],
        shares_outstanding=[284.21e6, 303.74e6, 333.62e6, 361.30e6, 591.26e6, 660.30e6, 752.46e6, 891.51e6, 933.98e6],
        eps_diluted=[1.10, 1.26, 1.38, 1.14, 0.87, 1.42, 1.26, 0.98, 1.17],
        dps=[2.527, 2.6305, 2.7105, 2.794, 2.833, 2.967, 3.051, 3.1255, 3.217],
        dividends_paid=[689.3e6, 761.6e6, 852.1e6, 964.2e6, 1169.0e6, 1813.4e6, 2111.8e6, 2691.7e6, 2920.9e6],
        net_income=[301.5e6, 363.6e6, 436.5e6, 395.5e6, 359.5e6, 869.4e6, 872.3e6, 847.9e6, 1058.6e6],
        revenue=[1215.8e6, 1327.8e6, 1488.2e6, 1647.1e6, 2080.5e6, 3343.7e6, 4079.0e6, 5271.1e6, 5749.4e6],
        total_assets=[14.06e9, 15.26e9, 18.55e9, 20.74e9, 43.14e9, 49.67e9, 57.78e9, 68.84e9, 72.80e9],
        equity=[8e9] * 9,
    )
    v, _ = compute_metrics(_tables(rows), AS_OF)
    assert v.loc["O", "splits_detected"] == 0
    assert v.loc["O", "dividend_growth_streak_years"] == 8
    assert v.loc["O", "eps_cagr_3y_pct"] == pytest.approx(((1.17 / 1.42) ** (1 / 3) - 1) * 100)
