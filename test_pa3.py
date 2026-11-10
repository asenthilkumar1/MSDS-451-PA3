#!/usr/bin/env python
# coding: utf-8

# Tests for pa3.py (Programming Assignment 3, MSDS 451)
# Run from the repository folder with:  python -m unittest -v test_pa3
# The tests use the price files in the repository and some small made-up
# series where the right answer can be worked out by hand.

import unittest

import numpy as np
import pandas as pd

import pa3

prices, dailyReturns = pa3.loadPrices()

# An independent version of the backtest that keeps track of dollars in
# each asset and in cash, instead of shares of the fund. It is slower and
# longer, and it is only here to check runBacktest in pa3.py.
def referenceBacktest(targets, returns, bps, lag):
    orders = targets.shift(lag + 1).to_numpy()
    R = returns.to_numpy()
    numDays, numAssets = R.shape
    positions = np.zeros(numAssets)
    cash = 1.0
    previousTotal = 1.0
    out = np.zeros(numDays)
    for t in range(numDays):
        positions = positions * (1 + R[t, :])
        total = positions.sum() + cash
        wanted = positions.copy()
        known = ~np.isnan(orders[t, :])
        wanted[known] = orders[t, known] * total
        trade = wanted - positions
        fee = bps / 10000 * np.abs(trade).sum()
        cash = cash - trade.sum() - fee
        positions = wanted
        newTotal = positions.sum() + cash
        if cash < 0: # no borrowing: sell a little of everything to cover the fee
            positions = positions * (newTotal / positions.sum())
            cash = 0.0
        out[t] = newTotal / previousTotal - 1
        previousTotal = newTotal
    return pd.Series(out, index = returns.index)

class TestData(unittest.TestCase):
    def test_four_assets_lined_up(self):
        self.assertEqual(list(prices.columns), pa3.myAssetNames)
        self.assertEqual(prices.shape, dailyReturns.shape)
        self.assertTrue((prices.index == dailyReturns.index).all())
        self.assertFalse(prices.isna().any().any())
        self.assertFalse(dailyReturns.isna().any().any())
        self.assertEqual(len(dailyReturns), 2765)

    def test_returns_match_prices(self):
        again = prices.pct_change().iloc[1:]
        np.testing.assert_allclose(dailyReturns.iloc[1:].to_numpy(), again.to_numpy(), atol = 1e-12)

class TestSignals(unittest.TestCase):
    def test_regression_score_on_a_perfect_trend(self):
        t = np.arange(300)
        trend = pd.DataFrame({"A": 100 * np.exp(0.001 * t)})
        score = pa3.regressionScore(trend, 90)
        expected = np.exp(0.001 * pa3.tradingDays) - 1 # R squared is 1
        self.assertTrue(score.iloc[:89].isna().all().all())
        np.testing.assert_allclose(score.iloc[89:, 0], expected, rtol = 1e-9)

    def test_regression_score_is_negative_in_a_downtrend(self):
        t = np.arange(300)
        down = pd.DataFrame({"A": 100 * np.exp(-0.001 * t)})
        self.assertTrue((pa3.regressionScore(down, 90).iloc[89:, 0] < 0).all())

    def test_momentum_weights_are_valid(self):
        targets = pa3.momentumTargets(prices, dailyReturns).dropna(how = "all")
        self.assertGreater(len(targets), 100)
        self.assertTrue((targets.to_numpy() >= 0).all())
        self.assertTrue((targets.sum(axis = 1) <= 1 + 1e-12).all())

    def test_momentum_holds_nothing_in_a_downtrend(self):
        t = np.arange(400)
        down = pd.DataFrame({"A": 100 * np.exp(-0.002 * t), "B": 50 * np.exp(-0.001 * t)})
        returns = down.pct_change().fillna(0)
        targets = pa3.momentumTargets(down, returns).dropna(how = "all")
        self.assertTrue((targets.to_numpy() == 0).all())

    def test_mean_reversion_buys_the_dip_and_sells_the_recovery(self):
        # flat for 40 days, a sharp drop, a slow climb back, then flat again
        path = np.r_[np.full(40, 100.0), np.linspace(100, 85, 3), np.linspace(85, 101, 20), np.full(20, 101.0)]
        series = pd.DataFrame({"A": path})
        targets = pa3.meanReversionTargets(series, window = 20, entryZ = 1.5, exitZ = 0.0)
        buys = targets.index[targets["A"] > 0] # one asset, so it gets the whole fund
        sells = targets.index[targets["A"] == 0.0]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(sells), 1)
        self.assertGreater(buys[0], 39) # only after the drop starts
        self.assertGreater(sells[0], buys[0])

    def test_signals_do_not_use_future_prices(self):
        cut = 1500
        fullMom = pa3.momentumTargets(prices, dailyReturns)
        partMom = pa3.momentumTargets(prices.iloc[:cut], dailyReturns.iloc[:cut])
        np.testing.assert_allclose(fullMom.iloc[:cut].to_numpy(), partMom.to_numpy(), equal_nan = True)
        fullMr = pa3.meanReversionTargets(prices)
        partMr = pa3.meanReversionTargets(prices.iloc[:cut])
        np.testing.assert_allclose(fullMr.iloc[:cut].to_numpy(), partMr.to_numpy(), equal_nan = True)

class TestBacktest(unittest.TestCase):
    def test_matches_independent_version_for_momentum(self):
        targets = pa3.momentumTargets(prices, dailyReturns)
        for lag in [0, 1]:
            mine = pa3.runBacktest(targets, dailyReturns, 10, lag)["net"]
            other = referenceBacktest(targets, dailyReturns, 10, lag)
            np.testing.assert_allclose(mine.to_numpy(), other.to_numpy(), atol = 1e-10)

    def test_matches_independent_version_for_mean_reversion(self):
        targets = pa3.meanReversionTargets(prices)
        mine = pa3.runBacktest(targets, dailyReturns, 10, 0)["net"]
        other = referenceBacktest(targets, dailyReturns, 10, 0)
        np.testing.assert_allclose(mine.to_numpy(), other.to_numpy(), atol = 1e-10)

    def test_matches_independent_version_for_equal_weights(self):
        targets = pa3.equalWeightTargets(prices)
        mine = pa3.runBacktest(targets, dailyReturns, 10, -1)["net"]
        other = referenceBacktest(targets, dailyReturns, 10, -1)
        np.testing.assert_allclose(mine.to_numpy(), other.to_numpy(), atol = 1e-10)

    def test_buy_and_hold_without_costs_is_the_asset(self):
        r = pa3.runBacktest(pa3.buyAndHoldTargets(prices, "SP500"), dailyReturns, 0, -1)
        # bought at the close on day one, so the day one return is not earned
        self.assertEqual(r["net"].iloc[0], 0.0)
        np.testing.assert_allclose(r["net"].iloc[1:].to_numpy(), dailyReturns["SP500"].iloc[1:].to_numpy(), atol = 1e-12)

    def test_a_trade_costs_what_it_should(self):
        # one purchase of 50% in an asset, then nothing
        index = ["d0", "d1", "d2", "d3"]
        flat = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0]}, index = index)
        targets = pd.DataFrame({"A": [0.5, np.nan, np.nan, np.nan]}, index = index)
        r = pa3.runBacktest(targets, flat, costBps = 100, lag = -1)
        self.assertAlmostEqual(r["net"].iloc[0], -0.5 * 0.01, places = 12)
        self.assertAlmostEqual(r["traded"].sum(), 0.5, places = 12)
        self.assertTrue((r["net"].iloc[1:] == 0).all())

    def test_orders_wait_for_the_lag(self):
        index = [f"d{i}" for i in range(6)]
        up = pd.DataFrame({"A": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]}, index = index)
        targets = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan]}, index = index)
        # signal on d0, filled at the close of d1 (lag 0), first return earned is d2
        r = pa3.runBacktest(targets, up, costBps = 0, lag = 0)
        self.assertEqual(r["net"].iloc[1], 0.0)
        self.assertAlmostEqual(r["net"].iloc[2], 0.1, places = 12)

    def test_weights_never_leave_zero_to_one(self):
        for targets in [pa3.momentumTargets(prices, dailyReturns), pa3.meanReversionTargets(prices)]:
            r = pa3.runBacktest(targets, dailyReturns)
            self.assertLessEqual(r["exposure"].max(), 1 + 1e-9)
            self.assertGreaterEqual(r["weights"].to_numpy().min(), -1e-12)

class TestMeasures(unittest.TestCase):
    def test_steady_gains(self):
        steady = pd.Series(np.full(504, 0.001)) # two years
        m = pa3.performanceMetrics(steady)
        self.assertAlmostEqual(m["CAGR"], 1.001 ** 252 - 1, places = 9)
        self.assertEqual(m["Max Drawdown"], 0.0)

    def test_sharpe_and_drawdown_by_hand(self):
        r = pd.Series([0.10, -0.10, 0.05, -0.05, 0.02, 0.01])
        m = pa3.performanceMetrics(r)
        expected = r.mean() / r.std() * np.sqrt(252)
        self.assertAlmostEqual(m["Sharpe"], expected, places = 12)
        equity = np.cumprod(1 + r.to_numpy())
        worst = (equity / np.maximum.accumulate(equity) - 1).min()
        self.assertAlmostEqual(m["Max Drawdown"], worst, places = 12)
        # the worst fall is from the peak after day one (1.10) to the end of day four (0.9 * 1.05 * 0.95 of it)
        self.assertAlmostEqual(m["Max Drawdown"], 0.9 * 1.05 * 0.95 - 1, places = 12)

    def test_alpha_and_beta(self):
        rng = np.random.default_rng(5)
        bench = pd.Series(rng.normal(0.0004, 0.01, 1000))
        mine = 0.5 * bench + 0.0002
        ab = pa3.alphaBeta(mine, bench)
        self.assertAlmostEqual(ab["Beta"], 0.5, places = 9)
        self.assertAlmostEqual(ab["Alpha"], 0.0002 * 252, places = 9)
        self.assertAlmostEqual(ab["Correlation"], 1.0, places = 9)

    def test_yearly_returns(self):
        frame = pd.DataFrame({"A": [0.1, 0.1, -0.5]}, index = ["2020-06-01", "2020-06-02", "2021-01-04"])
        y = pa3.yearlyReturns(frame)
        self.assertAlmostEqual(y.loc["2020", "A"], 1.1 * 1.1 - 1, places = 12)
        self.assertAlmostEqual(y.loc["2021", "A"], -0.5, places = 12)

    def test_bootstrap_is_reproducible(self):
        frame = pd.DataFrame(np.random.default_rng(2).normal(0.0005, 0.01, (400, 2)), columns = ["a", "b"])
        one = pa3.bootstrapSharpe(frame, reps = 50, block = 10, seed = 1)
        two = pa3.bootstrapSharpe(frame, reps = 50, block = 10, seed = 1)
        three = pa3.bootstrapSharpe(frame, reps = 50, block = 10, seed = 2)
        np.testing.assert_array_equal(one.to_numpy(), two.to_numpy())
        self.assertFalse(np.array_equal(one.to_numpy(), three.to_numpy()))
        self.assertEqual(one.shape, (50, 2))

if __name__ == "__main__":
    unittest.main(verbosity = 2)
