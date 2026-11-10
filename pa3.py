#!/usr/bin/env python
# coding: utf-8

# Algorithmic Trading: A Momentum and Mean-Reversion Backtest

# Programming Assignment 3, MSDS 451
# Written by Ashwin Senthilkumar, November 2026

# This builds on Programming Assignment 2. It uses the same four assets
# (a bond fund, gold, an S&P 500 fund, and bitcoin), the same Yahoo! Finance
# data files, and the same date alignment as the PA2 program (pa2.py).
# The trading ideas follow Clenow (2019), Trading Evolved.

# Two automated strategies are backtested on daily data:
#   1. Momentum: hold an asset while its 90 day trend is up (Clenow's
#      exponential regression slope times R squared), sized by volatility.
#   2. Mean reversion: buy an asset when it falls well below its 20 day
#      average and sell when it gets back to the average.
# A third "combined" fund puts half of its money in each strategy.
# The results are compared with buy-and-hold benchmarks: the S&P 500 (SPY),
# the four assets held in equal weights, and the bond fund (BND).

import os
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # figures are saved to files, no windows open
import matplotlib.pyplot as plt
import matplotlib.ticker
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# folder that holds this program, the data files, and the output files
baseDir = os.path.dirname(os.path.abspath(__file__))

# Yahoo! Finance tickers and the names used in the output (same as PA2)
# (the data files are made by getdata_yfinance.py)
myTickers = ["BND", "GLD", "SPY", "BTC-USD"]
myAssetNames = ["Bonds", "Gold", "SP500", "Bitcoin"]
startfile = "msds_getdata_yfinance_"
tradingDays = 252 # trading days per year, to annualize daily returns

# The backtest results start after this many days, so that every signal
# (the longest lookback tested is 250 days) has enough history.
warmUpDays = 260

# Trading costs in basis points of the amount traded. 10 basis points is
# 0.10%, charged on every dollar bought or sold.
costBps = 10

# Orders are decided from the closing prices on day t and filled at the
# close on day t + 1 + executionLag. With 0, a signal made from today's
# close is traded at tomorrow's close, so a trade never happens at a price
# that was used to make the signal.
executionLag = 0

# momentum strategy settings (fixed before looking at any results)
momentumWindow = 90 # days of prices in the trend regression
volWindow = 60 # days of returns used to measure volatility
rebalanceDays = 5 # trade once a week
riskBudget = 0.05 # each asset in an uptrend gets 5% annual volatility

# mean reversion strategy settings (also fixed ahead of time)
mrWindow = 20 # days in the moving average
mrEntryZ = 1.5 # buy when the price is this many standard deviations below
mrExitZ = 0.0 # sell when the price is back at the average

# bootstrap settings for the Sharpe ratio confidence intervals
bootstrapReps = 2000
bootstrapBlock = 21 # days per block, about one month
bootstrapSeed = 3333

# names of the strategies and benchmarks, in the order they are shown
strategyNames = ["Momentum", "Mean Reversion", "Combined"]
benchmarkNames = ["SPY Buy-and-Hold", "Equal Weight", "BND Buy-and-Hold"]
allNames = strategyNames + benchmarkNames

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

# Read the closing prices for each asset, keep the dates that all four
# have, and compute daily returns. This is the same pipeline as pa2.py.
# Returns prices lined up with the returns: row t of the prices is the
# close on day t and row t of the returns is the change from day t-1.
def loadPrices(tickers = myTickers, assetNames = myAssetNames):
    prices = pd.DataFrame()
    for ticker, assetName in zip(tickers, assetNames):
        filename = os.path.join(baseDir, startfile + ticker.lower() + ".csv")
        if not os.path.exists(filename):
            # pull the price history first if the data files are not here yet
            print("Missing " + filename + ", running getdata_yfinance.py")
            subprocess.run([sys.executable, "getdata_yfinance.py"], check = True, cwd = baseDir)
        priceData = pd.read_csv(filename)
        # keep only the calendar date (the files have a time zone on the date)
        priceData["Date"] = priceData["Date"].str[:10]
        prices[assetName] = priceData.set_index("Date")["Close"]
    prices = prices.dropna() # only dates when all four assets have a price
    dailyReturns = prices.pct_change().dropna()
    prices = prices.loc[dailyReturns.index]
    return prices, dailyReturns

# ----------------------------------------------------------------------
# Trading signals
# ----------------------------------------------------------------------

# Clenow's momentum score: fit a straight line to the log prices over the
# last "window" days, turn the slope into an annual rate, and multiply by
# R squared so that smooth trends score higher than jumpy ones.
# The score on day t only uses prices up to and including day t.
def regressionScore(prices, window):
    logPrices = np.log(prices.to_numpy())
    x = np.arange(window) - (window - 1) / 2
    sxx = (x ** 2).sum()
    windows = sliding_window_view(logPrices, window, axis = 0) # days, assets, window
    yMean = windows.mean(axis = 2)
    slope = (windows * x).sum(axis = 2) / sxx
    syy = ((windows - yMean[:, :, None]) ** 2).sum(axis = 2)
    rSquared = slope ** 2 * sxx / syy
    annualReturn = np.exp(slope * tradingDays) - 1
    score = np.full(logPrices.shape, np.nan)
    score[window - 1:, :] = annualReturn * rSquared
    return pd.DataFrame(score, index = prices.index, columns = prices.columns)

# Momentum: every rebalanceDays days, hold the assets whose score is
# positive. Each one is sized so that it adds about "budget" of annual
# volatility (budget divided by its own volatility), so risky assets get
# smaller positions. If the positions add up to more than 100% of the fund,
# they are scaled down. Any money not invested stays in cash.
# Returns the target weights on rebalance days and NaN (do nothing) on
# the other days.
def momentumTargets(prices, dailyReturns, window = momentumWindow, volWin = volWindow,
        rebalDays = rebalanceDays, budget = riskBudget):
    score = regressionScore(prices, window)
    vol = dailyReturns.rolling(volWin).std() * np.sqrt(tradingDays)
    firstDay = max(window, volWin) - 1
    targets = pd.DataFrame(np.nan, index = prices.index, columns = prices.columns)
    for t in range(firstDay, len(prices), rebalDays):
        uptrend = (score.iloc[t] > 0).to_numpy()
        weights = np.where(uptrend, budget / vol.iloc[t].to_numpy(), 0.0)
        if weights.sum() > 1:
            weights = weights / weights.sum() # no borrowing
        targets.iloc[t] = weights
    return targets

# Mean reversion: the z score is how many standard deviations the price is
# from its moving average. Buy an asset when z drops below -entryZ and sell
# when it is back to exitZ. Each asset gets its own equal share of the fund
# (a quarter each), which sits in cash while the asset is not bought.
# Returns the target weights on the days when an asset's position changes.
def meanReversionTargets(prices, window = mrWindow, entryZ = mrEntryZ, exitZ = mrExitZ):
    z = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
    zValues = z.to_numpy()
    numDays, numAssets = zValues.shape
    share = 1 / numAssets
    targets = np.full((numDays, numAssets), np.nan)
    holding = np.zeros(numAssets, dtype = bool)
    for t in range(numDays):
        for i in range(numAssets):
            if np.isnan(zValues[t, i]):
                continue
            if (not holding[i]) and zValues[t, i] < -entryZ:
                holding[i] = True
                targets[t, i] = share
            elif holding[i] and zValues[t, i] >= exitZ:
                holding[i] = False
                targets[t, i] = 0.0
    return pd.DataFrame(targets, index = prices.index, columns = prices.columns)

# Buy-and-hold: buy on the first day and never trade again.
def buyAndHoldTargets(prices, assetName):
    targets = pd.DataFrame(np.nan, index = prices.index, columns = prices.columns)
    targets.iloc[0] = 0.0
    targets.iloc[0, targets.columns.get_loc(assetName)] = 1.0
    return targets

# Equal weights: put a quarter in each asset on the first trading day of
# every month (the benchmark for the fund as a whole).
def equalWeightTargets(prices):
    targets = pd.DataFrame(np.nan, index = prices.index, columns = prices.columns)
    month = prices.index.str[:7]
    firstOfMonth = np.r_[True, month[1:] != month[:-1]]
    targets.loc[firstOfMonth, :] = 1 / prices.shape[1]
    return targets

# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------

# Run a strategy day by day. The targets are decided at the close on day t
# and traded lag + 1 days later (lag of -1 means the same close, which is
# only used for the buy-and-hold benchmarks and one sensitivity check). Between trades the positions drift with prices, and
# each trade pays costBps on the amount bought or sold.
# Returns the daily return after costs, the daily return before costs,
# the money invested (as a share of the fund), the amount traded each day,
# and the weights held each day.
def runBacktest(targets, dailyReturns, costBps = costBps, lag = executionLag):
    orders = targets.shift(lag + 1).to_numpy() # row t holds the orders filled at the close on day t
    returns = dailyReturns.to_numpy()
    numDays, numAssets = returns.shape
    held = np.zeros(numAssets) # share of the fund in each asset
    netReturn = np.zeros(numDays)
    grossReturn = np.zeros(numDays)
    traded = np.zeros(numDays)
    exposure = np.zeros(numDays)
    weightsHeld = np.zeros((numDays, numAssets))
    for t in range(numDays):
        exposure[t] = held.sum()
        weightsHeld[t, :] = held
        gross = held @ returns[t, :]
        # positions drift with the day's returns
        if 1 + gross > 0:
            held = held * (1 + returns[t, :]) / (1 + gross)
        # trade at the close
        order = orders[t, :]
        change = np.zeros(numAssets)
        known = ~np.isnan(order)
        change[known] = order[known] - held[known]
        cost = costBps / 10000 * np.abs(change).sum()
        # the cost comes out of the fund's value, so weights are measured
        # against what is left; the fund never borrows to pay it
        held = (held + change) / (1 - cost)
        if held.sum() > 1:
            held = held / held.sum()
        grossReturn[t] = gross
        netReturn[t] = (1 + gross) * (1 - cost) - 1
        traded[t] = np.abs(change).sum()
    index = dailyReturns.index
    return {"net": pd.Series(netReturn, index = index),
            "gross": pd.Series(grossReturn, index = index),
            "exposure": pd.Series(exposure, index = index),
            "traded": pd.Series(traded, index = index),
            "weights": pd.DataFrame(weightsHeld, index = index, columns = dailyReturns.columns)}

# The combined fund keeps half of its money in each strategy. Both halves
# pay their own trading costs.
def combineResults(first, second):
    return {key: (first[key] + second[key]) / 2 for key in first}

# Run the whole set of strategies and benchmarks for one set of settings.
def runAll(prices, dailyReturns, costBps = costBps, lag = executionLag,
        momWindow = momentumWindow, rebalDays = rebalanceDays, budget = riskBudget,
        mrWin = mrWindow, entryZ = mrEntryZ):
    results = {}
    results["Momentum"] = runBacktest(
        momentumTargets(prices, dailyReturns, momWindow, volWindow, rebalDays, budget),
        dailyReturns, costBps, lag)
    results["Mean Reversion"] = runBacktest(
        meanReversionTargets(prices, mrWin, entryZ), dailyReturns, costBps, lag)
    results["Combined"] = combineResults(results["Momentum"], results["Mean Reversion"])
    # The benchmarks are bought on day one, so the first trade happens on
    # the same day (there is no signal to wait for).
    results["SPY Buy-and-Hold"] = runBacktest(
        buyAndHoldTargets(prices, "SP500"), dailyReturns, costBps, lag = -1)
    results["Equal Weight"] = runBacktest(
        equalWeightTargets(prices), dailyReturns, costBps, lag = -1)
    results["BND Buy-and-Hold"] = runBacktest(
        buyAndHoldTargets(prices, "Bonds"), dailyReturns, costBps, lag = -1)
    return results

# ----------------------------------------------------------------------
# Performance measures
# ----------------------------------------------------------------------

# Measures for a series of daily returns. The risk free rate is taken as
# zero, so the Sharpe ratio is the average daily return divided by its
# standard deviation, scaled up to a year.
def performanceMetrics(dailyReturn, exposure = None, traded = None):
    numDays = len(dailyReturn)
    years = numDays / tradingDays
    equity = (1 + dailyReturn).cumprod()
    cagr = equity.iloc[-1] ** (1 / years) - 1
    annualVol = dailyReturn.std() * np.sqrt(tradingDays)
    sharpe = dailyReturn.mean() / dailyReturn.std() * np.sqrt(tradingDays)
    downsideDev = np.sqrt((np.minimum(dailyReturn, 0) ** 2).mean())
    sortino = dailyReturn.mean() / downsideDev * np.sqrt(tradingDays) if downsideDev > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    maxDrawdown = drawdown.min()
    calmar = cagr / abs(maxDrawdown) if maxDrawdown < 0 else np.nan
    metrics = {"Total Return": equity.iloc[-1] - 1, "CAGR": cagr, "Volatility": annualVol,
               "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": maxDrawdown,
               "Calmar": calmar}
    metrics["Avg Exposure"] = exposure.mean() if exposure is not None else np.nan
    metrics["Annual Turnover"] = traded.sum() / years if traded is not None else np.nan
    return metrics

# Beta, alpha, and correlation of a return series against a benchmark.
# Alpha is the annualized return left over after removing beta times the
# benchmark's return.
def alphaBeta(dailyReturn, benchmarkReturn):
    covariance = np.cov(dailyReturn, benchmarkReturn)
    beta = covariance[0, 1] / covariance[1, 1]
    alpha = (dailyReturn.mean() - beta * benchmarkReturn.mean()) * tradingDays
    correlation = np.corrcoef(dailyReturn, benchmarkReturn)[0, 1]
    return {"Beta": beta, "Alpha": alpha, "Correlation": correlation}

# Return for each calendar year.
def yearlyReturns(returnsFrame):
    years = returnsFrame.index.str[:4]
    return (1 + returnsFrame).groupby(years).prod() - 1

# Block bootstrap of the Sharpe ratio. Days are resampled in blocks (so
# that runs of good or bad days stay together), the same days for every
# series, and the Sharpe ratio is computed on each resample.
# Returns one row per resample and one column per series.
def bootstrapSharpe(returnsFrame, reps = bootstrapReps, block = bootstrapBlock,
        seed = bootstrapSeed):
    values = returnsFrame.to_numpy()
    numDays = values.shape[0]
    rng = np.random.default_rng(seed)
    numBlocks = int(np.ceil(numDays / block))
    starts = rng.integers(0, numDays, size = (reps, numBlocks))
    sharpes = np.zeros((reps, values.shape[1]))
    chunk = 100
    for first in range(0, reps, chunk):
        s = starts[first:first + chunk]
        index = (s[:, :, None] + np.arange(block)[None, None, :]) % numDays
        index = index.reshape(s.shape[0], -1)[:, :numDays]
        sample = values[index] # resamples, days, series
        sharpes[first:first + chunk, :] = sample.mean(axis = 1) / sample.std(axis = 1, ddof = 1) * np.sqrt(tradingDays)
    return pd.DataFrame(sharpes, columns = returnsFrame.columns)

# Metrics on the evaluation window (after the warm up) for a set of results.
def evaluate(results, startDay = warmUpDays):
    rows = {}
    for name in allNames:
        r = results[name]
        rows[name] = performanceMetrics(r["net"].iloc[startDay:],
            r["exposure"].iloc[startDay:], r["traded"].iloc[startDay:])
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

# colors: strategies in color, benchmarks in grays
lineColors = {"Momentum": "#2a78d6", "Mean Reversion": "#eb6834", "Combined": "#1baf7a",
              "SPY Buy-and-Hold": "#2b2b29", "Equal Weight": "#7d7c77", "BND Buy-and-Hold": "#b5b4ae"}
lineStyles = {"Momentum": "-", "Mean Reversion": "-", "Combined": "-",
              "SPY Buy-and-Hold": "-", "Equal Weight": "--", "BND Buy-and-Hold": ":"}
assetColors = {"Bonds": "#2a78d6", "Gold": "#eda100", "SP500": "#1baf7a", "Bitcoin": "#eb6834"}

def styleAxis(ax):
    ax.grid(color = "#e4e3df", linewidth = 0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9a9992")
    ax.tick_params(colors = "#52514e")

# Growth of $1 (log scale) on top and drawdowns underneath.
def plotEquityAndDrawdown(returnsFrame, filename):
    equity = (1 + returnsFrame).cumprod()
    drawdown = equity / equity.cummax() - 1
    dates = pd.to_datetime(returnsFrame.index)
    fig, axes = plt.subplots(2, 1, figsize = (10, 8.5), sharex = True,
        gridspec_kw = {"height_ratios": [2.2, 1]})
    for name in allNames:
        axes[0].plot(dates, equity[name], color = lineColors[name], linestyle = lineStyles[name],
            linewidth = 1.8, label = name)
        axes[1].plot(dates, drawdown[name], color = lineColors[name], linestyle = lineStyles[name],
            linewidth = 1.4)
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 2, 5, 10, 20, 50])
    axes[0].set_yticklabels(["$1", "$2", "$5", "$10", "$20", "$50"])
    axes[0].set_ylabel("Growth of $1 (log scale)", fontsize = 11)
    axes[1].set_ylabel("Drawdown from previous peak", fontsize = 11)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals = 0))
    # label the end of each line, spread out so the labels do not overlap
    ends = equity.iloc[-1].sort_values()
    minGap = 0.07
    placed = []
    for name, value in ends.items():
        y = np.log10(value)
        if placed and y < placed[-1] + minGap:
            y = placed[-1] + minGap
        placed.append(y)
        axes[0].annotate(name, (dates[-1], value), xytext = (dates[-1] + pd.Timedelta(days = 25), 10 ** y),
            fontsize = 9, color = "#2b2b29", va = "center",
            arrowprops = {"arrowstyle": "-", "color": "#b5b4ae", "linewidth": 0.6})
    axes[0].legend(loc = "upper left", frameon = False, fontsize = 9, ncol = 2)
    for ax in axes:
        styleAxis(ax)
    axes[1].set_xlim(dates[0], dates[-1] + pd.Timedelta(days = 330))
    fig.tight_layout()
    fig.savefig(filename, dpi = 150)
    plt.close(fig)

# What the momentum fund held over time, with the rest in cash.
def plotMomentumWeights(weights, filename):
    dates = pd.to_datetime(weights.index)
    cash = (1 - weights.sum(axis = 1)).clip(lower = 0)
    fig, ax = plt.subplots(figsize = (10, 4.8))
    layers = [weights[a].to_numpy() for a in weights.columns] + [cash.to_numpy()]
    colors = [assetColors[a] for a in weights.columns] + ["#dcdbd6"]
    labels = list(weights.columns) + ["Cash"]
    ax.stackplot(dates, layers, colors = colors, labels = labels, linewidth = 0.4,
        edgecolor = "#fcfcfb")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals = 0))
    ax.set_ylabel("Share of the momentum fund", fontsize = 11)
    ax.legend(loc = "upper center", ncol = 5, frameon = False, fontsize = 9,
        bbox_to_anchor = (0.5, 1.12))
    styleAxis(ax)
    fig.tight_layout()
    fig.savefig(filename, dpi = 150)
    plt.close(fig)

# Calendar year returns for the three strategies and the S&P 500.
def plotYearlyReturns(yearly, filename):
    names = ["Momentum", "Mean Reversion", "Combined", "SPY Buy-and-Hold"]
    fig, ax = plt.subplots(figsize = (10, 4.8))
    width = 0.2
    x = np.arange(len(yearly))
    for k, name in enumerate(names):
        ax.bar(x + (k - 1.5) * width, yearly[name], width = width * 0.9,
            color = lineColors[name], label = name)
    ax.axhline(0, color = "#52514e", linewidth = 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals = 0))
    ax.set_ylabel("Return for the year", fontsize = 11)
    ax.legend(loc = "upper left", frameon = False, fontsize = 9, ncol = 2)
    styleAxis(ax)
    fig.tight_layout()
    fig.savefig(filename, dpi = 150)
    plt.close(fig)

# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pct = lambda v: f"{v * 100:.1f}%"

    prices, dailyReturns = loadPrices()
    startDate = dailyReturns.index[warmUpDays]
    print("\nData: daily closes for", ", ".join(myAssetNames))
    print("Dates with all four prices:", dailyReturns.index[0], "to", dailyReturns.index[-1],
        "(" + str(len(dailyReturns)) + " daily returns)")
    print("The first", warmUpDays, "days are used to warm up the signals.")
    print("Backtest results cover", startDate, "to", dailyReturns.index[-1],
        "(" + str(len(dailyReturns) - warmUpDays) + " days)")

    print("\nSettings")
    print("  Momentum:", momentumWindow, "day regression, volatility from", volWindow,
        "days, rebalance every", rebalanceDays, "days, risk budget", riskBudget, "per asset")
    print("  Mean reversion:", mrWindow, "day average, buy at z below", -mrEntryZ,
        "and sell at z of", mrExitZ)
    print("  Trading cost:", costBps, "basis points per amount traded")
    print("  Orders are filled at the close", executionLag + 1, "trading day(s) after the signal day")
    print("  Risk free rate: 0")

    # ---- main backtest
    results = runAll(prices, dailyReturns)
    window = dailyReturns.index[warmUpDays:]
    netReturns = pd.DataFrame({name: results[name]["net"].loc[window] for name in allNames})
    metrics = evaluate(results)

    print("\n=== Backtest results, after trading costs ===")
    shown = metrics.astype(object)
    for row in ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Avg Exposure"]:
        shown.loc[row] = shown.loc[row].map(pct)
    for row in ["Sharpe", "Sortino", "Calmar", "Annual Turnover"]:
        shown.loc[row] = shown.loc[row].map(lambda v: f"{v:.2f}")
    print(shown.to_string())

    print("\n=== Against the S&P 500 (SPY Buy-and-Hold) ===")
    relative = pd.DataFrame({name: alphaBeta(netReturns[name], netReturns["SPY Buy-and-Hold"])
        for name in allNames if name != "SPY Buy-and-Hold"})
    shownRel = relative.astype(object)
    shownRel.loc["Alpha"] = shownRel.loc["Alpha"].map(pct)
    shownRel.loc["Beta"] = shownRel.loc["Beta"].map(lambda v: f"{v:.2f}")
    shownRel.loc["Correlation"] = shownRel.loc["Correlation"].map(lambda v: f"{v:.2f}")
    print(shownRel.to_string())

    # ---- calendar years
    yearly = yearlyReturns(netReturns)
    print("\n=== Return by calendar year ===")
    print(yearly.map(pct).to_string())

    # ---- how sure are we about the Sharpe ratios
    boot = bootstrapSharpe(netReturns)
    print("\n=== Block bootstrap of the Sharpe ratio (" + str(bootstrapReps) + " resamples of",
        bootstrapBlock, "day blocks) ===")
    bootRows = []
    for name in allNames:
        low, high = boot[name].quantile([0.025, 0.975])
        bootRows.append({"Series": name, "Sharpe": metrics.loc["Sharpe", name],
            "CI Low": low, "CI High": high, "Difference vs SPY": np.nan,
            "Diff CI Low": np.nan, "Diff CI High": np.nan, "Share Below SPY": np.nan})
        if name != "SPY Buy-and-Hold":
            diff = boot[name] - boot["SPY Buy-and-Hold"]
            dLow, dHigh = diff.quantile([0.025, 0.975])
            bootRows[-1].update({"Difference vs SPY": metrics.loc["Sharpe", name] - metrics.loc["Sharpe", "SPY Buy-and-Hold"],
                "Diff CI Low": dLow, "Diff CI High": dHigh,
                "Share Below SPY": (diff <= 0).mean()})
    bootFrame = pd.DataFrame(bootRows).set_index("Series")
    print(bootFrame.round(2).to_string())

    # ---- sensitivity checks
    print("\n=== Sensitivity checks (Sharpe ratio unless noted) ===")
    sensitivityRows = []

    def addRow(study, setting, results2):
        m = evaluate(results2)
        for name in strategyNames:
            sensitivityRows.append({"Study": study, "Setting": setting, "Series": name,
                "Sharpe": m.loc["Sharpe", name], "CAGR": m.loc["CAGR", name],
                "Max Drawdown": m.loc["Max Drawdown", name]})

    # momentum lookback and rebalance frequency
    print("\nMomentum: regression window (days) and rebalance interval (days)")
    momTable = pd.DataFrame(index = [40, 60, 90, 120, 180, 250], columns = [5, 21], dtype = float)
    for win in momTable.index:
        for rebal in momTable.columns:
            r = runBacktest(momentumTargets(prices, dailyReturns, win, volWindow, rebal, riskBudget),
                dailyReturns)
            m = performanceMetrics(r["net"].iloc[warmUpDays:])
            momTable.loc[win, rebal] = m["Sharpe"]
            sensitivityRows.append({"Study": "Momentum window and rebalance",
                "Setting": str(win) + " days, every " + str(rebal), "Series": "Momentum",
                "Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "Max Drawdown": m["Max Drawdown"]})
    momTable.index.name = "window"
    momTable.columns = ["rebalance " + str(c) for c in momTable.columns]
    print(momTable.round(2).to_string())

    # momentum risk budget
    print("\nMomentum: risk budget per asset (annual volatility)")
    budgetRow = {}
    for budget in [0.03, 0.05, 0.08, 0.12]:
        r = runBacktest(momentumTargets(prices, dailyReturns, momentumWindow, volWindow,
            rebalanceDays, budget), dailyReturns)
        m = performanceMetrics(r["net"].iloc[warmUpDays:])
        budgetRow[budget] = {"Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "Max Drawdown": m["Max Drawdown"]}
        sensitivityRows.append({"Study": "Momentum risk budget", "Setting": str(budget),
            "Series": "Momentum", "Sharpe": m["Sharpe"], "CAGR": m["CAGR"],
            "Max Drawdown": m["Max Drawdown"]})
    budgetFrame = pd.DataFrame(budgetRow)
    print(budgetFrame.round(3).to_string())

    # mean reversion window and entry threshold
    print("\nMean reversion: moving average window (days) and entry z score")
    mrTable = pd.DataFrame(index = [10, 20, 40], columns = [1.0, 1.5, 2.0], dtype = float)
    for win in mrTable.index:
        for entry in mrTable.columns:
            r = runBacktest(meanReversionTargets(prices, win, entry), dailyReturns)
            m = performanceMetrics(r["net"].iloc[warmUpDays:])
            mrTable.loc[win, entry] = m["Sharpe"]
            sensitivityRows.append({"Study": "Mean reversion window and entry",
                "Setting": str(win) + " days, z " + str(entry), "Series": "Mean Reversion",
                "Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "Max Drawdown": m["Max Drawdown"]})
    mrTable.index.name = "window"
    mrTable.columns = ["entry z " + str(c) for c in mrTable.columns]
    print(mrTable.round(2).to_string())

    # trading costs
    print("\nTrading cost (basis points per amount traded)")
    costTable = pd.DataFrame(index = strategyNames + ["SPY Buy-and-Hold"],
        columns = [0, 5, 10, 25, 50], dtype = float)
    for bps in costTable.columns:
        r2 = runAll(prices, dailyReturns, costBps = bps)
        addRow("Trading cost", str(bps) + " bps", r2)
        m = evaluate(r2)
        for name in costTable.index:
            costTable.loc[name, bps] = m.loc["Sharpe", name]
    costTable.columns = [str(c) + " bps" for c in costTable.columns]
    print(costTable.round(2).to_string())

    # delay between the signal and the trade
    print("\nExtra days between the signal day and the trade (0 = traded at the next close, as in the main results)")
    lagTable = pd.DataFrame(index = strategyNames, columns = [-1, 0, 1, 2], dtype = float)
    for lag in lagTable.columns:
        r2 = runAll(prices, dailyReturns, lag = lag)
        addRow("Execution lag", str(lag), r2)
        m = evaluate(r2)
        for name in lagTable.index:
            lagTable.loc[name, lag] = m.loc["Sharpe", name]
    lagTable.columns = ["same close" if c == -1 else str(c) + " extra" for c in lagTable.columns]
    print(lagTable.round(2).to_string())
    print("(\"same close\" trades at the close that makes the signal, which is not realistic")
    print(" and is shown only to see how much the one day wait matters.)")

    # ---- two halves of the sample
    print("\nTwo halves of the backtest period (Sharpe ratio)")
    half = len(window) // 2
    halves = {}
    for label, part in [("First half: " + window[0] + " to " + window[half - 1], netReturns.iloc[:half]),
                        ("Second half: " + window[half] + " to " + window[-1], netReturns.iloc[half:])]:
        halves[label] = {name: performanceMetrics(part[name])["Sharpe"] for name in allNames}
    print(pd.DataFrame(halves).round(2).to_string())

    # ---- save outputs
    out = lambda name: os.path.join(baseDir, name)
    netReturns.to_csv(out("pa3-daily-returns.csv"), index_label = "Date")
    metrics.to_csv(out("pa3-metrics.csv"))
    relative.to_csv(out("pa3-vs-spy.csv"))
    yearly.to_csv(out("pa3-yearly-returns.csv"), index_label = "Year")
    bootFrame.to_csv(out("pa3-bootstrap.csv"))
    pd.DataFrame(sensitivityRows).to_csv(out("pa3-sensitivity.csv"), index = False)
    results["Momentum"]["weights"].loc[window].to_csv(out("pa3-momentum-weights.csv"), index_label = "Date")
    plotEquityAndDrawdown(netReturns, out("pa3-figure-equity.png"))
    plotMomentumWeights(results["Momentum"]["weights"].loc[window], out("pa3-figure-weights.png"))
    plotYearlyReturns(yearly, out("pa3-figure-yearly.png"))

    print("\nRun Complete")

if __name__ == "__main__":
    main()
