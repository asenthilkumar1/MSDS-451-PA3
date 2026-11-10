# MSDS-451-PA3

MSDS 451 Financial Engineering, Programming Assignment 3.

An automated trading program with a backtest. It trades the four assets from PA2 (BND bonds, GLD gold, SPY S&P 500, BTC-USD bitcoin) with two rule-based strategies, momentum and mean reversion, tests them on ten years of daily data, and compares them with buy-and-hold using the Sharpe ratio and other measures. There is no machine learning or regime detection, which the assignment treats as optional.

## The strategies

**Momentum.** Every 5 trading days, fit a trend line to the last 90 days of each asset's log prices. The score is the annualized slope times R squared. Hold an asset only when its score is positive, sized so it adds about 5% annual volatility to the fund (5% divided by its volatility over the last 60 days). Positions are scaled down if they add up to more than 100%, and the rest stays in cash. The score follows the method in Clenow (2019), *Trading Evolved*. The windows, the use of standard deviation instead of average true range for sizing, the 5% risk budget, and weekly rebalancing are my own choices and are not taken from the book.

**Mean reversion.** For each asset, compute the z score of the price against its 20-day average. Buy below -1.5, sell at 0. Each asset gets a quarter of the fund, which stays in cash while the asset is not held. This is a simple z score rule, not taken from the book.

**Combined.** Half the fund in each strategy.

**Benchmarks.** SPY buy-and-hold (main comparison), the four assets in equal weights rebalanced monthly, and BND buy-and-hold.

**Backtest rules.**
- An order decided at the close on day t is filled at the close on day t + 1.
- 10 basis points cost on every amount traded.
- No shorting or borrowing. Cash earns nothing and the risk-free rate is zero.
- Settings were fixed before running any results and not tuned. The sensitivity checks show other settings.
- The first 260 days only warm up the signals. Everything is scored from January 15, 2016 to December 31, 2025 (2,505 trading days).

## Repository contents

| File | Purpose |
|---|---|
| `pa3.py` | Main program: loads data, runs strategies and benchmarks, computes measures, saves outputs |
| `test_pa3.py` | 20 unit tests |
| `getdata_yfinance.py` | Downloads the price history (called by `pa3.py` if the CSVs are missing) |
| `msds_getdata_yfinance_*.csv` | Daily prices for the four assets, 2015 to 2025, same files as PA2 |
| `pa3-report.pdf` | Research report |
| `pa3-output.txt` | Console output from a full run |
| `pa3-figure-equity.png`, `pa3-figure-weights.png`, `pa3-figure-yearly.png` | Growth and drawdowns, momentum holdings over time, calendar-year returns |
| `pa3-daily-returns.csv`, `pa3-metrics.csv`, `pa3-vs-spy.csv`, `pa3-yearly-returns.csv`, `pa3-bootstrap.csv`, `pa3-sensitivity.csv`, `pa3-momentum-weights.csv` | Result tables |
| `requirements.txt` | Python packages |

## Setup and running

Needs Python 3 (developed on 3.14) and `numpy`, `pandas`, `matplotlib`, `yfinance`. The data files are included, so the program runs offline.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python pa3.py | tee pa3-output.txt
```

It takes about 10 seconds and opens no windows. It prints the settings, results, comparison with SPY, yearly returns, bootstrap intervals, sensitivity checks, and ends with `Run Complete`. It also rewrites the figures and CSV files. If the data files are missing it runs `getdata_yfinance.py` first. On Windows, leave out `| tee pa3-output.txt`.

## Testing

```bash
python -m unittest -v test_pa3
```

All 20 tests should pass. They cover:
- Data alignment, no missing values, returns matching prices.
- The momentum score on a perfect uptrend and downtrend, valid weights (never negative, never above 100%), and holding nothing when everything falls.
- The mean reversion rule buying a drop and selling the recovery.
- No lookahead: signals from the first 1,500 days match the first 1,500 days of signals from all the data.
- The backtest matching a separately written one that tracks dollars instead of fund shares (momentum, mean reversion, equal weights).
- Costs, order delay, and holdings staying between 0% and 100%.
- Sharpe, drawdown, alpha, beta, and yearly returns against hand calculations, and a reproducible bootstrap.

Other checks:
- `python pa3.py` should finish with exit code 0 (check with `echo $?`, without `tee`) and print `Run Complete`.
- A new run should match `pa3-output.txt`. The only random step is the bootstrap, with a fixed seed (3333).
- The `SPY Buy-and-Hold` column in `pa3-daily-returns.csv` should grow by the same amount as SPY's closing price in `msds_getdata_yfinance_spy.csv` over the same dates.

## Results

January 15, 2016 to December 31, 2025, after trading costs.

| | Momentum | Mean Reversion | Combined | SPY | Equal Weight | BND |
|---|---|---|---|---|---|---|
| CAGR | 9.4% | 2.6% | 6.1% | 15.5% | 29.7% | 2.0% |
| Volatility | 7.0% | 10.5% | 7.5% | 18.0% | 19.6% | 5.5% |
| Sharpe ratio | 1.32 | 0.30 | 0.82 | 0.89 | 1.42 | 0.39 |
| Sortino ratio | 1.84 | 0.41 | 1.13 | 1.26 | 2.13 | 0.54 |
| Max drawdown | -11.6% | -24.3% | -14.3% | -33.7% | -35.2% | -18.6% |
| Calmar ratio | 0.81 | 0.11 | 0.42 | 0.46 | 0.84 | 0.11 |
| Average exposure | 80% | 25% | 53% | 100% | 100% | 100% |
| Annual turnover | 4.9x | 10.4x | 7.7x | 0 | 0.8x | 0 |

- Momentum beat SPY on Sharpe ratio (1.32 vs 0.89) with about a third of the volatility and a much smaller drawdown, but earned less (9.4% vs 15.5% a year). Beta to SPY is 0.17 and alpha is 6.5% a year. In 2022 it lost 4.9% against SPY's 18.2%.
- The Sharpe gap is not statistically clear. The 95% bootstrap interval for momentum minus SPY is -0.16 to 0.99.
- Mean reversion did not work on these assets (Sharpe 0.30), mostly from buying dips that kept falling in 2018 and 2022. It also trades the most, so costs hurt it.
- Equal-weight buy-and-hold had the best Sharpe ratio (1.42) and return. Most of the return comes from bitcoin (a rough check without it gives about 11% a year instead of about 29%), though the Sharpe ratio stays around 1.3 even without it. Neither strategy beat it, so the rules cut risk but did not add return over holding the assets.

Sensitivity (full tables in `pa3-output.txt` and `pa3-sensitivity.csv`):
- Momentum's Sharpe ratio stays between 1.16 and 1.41 across regression windows of 40 to 250 days and weekly or monthly trading, always above SPY's 0.89.
- Mean reversion stays between 0.20 and 0.45 under every window and entry level tested.
- At 50 basis points of cost, momentum's Sharpe falls to 1.03 and mean reversion's to -0.10.
- Momentum's Sharpe was 1.57 in the first half of the test and 1.06 in the second, above SPY's in both.

## Assumptions and limits

- Bitcoin was picked in PA2 for contrast, and ten years of bitcoin history is a very good ten years, so the equal-weight result should be read with that in mind.
- Cash earns zero, which slightly understates momentum because it held cash while rates were higher in 2023 to 2025.
- Only days when all four assets have a price are used. Prices are Yahoo Finance adjusted closes. Costs are a flat 10 basis points, with no taxes or price impact.
- This is one historical path with no separate test period. The bootstrap and sensitivity checks show how much luck could matter but cannot remove it.

## Reuse from PA2

The data files, `getdata_yfinance.py`, and `loadPrices` in `pa3.py` (reading the CSVs, aligning dates, computing returns) come from PA2. The strategies, backtest, measures, plots, and tests are new for PA3.

## Use of AI assistants
I used Claude to generate the remaining code needed for this programming assignment. I provided it with the previous programming assignment's code and asked it to fill any blanks needed to complete this assignment. I made the decisions, reviewed the work, and wrote the report. I chose to keep the same four assets from PA2, decided what the project should include and leave out. Claude also wrote up this readme for ease of onboarding for newcomers to this repository.
