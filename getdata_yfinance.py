#!/usr/bin/env python
# coding: utf-8

# Get Data from Yahoo! Finance for the four assets used in PA2 and PA3
# Adapted from the getdata_yfinance.py script used in PA1

# import Python packages
import pandas as pd

# package for downloading historical market data
import yfinance as yf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Settings for all data gathering
# Bitcoin only has price history from September 2014, so the
# sample for all four assets starts in January 2015.
start_date = '2015-01-01'
end_date = '2026-01-01'  # end date is not included
startfile = "msds_getdata_yfinance_"

# BND = bond fund, GLD = gold, SPY = S&P 500 index fund, BTC-USD = bitcoin
symbols = ['BND', 'GLD', 'SPY', 'BTC-USD']

for symbol in symbols:
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(start = start_date, end = end_date)
    print(symbol)
    print(historical_data.head())
    print("rows:", len(historical_data))
    historical_data.to_csv(startfile + symbol.lower() + ".csv")
