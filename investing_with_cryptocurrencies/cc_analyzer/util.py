# util functions
import pandas as pd
import numpy as np
import time
import yfinance as yf

## getting data
def get_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        print(ticker)
        data[ticker] = yf.download(ticker, start_date, end_date)
        time.sleep(1)
    return data

def compute_ret(raw_data_dict):
    res = pd.DataFrame()
    tickers = raw_data_dict.keys()
    for ticker in tickers:
        adj_close = raw_data_dict[ticker]["Adj Close"]
        #adj_close = adj_close[adj_close.index.dayofweek<5] #uncomment if ignore weekends
        log_ret = np.log(adj_close/adj_close.shift(1))
        log_ret.name = ticker
        res = pd.concat([res, log_ret], axis=1)
    res.dropna(how="all", inplace=True)
    res.fillna(0, inplace=True)
    return res