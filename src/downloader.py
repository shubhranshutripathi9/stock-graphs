import yfinance as yf
import os
from datetime import datetime
import pandas as pd


def download_stocks():

    end = datetime.now().strftime("%Y-%m-%d")
    tickers = [
        "SPY",
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "AAPL",
        "NVDA",
        "INTC",
        "F",
        "NOK",
    ]
    os.makedirs("data", exist_ok=True)

    for ticker in tickers:
        print(f"downloading {ticker}")
        try:
            df = yf.download(ticker, start="2018-01-01", end=end)
            if df.empty:
                print(f"No stock data retrieved for {ticker}")
                continue
            filepath = f"data/{ticker.replace('.', '_')}.csv"
            df.to_csv(filepath)
            print(f"saved {ticker} data to {filepath}, shape is {df.shape}")
        except Exception as e:
            print(f"Error in {ticker} as {e}")


if __name__ == "__main__":
    download_stocks()
