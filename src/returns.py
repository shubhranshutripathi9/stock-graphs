import pandas as pd
import numpy as np


def compute_log_returns(filepath, verbose=False):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")

    close = pd.to_numeric(df["Close"], errors="coerce")
    log_returns = np.log(close / close.shift(1))
    log_returns = log_returns.dropna()

    if verbose:
        print(f"shape = {log_returns.shape}")
        print(f"mean = {log_returns.mean():.6f}")
        print(f"Standard Deviation = {log_returns.std():.6f}")
    return log_returns


if __name__ == "__main__":
    returns = compute_log_returns("data/HDFCBANK_NS.csv")
    print(returns.head())
