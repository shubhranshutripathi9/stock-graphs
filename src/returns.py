import pandas as pd
import numpy as np


def compute_log_returns(filepath, verbose=False):
    df = pd.read_csv(
        filepath,
        header=[0],
        skiprows=[1, 2],
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )
    df.index.name = "Date"

    close = pd.to_numeric(df["Close"], errors="coerce")
    log_returns = np.log(close / close.shift(1)).dropna()

    if verbose:
        print("--- Dataset Summary ---")
        print(f"Total Rows Processed : {log_returns.shape[0]}")
        print(f"Log Return Mean      : {log_returns.mean():.6f}")
        print(f"Log Return Std Dev   : {log_returns.std():.6f}")

    return log_returns


if __name__ == "__main__":
    returns = compute_log_returns("data/HDFCBANK_NS.csv", verbose=True)
    print("returns for 1st 5 entries")
    print(returns.head())
