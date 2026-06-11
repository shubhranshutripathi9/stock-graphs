import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np
import pandas as pd
from returns import compute_log_returns


def plot_qq(log_returns, results, ticker):
    os.makedirs(f"outputs/qq_{ticker}", exist_ok=True)
    # results from fit_distributions for sparams of probplot

    # mapping distributions to their parametes
    dist_map = {
        "norm": ("mu", "sigma"),
        "t": ("df",),
        "skewnorm": ("a",),
        "gamma": ("a",),
        "genlogistic": ("c",),
    }
    for dist_name, param_keys in dist_map.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        sparams = tuple(results[dist_name][k] for k in param_keys)
        stats.probplot(log_returns, sparams=sparams, dist=dist_name, fit=True, plot=ax)
        ax.set_title(f"QQ plot- {ticker}-{dist_name}")
        plt.savefig(f"outputs/qq_{ticker}/{ticker}-{dist_name}.png")
        plt.close()
        print(f"Saved: outputs/{ticker} {dist_name}.png")
        print(f"{dist_name}: sparams={sparams}")


def plot_rolling_vol(returns_dict):
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker, returns in returns_dict.items():
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        ax.plot(rolling_vol, label=ticker)

    ax.set_title("30 Day rolling annualized volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    plt.savefig(f"outputs/rolling_vol.png")
    plt.close()


if __name__ == "__main__":
    # essentially returns.py had to be rebuilt here for the sake of testing
    data = pd.read_csv(
        r"F:\stocks_returns_analyzer\data\AAPL.csv",
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
    )
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    test_params = {
        "norm": dict(zip(("mu", "sigma"), stats.norm.fit(log_returns))),
        "t": dict(zip(("df", "loc", "scale"), stats.t.fit(log_returns))),
        "skewnorm": dict(zip(("a", "loc", "scale"), stats.skewnorm.fit(log_returns))),
        "gamma": dict(zip(("a", "loc", "scale"), stats.gamma.fit(log_returns))),
        "genlogistic": dict(
            zip(("c", "loc", "scale"), stats.genlogistic.fit(log_returns))
        ),
    }

    plot_qq(log_returns, test_params, "AAPL")

    tickers = ["SPY", "RELIANCE_NS", "TCS_NS", "INFY_NS", "HDFCBANK_NS"]
    returns_dict = {}
    for ticker in tickers:
        returns_dict[ticker] = compute_log_returns(f"data/{ticker}.csv")

    plot_rolling_vol(returns_dict)
