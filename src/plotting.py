import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np
import pandas as pd



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
    plt.savefig("outputs/rolling_vol.png")
    plt.close()
