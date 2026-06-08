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
