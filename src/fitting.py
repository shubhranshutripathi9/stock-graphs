from scipy import stats
import numpy as np


def fit_distributions(log_returns):
    results = {}

    # fit to normal
    mu, sigma = stats.norm.fit(log_returns)
    # mu and sigma are mean and sd
    results["norm"] = {"mu": mu, "sigma": sigma}

    # fit to students t
    df, loc, scale = stats.t.fit(log_returns)
    # df is deg of freedom, loc means location(axis shift) and scale is the scale parameter
    results["t"] = {"df": df, "loc": loc, "scale": scale}

    # fit to skew-normal
    a, loc, scale = stats.skewnorm.fit(log_returns)
    # a is shape like alpha in chi^2, rest same
    results["skewnorm"] = {"a": a, "loc": loc, "scale": scale}

    # fit to gamma
    a, loc, scale = stats.gamma.fit(log_returns)
    results["gamma"] = {"a": a, "loc": loc, "scale": scale}

    # fit to generalised logistic
    c, loc, scale = stats.genlogistic.fit(log_returns)
    results["genlogistic"] = {"c": c, "loc": loc, "scale": scale}

    # all adjusted, the parameters are returned as a dictionary
    return results


if __name__ == "__main__":
    # test data, just to verify if code works; irrelevant otherwise
    print(fit_distributions(data))
