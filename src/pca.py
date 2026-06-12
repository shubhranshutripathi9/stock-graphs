import numpy as np
import pandas as pd


def run_pca(returns_dict):
    df = pd.DataFrame(returns_dict).dropna()
    cov_matrix = np.cov(df.values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total

    n_components = np.argmax(cumulative >= 0.8) + 1
    print(f"required {n_components} for 80% variance")

    return eigenvalues, eigenvectors, cumulative
