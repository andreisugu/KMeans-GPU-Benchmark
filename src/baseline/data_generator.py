"""
data_generator.py
-----------------
Generation of synthetic datasets for K-Means benchmarking.
Supports both synthetic data (make_blobs) and loading from CSV (Kaggle).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from pathlib import Path
import os


# ── Recommended test scenarios ──────────────────────────────────
BENCHMARK_SCENARIOS = [
    {"name": "Small",    "n_samples": 10_000,    "n_features": 2,   "k": 5,  "desc": "Quick debugging"},
    {"name": "Medium",   "n_samples": 100_000,   "n_features": 16,  "k": 10, "desc": "Moderate data"},
    {"name": "Large",    "n_samples": 1_000_000, "n_features": 64,  "k": 20, "desc": "CPU limit"},
    {"name": "High-Dim", "n_samples": 100_000,   "n_features": 512, "k": 10, "desc": "Memory bandwidth"},
]


def generate_synthetic(n_samples: int,
                       n_features: int,
                       k: int,
                       random_state: int = 42) -> np.ndarray:
    """
    Generates a synthetic dataset using sklearn.datasets.make_blobs.

    Parameters
    ----------
    n_samples    : number of points
    n_features   : number of dimensions / features
    k            : number of clusters (Gaussian centers)
    random_state : seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float32
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=k,
        cluster_std=1.0,
        random_state=random_state,
    )
    return X.astype(np.float32)


def load_from_csv(filepath: str,
                  feature_columns: list = None,
                  max_rows: int = None) -> np.ndarray:
    """
    Loads a dataset from CSV file (e.g. Kaggle dataset).

    Parameters
    ----------
    filepath        : path to CSV file
    feature_columns : list of numeric columns to use (None = all)
    max_rows        : row limit (None = all)

    Returns
    -------
    X : np.ndarray float32
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    df = pd.read_csv(filepath, nrows=max_rows)

    if feature_columns:
        df = df[feature_columns]

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).dropna()

    print(f"[DataGenerator] Loaded: {df.shape[0]} rows x {df.shape[1]} columns from {path.name}")
    return df.values.astype(np.float32)


def save_to_csv(X: np.ndarray, output_path: str, scenario_name: str = "data"):
    """
    Saves a numpy dataset as CSV file (useful to be read by C++).

    Parameters
    ----------
    X            : array (n_samples, n_features)
    output_path  : output directory
    scenario_name: file name
    """
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{scenario_name}.csv")
    pd.DataFrame(X).to_csv(filepath, index=False, header=False)
    print(f"[DataGenerator] Saved: {filepath}  ({X.shape[0]} x {X.shape[1]})")
    return filepath


def generate_all_scenarios(output_dir: str = "data/generated",
                           random_state: int = 42,
                           save: bool = False) -> dict:
    """
    Generates all scenarios defined in BENCHMARK_SCENARIOS.

    Returns
    -------
    dict { scenario_name -> np.ndarray }
    """
    datasets = {}
    for scenario in BENCHMARK_SCENARIOS:
        name = scenario["name"]
        print(f"[DataGenerator] Generating '{name}' — "
              f"N={scenario['n_samples']:,}, D={scenario['n_features']}, K={scenario['k']}")

        X = generate_synthetic(
            n_samples=scenario["n_samples"],
            n_features=scenario["n_features"],
            k=scenario["k"],
            random_state=random_state,
        )
        datasets[name] = X

        if save:
            save_to_csv(X, output_dir, name)

    return datasets


# ── Rulare directa ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Data Generator — K-Means Benchmark ===\n")
    datasets = generate_all_scenarios(output_dir="../../data/generated", save=True)

    print("\nSummary of generated datasets:")
    for name, X in datasets.items():
        mb = X.nbytes / (1024 ** 2)
        print(f"  {name:10s}: shape={X.shape}, memory={mb:.1f} MB, dtype={X.dtype}")
