"""
data_generator.py
-----------------
Generare seturi de date sintetice pentru benchmarking K-Means.
Suporta atat date sintetice (make_blobs) cat si incarcarea din CSV (Kaggle).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from pathlib import Path
import os


# ── Scenariile de testare recomandate ─────────────────────────────────────────
BENCHMARK_SCENARIOS = [
    {"name": "Small",    "n_samples": 10_000,    "n_features": 2,   "k": 5,  "desc": "Debugging rapid"},
    {"name": "Medium",   "n_samples": 100_000,   "n_features": 16,  "k": 10, "desc": "Date moderate"},
    {"name": "Large",    "n_samples": 1_000_000, "n_features": 64,  "k": 20, "desc": "Limita CPU"},
    {"name": "High-Dim", "n_samples": 100_000,   "n_features": 512, "k": 10, "desc": "Bandwidth memorie"},
]


def generate_synthetic(n_samples: int,
                       n_features: int,
                       k: int,
                       random_state: int = 42) -> np.ndarray:
    """
    Genereaza un dataset sintetic folosind sklearn.datasets.make_blobs.

    Parametri
    ---------
    n_samples    : numarul de puncte
    n_features   : numarul de dimensiuni / features
    k            : numarul de clustere (centre Gaussiene)
    random_state : seed pentru reproductibilitate

    Returneaza
    ----------
    X : np.ndarray de forma (n_samples, n_features), dtype float32
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
    Incarca un dataset din fisier CSV (ex: dataset Kaggle).

    Parametri
    ---------
    filepath        : calea catre fisierul CSV
    feature_columns : lista de coloane numerice de folosit (None = toate)
    max_rows        : limita de randuri (None = toate)

    Returneaza
    ----------
    X : np.ndarray float32
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fisierul nu exista: {filepath}")

    df = pd.read_csv(filepath, nrows=max_rows)

    if feature_columns:
        df = df[feature_columns]

    # Pastram doar coloanele numerice
    df = df.select_dtypes(include=[np.number]).dropna()

    print(f"[DataGenerator] Incarcat: {df.shape[0]} randuri x {df.shape[1]} coloane din {path.name}")
    return df.values.astype(np.float32)


def save_to_csv(X: np.ndarray, output_path: str, scenario_name: str = "data"):
    """
    Salveaza un dataset numpy ca fisier CSV (util pentru a fi citit de C++).

    Parametri
    ---------
    X            : array (n_samples, n_features)
    output_path  : director de output
    scenario_name: numele fisierului
    """
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{scenario_name}.csv")
    pd.DataFrame(X).to_csv(filepath, index=False, header=False)
    print(f"[DataGenerator] Salvat: {filepath}  ({X.shape[0]} x {X.shape[1]})")
    return filepath


def generate_all_scenarios(output_dir: str = "data/generated",
                           random_state: int = 42,
                           save: bool = False) -> dict:
    """
    Genereaza toate scenariile definite in BENCHMARK_SCENARIOS.

    Returneaza
    ----------
    dict { scenario_name -> np.ndarray }
    """
    datasets = {}
    for scenario in BENCHMARK_SCENARIOS:
        name = scenario["name"]
        print(f"[DataGenerator] Generare '{name}' — "
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

    print("\nSumar dataset-uri generate:")
    for name, X in datasets.items():
        mb = X.nbytes / (1024 ** 2)
        print(f"  {name:10s}: shape={X.shape}, memorie={mb:.1f} MB, dtype={X.dtype}")
