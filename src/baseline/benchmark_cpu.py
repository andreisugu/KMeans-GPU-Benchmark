"""
benchmark_cpu.py
----------------
Scriptul principal de benchmarking pentru varianta CPU (scikit-learn).
Itereaza prin toate scenariile, salveaza rezultatele in CSV si afiseaza un sumar.

Rulare:
    python benchmark_cpu.py
    python benchmark_cpu.py --output ../../results/results_cpu.csv
    python benchmark_cpu.py --skip-large   # sari Large + High-Dim pentru test rapid
"""

import argparse
import os
import time
import pandas as pd
import numpy as np

from data_generator import generate_synthetic, BENCHMARK_SCENARIOS
from kmeans_cpu import run_kmeans_cpu, KMeansResult


# ── Configuratie globala ───────────────────────────────────────────────────────
RANDOM_STATE   = 42
INIT_METHOD    = "k-means++"   # schimba in "random" daca CUDA nu are k-means++
MAX_ITER       = 300
TOL            = 1e-4
N_INIT         = 1             # 1 rulare per scenarui (consistent cu CUDA)


def run_benchmark(scenarios: list,
                  random_state: int = RANDOM_STATE) -> list[KMeansResult]:
    """
    Ruleaza K-Means CPU pe lista de scenarii data.

    Returneaza lista de KMeansResult.
    """
    results = []

    print("=" * 70)
    print(" K-Means CPU Benchmark — scikit-learn")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        name       = scenario["name"]
        n_samples  = scenario["n_samples"]
        n_features = scenario["n_features"]
        k          = scenario["k"]
        desc       = scenario.get("desc", "")

        print(f"\n[{i}/{len(scenarios)}] Scenariu: {name} ({desc})")
        print(f"  N={n_samples:,}  D={n_features}  K={k}")

        # Generare date (nu este inclus in timp)
        print("  Generare date...", end=" ", flush=True)
        t0 = time.perf_counter()
        X = generate_synthetic(n_samples, n_features, k, random_state)
        print(f"gata in {time.perf_counter()-t0:.2f}s  ({X.nbytes/1e6:.0f} MB)")

        # Rulare K-Means
        print("  Rulare K-Means CPU...", end=" ", flush=True)
        result = run_kmeans_cpu(
            X,
            k=k,
            init=INIT_METHOD,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=N_INIT,
            random_state=random_state,
        )
        print(f"gata!")
        print(f"  → {result.summary()}")

        results.append(result)

    return results


def results_to_dataframe(results: list[KMeansResult]) -> pd.DataFrame:
    """Converteste lista de rezultate intr-un DataFrame pandas."""
    rows = []
    for r in results:
        rows.append({
            "Platform":    r.platform,
            "N_Samples":   r.n_samples,
            "D_Features":  r.n_features,
            "K_Clusters":  r.k,
            "Time_Seconds": round(r.time_seconds, 6),
            "Iterations":  r.iterations,
            "Inertia":     round(r.inertia, 4),
        })
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Afiseaza un tabel sumar formatat in consola."""
    print("\n" + "=" * 70)
    print(" SUMAR REZULTATE CPU")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


def save_results(df: pd.DataFrame, output_path: str):
    """Salveaza rezultatele in CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Rezultate salvate in: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="K-Means CPU Benchmark")
    parser.add_argument(
        "--output", default="../../results/results_cpu.csv",
        help="Calea fisierului CSV de output"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Sari scenariile Large si High-Dim (test rapid)"
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help="Seed pentru reproductibilitate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Filtrare scenarii
    scenarios = BENCHMARK_SCENARIOS
    if args.skip_large:
        scenarios = [s for s in scenarios if s["name"] in ("Small", "Medium")]
        print("[INFO] Mod rapid: doar Small si Medium\n")

    # Rulare benchmark
    results = run_benchmark(scenarios, random_state=args.random_state)

    # Salvare si afisare
    df = results_to_dataframe(results)
    print_summary_table(df)
    save_results(df, args.output)

    print("\nBenchmark complet! Foloseste rezultatele din CSV pentru comparatie cu GPU.\n")
