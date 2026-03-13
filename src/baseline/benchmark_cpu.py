"""
benchmark_cpu.py
----------------
Main benchmarking script for CPU variant (scikit-learn).
Iterates through all scenarios, saves results to CSV and displays a summary.

Usage:
    python benchmark_cpu.py
    python benchmark_cpu.py --output ../../results/results_cpu.csv
    python benchmark_cpu.py --skip-large   # skip Large + High-Dim for quick test
"""

import argparse
import os
import time
import pandas as pd
import numpy as np

from data_generator import generate_synthetic, BENCHMARK_SCENARIOS
from kmeans_cpu import run_kmeans_cpu, KMeansResult


# ── Global configuration ───────────────────────────────────────────────────────
RANDOM_STATE   = 42
INIT_METHOD    = "k-means++"   # change to "random" if CUDA doesn't support k-means++
MAX_ITER       = 300
TOL            = 1e-4
N_INIT         = 1             # 1 run per scenario (consistent with CUDA)


def run_benchmark(scenarios: list,
                  random_state: int = RANDOM_STATE) -> list[KMeansResult]:
    """
    Runs K-Means CPU on the given list of scenarios.

    Returns list of KMeansResult.
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

        print(f"\n[{i}/{len(scenarios)}] Scenario: {name} ({desc})")
        print(f"  N={n_samples:,}  D={n_features}  K={k}")

        # Data generation (not included in timing)
        print("  Generating data...", end=" ", flush=True)
        t0 = time.perf_counter()
        X = generate_synthetic(n_samples, n_features, k, random_state)
        print(f"done in {time.perf_counter()-t0:.2f}s  ({X.nbytes/1e6:.0f} MB)")

        # Running K-Means
        print("  Running K-Means CPU...", end=" ", flush=True)
        result = run_kmeans_cpu(
            X,
            k=k,
            init=INIT_METHOD,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=N_INIT,
            random_state=random_state,
        )
        print(f"done!")
        print(f"  → {result.summary()}")

        results.append(result)

    return results


def results_to_dataframe(results: list[KMeansResult]) -> pd.DataFrame:
    """Converts list of results to pandas DataFrame."""
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
    """Displays formatted summary table in console."""
    print("\n" + "=" * 70)
    print(" CPU RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


def save_results(df: pd.DataFrame, output_path: str):
    """Saves results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="K-Means CPU Benchmark")
    parser.add_argument(
        "--output", default="../../results/results_cpu.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip Large and High-Dim scenarios (quick test)"
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help="Seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Filter scenarios
    scenarios = BENCHMARK_SCENARIOS
    if args.skip_large:
        scenarios = [s for s in scenarios if s["name"] in ("Small", "Medium")]
        print("[INFO] Quick mode: Small and Medium only\n")

    # Run benchmark
    results = run_benchmark(scenarios, random_state=args.random_state)

    # Save and display
    df = results_to_dataframe(results)
    print_summary_table(df)
    save_results(df, args.output)

    print("\nBenchmark complete! Use the CSV results for comparison with GPU.\n")
