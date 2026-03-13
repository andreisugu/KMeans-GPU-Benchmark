"""
benchmark_rapids.py
-------------------
Benchmark GPU cuML — interfata identica cu benchmark_cpu.py.
Salveaza rezultatele in results/results_rapids.csv.

Rulare in Google Colab (GPU runtime):
    !python src/rapids/benchmark_rapids.py
    !python src/rapids/benchmark_rapids.py --skip-large
"""

import argparse
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from data_generator import generate_synthetic, BENCHMARK_SCENARIOS
from kmeans_cpu import KMeansResult, validate_against_reference, run_kmeans_cpu
from kmeans_rapids import run_kmeans_rapids


def run_benchmark(scenarios, random_state=42, validate=True):
    results      = []
    cpu_results  = {}   # pentru validare inertia

    print("=" * 70)
    print(" K-Means GPU Benchmark — RAPIDS cuML")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        name, n, d, k = scenario["name"], scenario["n_samples"], scenario["n_features"], scenario["k"]
        print(f"\n[{i}/{len(scenarios)}] {name} — N={n:,} D={d} K={k}")

        print("  Generare date...", end=" ", flush=True)
        X = generate_synthetic(n, d, k, random_state)
        print("gata")

        # Referinta CPU (pentru validare)
        if validate:
            cpu_res = run_kmeans_cpu(X, k=k, n_init=1, random_state=random_state)
            cpu_results[name] = cpu_res

        print("  Rulare cuML GPU...", end=" ", flush=True)
        res = run_kmeans_rapids(X, k=k, random_state=random_state)
        print("gata!")
        print(f"  → {res.summary()}")

        if validate:
            validate_against_reference(res, cpu_results[name])

        results.append(res)

    return results


def results_to_df(results):
    return pd.DataFrame([{
        "Platform": r.platform, "N_Samples": r.n_samples,
        "D_Features": r.n_features, "K_Clusters": r.k,
        "Time_Seconds": round(r.time_seconds, 6),
        "Iterations": r.iterations, "Inertia": round(r.inertia, 4),
    } for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../results/results_rapids.csv")
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    scenarios = BENCHMARK_SCENARIOS
    if args.skip_large:
        scenarios = [s for s in scenarios if s["name"] in ("Small", "Medium")]

    results = run_benchmark(scenarios, validate=not args.no_validate)
    df = results_to_df(results)
    print("\n" + df.to_string(index=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Salvat: {args.output}")
