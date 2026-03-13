"""
benchmark_taichi.py
-------------------
Benchmark runner for AMD iGPU via Taichi (Vulkan backend).
Saves to results/results_taichi.csv.

Setup:  pip install taichi
Run:    python benchmark_taichi.py
        python benchmark_taichi.py --skip-large
        python benchmark_taichi.py --validate
        python benchmark_taichi.py --arch cpu    # test without GPU
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from data_generator import generate_synthetic, BENCHMARK_SCENARIOS
from kmeans_cpu import run_kmeans_cpu, validate_against_reference
from kmeans_taichi import init_taichi, run_kmeans_taichi


def run_benchmark(scenarios, random_state=42, validate=False):
    print("=" * 70)
    print(" K-Means Taichi Benchmark — AMD Radeon 780M (Vulkan backend)")
    print("=" * 70)

    results = []
    for i, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        n    = scenario["n_samples"]
        d    = scenario["n_features"]
        k    = scenario["k"]
        desc = scenario.get("desc", "")

        print(f"\n[{i}/{len(scenarios)}] {name} ({desc})")
        print(f"  N={n:,}  D={d}  K={k}")

        print("  Generating data...", end=" ", flush=True)
        X = generate_synthetic(n, d, k, random_state)
        print(f"done  ({X.nbytes / 1e6:.0f} MB)")

        cpu_result = None
        if validate:
            print("  Running CPU reference...", end=" ", flush=True)
            cpu_result = run_kmeans_cpu(X, k=k, n_init=1, random_state=random_state)
            print("done")

        print("  Running Taichi iGPU...", end=" ", flush=True)
        result = run_kmeans_taichi(X, k=k, random_state=random_state)
        print("done!")
        print(f"  -> {result.summary()}")

        if validate and cpu_result:
            validate_against_reference(result, cpu_result, inertia_tol_pct=5.0)

        results.append(result)

    return results


def results_to_df(results):
    return pd.DataFrame([{
        "Platform":     r.platform,
        "N_Samples":    r.n_samples,
        "D_Features":   r.n_features,
        "K_Clusters":   r.k,
        "Time_Seconds": round(r.time_seconds, 6),
        "Iterations":   r.iterations,
        "Inertia":      round(r.inertia, 4),
    } for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means Taichi iGPU Benchmark")
    parser.add_argument("--output",       default="../../results/results_taichi.csv")
    parser.add_argument("--skip-large",   action="store_true")
    parser.add_argument("--validate",     action="store_true")
    parser.add_argument("--arch",         default="vulkan",
                        choices=["vulkan", "cpu", "cuda", "opengl"])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    arch_map = {
        "vulkan": ti.vulkan, "cpu": ti.cpu,
        "cuda":   ti.cuda,   "opengl": ti.opengl,
    }

    # Init Taichi ONCE — kernels compiled once, reused across all 4 scenarios
    init_taichi(arch=arch_map[args.arch])

    scenarios = BENCHMARK_SCENARIOS
    if args.skip_large:
        scenarios = [s for s in scenarios if s["name"] in ("Small", "Medium")]
        print("[INFO] Quick mode: Small + Medium only")

    results = run_benchmark(scenarios,
                            random_state=args.random_state,
                            validate=args.validate)

    df = results_to_df(results)
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
