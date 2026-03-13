"""
benchmark_igpu.py
-----------------
Benchmark runner for AMD iGPU (Radeon 780M) via PyOpenCL.
Interface identical to benchmark_cpu.py and benchmark_rapids.py.
Saves results to results/results_igpu.csv — compatible with plot_results.py.

Usage:
    python benchmark_igpu.py
    python benchmark_igpu.py --skip-large
    python benchmark_igpu.py --validate          # compare inertia vs CPU
    python benchmark_igpu.py --list-devices      # show all OpenCL devices
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from data_generator import generate_synthetic, BENCHMARK_SCENARIOS
from kmeans_cpu import KMeansResult, validate_against_reference, run_kmeans_cpu
from kmeans_pyopencl import run_kmeans_pyopencl, get_amd_igpu_device

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False


def list_opencl_devices():
    """Print all available OpenCL platforms and devices."""
    print("=== Available OpenCL Devices ===\n")
    for platform in cl.get_platforms():
        print(f"Platform: {platform.name}")
        for dev in platform.get_devices():
            dtype = cl.device_type.to_string(dev.type)
            print(f"  [{dtype}] {dev.name}")
            print(f"    Max CUs     : {dev.max_compute_units}")
            print(f"    Max freq    : {dev.max_clock_frequency} MHz")
            print(f"    Global mem  : {dev.global_mem_size // (1024**2)} MB")
            print(f"    Local mem   : {dev.local_mem_size // 1024} KB")
            print(f"    Max work-group: {dev.max_work_group_size}")
            print(f"    OpenCL ver  : {dev.version}")
            print()


def run_benchmark(scenarios: list,
                  random_state: int = 42,
                  validate: bool = False) -> list:
    results = []

    # Detect device once
    device = get_amd_igpu_device()
    print(f"\nTarget device: {device.name}")
    print(f"Compute units: {device.max_compute_units}")
    print(f"Global memory: {device.global_mem_size // (1024**2)} MB")

    print("\n" + "=" * 70)
    print(" K-Means iGPU Benchmark — AMD Radeon 780M (PyOpenCL)")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        n    = scenario["n_samples"]
        d    = scenario["n_features"]
        k    = scenario["k"]
        desc = scenario.get("desc", "")

        print(f"\n[{i}/{len(scenarios)}] {name} ({desc})")
        print(f"  N={n:,}  D={d}  K={k}")

        # Generate data (not timed)
        print("  Generating data...", end=" ", flush=True)
        X = generate_synthetic(n, d, k, random_state)
        print(f"done  ({X.nbytes / 1e6:.0f} MB)")

        # Optional CPU reference for validation
        cpu_result = None
        if validate:
            print("  Running CPU reference...", end=" ", flush=True)
            cpu_result = run_kmeans_cpu(X, k=k, n_init=1, random_state=random_state)
            print("done")

        # Run iGPU benchmark
        print("  Running AMD iGPU (PyOpenCL)...", end=" ", flush=True)
        result = run_kmeans_pyopencl(X, k=k, random_state=random_state, device=device)
        print("done!")
        print(f"  → {result.summary()}")

        if validate and cpu_result:
            validate_against_reference(result, cpu_result, inertia_tol_pct=5.0)

        results.append(result)

    return results


def results_to_df(results: list) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="K-Means AMD iGPU Benchmark")
    parser.add_argument("--output",       default="../../results/results_igpu.csv")
    parser.add_argument("--skip-large",   action="store_true",
                        help="Run only Small + Medium scenarios")
    parser.add_argument("--validate",     action="store_true",
                        help="Compare inertia against CPU scikit-learn")
    parser.add_argument("--list-devices", action="store_true",
                        help="List all OpenCL devices and exit")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not PYOPENCL_AVAILABLE:
        print("ERROR: PyOpenCL not installed. Run: pip install pyopencl")
        sys.exit(1)

    if args.list_devices:
        list_opencl_devices()
        sys.exit(0)

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
    print(f"\n✅ Saved to: {args.output}")
