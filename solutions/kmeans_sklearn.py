# ============================================================================
# KMeans-GPU-Benchmark: Scikit-Learn Variant (Optimized Library Baseline)
# Uses sklearn's highly-optimized, vectorized KMeans under the hood.
# ============================================================================

import sys
import time
import numpy as np
from sklearn.cluster import KMeans

# ============================================================================
# WHY SKLEARN?
# Scikit-Learn's KMeans uses NumPy's vectorized BLAS routines (Intel MKL or
# OpenBLAS) which automatically parallelize across CPU cores using SIMD
# instructions (AVX2/AVX-512). This makes it drastically faster than a
# naive sequential loop, even though it runs on the CPU.
# Under the hood, sklearn computes all N*K distances in a single matrix
# operation: ||X - C||^2 = ||X||^2 - 2*X*C^T + ||C||^2 (the "kernel trick").
# This avoids the explicit O(N*K*D) inner loop and leverages cache-friendly
# Level-3 BLAS (matrix-matrix multiply).
# ============================================================================


def main():
    # 1. Parse custom header (Points, Dimensions, Clusters, Iterations, Name)
    header = sys.stdin.readline().split()
    if len(header) < 5:
        print("Error reading dataset header. Ensure format is: "
              "Points Attributes Clusters Max_Iterations Has_Name",
              file=sys.stderr)
        return 1

    num_points = int(header[0])
    num_features = int(header[1])
    k = int(header[2])
    max_iterations = int(header[3])
    has_name = int(header[4])

    # ============================================================================
    # DATA LOADING
    # We use NumPy to load all remaining lines into a contiguous C-order array.
    # This is equivalent to the flat-array DOD approach in the C++ version:
    # a single contiguous block of memory for maximum cache locality.
    # ============================================================================
    lines = sys.stdin.readlines()

    data = np.empty((num_points, num_features), dtype=np.float64)
    names = []

    for i in range(num_points):
        parts = lines[i].split()
        data[i] = [float(x) for x in parts[:num_features]]
        if has_name == 1:
            names.append(parts[num_features])

    # ============================================================================
    # 2. SKLEARN KMEANS CONFIGURATION
    # We match the sequential C++ variant as closely as possible:
    #   - init='random'     : Forgy initialization (same as C++ random pick)
    #   - n_init=1          : Single run (no restarts), same as C++ single pass
    #   - algorithm='lloyd'  : Classic Lloyd's algorithm (same as C++ implementation)
    #   - max_iter=maxIter  : Same iteration cap as the dataset header specifies
    #   - tol=0             : Disable early convergence by tolerance; only stop
    #                         when assignments no longer change (like C++ `changed`)
    # ============================================================================
    kmeans = KMeans(
        n_clusters=k,
        init='random',
        n_init=1,
        algorithm='lloyd',
        max_iter=max_iterations,
        tol=0,
        random_state=None  # Truly random each run, matching C++ random_device
    )

    print(f"Starting K-Means clustering on {num_points} points...")

    # --- START BENCHMARK TIMER ---
    # We strictly measure the mathematical algorithm, excluding I/O (disk reading).
    # time.perf_counter() is the highest-resolution timer available in Python.
    start_time = time.perf_counter()

    # ============================================================================
    # 3. FIT THE MODEL (THE ENTIRE ALGORITHM RUNS HERE)
    # Internally, sklearn performs:
    #   STEP A (Assignment): Vectorized distance matrix via BLAS  → O(N*K*D)
    #   STEP B (Update):     np.bincount + np.add.at for centroids → O(N*D)
    # All of this is parallelized across threads by OpenBLAS/MKL automatically.
    # ============================================================================
    kmeans.fit(data)

    # --- STOP BENCHMARK TIMER ---
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000.0

    # 4. Display results (matching sequential C++ output format)
    print("------------------------------------------------")
    print(f"Convergence reached in {kmeans.n_iter_} iterations.")
    print(f"Execution Time (Core Loop): {duration_ms:.4f} ms")
    print("------------------------------------------------")

    # Count points per cluster (equivalent to C++ counts_helper)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for j in range(k):
        coords = " ".join(f"{c}" for c in centroids[j])
        count = int(np.sum(labels == j))
        print(f"Centroid {j}: {coords} (Points: {count})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
