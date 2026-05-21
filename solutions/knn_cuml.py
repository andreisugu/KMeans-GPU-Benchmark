# ============================================================================
# KMeans-GPU-Benchmark: RAPIDS cuML Variant (GPU-Accelerated k-NN)
# Uses NVIDIA RAPIDS cuML's NearestNeighbors which executes entirely on the
# GPU using CUDA — no data ever leaves the GPU during the query phase.
#
# NOTE: This variant requires an NVIDIA GPU with CUDA support and the RAPIDS
# cuML library installed. It CANNOT run on AMD GPUs (ROCm/HIP).
# To install: https://rapids.ai/start.html
#   conda install -c rapidsai -c conda-forge cuml cudatoolkit=12.0
# ============================================================================

import sys
import time
import numpy as np

# ============================================================================
# WHY RAPIDS cuML?
# cuML provides GPU-accelerated drop-in replacements for scikit-learn
# estimators. Its NearestNeighbors implementation (brute-force L2) maps
# directly onto a GPU GEMM: distances for all N query points against all N
# reference points are computed in a single matrix multiplication on the GPU,
# then a top-k reduction (via a GPU-native radix-select) finds the k closest
# neighbours for each point.
#
# The key advantage over a manual CUDA kernel is that cuML leverages:
#   - cuBLAS GEMM for batched distance computation (FP32/FP64 tensor cores)
#   - Thrust's GPU-parallel radix sort for top-k selection
#   - FAISS-compatible indexing for large-scale ANN (approximate k-NN)
# This makes cuML typically faster than hand-written CUDA for k-NN because
# vendor-tuned BLAS operations are heavily optimised for each GPU architecture.
# ============================================================================

try:
    import cuml
    from cuml.neighbors import NearestNeighbors
    import cudf
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


def main():
    if not CUML_AVAILABLE:
        print("ERROR: RAPIDS cuML is not installed or not available on this machine.", file=sys.stderr)
        print("       Install via: conda install -c rapidsai -c conda-forge cuml", file=sys.stderr)
        print("       Requires: NVIDIA GPU + CUDA toolkit", file=sys.stderr)
        print("       See: https://rapids.ai/start.html", file=sys.stderr)
        return 1

    # 1. Parse custom header (Points, Dimensions, k-Neighbours, unused, hasName)
    # We reuse the project's standard header format:
    #   N D K maxIter hasName
    # For k-NN:  K = number of nearest neighbours to find (replaces cluster count)
    #            maxIter is not used (k-NN is non-iterative) but preserved for
    #            compatibility with the shared dataset format.
    header = sys.stdin.readline().split()
    if len(header) < 5:
        print("Error reading dataset header. Expected format: N D K maxIter hasName",
              file=sys.stderr)
        return 1

    num_points    = int(header[0])
    num_features  = int(header[1])
    k_neighbours  = int(header[2])  # K = neighbours, not clusters
    _max_iter     = int(header[3])  # Not used for k-NN; kept for format compat
    has_name      = int(header[4])

    # ============================================================================
    # DATA LOADING
    # Read all point data from stdin into a NumPy array first, then move to GPU.
    # The transfer to GPU (hipMalloc equivalent: cudaMalloc + cudaMemcpy) happens
    # implicitly when we construct the cuDF DataFrame from the NumPy array.
    # ============================================================================
    lines = sys.stdin.readlines()

    data = np.empty((num_points, num_features), dtype=np.float32)
    names = []

    for i in range(num_points):
        parts = lines[i].split()
        data[i] = [float(x) for x in parts[:num_features]]
        if has_name == 1:
            names.append(parts[num_features])

    print(f"Starting k-NN search on {num_points} points (k={k_neighbours})...")

    # ============================================================================
    # GPU TRANSFER
    # cuML works best with cuDF DataFrames backed by GPU memory (device arrays).
    # The data is moved from host RAM → GPU VRAM here. This transfer is excluded
    # from the benchmark timer, which mirrors how the HIP/GPU K-Means variant
    # starts timing AFTER hipMemcpy(HostToDevice).
    # ============================================================================
    data_gpu = cudf.DataFrame(data)

    # ============================================================================
    # CUML NEARESTNEIGHBORS CONFIGURATION
    # algorithm='brute' : exhaustive L2 search (equivalent to the HIP kernel)
    # metric='euclidean': L2 distance, same as all other project variants
    # n_jobs=1           : single GPU stream, reproducible timing
    # output_type='numpy': return indices/distances as host arrays for printing
    # ============================================================================
    nn_model = NearestNeighbors(
        n_neighbors=k_neighbours,
        algorithm='brute',
        metric='euclidean',
        output_type='numpy'
    )

    # --- START BENCHMARK TIMER ---
    # Everything from here is the "Core Loop": GPU fit + GPU query.
    # We call cuml.common.cuda.synchronize() before starting and after finishing
    # to ensure we measure actual GPU wall time, not just kernel launch time.
    cuml.common.cuda.synchronize()
    start_time = time.perf_counter()

    # ============================================================================
    # STEP A: FIT (Index Phase)
    # For brute-force k-NN, fit() simply stores a reference to the training data
    # on the GPU. No index structure is built. O(1) work, but ensures the data
    # is pinned in GPU memory for the query phase.
    # ============================================================================
    nn_model.fit(data_gpu)

    # ============================================================================
    # STEP B: QUERY (Search Phase — the actual parallelism)
    # kneighbors() launches a GPU GEMM to compute the full N×N distance matrix
    # in tiles, then a parallel radix-select to find the k smallest distances
    # per row. This is the embarrassingly parallel part:
    #   - Each query point's k-NN search is independent of all others
    #   - All N×N distance computations execute simultaneously on GPU threads
    # Total work: O(N² × D) but with GPU parallelism factor = SM_count × warp_size
    # ============================================================================
    distances, indices = nn_model.kneighbors(data_gpu)

    # Synchronize GPU before stopping the timer: ensures all CUDA kernels have
    # completed execution, not just been launched asynchronously.
    cuml.common.cuda.synchronize()

    # --- STOP BENCHMARK TIMER ---
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000.0

    # ============================================================================
    # OUTPUT (matching project conventions)
    # k-NN is non-iterative, so we report query count instead of iterations.
    # We sample 5 points to show example nearest-neighbour results.
    # ============================================================================
    print("------------------------------------------------")
    print(f"Query completed for {num_points} points.")
    print(f"Execution Time (Core Loop): {duration_ms:.4f} ms")
    print("------------------------------------------------")

    # Show a sample of results (first 5 query points)
    sample_size = min(5, num_points)
    print(f"Sample Results (first {sample_size} query points):")
    for i in range(sample_size):
        neighbour_ids  = " ".join(str(idx) for idx in indices[i])
        neighbour_dist = " ".join(f"{d:.4f}" for d in distances[i])
        label = f" [{names[i]}]" if has_name == 1 else ""
        print(f"  Point {i}{label}: neighbours=[{neighbour_ids}]  dists=[{neighbour_dist}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
