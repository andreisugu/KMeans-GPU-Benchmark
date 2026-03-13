#pragma once
/**
 * kmeans_sycl.h
 * -------------
 * K-Means on AMD iGPU (Radeon 780M) using SYCL 2020.
 *
 * Why SYCL over OpenCL:
 *   - Kernels are C++ lambdas — no separate .cl files, no runtime compilation
 *   - Native float atomics via sycl::atomic_ref<float> (no CAS loop hack)
 *   - Automatic kernel dependency tracking via the SYCL runtime
 *   - Single-source: host + device code in the same .cpp file
 *   - Portable: same code runs on AMD (via AdaptiveCpp), Intel, NVIDIA
 *
 * Compiler: AdaptiveCpp (formerly hipSYCL) — recommended for AMD iGPU
 *   Install: https://github.com/AdaptiveCpp/AdaptiveCpp
 *   Compile: acpp --acpp-targets="hip:gfx1103" -O2 -o kmeans_sycl main_sycl.cpp kmeans_sycl.cpp
 *
 * Alternative: Intel oneAPI DPC++ with OpenCL backend
 *   Compile: icpx -fsycl -O2 -o kmeans_sycl main_sycl.cpp kmeans_sycl.cpp
 *
 * Memory model (buffer/accessor — portable SYCL 2020):
 *   Dataset    (N×D float) → sycl::buffer<float>  READ       — copied once to device
 *   Labels     (N   int)   → sycl::buffer<int>    READ_WRITE — written by assignment kernel
 *   Centroids  (K×D float) → sycl::buffer<float>  READ       — updated each iteration
 *   New sums   (K×D float) → sycl::buffer<float>  READ_WRITE — atomic accumulation
 *   Counts     (K   int)   → sycl::buffer<int>    READ_WRITE — atomic increment
 */

#include <sycl/sycl.hpp>
#include <string>
#include <vector>

// Reuse Dataset and KMeansResult from baseline
#include "../baseline/kmeans.h"


/**
 * SYCL K-Means implementation.
 * Algorithm: Lloyd's — identical to CPU baseline and OpenCL variant.
 */
class KMeansSYCL {
public:
    /**
     * @param k            Number of clusters
     * @param max_iter     Maximum iterations
     * @param tol          Convergence threshold (max centroid shift)
     * @param random_state Seed for K-Means++ init (runs on CPU)
     */
    KMeansSYCL(int   k,
               int   max_iter     = 300,
               float tol          = 1e-4f,
               int   random_state = 42);

    /**
     * Fit to dataset. Times GPU execution only (excludes data generation/I/O).
     * @param dataset  Input data (N×D float32, row-major flat array)
     * @return         KMeansResult with GPU timing, inertia, labels
     */
    KMeansResult fit(const Dataset& dataset);

    /**
     * Print the SYCL device that will be used.
     */
    static void print_device_info();

private:
    int   k_;
    int   max_iter_;
    float tol_;
    int   random_state_;

    // SYCL queue — created once, reused across calls
    sycl::queue queue_;

    // K-Means++ init (CPU side, called once before GPU loop)
    std::vector<float32_t> init_kmeans_plus_plus(const Dataset& ds,
                                                  std::mt19937& rng) const;

    // Inertia (WCSS) — computed on CPU after GPU loop
    float compute_inertia(const Dataset& ds,
                          const std::vector<int>&       labels,
                          const std::vector<float32_t>& centers) const;
};


// ── I/O ───────────────────────────────────────────────────────────────────────
namespace sycl_io {
    void save_benchmark_row(const std::string& filepath,
                            const std::string& platform,
                            int n_samples, int n_features, int k,
                            double time_ms, int n_iter, float inertia);
}
