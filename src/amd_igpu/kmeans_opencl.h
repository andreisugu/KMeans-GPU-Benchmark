#pragma once
/**
 * kmeans_opencl.h
 * ---------------
 * C++ OpenCL K-Means for AMD iGPU (Radeon 780M).
 * Low-level equivalent of the CUDA kernel implementation,
 * but portable across AMD / Intel / NVIDIA via OpenCL API.
 *
 * Memory architecture (mirrors CUDA implementation):
 *   Dataset    (N×D float32) → clCreateBuffer READ_ONLY  — transferred once
 *   Labels     (N   int32)   → clCreateBuffer READ_WRITE — written by assignment kernel
 *   Centroids  (K×D float32) → clCreateBuffer READ_ONLY  — updated each iteration
 *   New sums   (K×D float32) → clCreateBuffer READ_WRITE — reset each iteration
 *   Counts     (K   int32)   → clCreateBuffer READ_WRITE — reset each iteration
 *
 * Timing: clGetEventProfilingInfo on GPU events (equivalent to cudaEventRecord)
 * This correctly measures GPU execution time, unlike std::chrono which
 * would stop before async GPU work completes.
 */

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <stdexcept>

// Reuse Dataset and KMeansResult from the baseline
#include "../baseline/kmeans.h"


// ── OpenCL error checking macro ───────────────────────────────────────────────
#define CL_CHECK(err) \
    if ((err) != CL_SUCCESS) { \
        throw std::runtime_error(std::string("OpenCL error ") + \
                                 std::to_string(err) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }


/**
 * OpenCL K-Means implementation.
 * Algorithm: Lloyd's, same as CPU baseline and CUDA kernel.
 */
class KMeansOpenCL {
public:
    /**
     * @param k            Number of clusters
     * @param max_iter     Maximum iterations
     * @param tol          Convergence threshold (max centroid shift)
     * @param random_state Seed for K-Means++ initialization
     * @param kernel_dir   Path to directory containing .cl kernel files
     */
    KMeansOpenCL(int k,
                 int max_iter         = 300,
                 float tol            = 1e-4f,
                 int random_state     = 42,
                 std::string kernel_dir = "kernels");

    ~KMeansOpenCL();

    /**
     * Fit the model to the dataset.
     * @param dataset  Input data (N×D, float32, row-major)
     * @return         KMeansResult with timing (GPU events), inertia, labels
     */
    KMeansResult fit(const Dataset& dataset);

    /**
     * Print detected OpenCL device info.
     */
    static void print_device_info();

private:
    int         k_;
    int         max_iter_;
    float       tol_;
    int         random_state_;
    std::string kernel_dir_;

    // OpenCL objects
    cl_platform_id   platform_  = nullptr;
    cl_device_id     device_    = nullptr;
    cl_context       context_   = nullptr;
    cl_command_queue queue_     = nullptr;
    cl_program       program_   = nullptr;
    cl_kernel        k_assign_  = nullptr;
    cl_kernel        k_update_  = nullptr;
    cl_kernel        k_divide_  = nullptr;

    void setup_opencl();
    void build_kernels();
    std::string load_kernel_source(const std::string& filename);

    // K-Means++ initialization (CPU side, called once)
    std::vector<float32_t> init_kmeans_plus_plus(const Dataset& ds, std::mt19937& rng);

    // Compute inertia on CPU after GPU run
    float compute_inertia(const Dataset& ds,
                          const std::vector<int>& labels,
                          const std::vector<float32_t>& centers) const;
};


// ── I/O helpers (shared with baseline) ───────────────────────────────────────
namespace igpu_io {
    void save_benchmark_row(const std::string& filepath,
                            const std::string& platform,
                            int n_samples, int n_features, int k,
                            double time_ms, int n_iter, float inertia);
}
