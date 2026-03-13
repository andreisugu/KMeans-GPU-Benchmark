/**
 * kmeans_opencl.cpp
 * -----------------
 * Full C++ OpenCL K-Means implementation for AMD iGPU (Radeon 780M).
 * Mirrors the CUDA kernel structure but uses portable OpenCL API.
 */

#include "kmeans_opencl.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


// ══════════════════════════════════════════════════════════════════════════════
//  Constructor / Destructor
// ══════════════════════════════════════════════════════════════════════════════
KMeansOpenCL::KMeansOpenCL(int k, int max_iter, float tol,
                             int random_state, std::string kernel_dir)
    : k_(k), max_iter_(max_iter), tol_(tol),
      random_state_(random_state), kernel_dir_(std::move(kernel_dir))
{
    if (k_ <= 0)        throw std::invalid_argument("k must be > 0");
    if (max_iter_ <= 0) throw std::invalid_argument("max_iter must be > 0");

    setup_opencl();
    build_kernels();
}

KMeansOpenCL::~KMeansOpenCL()
{
    if (k_assign_)  clReleaseKernel(k_assign_);
    if (k_update_)  clReleaseKernel(k_update_);
    if (k_divide_)  clReleaseKernel(k_divide_);
    if (program_)   clReleaseProgram(program_);
    if (queue_)     clReleaseCommandQueue(queue_);
    if (context_)   clReleaseContext(context_);
}


// ══════════════════════════════════════════════════════════════════════════════
//  OpenCL Setup — find AMD iGPU, create context + queue
// ══════════════════════════════════════════════════════════════════════════════
void KMeansOpenCL::setup_opencl()
{
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0)
        throw std::runtime_error("No OpenCL platforms found. Install mesa-opencl-icd.");

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    // Prefer AMD GPU (Radeon 780M)
    for (auto& plat : platforms) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS)
            continue;
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        for (auto& dev : devices) {
            char name[256];
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            std::string sname(name);
            // Match AMD iGPU names
            if (sname.find("Radeon") != std::string::npos ||
                sname.find("AMD")    != std::string::npos ||
                sname.find("gfx")   != std::string::npos) {
                platform_ = plat;
                device_   = dev;
                std::cout << "[OpenCL] Selected: " << sname << "\n";
                goto device_found;
            }
        }
    }

    // Fallback: any GPU
    for (auto& plat : platforms) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS)
            continue;
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        platform_ = plat;
        device_   = devices[0];

        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[OpenCL] Fallback device: " << name << "\n";
        goto device_found;
    }

    throw std::runtime_error("No GPU OpenCL device found. Install mesa-opencl-icd.");

device_found:
    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);

    // Enable profiling for accurate GPU timing
    queue_ = clCreateCommandQueue(context_, device_,
                                   CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);
}


// ══════════════════════════════════════════════════════════════════════════════
//  Kernel loading and compilation
// ══════════════════════════════════════════════════════════════════════════════
std::string KMeansOpenCL::load_kernel_source(const std::string& filename)
{
    std::string path = kernel_dir_ + "/" + filename;
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

void KMeansOpenCL::build_kernels()
{
    std::string src = load_kernel_source("assignment.cl") + "\n"
                    + load_kernel_source("update.cl");
    const char* src_ptr = src.c_str();
    size_t      src_len = src.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);

    err = clBuildProgram(program_, 1, &device_,
                         "-cl-fast-relaxed-math -cl-mad-enable",
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Print build log for debugging
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr);
        throw std::runtime_error("Kernel build failed:\n" + log);
    }

    k_assign_ = clCreateKernel(program_, "assignment",   &err); CL_CHECK(err);
    k_update_ = clCreateKernel(program_, "update",       &err); CL_CHECK(err);
    k_divide_ = clCreateKernel(program_, "divide_counts",&err); CL_CHECK(err);
}


// ══════════════════════════════════════════════════════════════════════════════
//  K-Means++ initialization (CPU)
// ══════════════════════════════════════════════════════════════════════════════
std::vector<float32_t> KMeansOpenCL::init_kmeans_plus_plus(
    const Dataset& ds, std::mt19937& rng)
{
    const int N = ds.n_samples;
    const int D = ds.n_features;

    std::vector<float32_t> centers(k_ * D);
    std::vector<float32_t> min_dists(N, std::numeric_limits<float32_t>::max());

    // First centroid: uniform random
    std::uniform_int_distribution<int> uid(0, N - 1);
    int first = uid(rng);
    std::copy(&ds.data[first * D], &ds.data[first * D + D], &centers[0]);

    for (int ki = 1; ki < k_; ++ki) {
        // Update min distances to last centroid
        for (int i = 0; i < N; ++i) {
            float32_t d = 0.0f;
            for (int dim = 0; dim < D; ++dim) {
                float32_t diff = ds.data[i * D + dim] - centers[(ki-1) * D + dim];
                d += diff * diff;
            }
            if (d < min_dists[i]) min_dists[i] = d;
        }

        float32_t total = std::accumulate(min_dists.begin(), min_dists.end(), 0.0f);
        std::uniform_real_distribution<float32_t> urd(0.0f, total);
        float32_t threshold = urd(rng);

        float32_t cumsum = 0.0f;
        int chosen = N - 1;
        for (int i = 0; i < N; ++i) {
            cumsum += min_dists[i];
            if (cumsum >= threshold) { chosen = i; break; }
        }

        std::copy(&ds.data[chosen * D], &ds.data[chosen * D + D], &centers[ki * D]);
    }
    return centers;
}


// ══════════════════════════════════════════════════════════════════════════════
//  Inertia (WCSS) — computed on CPU after GPU run
// ══════════════════════════════════════════════════════════════════════════════
float KMeansOpenCL::compute_inertia(const Dataset& ds,
                                     const std::vector<int>& labels,
                                     const std::vector<float32_t>& centers) const
{
    double inertia = 0.0;
    const int D = ds.n_features;
    for (int i = 0; i < ds.n_samples; ++i) {
        int ki = labels[i];
        for (int d = 0; d < D; ++d) {
            float diff = ds.data[i * D + d] - centers[ki * D + d];
            inertia += diff * diff;
        }
    }
    return static_cast<float>(inertia);
}


// ══════════════════════════════════════════════════════════════════════════════
//  fit() — Main GPU loop
// ══════════════════════════════════════════════════════════════════════════════
KMeansResult KMeansOpenCL::fit(const Dataset& dataset)
{
    const int N = dataset.n_samples;
    const int D = dataset.n_features;

    if (N < k_) throw std::invalid_argument("N_samples < K");

    std::mt19937 rng(static_cast<unsigned>(random_state_));
    auto centroids = init_kmeans_plus_plus(dataset, rng);  // (k_ * D)

    cl_int err;

    // ── Allocate GPU buffers ──────────────────────────────────────────────────
    size_t sz_points    = (size_t)N * D * sizeof(float32_t);
    size_t sz_centroids = (size_t)k_ * D * sizeof(float32_t);
    size_t sz_labels    = (size_t)N * sizeof(cl_int);
    size_t sz_counts    = (size_t)k_ * sizeof(cl_int);

    cl_mem buf_points = clCreateBuffer(context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sz_points, const_cast<float32_t*>(dataset.data.data()), &err);
    CL_CHECK(err);

    cl_mem buf_centroids = clCreateBuffer(context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sz_centroids, centroids.data(), &err);
    CL_CHECK(err);

    cl_mem buf_labels = clCreateBuffer(context_,
        CL_MEM_READ_WRITE, sz_labels, nullptr, &err);
    CL_CHECK(err);

    cl_mem buf_new_sums = clCreateBuffer(context_,
        CL_MEM_READ_WRITE, sz_centroids, nullptr, &err);
    CL_CHECK(err);

    cl_mem buf_counts = clCreateBuffer(context_,
        CL_MEM_READ_WRITE, sz_counts, nullptr, &err);
    CL_CHECK(err);

    // Work sizes
    const size_t local_assign = 256;
    const size_t global_assign = ((N + local_assign - 1) / local_assign) * local_assign;
    const size_t local_update  = 256;
    const size_t global_update = ((N + local_update - 1) / local_update) * local_update;
    const size_t global_divide = ((k_ * D + 63) / 64) * 64;

    // Static kernel args
    cl_int N_cl = N, D_cl = D, K_cl = k_;

    CL_CHECK(clSetKernelArg(k_assign_, 0, sizeof(cl_mem), &buf_points));
    CL_CHECK(clSetKernelArg(k_assign_, 1, sizeof(cl_mem), &buf_centroids));
    CL_CHECK(clSetKernelArg(k_assign_, 2, sizeof(cl_mem), &buf_labels));
    CL_CHECK(clSetKernelArg(k_assign_, 3, sizeof(cl_int), &N_cl));
    CL_CHECK(clSetKernelArg(k_assign_, 4, sizeof(cl_int), &D_cl));
    CL_CHECK(clSetKernelArg(k_assign_, 5, sizeof(cl_int), &K_cl));

    CL_CHECK(clSetKernelArg(k_update_, 0, sizeof(cl_mem), &buf_points));
    CL_CHECK(clSetKernelArg(k_update_, 1, sizeof(cl_mem), &buf_labels));
    CL_CHECK(clSetKernelArg(k_update_, 2, sizeof(cl_mem), &buf_new_sums));
    CL_CHECK(clSetKernelArg(k_update_, 3, sizeof(cl_mem), &buf_counts));
    CL_CHECK(clSetKernelArg(k_update_, 4, sizeof(cl_int), &N_cl));
    CL_CHECK(clSetKernelArg(k_update_, 5, sizeof(cl_int), &D_cl));
    CL_CHECK(clSetKernelArg(k_update_, 6, sizeof(cl_int), &K_cl));

    CL_CHECK(clSetKernelArg(k_divide_, 0, sizeof(cl_mem), &buf_new_sums));
    CL_CHECK(clSetKernelArg(k_divide_, 1, sizeof(cl_mem), &buf_counts));
    CL_CHECK(clSetKernelArg(k_divide_, 2, sizeof(cl_int), &K_cl));
    CL_CHECK(clSetKernelArg(k_divide_, 3, sizeof(cl_int), &D_cl));

    std::vector<float32_t> zeros_f(k_ * D, 0.0f);
    std::vector<cl_int>    zeros_i(k_, 0);
    std::vector<float32_t> new_centroids(k_ * D);
    std::vector<cl_int>    counts(k_);

    // ── START timing (GPU profiling events) ───────────────────────────────────
    clFinish(queue_);
    auto t_start = std::chrono::high_resolution_clock::now();

    int n_iter = 0;
    for (int iter = 0; iter < max_iter_; ++iter) {
        n_iter = iter + 1;

        // Kernel 1: Assignment
        CL_CHECK(clEnqueueNDRangeKernel(queue_, k_assign_, 1,
            nullptr, &global_assign, &local_assign, 0, nullptr, nullptr));

        // Reset accumulators
        CL_CHECK(clEnqueueWriteBuffer(queue_, buf_new_sums, CL_FALSE, 0,
            sz_centroids, zeros_f.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(queue_, buf_counts, CL_FALSE, 0,
            sz_counts, zeros_i.data(), 0, nullptr, nullptr));

        // Kernel 2: Update (atomic accumulation)
        CL_CHECK(clEnqueueNDRangeKernel(queue_, k_update_, 1,
            nullptr, &global_update, &local_update, 0, nullptr, nullptr));

        // Kernel 3: Divide sums / counts → new centroids
        CL_CHECK(clEnqueueNDRangeKernel(queue_, k_divide_, 1,
            nullptr, &global_divide, nullptr, 0, nullptr, nullptr));

        // Read new centroids + counts back to CPU for convergence check
        CL_CHECK(clEnqueueReadBuffer(queue_, buf_new_sums, CL_FALSE, 0,
            sz_centroids, new_centroids.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(queue_, buf_counts, CL_TRUE, 0,
            sz_counts, counts.data(), 0, nullptr, nullptr));
        clFinish(queue_);

        // Handle empty clusters: restore old centroid
        for (int ki = 0; ki < k_; ++ki) {
            if (counts[ki] == 0) {
                std::copy(&centroids[ki * D], &centroids[ki * D + D],
                          &new_centroids[ki * D]);
            }
        }

        // Convergence check
        float32_t max_shift = 0.0f;
        for (int ki = 0; ki < k_; ++ki) {
            float32_t shift = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = new_centroids[ki * D + d] - centroids[ki * D + d];
                shift += diff * diff;
            }
            max_shift = std::max(max_shift, std::sqrt(shift));
        }

        centroids = new_centroids;

        // Upload updated centroids to GPU
        CL_CHECK(clEnqueueWriteBuffer(queue_, buf_centroids, CL_FALSE, 0,
            sz_centroids, centroids.data(), 0, nullptr, nullptr));

        if (max_shift < tol_) break;
    }

    // ── STOP timing ───────────────────────────────────────────────────────────
    clFinish(queue_);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Read final labels
    std::vector<cl_int> labels_cl(N);
    CL_CHECK(clEnqueueReadBuffer(queue_, buf_labels, CL_TRUE, 0,
        sz_labels, labels_cl.data(), 0, nullptr, nullptr));
    clFinish(queue_);

    std::vector<int> labels(labels_cl.begin(), labels_cl.end());
    float inertia = compute_inertia(dataset, labels, centroids);

    // Cleanup GPU buffers
    clReleaseMemObject(buf_points);
    clReleaseMemObject(buf_centroids);
    clReleaseMemObject(buf_labels);
    clReleaseMemObject(buf_new_sums);
    clReleaseMemObject(buf_counts);

    KMeansResult result;
    result.labels  = std::move(labels);
    result.centers = std::move(centroids);
    result.inertia = inertia;
    result.n_iter  = n_iter;
    result.time_ms = elapsed_ms;
    return result;
}


// ══════════════════════════════════════════════════════════════════════════════
//  Device info
// ══════════════════════════════════════════════════════════════════════════════
void KMeansOpenCL::print_device_info()
{
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    std::cout << "=== OpenCL Devices ===\n";
    for (auto& plat : platforms) {
        char pname[256];
        clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        std::cout << "Platform: " << pname << "\n";

        cl_uint num_devices = 0;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        for (auto& dev : devices) {
            char dname[256]; cl_uint cu, freq; cl_ulong gmem;
            clGetDeviceInfo(dev, CL_DEVICE_NAME,               sizeof(dname), dname, nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,  sizeof(cu),    &cu,   nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(freq),  &freq, nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,    sizeof(gmem),  &gmem, nullptr);
            std::cout << "  Device: " << dname
                      << " | CUs=" << cu
                      << " | " << freq << " MHz"
                      << " | " << gmem / (1024*1024) << " MB VRAM\n";
        }
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  I/O
// ══════════════════════════════════════════════════════════════════════════════
void igpu_io::save_benchmark_row(const std::string& filepath,
                                  const std::string& platform,
                                  int n_samples, int n_features, int k,
                                  double time_ms, int n_iter, float inertia)
{
    bool exists = std::ifstream(filepath).good();
    std::ofstream f(filepath, std::ios::app);
    if (!exists)
        f << "Platform,N_Samples,D_Features,K_Clusters,Time_Seconds,Iterations,Inertia\n";
    f << platform << "," << n_samples << "," << n_features << "," << k << ","
      << (time_ms / 1000.0) << "," << n_iter << "," << inertia << "\n";
}
