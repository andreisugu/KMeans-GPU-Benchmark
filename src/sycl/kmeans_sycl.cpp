/**
 * kmeans_sycl.cpp
 * ---------------
 * Full SYCL 2020 K-Means implementation for AMD iGPU (Radeon 780M).
 *
 * Two GPU kernels, submitted to the same sycl::queue:
 *
 *   Kernel 1 — assignment:
 *     Maps 1 work-item → 1 data point.
 *     Reads point from global memory, iterates over K centroids
 *     (from a read accessor, ideally cached), writes nearest label.
 *     Zero race conditions — each work-item writes to a unique index.
 *
 *   Kernel 2 — update:
 *     Maps 1 work-item → 1 data point.
 *     Uses sycl::atomic_ref<float> for native float atomics (SYCL 2020).
 *     Accumulates coordinates into new_sums and increments counts.
 *     Host then divides sums/counts to get new centroid positions.
 *
 * Convergence check runs on the host after reading back centroids each iter.
 * Only K×D×4 bytes are transferred per iteration (centroid data only).
 */

#include "kmeans_sycl.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sycl;


// ══════════════════════════════════════════════════════════════════════════════
//  Device selection — prefer AMD GPU (Radeon 780M)
// ══════════════════════════════════════════════════════════════════════════════
static sycl::queue make_queue()
{
    // Try to find AMD GPU explicitly
    for (const auto& platform : sycl::platform::get_platforms()) {
        for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
            std::string name = device.get_info<sycl::info::device::name>();
            std::string vendor = device.get_info<sycl::info::device::vendor>();
            if (vendor.find("AMD")    != std::string::npos ||
                vendor.find("Advanced") != std::string::npos ||
                name.find("Radeon")   != std::string::npos ||
                name.find("gfx")      != std::string::npos) {
                std::cout << "[SYCL] Selected: " << name
                          << " (" << vendor << ")\n";
                return sycl::queue(device, sycl::property::queue::enable_profiling{});
            }
        }
    }

    // Fallback: any GPU
    try {
        sycl::queue q(sycl::gpu_selector_v,
                      sycl::property::queue::enable_profiling{});
        auto dev = q.get_device();
        std::cout << "[SYCL] Fallback GPU: "
                  << dev.get_info<sycl::info::device::name>() << "\n";
        return q;
    } catch (...) {}

    // Last resort: default device (may be CPU — useful for testing)
    std::cout << "[SYCL] WARNING: No GPU found, using default device.\n";
    return sycl::queue(sycl::default_selector_v,
                       sycl::property::queue::enable_profiling{});
}


// ══════════════════════════════════════════════════════════════════════════════
//  Constructor
// ══════════════════════════════════════════════════════════════════════════════
KMeansSYCL::KMeansSYCL(int k, int max_iter, float tol, int random_state)
    : k_(k), max_iter_(max_iter), tol_(tol), random_state_(random_state),
      queue_(make_queue())
{
    if (k_ <= 0)        throw std::invalid_argument("k must be > 0");
    if (max_iter_ <= 0) throw std::invalid_argument("max_iter must be > 0");
}


// ══════════════════════════════════════════════════════════════════════════════
//  K-Means++ initialization (CPU — called once before GPU loop)
// ══════════════════════════════════════════════════════════════════════════════
std::vector<float32_t> KMeansSYCL::init_kmeans_plus_plus(
    const Dataset& ds, std::mt19937& rng) const
{
    const int N = ds.n_samples;
    const int D = ds.n_features;

    std::vector<float32_t> centers(k_ * D);
    std::vector<float32_t> min_dists(N, std::numeric_limits<float32_t>::max());

    std::uniform_int_distribution<int> uid(0, N - 1);
    int first = uid(rng);
    std::copy(&ds.data[first * D], &ds.data[first * D + D], &centers[0]);

    for (int ki = 1; ki < k_; ++ki) {
        for (int i = 0; i < N; ++i) {
            float32_t d = 0.0f;
            for (int dim = 0; dim < D; ++dim) {
                float32_t diff = ds.data[i * D + dim] - centers[(ki - 1) * D + dim];
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
//  Inertia (WCSS) — computed on CPU after the GPU loop
// ══════════════════════════════════════════════════════════════════════════════
float KMeansSYCL::compute_inertia(const Dataset& ds,
                                   const std::vector<int>& labels,
                                   const std::vector<float32_t>& centers) const
{
    double inertia = 0.0;
    const int D = ds.n_features;
    for (int i = 0; i < ds.n_samples; ++i) {
        int ki = labels[i];
        for (int d = 0; d < D; ++d) {
            float diff = ds.data[i * D + d] - centers[ki * D + d];
            inertia += static_cast<double>(diff * diff);
        }
    }
    return static_cast<float>(inertia);
}


// ══════════════════════════════════════════════════════════════════════════════
//  fit() — Main GPU Lloyd loop
// ══════════════════════════════════════════════════════════════════════════════
KMeansResult KMeansSYCL::fit(const Dataset& dataset)
{
    const int N = dataset.n_samples;
    const int D = dataset.n_features;
    const int K = k_;

    if (N < K) throw std::invalid_argument("N_samples < K");

    std::mt19937 rng(static_cast<unsigned>(random_state_));
    auto centroids = init_kmeans_plus_plus(dataset, rng);  // (K * D) float32

    // ── Allocate SYCL buffers (host data is copied to device automatically) ──
    //
    // buffer<T>(ptr, range) — wraps host memory; SYCL manages host↔device sync
    // The dataset is READ_ONLY on the GPU — transferred once, never modified.

    sycl::buffer<float, 1> buf_points(dataset.data.data(),
                                       sycl::range<1>(N * D));
    sycl::buffer<float, 1> buf_centroids(centroids.data(),
                                          sycl::range<1>(K * D));
    sycl::buffer<int,   1> buf_labels(sycl::range<1>(N));
    sycl::buffer<float, 1> buf_new_sums(sycl::range<1>(K * D));
    sycl::buffer<int,   1> buf_counts(sycl::range<1>(K));

    // Work group size — 256 is a safe default for most AMD RDNA GPUs
    const size_t WG_SIZE = 256;
    const size_t global_assign = ((N + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
    const size_t global_update = global_assign;
    const size_t global_divide = ((K * D + 63) / 64) * 64;

    // ── START timing ─────────────────────────────────────────────────────────
    queue_.wait();
    auto t_start = std::chrono::high_resolution_clock::now();

    int n_iter = 0;
    std::vector<float> new_centroids(K * D);
    std::vector<int>   counts_host(K);

    for (int iter = 0; iter < max_iter_; ++iter) {
        n_iter = iter + 1;

        // ── Kernel 1: Assignment ─────────────────────────────────────────────
        // Each work-item finds the nearest centroid for one data point.
        // No writes to shared memory — embarrassingly parallel.
        queue_.submit([&](sycl::handler& h) {
            auto pts  = buf_points.get_access<sycl::access::mode::read>(h);
            auto ctrs = buf_centroids.get_access<sycl::access::mode::read>(h);
            auto lbl  = buf_labels.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::nd_range<1>(global_assign, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int i = item.get_global_id(0);
                    if (i >= N) return;

                    float best_dist = std::numeric_limits<float>::max();
                    int   best_k    = 0;

                    for (int k = 0; k < K; ++k) {
                        float dist = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            float diff = pts[i * D + d] - ctrs[k * D + d];
                            dist += diff * diff;  // squared dist — no sqrt needed
                        }
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k    = k;
                        }
                    }
                    lbl[i] = best_k;
                });
        });

        // ── Reset accumulators ────────────────────────────────────────────────
        queue_.submit([&](sycl::handler& h) {
            auto sums   = buf_new_sums.get_access<sycl::access::mode::write>(h);
            auto counts = buf_counts.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(K * D), [=](sycl::id<1> idx) {
                sums[idx] = 0.0f;
                if (idx < (size_t)K) counts[idx] = 0;
            });
        });

        // ── Kernel 2: Update ─────────────────────────────────────────────────
        // Each work-item atomically accumulates its point's coordinates.
        // Uses sycl::atomic_ref<float> — SYCL 2020 native float atomics.
        // This is cleaner than the CAS loop required in OpenCL 1.2.
        queue_.submit([&](sycl::handler& h) {
            auto pts    = buf_points.get_access<sycl::access::mode::read>(h);
            auto lbl    = buf_labels.get_access<sycl::access::mode::read>(h);
            auto sums   = buf_new_sums.get_access<sycl::access::mode::read_write>(h);
            auto counts = buf_counts.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for(sycl::nd_range<1>(global_update, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int i = item.get_global_id(0);
                    if (i >= N) return;

                    int k = lbl[i];

                    // Atomic increment of count for this cluster
                    sycl::atomic_ref<int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        atomic_count(counts[k]);
                    atomic_count.fetch_add(1);

                    // Atomic accumulation of coordinates
                    for (int d = 0; d < D; ++d) {
                        sycl::atomic_ref<float,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_sum(sums[k * D + d]);
                        atomic_sum.fetch_add(pts[i * D + d]);
                    }
                });
        });

        // ── Read back new sums + counts for convergence check ─────────────────
        // Only K×D×4 bytes transferred per iteration — negligible overhead.
        {
            auto h_sums   = buf_new_sums.get_host_access();
            auto h_counts = buf_counts.get_host_access();
            queue_.wait();

            for (int k = 0; k < K; ++k) {
                counts_host[k] = h_counts[k];
                for (int d = 0; d < D; ++d) {
                    if (h_counts[k] > 0) {
                        new_centroids[k * D + d] = h_sums[k * D + d]
                                                 / static_cast<float>(h_counts[k]);
                    } else {
                        // Empty cluster: keep old centroid
                        new_centroids[k * D + d] = centroids[k * D + d];
                    }
                }
            }
        }

        // ── Convergence check (CPU) ───────────────────────────────────────────
        float max_shift = 0.0f;
        for (int k = 0; k < K; ++k) {
            float shift = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = new_centroids[k * D + d] - centroids[k * D + d];
                shift += diff * diff;
            }
            max_shift = std::max(max_shift, std::sqrt(shift));
        }

        centroids = new_centroids;

        // Upload updated centroids to GPU buffer for next iteration
        {
            auto h_ctrs = buf_centroids.get_host_access();
            for (int i = 0; i < K * D; ++i) h_ctrs[i] = centroids[i];
        }

        if (max_shift < tol_) break;
    }

    // ── STOP timing ───────────────────────────────────────────────────────────
    queue_.wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    // Read final labels
    std::vector<int> labels(N);
    {
        auto h_lbl = buf_labels.get_host_access();
        for (int i = 0; i < N; ++i) labels[i] = h_lbl[i];
    }

    float inertia = compute_inertia(dataset, labels, centroids);

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
void KMeansSYCL::print_device_info()
{
    std::cout << "=== SYCL Devices ===\n";
    for (const auto& platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>() << "\n";
        for (const auto& dev : platform.get_devices()) {
            auto type = dev.get_info<sycl::info::device::device_type>();
            std::string type_str =
                (type == sycl::info::device_type::gpu)    ? "GPU" :
                (type == sycl::info::device_type::cpu)    ? "CPU" :
                (type == sycl::info::device_type::accelerator) ? "ACC" : "OTHER";
            std::cout << "  [" << type_str << "] "
                      << dev.get_info<sycl::info::device::name>()
                      << " | " << dev.get_info<sycl::info::device::vendor>()
                      << " | CUs=" << dev.get_info<sycl::info::device::max_compute_units>()
                      << " | " << dev.get_info<sycl::info::device::max_clock_frequency>() << " MHz"
                      << " | " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024)
                      << " MB\n";
        }
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  I/O
// ══════════════════════════════════════════════════════════════════════════════
void sycl_io::save_benchmark_row(const std::string& filepath,
                                  const std::string& platform,
                                  int n_samples, int n_features, int k,
                                  double time_ms, int n_iter, float inertia)
{
    bool exists = std::ifstream(filepath).good();
    std::ofstream f(filepath, std::ios::app);
    if (!exists)
        f << "Platform,N_Samples,D_Features,K_Clusters,"
             "Time_Seconds,Iterations,Inertia\n";
    f << platform << "," << n_samples << "," << n_features << "," << k << ","
      << (time_ms / 1000.0) << "," << n_iter << "," << inertia << "\n";
}
