/**
 * main_opencl.cpp
 * ---------------
 * C++ OpenCL benchmark entry point for AMD iGPU (Radeon 780M).
 * Mirrors src/baseline/main.cpp but uses OpenCL kernels instead of CPU.
 *
 * Usage:
 *   ./kmeans_igpu --all
 *   ./kmeans_igpu --n 100000 --d 16 --k 10
 *   ./kmeans_igpu --list-devices
 *   ./kmeans_igpu --csv ../../data/generated/Medium.csv --k 10
 */

#include "kmeans_opencl.h"
#include "../baseline/kmeans.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <numeric>


// ── Benchmark scenarios (identical to baseline) ───────────────────────────────
struct Scenario {
    std::string name;
    int n_samples, n_features, k;
    std::string desc;
};

static const std::vector<Scenario> SCENARIOS = {
    {"Small",    10'000,    2,   5,  "Correctness validation"},
    {"Medium",   100'000,   16,  10, "Standard workload"},
    {"Large",    1'000'000, 64,  20, "CPU stress test"},
    {"High-Dim", 100'000,   512, 10, "Memory bandwidth test"},
};


// ── Synthetic data generation (identical to baseline/main.cpp) ────────────────
Dataset generate_synthetic(int n_samples, int n_features, int k,
                            int random_state = 42)
{
    std::mt19937 rng(static_cast<unsigned>(random_state));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> cluster_pick(0, k - 1);
    std::uniform_real_distribution<float> center_dist(-10.0f, 10.0f);

    std::vector<std::vector<float>> centers(k, std::vector<float>(n_features));
    for (int ki = 0; ki < k; ++ki)
        for (int d = 0; d < n_features; ++d)
            centers[ki][d] = center_dist(rng);

    Dataset ds;
    ds.n_samples  = n_samples;
    ds.n_features = n_features;
    ds.data.resize((size_t)n_samples * n_features);

    for (int i = 0; i < n_samples; ++i) {
        int ki = cluster_pick(rng);
        for (int d = 0; d < n_features; ++d)
            ds.data[i * n_features + d] = centers[ki][d] + normal(rng);
    }
    return ds;
}


// ── Print result ──────────────────────────────────────────────────────────────
void print_result(const std::string& name, const KMeansResult& res,
                  int n, int d, int k)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[OpenCL-iGPU | " << name << "] "
              << "N=" << n << " D=" << d << " K=" << k
              << " | Time=" << res.time_ms << "ms"
              << " | Iter=" << res.n_iter
              << " | Inertia=" << res.inertia << "\n";
}


// ── Run one scenario ──────────────────────────────────────────────────────────
void run_scenario(const Scenario& sc, const std::string& output_csv,
                  const std::string& kernel_dir, int seed = 42)
{
    std::cout << "\nScenario: " << sc.name << " (" << sc.desc << ")\n";
    std::cout << "  N=" << sc.n_samples << "  D=" << sc.n_features << "  K=" << sc.k << "\n";

    std::cout << "  Generating data...";
    auto t0 = std::chrono::high_resolution_clock::now();
    Dataset ds = generate_synthetic(sc.n_samples, sc.n_features, sc.k, seed);
    double gen_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << " done (" << gen_ms << "ms)\n";

    std::cout << "  Running OpenCL iGPU...";
    KMeansOpenCL model(sc.k, 300, 1e-4f, seed, kernel_dir);
    KMeansResult res = model.fit(ds);
    std::cout << " done!\n";

    print_result(sc.name, res, sc.n_samples, sc.n_features, sc.k);

    igpu_io::save_benchmark_row(output_csv, "AMD-iGPU-780M-OpenCL-Cpp",
                                 sc.n_samples, sc.n_features, sc.k,
                                 res.time_ms, res.n_iter, res.inertia);
}


// ── Argument parsing ──────────────────────────────────────────────────────────
struct Args {
    bool        run_all      = false;
    bool        list_devices = false;
    bool        from_csv     = false;
    std::string csv_path     = "";
    std::string kernel_dir   = "kernels";
    std::string output       = "../../results/results_igpu_cpp.csv";
    int n = 10000, d = 2, k = 5, seed = 42;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--all")          args.run_all      = true;
        else if (a == "--list-devices") args.list_devices = true;
        else if (a == "--n")            args.n            = std::atoi(argv[++i]);
        else if (a == "--d")            args.d            = std::atoi(argv[++i]);
        else if (a == "--k")            args.k            = std::atoi(argv[++i]);
        else if (a == "--seed")         args.seed         = std::atoi(argv[++i]);
        else if (a == "--output")       args.output       = argv[++i];
        else if (a == "--kernel-dir")   args.kernel_dir   = argv[++i];
        else if (a == "--csv") {
            args.from_csv = true;
            args.csv_path = argv[++i];
        }
    }
    return args;
}


// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::cout << "============================================================\n";
    std::cout << " K-Means OpenCL — AMD iGPU (Radeon 780M) Benchmark\n";
    std::cout << "============================================================\n";

    Args args = parse_args(argc, argv);

    if (args.list_devices) {
        KMeansOpenCL::print_device_info();
        return 0;
    }

    try {
        if (args.run_all) {
            for (const auto& sc : SCENARIOS)
                run_scenario(sc, args.output, args.kernel_dir, args.seed);

        } else if (args.from_csv) {
            Dataset ds = io::load_csv(args.csv_path);
            KMeansOpenCL model(args.k, 300, 1e-4f, args.seed, args.kernel_dir);
            KMeansResult res = model.fit(ds);
            print_result("CSV", res, ds.n_samples, ds.n_features, args.k);
            igpu_io::save_benchmark_row(args.output, "AMD-iGPU-780M-OpenCL-Cpp",
                                         ds.n_samples, ds.n_features, args.k,
                                         res.time_ms, res.n_iter, res.inertia);
        } else {
            Scenario sc{"Custom", args.n, args.d, args.k, "manual"};
            run_scenario(sc, args.output, args.kernel_dir, args.seed);
        }

        std::cout << "\nResults saved to: " << args.output << "\n";
        std::cout << "============================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        std::cerr << "Tip: Run with --list-devices to check OpenCL availability.\n";
        return 1;
    }

    return 0;
}
