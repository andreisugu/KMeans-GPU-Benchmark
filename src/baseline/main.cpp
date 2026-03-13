/**
 * main.cpp
 * --------
 * Entry point for sequential C++ K-Means benchmark.
 *
 * Usage:
 *   ./kmeans_seq --n 100000 --d 16 --k 10
 *   ./kmeans_seq --csv ../../data/generated/Medium.csv --k 10
 *   ./kmeans_seq --all                  (run all scenarios)
 *
 * Results are saved to ../../results/results_cpp.csv
 */

#include "kmeans.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>


// ── Benchmark scenarios (equivalent to Python) ───────────────────────────
struct Scenario {
    std::string name;
    int n_samples;
    int n_features;
    int k;
    std::string desc;
};

static const std::vector<Scenario> SCENARIOS = {
    {"Small",    10'000,    2,   5,  "Quick debugging"},
    {"Medium",   100'000,   16,  10, "Moderate data"},
    {"Large",    1'000'000, 64,  20, "CPU limit"},
    {"High-Dim", 100'000,   512, 10, "Memory bandwidth"},
};


// ── Generate synthetic data in C++ (no Python dependency) ────────────────
/**
 * Genereaza N puncte in D dimensiuni cu K clustere gaussiene.
 * Echivalent cu sklearn.datasets.make_blobs (acelasi seed = rezultate similare).
 */
Dataset generate_synthetic(int n_samples, int n_features, int k,
                            int random_state = 42)
{
    std::mt19937 rng(static_cast<unsigned>(random_state));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> cluster_pick(0, k - 1);

    // Generate random cluster centers
    std::vector<std::vector<float>> cluster_centers(k, std::vector<float>(n_features));
    std::uniform_real_distribution<float> center_dist(-10.0f, 10.0f);
    for (int ki = 0; ki < k; ++ki)
        for (int d = 0; d < n_features; ++d)
            cluster_centers[ki][d] = center_dist(rng);

    Dataset ds;
    ds.n_samples  = n_samples;
    ds.n_features = n_features;
    ds.data.resize(static_cast<size_t>(n_samples) * n_features);

    for (int i = 0; i < n_samples; ++i) {
        int ki = cluster_pick(rng);
        for (int d = 0; d < n_features; ++d) {
            ds.data[i * n_features + d] = cluster_centers[ki][d] + normal(rng);
        }
    }
    return ds;
}


// ── Display summary ─────────────────────────────────────────────────────────────
void print_result(const std::string& scenario_name,
                  const KMeansResult& res,
                  int n_samples, int n_features, int k)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[C++-seq | " << scenario_name << "] "
              << "N=" << n_samples << " D=" << n_features << " K=" << k
              << " | Time=" << res.time_ms << "ms"
              << " | Iter=" << res.n_iter
              << " | Inertia=" << res.inertia
              << "\n";
}


// ── Run a single scenario ─────────────────────────────────────────────────
void run_scenario(const Scenario& sc,
                  const std::string& output_csv,
                  int random_state = 42)
{
    std::cout << "\nScenario: " << sc.name << " (" << sc.desc << ")\n";
    std::cout << "  N=" << sc.n_samples
              << "  D=" << sc.n_features
              << "  K=" << sc.k << "\n";

    std::cout << "  Generating data...";
    auto t0 = std::chrono::high_resolution_clock::now();
    Dataset ds = generate_synthetic(sc.n_samples, sc.n_features, sc.k, random_state);
    double gen_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << " done (" << gen_ms << "ms, "
              << (ds.data.size() * 4 / 1e6) << " MB)\n";

    std::cout << "  Running K-Means C++...";
    KMeans model(sc.k, 300, 1e-4f, "kmeans++", random_state);
    KMeansResult res = model.fit(ds);
    std::cout << " done!\n";

    print_result(sc.name, res, sc.n_samples, sc.n_features, sc.k);

    // Save to CSV
    io::save_benchmark_row(output_csv, "C++-sequential",
                           sc.n_samples, sc.n_features, sc.k,
                           res.time_ms, res.n_iter, res.inertia);
}


// ── Parse simple arguments ──────────────────────────────────────────────────
struct Args {
    bool        run_all    = false;
    bool        from_csv   = false;
    std::string csv_path   = "";
    int         n          = 10000;
    int         d          = 2;
    int         k          = 5;
    int         seed       = 42;
    std::string output     = "../../results/results_cpp.csv";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--all")          args.run_all = true;
        else if (a == "--n")       args.n       = std::atoi(argv[++i]);
        else if (a == "--d")       args.d       = std::atoi(argv[++i]);
        else if (a == "--k")       args.k       = std::atoi(argv[++i]);
        else if (a == "--seed")    args.seed    = std::atoi(argv[++i]);
        else if (a == "--output")  args.output  = argv[++i];
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
    std::cout << " K-Means C++ Secvential — Benchmark\n";
    std::cout << "============================================================\n";

    Args args = parse_args(argc, argv);

    if (args.run_all) {
        // Ruleaza toate scenariile
        for (const auto& sc : SCENARIOS) {
            run_scenario(sc, args.output, args.seed);
        }
    } else if (args.from_csv) {
        // Incarca din CSV (generat de Python)
        Dataset ds = io::load_csv(args.csv_path);
        KMeans model(args.k, 300, 1e-4f, "kmeans++", args.seed);
        KMeansResult res = model.fit(ds);
        print_result("CSV", res, ds.n_samples, ds.n_features, args.k);
        io::save_benchmark_row(args.output, "C++-sequential",
                               ds.n_samples, ds.n_features, args.k,
                               res.time_ms, res.n_iter, res.inertia);
    } else {
        // Scenariu manual (--n, --d, --k)
        Scenario sc{"Custom", args.n, args.d, args.k, "scenariu manual"};
        run_scenario(sc, args.output, args.seed);
    }

    std::cout << "\nRezultate salvate in: " << args.output << "\n";
    std::cout << "============================================================\n";
    return 0;
}
