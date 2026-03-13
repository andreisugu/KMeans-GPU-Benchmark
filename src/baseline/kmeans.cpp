/**
 * kmeans.cpp
 * ----------
 * Implementare manuala K-Means secvential in C++ pur.
 * Algoritmul Lloyd — fara dependinte externe, fara SIMD, fara threading.
 * Acesta este baseline-ul "cel mai lent" — va fi comparat cu CUDA.
 */

#include "kmeans.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <limits>


// ══════════════════════════════════════════════════════════════════════════════
//  Constructor
// ══════════════════════════════════════════════════════════════════════════════
KMeans::KMeans(int k, int max_iter, float32_t tol,
               std::string init, int random_state)
    : k_(k), max_iter_(max_iter), tol_(tol),
      init_(std::move(init)), random_state_(random_state)
{
    if (k_ <= 0)       throw std::invalid_argument("k trebuie sa fie > 0");
    if (max_iter_ <= 0) throw std::invalid_argument("max_iter trebuie sa fie > 0");
    if (init_ != "kmeans++" && init_ != "random")
        throw std::invalid_argument("init trebuie sa fie 'kmeans++' sau 'random'");
}


// ══════════════════════════════════════════════════════════════════════════════
//  Distanta euclidiana la patrat
// ══════════════════════════════════════════════════════════════════════════════
float32_t KMeans::sq_dist(const Dataset& ds, int point_idx,
                           const std::vector<float32_t>& centers,
                           int center_idx) const
{
    float32_t dist = 0.0f;
    const int D = ds.n_features;
    const float32_t* p = &ds.data[point_idx * D];
    const float32_t* c = &centers[center_idx * D];

    for (int d = 0; d < D; ++d) {
        float32_t diff = p[d] - c[d];
        dist += diff * diff;
    }
    return dist;
}


// ══════════════════════════════════════════════════════════════════════════════
//  Initializare — Random
// ══════════════════════════════════════════════════════════════════════════════
void KMeans::init_random(const Dataset& ds, std::mt19937& rng)
{
    const int N = ds.n_samples;
    const int D = ds.n_features;

    result_.centers.resize(k_ * D);

    // Alegem k indici unici din [0, N)
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int ki = 0; ki < k_; ++ki) {
        int idx = indices[ki];
        std::copy(&ds.data[idx * D],
                  &ds.data[idx * D + D],
                  &result_.centers[ki * D]);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Initializare — K-Means++
//  Algoritm:
//    1. Alege primul centroid uniform la intamplare
//    2. Calculeaza distanta D(x)^2 de la fiecare punct la cel mai apropiat centroid
//    3. Alege urmatorul centroid cu probabilitate proportionala cu D(x)^2
//    4. Repeta pana la k centroizi
// ══════════════════════════════════════════════════════════════════════════════
void KMeans::init_kmeans_plus_plus(const Dataset& ds, std::mt19937& rng)
{
    const int N = ds.n_samples;
    const int D = ds.n_features;

    result_.centers.resize(k_ * D);

    // 1. Primul centroid — uniform random
    std::uniform_int_distribution<int> uniform(0, N - 1);
    int first_idx = uniform(rng);
    std::copy(&ds.data[first_idx * D],
              &ds.data[first_idx * D + D],
              &result_.centers[0]);

    // Distante minime la centroizii alesi pana acum
    std::vector<float32_t> min_dists(N, std::numeric_limits<float32_t>::max());

    for (int ki = 1; ki < k_; ++ki) {
        // Actualizeaza distantele fata de ultimul centroid adaugat
        for (int i = 0; i < N; ++i) {
            float32_t d = sq_dist(ds, i, result_.centers, ki - 1);
            if (d < min_dists[i]) min_dists[i] = d;
        }

        // Selectie proportionala cu D(x)^2
        float32_t total = std::accumulate(min_dists.begin(), min_dists.end(), 0.0f);
        std::uniform_real_distribution<float32_t> dist_real(0.0f, total);
        float32_t threshold = dist_real(rng);

        float32_t cumsum = 0.0f;
        int chosen = N - 1;  // fallback
        for (int i = 0; i < N; ++i) {
            cumsum += min_dists[i];
            if (cumsum >= threshold) {
                chosen = i;
                break;
            }
        }

        std::copy(&ds.data[chosen * D],
                  &ds.data[chosen * D + D],
                  &result_.centers[ki * D]);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Assignment step
//  Pentru fiecare punct: cauta centroidul cel mai apropiat (distanta^2 minima)
// ══════════════════════════════════════════════════════════════════════════════
bool KMeans::assignment_step(const Dataset& ds,
                              const std::vector<float32_t>& centers,
                              std::vector<int>& labels) const
{
    const int N = ds.n_samples;
    bool changed = false;

    for (int i = 0; i < N; ++i) {
        float32_t best_dist = std::numeric_limits<float32_t>::max();
        int best_k = 0;

        for (int ki = 0; ki < k_; ++ki) {
            float32_t d = sq_dist(ds, i, centers, ki);
            if (d < best_dist) {
                best_dist = d;
                best_k    = ki;
            }
        }

        if (labels[i] != best_k) {
            labels[i] = best_k;
            changed = true;
        }
    }
    return changed;
}


// ══════════════════════════════════════════════════════════════════════════════
//  Update step
//  Recalculeaza fiecare centroid ca medie aritmetica a punctelor din cluster.
//  Returneaza deplasarea maxima (pentru convergenta).
// ══════════════════════════════════════════════════════════════════════════════
float32_t KMeans::update_step(const Dataset& ds,
                               const std::vector<int>& labels,
                               std::vector<float32_t>& centers) const
{
    const int N = ds.n_samples;
    const int D = ds.n_features;

    std::vector<float32_t> new_centers(k_ * D, 0.0f);
    std::vector<int>       counts(k_, 0);

    // Acumulare sume per cluster
    for (int i = 0; i < N; ++i) {
        int ki = labels[i];
        counts[ki]++;
        for (int d = 0; d < D; ++d) {
            new_centers[ki * D + d] += ds.data[i * D + d];
        }
    }

    // Impartire la numar de puncte => medie
    for (int ki = 0; ki < k_; ++ki) {
        if (counts[ki] > 0) {
            float32_t inv = 1.0f / static_cast<float32_t>(counts[ki]);
            for (int d = 0; d < D; ++d) {
                new_centers[ki * D + d] *= inv;
            }
        }
        // Daca un cluster e gol, pastram centroidul vechi (comportament standard)
    }

    // Calculam deplasarea maxima
    float32_t max_shift = 0.0f;
    for (int ki = 0; ki < k_; ++ki) {
        float32_t shift = 0.0f;
        for (int d = 0; d < D; ++d) {
            float32_t diff = new_centers[ki * D + d] - centers[ki * D + d];
            shift += diff * diff;
        }
        max_shift = std::max(max_shift, std::sqrt(shift));
    }

    centers = std::move(new_centers);
    return max_shift;
}


// ══════════════════════════════════════════════════════════════════════════════
//  Inertia (WCSS)
// ══════════════════════════════════════════════════════════════════════════════
float KMeans::compute_inertia(const Dataset& ds,
                               const std::vector<int>& labels,
                               const std::vector<float32_t>& centers) const
{
    double inertia = 0.0;
    for (int i = 0; i < ds.n_samples; ++i) {
        inertia += sq_dist(ds, i, centers, labels[i]);
    }
    return static_cast<float>(inertia);
}


// ══════════════════════════════════════════════════════════════════════════════
//  fit() — Algoritmul principal Lloyd
// ══════════════════════════════════════════════════════════════════════════════
KMeansResult KMeans::fit(const Dataset& dataset)
{
    const int N = dataset.n_samples;
    const int D = dataset.n_features;

    if (N < k_)
        throw std::invalid_argument("N_samples < K — imposibil de a forma clustere");

    // Initializare RNG
    std::mt19937 rng(static_cast<unsigned>(random_state_));

    // Initializare centroizi
    result_.labels.assign(N, -1);
    if (init_ == "kmeans++") {
        init_kmeans_plus_plus(dataset, rng);
    } else {
        init_random(dataset, rng);
    }

    // ── START cronometru — masuram doar algoritmul Lloyd ─────────────────────
    auto t_start = std::chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 0; iter < max_iter_; ++iter) {
        // Assignment
        bool changed = assignment_step(dataset, result_.centers, result_.labels);
        if (!changed) break;   // convergenta prin etichete stabile

        // Update
        float32_t shift = update_step(dataset, result_.labels, result_.centers);
        if (shift < tol_) break;  // convergenta prin deplasare mica
    }

    // ── STOP cronometru ───────────────────────────────────────────────────────
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Calcul inertia finala
    result_.inertia  = compute_inertia(dataset, result_.labels, result_.centers);
    result_.n_iter   = iter + 1;
    result_.time_ms  = elapsed_ms;

    return result_;
}


// ══════════════════════════════════════════════════════════════════════════════
//  I/O Utilitare
// ══════════════════════════════════════════════════════════════════════════════
Dataset io::load_csv(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Nu pot deschide fisierul: " + filepath);

    Dataset ds;
    ds.n_samples  = 0;
    ds.n_features = 0;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<float32_t> row;

        while (std::getline(ss, token, ',')) {
            row.push_back(std::stof(token));
        }

        if (first_line) {
            ds.n_features = static_cast<int>(row.size());
            first_line = false;
        } else if (static_cast<int>(row.size()) != ds.n_features) {
            throw std::runtime_error("CSV inconsistent: numar diferit de coloane la randul "
                                      + std::to_string(ds.n_samples + 1));
        }

        ds.data.insert(ds.data.end(), row.begin(), row.end());
        ds.n_samples++;
    }

    std::cout << "[io::load_csv] Incarcat " << ds.n_samples
              << " x " << ds.n_features << " din " << filepath << "\n";
    return ds;
}


void io::save_labels(const std::string& filepath, const std::vector<int>& labels)
{
    std::ofstream f(filepath);
    for (int l : labels) f << l << "\n";
    std::cout << "[io::save_labels] Salvat " << labels.size()
              << " etichete in " << filepath << "\n";
}


void io::save_benchmark_row(const std::string& filepath,
                             const std::string& platform,
                             int n_samples, int n_features, int k,
                             double time_ms, int n_iter, float inertia)
{
    bool exists = std::ifstream(filepath).good();
    std::ofstream f(filepath, std::ios::app);

    if (!exists) {
        f << "Platform,N_Samples,D_Features,K_Clusters,"
             "Time_Seconds,Iterations,Inertia\n";
    }

    f << platform << ","
      << n_samples << ","
      << n_features << ","
      << k << ","
      << (time_ms / 1000.0) << ","
      << n_iter << ","
      << inertia << "\n";
}
