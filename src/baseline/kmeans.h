#pragma once
/**
 * kmeans.h
 * --------
 * Implementare manuala K-Means in C++ pur (varianta secventiala).
 * Fara dependinte externe — doar STL standard.
 *
 * Algoritm: Lloyd's Algorithm
 *   1. Initializare centroizi (K-Means++ sau random)
 *   2. Assignment: fiecare punct → centroidul cel mai apropiat
 *   3. Update: recalculare centroizi ca medie a clusterelor
 *   4. Convergenta: oprire daca deplasarea maxima < tol sau max_iter atins
 */

#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <chrono>


// ── Tipuri de date ─────────────────────────────────────────────────────────────
using float32_t = float;

/**
 * Structura pentru date de intrare: matrice N x D stocata row-major.
 * data[i * n_features + j] = valoarea punctului i pe dimensiunea j.
 */
struct Dataset {
    std::vector<float32_t> data;   // matrice plata (N * D)
    int n_samples;                 // N — numarul de puncte
    int n_features;                // D — numarul de dimensiuni

    float32_t at(int i, int j) const { return data[i * n_features + j]; }
};

/**
 * Rezultatul unei rulari K-Means.
 */
struct KMeansResult {
    std::vector<int>        labels;    // eticheta clusterului pentru fiecare punct
    std::vector<float32_t>  centers;   // centroizi: K x D (row-major)
    float                   inertia;   // WCSS — Within-Cluster Sum of Squares
    int                     n_iter;    // iteratii pana la convergenta
    double                  time_ms;   // timp de executie in millisecunde (fara I/O)
};


// ── Clasa principala ───────────────────────────────────────────────────────────
class KMeans {
public:
    /**
     * Constructor.
     *
     * @param k           Numarul de clustere
     * @param max_iter    Iteratii maxime
     * @param tol         Prag convergenta (deplasare maxima centroid)
     * @param init        Metoda initializare: "kmeans++" sau "random"
     * @param random_state Seed reproductibil
     */
    KMeans(int k,
           int max_iter       = 300,
           float32_t tol      = 1e-4f,
           std::string init   = "kmeans++",
           int random_state   = 42);

    /**
     * Antreneaza modelul pe datele X.
     *
     * @param dataset  Datele de intrare (N x D)
     * @return         KMeansResult cu toate metricile
     */
    KMeansResult fit(const Dataset& dataset);

    // ── Getteri (valabili dupa fit) ─────────────────────────────────────────
    const std::vector<int>&       labels()  const { return result_.labels;  }
    const std::vector<float32_t>& centers() const { return result_.centers; }
    float                         inertia() const { return result_.inertia; }
    int                           n_iter()  const { return result_.n_iter;  }

private:
    // Hiperparametri
    int         k_;
    int         max_iter_;
    float32_t   tol_;
    std::string init_;
    int         random_state_;

    // Stare interna (dupa fit)
    KMeansResult result_;

    // ── Metode private ──────────────────────────────────────────────────────
    void init_random(const Dataset& ds, std::mt19937& rng);
    void init_kmeans_plus_plus(const Dataset& ds, std::mt19937& rng);

    /**
     * Distanta euclidiana la patrat intre punctul i si centroidul k.
     * Lucram cu distanta la patrat pentru a evita sqrt (mai rapid).
     */
    float32_t sq_dist(const Dataset& ds, int point_idx,
                      const std::vector<float32_t>& centers, int center_idx) const;

    /**
     * Pasul de assignment: atribuie fiecare punct celui mai apropiat centroid.
     * @return true daca s-a schimbat cel putin o eticheta
     */
    bool assignment_step(const Dataset& ds,
                         const std::vector<float32_t>& centers,
                         std::vector<int>& labels) const;

    /**
     * Pasul de update: recalculeaza centroizii ca medie a clusterelor.
     * @return deplasarea maxima a unui centroid (pentru criteriul de convergenta)
     */
    float32_t update_step(const Dataset& ds,
                          const std::vector<int>& labels,
                          std::vector<float32_t>& centers) const;

    /**
     * Calculeaza inertia (WCSS) finala.
     */
    float compute_inertia(const Dataset& ds,
                          const std::vector<int>& labels,
                          const std::vector<float32_t>& centers) const;
};


// ── Utilitare I/O ──────────────────────────────────────────────────────────────
namespace io {
    /**
     * Incarca un CSV fara header (format generat de data_generator.py).
     * Fiecare rand = un punct; valorile separate prin virgula.
     */
    Dataset load_csv(const std::string& filepath);

    /**
     * Salveaza etichetele clusterelor intr-un CSV (un label per rand).
     */
    void save_labels(const std::string& filepath, const std::vector<int>& labels);

    /**
     * Salveaza metrici de benchmark intr-un CSV (append sau creare).
     */
    void save_benchmark_row(const std::string& filepath,
                            const std::string& platform,
                            int n_samples, int n_features, int k,
                            double time_ms, int n_iter, float inertia);
}
