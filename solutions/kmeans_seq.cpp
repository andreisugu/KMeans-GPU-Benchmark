// ============================================================================
// KMeans-GPU-Benchmark: Sequential Variant (CPU Baseline)
// Optimized implementation using Data-Oriented Design and Lloyd's algorithm.
// ============================================================================

#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <chrono> // Used for precise benchmarking

using namespace std;

// ============================================================================
// DISTANCE FUNCTION (Performance Optimization)
// Mathematically, we use the Squared Euclidean Distance: d^2 = sum((A_i - B_i)^2)
// We intentionally omitted the sqrt() function because it is very slow on the CPU.
// Since the square root function is strictly increasing, the minimum remains the 
// same (if A^2 < B^2, then A < B). We only care about *which* centroid is closer.
// ============================================================================
inline double calculateSquaredDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Helper function to display the number of points in each cluster at the end
int counts_helper(const vector<int>& assignments, int targetCluster) {
    int count = 0;
    for (int cluster : assignments) {
        if (cluster == targetCluster) count++;
    }
    return count;
}

int main() {
    // 1. Parse custom header (Points, Dimensions, Clusters, Iterations, Name)
    int numPoints, numFeatures, k, maxIterations, hasName;
    if (!(cin >> numPoints >> numFeatures >> k >> maxIterations >> hasName)) {
        cerr << "Error reading dataset header. Ensure format is: Points Attributes Clusters Max_Iterations Has_Name" << endl;
        return 1;
    }

    // ============================================================================
    // DATA-ORIENTED DESIGN (DOD) & CACHE LOCALITY
    // We avoid object-oriented programming (e.g., `Point` classes with push_back/erase).
    // We use 'flat' (contiguous) arrays to keep the processor pipeline fed 
    // (L1/L2 Cache) and avoid memory thrashing (slow dynamic allocations).
    // ============================================================================
    vector<vector<double>> data(numPoints, vector<double>(numFeatures));
    vector<string> names(numPoints);

    // 2. Load points into memory
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            cin >> data[i][j];
        }
        if (hasName == 1) {
            cin >> names[i];
        }
    }

    // ============================================================================
    // 3. CENTROID INITIALIZATION (Random / Forgy Method)
    // We chose the classic random initialization instead of K-Means++ to keep 
    // the benchmark focused on the raw power of the parallel loop, eliminating a 
    // sequential initialization step that is hard to parallelize.
    // ============================================================================
    vector<vector<double>> centroids(k, vector<double>(numFeatures));
    vector<int> clusterAssignments(numPoints, -1);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numPoints - 1);
    
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[dis(gen)];
    }

    bool changed = true;
    int iter = 0;

    cout << "Starting K-Means clustering on " << numPoints << " points..." << endl;

    // --- START BENCHMARK TIMER ---
    // We strictly measure the mathematical algorithm, excluding I/O (disk reading).
    auto start_time = chrono::high_resolution_clock::now();

    // ============================================================================
    // 4. INERTIA MINIMIZATION LOOP (Lloyd's Algorithm)
    // Mathematically, at each step, WCSS (Within-Cluster Sum of Squares) decreases.
    // The algorithm is guaranteed to converge to an equilibrium (changed == false).
    // ============================================================================
    while (changed && iter < maxIterations) {
        changed = false;
        iter++;

        // ============================================================================
        // STEP A: ASSIGNMENT PHASE (EXPECTATION STEP) - COMPUTATIONAL BOTTLENECK
        // Complexity: O(N * K * D) per iteration. Here we partition the space into Voronoi Cells.
        // On CPU it runs sequentially, creating a massive bottleneck (billions of distances).
        // THIS is the "Embarrassingly Parallel" phase that will run in CUDA on the GPU.
        // ============================================================================
        for (int i = 0; i < numPoints; ++i) {
            double minDistance = numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; ++j) {
                double dist = calculateSquaredDistance(data[i], centroids[j]);
                if (dist < minDistance) {
                    minDistance = dist;
                    bestCluster = j;
                }
            }

            // Check if the point migrated to another cluster
            if (clusterAssignments[i] != bestCluster) {
                clusterAssignments[i] = bestCluster;
                changed = true;
            }
        }

        // ============================================================================
        // STEP B: UPDATE PHASE (MAXIMIZATION STEP)
        // Recalculate the new center of mass (arithmetic mean) of each cluster.
        // When moving to the GPU, this step will require Atomic Operations (atomicAdd)
        // to prevent "Race Conditions" (threads overwriting each other's data).
        // ============================================================================
        vector<vector<double>> newCentroids(k, vector<double>(numFeatures, 0.0));
        vector<int> counts(k, 0);

        // Sum up the coordinates of the points in their clusters
        for (int i = 0; i < numPoints; ++i) {
            int cluster = clusterAssignments[i];
            counts[cluster]++;
            for (int d = 0; d < numFeatures; ++d) {
                newCentroids[cluster][d] += data[i][d];
            }
        }

        // Divide by the number of points to find the average position (new centroid)
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) { // Protection against division by zero (empty clusters)
                for (int d = 0; d < numFeatures; ++d) {
                    centroids[j][d] = newCentroids[j][d] / counts[j];
                }
            }
        }
    }

    // --- STOP BENCHMARK TIMER ---
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end_time - start_time;

    // 5. Display results
    cout << "------------------------------------------------" << endl;
    cout << "Convergence reached in " << iter << " iterations." << endl;
    cout << "Execution Time (Core Loop): " << duration.count() << " ms" << endl;
    cout << "------------------------------------------------" << endl;
    
    for (int j = 0; j < k; ++j) {
        cout << "Centroid " << j << ": ";
        for (int d = 0; d < numFeatures; ++d) {
            cout << centroids[j][d] << " ";
        }
        cout << "(Points: " << counts_helper(clusterAssignments, j) << ")" << endl;
    }

    return 0;
}