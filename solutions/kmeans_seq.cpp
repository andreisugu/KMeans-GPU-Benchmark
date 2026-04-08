#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <chrono> // Added for benchmarking

using namespace std;

// Helper to calculate Squared Euclidean distance
// We omit the sqrt() because it is computationally expensive and unnecessary 
// for simply finding the *closest* centroid.
inline double calculateSquaredDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Simple helper to count points per cluster for the final output
// Moved above main() so it is declared before being called
int counts_helper(const vector<int>& assignments, int targetCluster) {
    int count = 0;
    for (int cluster : assignments) {
        if (cluster == targetCluster) count++;
    }
    return count;
}

int main() {
    // 1. Parse the specific header format from the dataset
    int numPoints, numFeatures, k, maxIterations, hasName;
    if (!(cin >> numPoints >> numFeatures >> k >> maxIterations >> hasName)) {
        cerr << "Error reading dataset header. Ensure format is: Points Attributes Clusters Max_Iterations Has_Name" << endl;
        return 1;
    }

    // Flat data structures to avoid memory thrashing (Data-Oriented Design)
    vector<vector<double>> data(numPoints, vector<double>(numFeatures));
    vector<string> names(numPoints);

    // 2. Load the data points
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            cin >> data[i][j];
        }
        if (hasName == 1) {
            cin >> names[i];
        }
    }

    // 3. Initialize centroids (randomly selecting K points from the dataset)
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
    auto start_time = chrono::high_resolution_clock::now();

    // 4. Main K-Means Loop
    while (changed && iter < maxIterations) {
        changed = false;
        iter++;

        // Step A: Assign points to the nearest centroid (Expectation)
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

            // Check if the point changed its cluster
            if (clusterAssignments[i] != bestCluster) {
                clusterAssignments[i] = bestCluster;
                changed = true;
            }
        }

        // Step B: Update centroids based on new assignments (Maximization)
        vector<vector<double>> newCentroids(k, vector<double>(numFeatures, 0.0));
        vector<int> counts(k, 0);

        // Sum up all points in each cluster
        for (int i = 0; i < numPoints; ++i) {
            int cluster = clusterAssignments[i];
            counts[cluster]++;
            for (int d = 0; d < numFeatures; ++d) {
                newCentroids[cluster][d] += data[i][d];
            }
        }

        // Divide by the count to get the mean (new centroid position)
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) { // Avoid division by zero if a cluster becomes empty
                for (int d = 0; d < numFeatures; ++d) {
                    centroids[j][d] = newCentroids[j][d] / counts[j];
                }
            }
        }
    }

    // --- STOP BENCHMARK TIMER ---
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end_time - start_time;

    // 5. Output the results
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