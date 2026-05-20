// ============================================================================
// KMeans-GPU-Benchmark: HIP/GPU Variant (AMD GPU Acceleration)
// Offloads the embarrassingly parallel assignment phase to the GPU using
// HIP (Heterogeneous-Compute Interface for Portability), AMD's CUDA equivalent.
// ============================================================================

#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>

using namespace std;

// ============================================================================
// MACRO: HIP ERROR CHECKING
// Every HIP API call can fail silently. This macro wraps each call and
// immediately terminates with a descriptive error if something goes wrong.
// ============================================================================
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            cerr << "HIP Error: " << hipGetErrorString(err)                    \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// KERNEL 1: ASSIGNMENT PHASE (EXPECTATION STEP) — GPU PARALLEL
// Each GPU thread handles ONE point. It computes the squared Euclidean
// distance to ALL K centroids and assigns the nearest one.
// Thread mapping: thread (blockIdx.x * blockDim.x + threadIdx.x) -> point[i]
// Memory layout: flat 1D arrays for coalesced memory access.
// ============================================================================
__global__ void assignClustersKernel(
    const double *data, const double *centroids,
    int *assignments, int *changedFlag,
    int numPoints, int numFeatures, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    double minDistance = 1e308;
    int bestCluster = -1;

    for (int j = 0; j < k; ++j)
    {
        double dist = 0.0;
        for (int d = 0; d < numFeatures; ++d)
        {
            double diff = data[i * numFeatures + d] - centroids[j * numFeatures + d];
            dist += diff * diff;
        }
        if (dist < minDistance)
        {
            minDistance = dist;
            bestCluster = j;
        }
    }

    if (assignments[i] != bestCluster)
    {
        assignments[i] = bestCluster;
        atomicExch(changedFlag, 1);
    }
}

// ============================================================================
// KERNEL 2: CENTROID ACCUMULATION (MAXIMIZATION STEP) — GPU PARALLEL
// Each thread handles ONE point: adds coordinates to its cluster's accumulator
// using atomicAdd to prevent race conditions.
// ============================================================================
__global__ void accumulateCentroidsKernel(
    const double *data, const int *assignments,
    double *centroidSums, int *counts,
    int numPoints, int numFeatures)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    int cluster = assignments[i];
    atomicAdd(&counts[cluster], 1);

    for (int d = 0; d < numFeatures; ++d)
    {
        atomicAdd(&centroidSums[cluster * numFeatures + d], data[i * numFeatures + d]);
    }
}

// Helper function to display the number of points in each cluster at the end
int counts_helper(const vector<int> &assignments, int targetCluster)
{
    int count = 0;
    for (int cluster : assignments)
    {
        if (cluster == targetCluster) count++;
    }
    return count;
}

int main()
{
    // 1. Parse custom header (Points, Dimensions, Clusters, Iterations, Name)
    int numPoints, numFeatures, k, maxIterations, hasName;
    if (!(cin >> numPoints >> numFeatures >> k >> maxIterations >> hasName))
    {
        cerr << "Error reading dataset header." << endl;
        return 1;
    }

    // ============================================================================
    // DATA-ORIENTED DESIGN (DOD) — FLAT ARRAYS
    // On the GPU, we MUST use flat 1D arrays. The GPU has no concept of
    // vector<vector<double>>. Flat arrays also enable "Coalesced Memory Access":
    // adjacent threads read adjacent memory addresses, maximizing bandwidth.
    // ============================================================================
    vector<double> data(numPoints * numFeatures);
    vector<string> names(numPoints);

    for (int i = 0; i < numPoints; ++i)
    {
        for (int j = 0; j < numFeatures; ++j)
            cin >> data[i * numFeatures + j];
        if (hasName == 1) cin >> names[i];
    }

    // ============================================================================
    // 3. CENTROID INITIALIZATION (Random / Forgy Method)
    // ============================================================================
    vector<double> centroids(k * numFeatures);
    vector<int> clusterAssignments(numPoints, -1);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numPoints - 1);

    for (int i = 0; i < k; ++i)
    {
        int idx = dis(gen);
        for (int d = 0; d < numFeatures; ++d)
            centroids[i * numFeatures + d] = data[idx * numFeatures + d];
    }

    // ============================================================================
    // 4. GPU MEMORY ALLOCATION
    // ============================================================================
    double *d_data, *d_centroids, *d_centroidSums;
    int *d_assignments, *d_counts, *d_changedFlag;

    HIP_CHECK(hipMalloc(&d_data, numPoints * numFeatures * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_centroids, k * numFeatures * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_centroidSums, k * numFeatures * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_assignments, numPoints * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_counts, k * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_changedFlag, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_data, data.data(),
                        numPoints * numFeatures * sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_assignments, clusterAssignments.data(),
                        numPoints * sizeof(int), hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    bool changed = true;
    int iter = 0;

    cout << "Starting K-Means clustering on " << numPoints << " points..." << endl;

    // --- START BENCHMARK TIMER ---
    auto start_time = chrono::high_resolution_clock::now();

    // ============================================================================
    // 5. INERTIA MINIMIZATION LOOP (Lloyd's Algorithm — GPU-Accelerated)
    // ============================================================================
    while (changed && iter < maxIterations)
    {
        iter++;

        HIP_CHECK(hipMemcpy(d_centroids, centroids.data(),
                            k * numFeatures * sizeof(double),
                            hipMemcpyHostToDevice));

        int zero = 0;
        HIP_CHECK(hipMemcpy(d_changedFlag, &zero, sizeof(int),
                            hipMemcpyHostToDevice));

        // STEP A: ASSIGNMENT PHASE — LAUNCH GPU KERNEL
        hipLaunchKernelGGL(assignClustersKernel,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           d_data, d_centroids, d_assignments, d_changedFlag,
                           numPoints, numFeatures, k);
        HIP_CHECK(hipGetLastError());

        int h_changed = 0;
        HIP_CHECK(hipMemcpy(&h_changed, d_changedFlag, sizeof(int),
                            hipMemcpyDeviceToHost));
        changed = (h_changed != 0);
        if (!changed) break;

        // STEP B: UPDATE PHASE — ACCUMULATE ON GPU, DIVIDE ON CPU
        HIP_CHECK(hipMemset(d_centroidSums, 0, k * numFeatures * sizeof(double)));
        HIP_CHECK(hipMemset(d_counts, 0, k * sizeof(int)));

        hipLaunchKernelGGL(accumulateCentroidsKernel,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           d_data, d_assignments, d_centroidSums, d_counts,
                           numPoints, numFeatures);
        HIP_CHECK(hipGetLastError());

        vector<double> centroidSums(k * numFeatures);
        vector<int> counts(k, 0);

        HIP_CHECK(hipMemcpy(centroidSums.data(), d_centroidSums,
                            k * numFeatures * sizeof(double),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(counts.data(), d_counts,
                            k * sizeof(int), hipMemcpyDeviceToHost));

        for (int j = 0; j < k; ++j)
        {
            if (counts[j] > 0)
            {
                for (int d = 0; d < numFeatures; ++d)
                    centroids[j * numFeatures + d] = centroidSums[j * numFeatures + d] / counts[j];
            }
        }
    }

    HIP_CHECK(hipDeviceSynchronize());

    // --- STOP BENCHMARK TIMER ---
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end_time - start_time;

    HIP_CHECK(hipMemcpy(clusterAssignments.data(), d_assignments,
                        numPoints * sizeof(int), hipMemcpyDeviceToHost));

    // 6. Display results
    cout << "------------------------------------------------" << endl;
    cout << "Convergence reached in " << iter << " iterations." << endl;
    cout << "Execution Time (Core Loop): " << duration.count() << " ms" << endl;
    cout << "------------------------------------------------" << endl;

    for (int j = 0; j < k; ++j)
    {
        cout << "Centroid " << j << ": ";
        for (int d = 0; d < numFeatures; ++d)
            cout << centroids[j * numFeatures + d] << " ";
        cout << "(Points: " << counts_helper(clusterAssignments, j) << ")" << endl;
    }

    // 7. Free GPU memory
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_centroids));
    HIP_CHECK(hipFree(d_centroidSums));
    HIP_CHECK(hipFree(d_assignments));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_changedFlag));

    return 0;
}
