// ============================================================================
// KMeans-GPU-Benchmark: MPI Variant (Distributed CPU Parallelism)
// Distributes the computational workload across multiple CPU processes using
// MPI (Message Passing Interface) for inter-process communication.
// ============================================================================

#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <chrono> // Used for precise benchmarking
#include <mpi.h>

using namespace std;

// ============================================================================
// DISTANCE FUNCTION (Performance Optimization)
// Mathematically, we use the Squared Euclidean Distance: d^2 = sum((A_i - B_i)^2)
// We intentionally omitted the sqrt() function because it is very slow on the CPU.
// Since the square root function is strictly increasing, the minimum remains the
// same (if A^2 < B^2, then A < B). We only care about *which* centroid is closer.
// ============================================================================
inline double calculateSquaredDistance(const double *a, const double *b, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char *argv[])
{
    // ============================================================================
    // MPI INITIALIZATION
    // MPI_Init sets up the communication infrastructure. Each process gets a
    // unique "rank" (ID) from 0 to P-1, where P is the total number of processes.
    // Rank 0 is the "master" — it handles I/O and distributes work to others.
    // ============================================================================
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int numPoints, numFeatures, k, maxIterations, hasName;
    vector<double> allData;     // Only rank 0 holds the full dataset
    vector<string> names;

    // ============================================================================
    // 1. RANK 0: READ INPUT DATA
    // Only the master process reads from stdin. Broadcasting raw file I/O across
    // all processes would cause contention and is not how MPI programs work.
    // ============================================================================
    if (rank == 0)
    {
        if (!(cin >> numPoints >> numFeatures >> k >> maxIterations >> hasName))
        {
            cerr << "Error reading dataset header. Ensure format is: "
                 << "Points Attributes Clusters Max_Iterations Has_Name" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // ============================================================================
        // DATA-ORIENTED DESIGN (DOD) — FLAT ARRAYS
        // We use a single contiguous 1D array instead of vector<vector<double>>.
        // This is critical for MPI: MPI_Scatterv requires a contiguous memory
        // buffer to slice and distribute across processes.
        //   Layout: allData[i * numFeatures + d] = feature d of point i
        // ============================================================================
        allData.resize(numPoints * numFeatures);
        names.resize(numPoints);

        for (int i = 0; i < numPoints; ++i)
        {
            for (int j = 0; j < numFeatures; ++j)
            {
                cin >> allData[i * numFeatures + j];
            }
            if (hasName == 1)
            {
                cin >> names[i];
            }
        }
    }

    // ============================================================================
    // 2. BROADCAST HEADER PARAMETERS
    // All processes need to know N, D, K, and maxIter to allocate their local
    // buffers. MPI_Bcast sends from rank 0 to all other ranks.
    // ============================================================================
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ============================================================================
    // 3. DATA DECOMPOSITION (Domain Decomposition)
    // We split the N points as evenly as possible across P processes.
    // If N is not perfectly divisible by P, the first (N % P) processes get
    // one extra point each. This ensures load balance within ±1 point.
    //
    // Example: 1,000,000 points on 4 processes → 250,000 each.
    //          1,000,003 points on 4 processes → 250,001, 250,001, 250,001, 250,000
    // ============================================================================
    vector<int> sendCounts(numProcs);
    vector<int> displacements(numProcs);

    int baseCount = numPoints / numProcs;
    int remainder = numPoints % numProcs;

    for (int p = 0; p < numProcs; ++p)
    {
        sendCounts[p] = (baseCount + (p < remainder ? 1 : 0)) * numFeatures;
        displacements[p] = (p == 0) ? 0 : displacements[p - 1] + sendCounts[p - 1];
    }

    int localNumPoints = sendCounts[rank] / numFeatures;
    vector<double> localData(localNumPoints * numFeatures);

    // ============================================================================
    // MPI_Scatterv: DISTRIBUTE DATA CHUNKS
    // Rank 0 sends each process its slice of the data array. Each process
    // receives only its localNumPoints rows (contiguous in memory).
    // This is O(N) communication — done only once at startup.
    // ============================================================================
    MPI_Scatterv(allData.data(), sendCounts.data(), displacements.data(), MPI_DOUBLE,
                 localData.data(), localNumPoints * numFeatures, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // ============================================================================
    // 4. CENTROID INITIALIZATION (Random / Forgy Method)
    // Rank 0 picks K random points, then broadcasts to all processes.
    // All processes must start with identical centroids for correctness.
    // ============================================================================
    vector<double> centroids(k * numFeatures);

    if (rank == 0)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, numPoints - 1);

        for (int i = 0; i < k; ++i)
        {
            int idx = dis(gen);
            for (int d = 0; d < numFeatures; ++d)
            {
                centroids[i * numFeatures + d] = allData[idx * numFeatures + d];
            }
        }
    }

    MPI_Bcast(centroids.data(), k * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int> localAssignments(localNumPoints, -1);
    bool globalChanged = true;
    int iter = 0;

    if (rank == 0)
    {
        cout << "Starting K-Means clustering on " << numPoints
             << " points with " << numProcs << " MPI processes..." << endl;
    }

    // --- START BENCHMARK TIMER ---
    // We strictly measure the mathematical algorithm, excluding I/O (disk reading).
    // MPI_Barrier ensures all processes start the timer at the same instant.
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = chrono::high_resolution_clock::now();

    // ============================================================================
    // 5. INERTIA MINIMIZATION LOOP (Lloyd's Algorithm — MPI-Distributed)
    // Same convergence guarantee as the sequential variant.
    // The key difference: the O(N*K*D) assignment work is split across P processes,
    // each computing only O(N/P * K * D) distances. The update phase uses
    // MPI_Allreduce to combine partial sums into global centroids.
    // ============================================================================
    while (globalChanged && iter < maxIterations)
    {
        int localChanged = 0;
        iter++;

        // ============================================================================
        // STEP A: ASSIGNMENT PHASE (EXPECTATION STEP) — EACH PROCESS HANDLES ITS SLICE
        // Complexity per process: O((N/P) * K * D) — linear speedup with P processes.
        // Each process independently assigns its local points to the nearest centroid.
        // No communication needed during this phase (embarrassingly parallel).
        // ============================================================================
        for (int i = 0; i < localNumPoints; ++i)
        {
            double minDistance = numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; ++j)
            {
                double dist = calculateSquaredDistance(
                    &localData[i * numFeatures],
                    &centroids[j * numFeatures],
                    numFeatures);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestCluster = j;
                }
            }

            if (localAssignments[i] != bestCluster)
            {
                localAssignments[i] = bestCluster;
                localChanged = 1;
            }
        }

        // ============================================================================
        // CONVERGENCE CHECK (Collective Operation)
        // MPI_Allreduce with MPI_LOR (Logical OR): if ANY process detected a change,
        // ALL processes agree to continue iterating. This ensures globally consistent
        // termination — no process can stop early while others are still working.
        // ============================================================================
        int globalChangedInt = 0;
        MPI_Allreduce(&localChanged, &globalChangedInt, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        globalChanged = (globalChangedInt != 0);

        if (!globalChanged)
            break;

        // ============================================================================
        // STEP B: UPDATE PHASE (MAXIMIZATION STEP) — DISTRIBUTED REDUCTION
        // Each process computes PARTIAL sums for the clusters it sees locally.
        // Then MPI_Allreduce(MPI_SUM) combines all partial sums into the global sum.
        // This is O(K*D) communication — very small compared to the O(N*D) computation.
        // ============================================================================
        vector<double> localCentroidSums(k * numFeatures, 0.0);
        vector<int> localCounts(k, 0);

        // Each process accumulates its local points
        for (int i = 0; i < localNumPoints; ++i)
        {
            int cluster = localAssignments[i];
            localCounts[cluster]++;
            for (int d = 0; d < numFeatures; ++d)
            {
                localCentroidSums[cluster * numFeatures + d] += localData[i * numFeatures + d];
            }
        }

        // ============================================================================
        // MPI_Allreduce: GLOBAL SUM
        // Combines partial sums from all processes. After this call, every process
        // has the identical global centroid sums and counts — no master needed.
        // This is far more efficient than Gather→Compute→Broadcast on rank 0.
        // ============================================================================
        vector<double> globalCentroidSums(k * numFeatures, 0.0);
        vector<int> globalCounts(k, 0);

        MPI_Allreduce(localCentroidSums.data(), globalCentroidSums.data(),
                      k * numFeatures, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localCounts.data(), globalCounts.data(),
                      k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Divide by count to get new centroid positions (identical on all processes)
        for (int j = 0; j < k; ++j)
        {
            if (globalCounts[j] > 0)
            { // Protection against division by zero (empty clusters)
                for (int d = 0; d < numFeatures; ++d)
                {
                    centroids[j * numFeatures + d] = globalCentroidSums[j * numFeatures + d] / globalCounts[j];
                }
            }
        }
    }

    // --- STOP BENCHMARK TIMER ---
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end_time - start_time;

    // ============================================================================
    // 6. GATHER RESULTS (Only for display)
    // Rank 0 collects all assignments to compute per-cluster point counts.
    // ============================================================================
    if (rank == 0)
    {
        // Prepare receive counts (in number of ints, not doubles)
        vector<int> recvCounts(numProcs);
        vector<int> recvDisplacements(numProcs);
        for (int p = 0; p < numProcs; ++p)
        {
            recvCounts[p] = sendCounts[p] / numFeatures; // number of points
            recvDisplacements[p] = (p == 0) ? 0 : recvDisplacements[p - 1] + recvCounts[p - 1];
        }

        vector<int> allAssignments(numPoints);
        MPI_Gatherv(localAssignments.data(), localNumPoints, MPI_INT,
                     allAssignments.data(), recvCounts.data(), recvDisplacements.data(),
                     MPI_INT, 0, MPI_COMM_WORLD);

        // Display results (matching sequential output format)
        cout << "------------------------------------------------" << endl;
        cout << "Convergence reached in " << iter << " iterations." << endl;
        cout << "Execution Time (Core Loop): " << duration.count() << " ms" << endl;
        cout << "------------------------------------------------" << endl;

        for (int j = 0; j < k; ++j)
        {
            cout << "Centroid " << j << ": ";
            for (int d = 0; d < numFeatures; ++d)
            {
                cout << centroids[j * numFeatures + d] << " ";
            }
            // Count points in this cluster
            int count = 0;
            for (int a : allAssignments)
            {
                if (a == j) count++;
            }
            cout << "(Points: " << count << ")" << endl;
        }
    }
    else
    {
        // Non-root processes just send their local assignments
        MPI_Gatherv(localAssignments.data(), localNumPoints, MPI_INT,
                     nullptr, nullptr, nullptr,
                     MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
