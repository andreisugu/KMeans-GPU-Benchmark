/*
 * assignment.cl
 * -------------
 * OpenCL Kernel 1: Assignment Step
 * Maps each point to its nearest centroid (squared Euclidean distance).
 *
 * Architecture: 1 work-item = 1 data point
 * Memory:
 *   - points   → Global Memory (read-only, coalesced access)
 *   - centroids → Constant Memory (__constant) — broadcast to all work-items
 *   - labels   → Global Memory (write-only, no race conditions)
 *
 * Equivalent to CUDA: __global__ void assignment_kernel(...)
 */

__kernel void assignment(
    __global  const float* points,      // [N * D] row-major
    __constant const float* centroids,  // [K * D] row-major — cached in constant mem
    __global  int*          labels,     // [N]     output: cluster id per point
    const int N,
    const int D,
    const int K
) {
    int i = get_global_id(0);   // point index
    if (i >= N) return;

    float best_dist = FLT_MAX;
    int   best_k    = 0;

    // Iterate over all centroids — centroids fit in constant cache
    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = points[i * D + d] - centroids[k * D + d];
            dist += diff * diff;   // squared Euclidean — no sqrt needed
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k    = k;
        }
    }

    labels[i] = best_k;
}
