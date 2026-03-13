/*
 * update.cl
 * ---------
 * OpenCL Kernel 2: Update Step (V1 — atomic adds)
 * Recomputes centroids as the mean of all points assigned to each cluster.
 *
 * Architecture: 1 work-item = 1 data point
 * Strategy: Each work-item atomically accumulates its coordinates into
 *           the new_centroids buffer and increments the cluster count.
 *           Host then divides sums by counts to get the new means.
 *
 * Note: OpenCL 1.2 only supports atomic_add on int/uint, not float.
 * Workaround: We accumulate as scaled integers (multiply by SCALE factor)
 * to avoid floating point atomics, then divide on the host side.
 * Alternatively, we use the atom_add extension for floats if available.
 *
 * Equivalent to CUDA: __global__ void update_kernel(...) with atomicAdd
 */

// Enable 64-bit atomics if available (for higher precision accumulation)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void update(
    __global const float* points,       // [N * D] input points
    __global const int*   labels,       // [N]     cluster assignment per point
    __global float*       new_sums,     // [K * D] accumulator (zeroed before call)
    __global int*         counts,       // [K]     point count per cluster (zeroed before call)
    const int N,
    const int D,
    const int K
) {
    int i = get_global_id(0);
    if (i >= N) return;

    int k = labels[i];

    // Atomic increment of count for this cluster
    atomic_add(&counts[k], 1);

    // Accumulate coordinates — use local accumulation to reduce atomic pressure
    for (int d = 0; d < D; d++) {
        // OpenCL does not have native float atomics in 1.2.
        // We use a compare-and-swap (CAS) loop to simulate atomic float add.
        __global float* addr = &new_sums[k * D + d];
        float val = points[i * D + d];

        // CAS loop for atomic float add (portable across AMD/Intel/NVIDIA OpenCL)
        volatile float old_val, new_val;
        do {
            old_val = *addr;
            new_val = old_val + val;
        } while (atomic_cmpxchg(
            (__global volatile int*)addr,
            as_int(old_val),
            as_int(new_val)
        ) != as_int(old_val));
    }
}

/*
 * Kernel: divide_counts
 * ---------------------
 * Divides accumulated sums by cluster counts to get final centroid positions.
 * Launched with K * D work-items (one per centroid coordinate).
 */
__kernel void divide_counts(
    __global float*     new_centroids,  // [K * D] in/out: sums → means
    __global const int* counts,         // [K]     point count per cluster
    const int K,
    const int D
) {
    int idx = get_global_id(0);  // k * D + d
    if (idx >= K * D) return;

    int k = idx / D;
    if (counts[k] > 0) {
        new_centroids[idx] /= (float)counts[k];
    }
    // If counts[k] == 0 (empty cluster), centroid stays unchanged (handled on host)
}
