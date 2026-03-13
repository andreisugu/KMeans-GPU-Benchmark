"""
kmeans_pyopencl.py
------------------
K-Means on AMD iGPU (Radeon 780M) using PyOpenCL.
Interface identical to kmeans_cpu.py for direct comparison.

Kernels:
  - kernels/assignment.cl : 1 work-item = 1 point → finds nearest centroid
  - kernels/update.cl     : atomic float accumulation → recomputes centroids

Memory strategy (mirrors CUDA implementation):
  - Dataset  (N×D float32) → GPU Global Memory, transferred ONCE
  - Labels   (N int32)     → GPU Global Memory, written by assignment kernel
  - Centroids(K×D float32) → GPU Constant Memory (__constant buffer)
  - New sums (K×D float32) → GPU Global Memory, reset each iteration

Requirements:
  pip install pyopencl numpy

Setup on Linux (AMD iGPU):
  sudo apt install mesa-opencl-icd ocl-icd-opencl-dev
  Check GPU detected: clinfo | grep "Device Name"
"""

import os
import time
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    print("[WARNING] PyOpenCL not installed. Run: pip install pyopencl")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from kmeans_cpu import KMeansResult

# Path to .cl kernel files
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")


def _load_kernel_source(filename: str) -> str:
    path = os.path.join(KERNEL_DIR, filename)
    with open(path, "r") as f:
        return f.read()


def get_amd_igpu_device():
    """
    Auto-detects the AMD iGPU (Radeon 780M) from available OpenCL platforms.
    Falls back to any GPU, then any available device.
    """
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(cl.device_type.GPU)
        for dev in devices:
            name = dev.name.lower()
            if "radeon" in name or "amd" in name or "780m" in name or "gfx" in name:
                print(f"[PyOpenCL] Found AMD GPU: {dev.name} on {platform.name}")
                return dev

    # Fallback: any GPU
    for platform in platforms:
        devices = platform.get_devices(cl.device_type.GPU)
        if devices:
            print(f"[PyOpenCL] Using GPU: {devices[0].name} on {platform.name}")
            return devices[0]

    # Last resort: any device (CPU OpenCL for testing)
    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            print(f"[PyOpenCL] WARNING: Using non-GPU device: {devices[0].name}")
            return devices[0]

    raise RuntimeError("No OpenCL device found. Install mesa-opencl-icd.")


def _init_kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    K-Means++ initialization on CPU (called once before GPU loop).
    Identical logic to C++ baseline for reproducibility.
    """
    N, D = X.shape
    centers = np.empty((k, D), dtype=np.float32)

    # First centroid: random
    centers[0] = X[rng.integers(0, N)]

    min_dists = np.full(N, np.finfo(np.float32).max, dtype=np.float32)

    for ki in range(1, k):
        # Update min distances to last chosen centroid
        diff = X - centers[ki - 1]
        dists = np.sum(diff * diff, axis=1)
        np.minimum(min_dists, dists, out=min_dists)

        # Sample proportional to D(x)^2
        probs = min_dists / min_dists.sum()
        chosen = rng.choice(N, p=probs)
        centers[ki] = X[chosen]

    return centers


def run_kmeans_pyopencl(X: np.ndarray,
                         k: int,
                         max_iter: int = 300,
                         tol: float = 1e-4,
                         random_state: int = 42,
                         device=None) -> KMeansResult:
    """
    Runs K-Means on AMD iGPU via PyOpenCL.

    Parameters
    ----------
    X            : input data (N, D) float32
    k            : number of clusters
    max_iter     : maximum iterations
    tol          : convergence threshold (max centroid shift)
    random_state : reproducible seed
    device       : OpenCL device (auto-detected if None)

    Returns
    -------
    KMeansResult with timing, inertia, iterations
    """
    if not PYOPENCL_AVAILABLE:
        raise RuntimeError("PyOpenCL not available. Run: pip install pyopencl")

    X = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X.shape
    rng  = np.random.default_rng(random_state)

    # ── OpenCL setup ──────────────────────────────────────────────────────────
    if device is None:
        device = get_amd_igpu_device()

    ctx   = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Compile kernels
    src_assignment = _load_kernel_source("assignment.cl")
    src_update     = _load_kernel_source("update.cl")
    program = cl.Program(ctx, src_assignment + "\n" + src_update).build(
        options=["-cl-fast-relaxed-math", "-cl-mad-enable"]
    )
    k_assign  = program.assignment
    k_update  = program.update
    k_divide  = program.divide_counts

    # ── K-Means++ init on CPU ─────────────────────────────────────────────────
    centroids = _init_kmeans_plus_plus(X, k, rng)  # (k, D) float32

    # ── Allocate GPU buffers ──────────────────────────────────────────────────
    mf = cl.mem_flags

    buf_points    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=X)
    buf_centroids = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=centroids)
    buf_labels    = cl.Buffer(ctx, mf.READ_WRITE,  size=N * np.dtype(np.int32).itemsize)
    buf_new_sums  = cl.Buffer(ctx, mf.READ_WRITE,  size=k * D * np.dtype(np.float32).itemsize)
    buf_counts    = cl.Buffer(ctx, mf.READ_WRITE,  size=k * np.dtype(np.int32).itemsize)

    # Work sizes
    local_size_assign = 256
    global_size_assign = int(np.ceil(N / local_size_assign)) * local_size_assign

    local_size_update = 256
    global_size_update = int(np.ceil(N / local_size_update)) * local_size_update

    global_size_divide = int(np.ceil((k * D) / 64)) * 64

    # Set static kernel args
    k_assign.set_args(buf_points, buf_centroids, buf_labels,
                      np.int32(N), np.int32(D), np.int32(k))
    k_update.set_args(buf_points, buf_labels, buf_new_sums, buf_counts,
                      np.int32(N), np.int32(D), np.int32(k))
    k_divide.set_args(buf_new_sums, buf_counts, np.int32(k), np.int32(D))

    zeros_sums   = np.zeros(k * D, dtype=np.float32)
    zeros_counts = np.zeros(k, dtype=np.int32)

    # ── START timing — GPU algorithm only ─────────────────────────────────────
    queue.finish()
    t_start = time.perf_counter()

    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Kernel 1: Assignment
        cl.enqueue_nd_range_kernel(queue, k_assign,
                                   (global_size_assign,), (local_size_assign,))

        # Reset accumulators
        cl.enqueue_copy(queue, buf_new_sums, zeros_sums)
        cl.enqueue_copy(queue, buf_counts,   zeros_counts)

        # Kernel 2: Update (atomic accumulation)
        cl.enqueue_nd_range_kernel(queue, k_update,
                                   (global_size_update,), (local_size_update,))

        # Kernel 3: Divide sums by counts → new centroid positions
        cl.enqueue_nd_range_kernel(queue, k_divide,
                                   (global_size_divide,), None)

        # Read new centroids back to CPU for convergence check
        new_centroids = np.empty((k, D), dtype=np.float32)
        cl.enqueue_copy(queue, new_centroids, buf_new_sums)
        queue.finish()

        # Handle empty clusters: keep old centroid
        counts = np.empty(k, dtype=np.int32)
        cl.enqueue_copy(queue, counts, buf_counts)
        queue.finish()
        empty = counts == 0
        if np.any(empty):
            new_centroids[empty] = centroids[empty]

        # Convergence check: max centroid shift
        shifts = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
        max_shift = shifts.max()

        centroids = new_centroids

        # Upload updated centroids to GPU
        cl.enqueue_copy(queue, buf_centroids, centroids)

        if max_shift < tol:
            break

    # ── STOP timing ───────────────────────────────────────────────────────────
    queue.finish()
    t_end = time.perf_counter()

    # Read final labels
    labels = np.empty(N, dtype=np.int32)
    cl.enqueue_copy(queue, labels, buf_labels)
    queue.finish()

    # Compute inertia on CPU
    inertia = 0.0
    for ki in range(k):
        mask = labels == ki
        if mask.any():
            diff = X[mask] - centroids[ki]
            inertia += float(np.sum(diff * diff))

    return KMeansResult(
        platform="AMD-iGPU-780M-PyOpenCL",
        n_samples=N,
        n_features=D,
        k=k,
        time_seconds=t_end - t_start,
        iterations=n_iter,
        inertia=inertia,
        labels=labels,
        centers=centroids,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
    from data_generator import generate_synthetic

    print("=== Test kmeans_pyopencl.py — AMD Radeon 780M ===\n")
    X = generate_synthetic(n_samples=50_000, n_features=8, k=10)
    result = run_kmeans_pyopencl(X, k=10)
    print(result.summary())
    print(f"  Labels[:5]  : {result.labels[:5]}")
    print(f"  Centers shape: {result.centers.shape}")
