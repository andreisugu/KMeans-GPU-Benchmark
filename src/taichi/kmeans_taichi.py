"""
kmeans_taichi.py
----------------
K-Means on AMD iGPU (Radeon 780M) using Taichi — Vulkan backend.

IMPORTANT: call init_taichi() ONCE before any benchmark runs.
  Taichi compiles kernels JIT on first call. If ti.init() is called
  multiple times (e.g. once per scenario), kernels are recompiled each
  time and the measured time includes compilation overhead.

Architecture — all GPU:
  Kernel 1 — assignment : 1 thread = 1 point, finds nearest centroid
  Kernel 2 — reset      : zeros sums and counts
  Kernel 3 — update     : ti.atomic_add accumulates coords per cluster
  Kernel 4 — divide     : sums / counts -> new centroid positions

  Convergence check (max centroid shift) runs on CPU after reading back
  only K*D*4 bytes per iteration — negligible on shared iGPU DDR5.

Requirements:
    pip install taichi

Verify AMD GPU:
    python -c "import taichi as ti; ti.init(arch=ti.vulkan); print('OK')"
"""

import os
import sys
import time
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from kmeans_cpu import KMeansResult

# ── Taichi init — call ONCE via init_taichi() before any benchmark ────────────
_ti_initialized = False

def init_taichi(arch=None):
    """
    Initialize Taichi runtime. Must be called ONCE before any benchmarks.
    Kernels are compiled JIT on first invocation and cached for all subsequent runs.

    Args:
        arch: ti.vulkan (default, AMD 780M), ti.cpu (testing), ti.cuda (NVIDIA)
    """
    global _ti_initialized
    if _ti_initialized:
        return
    if arch is None:
        arch = ti.vulkan
    ti.init(arch=arch, log_level=ti.WARN, device_memory_fraction=0.6)
    _ti_initialized = True


def _init_kmeans_plus_plus(X: np.ndarray, k: int,
                            rng: np.random.Generator) -> np.ndarray:
    """K-Means++ initialization on CPU — called once before the GPU loop."""
    N, D = X.shape
    centers   = np.empty((k, D), dtype=np.float32)
    centers[0] = X[rng.integers(0, N)]
    min_dists  = np.full(N, np.finfo(np.float32).max, dtype=np.float32)

    for ki in range(1, k):
        diff = X - centers[ki - 1]
        d    = np.sum(diff * diff, axis=1)
        np.minimum(min_dists, d, out=min_dists)
        probs  = min_dists / min_dists.sum()
        chosen = rng.choice(N, p=probs)
        centers[ki] = X[chosen]

    return centers


def run_kmeans_taichi(X: np.ndarray,
                      k: int,
                      max_iter: int = 300,
                      tol: float    = 1e-4,
                      random_state: int = 42) -> KMeansResult:
    """
    Run K-Means on AMD iGPU via Taichi (Vulkan backend).
    All steps run on GPU. init_taichi() must be called before this.
    """
    if not _ti_initialized:
        raise RuntimeError("Call init_taichi() before run_kmeans_taichi()")

    X = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X.shape

    rng       = np.random.default_rng(random_state)
    centroids = _init_kmeans_plus_plus(X, k, rng)

    # ── GPU fields ────────────────────────────────────────────────────────────
    f_points    = ti.field(dtype=ti.f32, shape=(N, D))
    f_centroids = ti.field(dtype=ti.f32, shape=(k, D))
    f_labels    = ti.field(dtype=ti.i32, shape=N)
    f_new_sums  = ti.field(dtype=ti.f32, shape=(k, D))
    f_counts    = ti.field(dtype=ti.i32, shape=k)

    f_points.from_numpy(X)            # uploaded once, never modified
    f_centroids.from_numpy(centroids)

    # ── Kernels — compiled on first call, cached for all subsequent iterations ─

    @ti.kernel
    def assignment():
        """1 thread = 1 point. Finds nearest centroid. Zero write conflicts."""
        for i in range(N):
            best_dist = float('inf')
            best_k    = 0
            for ki in range(k):
                dist = 0.0
                for d in range(D):
                    diff  = f_points[i, d] - f_centroids[ki, d]
                    dist += diff * diff       # squared dist — no sqrt needed
                if dist < best_dist:
                    best_dist = dist
                    best_k    = ki
            f_labels[i] = best_k

    @ti.kernel
    def reset():
        """Reset accumulators before each update step."""
        for ki in range(k):
            f_counts[ki] = 0
            for d in range(D):
                f_new_sums[ki, d] = 0.0

    @ti.kernel
    def update():
        """
        1 thread = 1 point. Atomically accumulates coords into cluster sums.
        ti.atomic_add is a native GPU atomic — no CAS loop needed.
        """
        for i in range(N):
            ki = f_labels[i]
            ti.atomic_add(f_counts[ki], 1)
            for d in range(D):
                ti.atomic_add(f_new_sums[ki, d], f_points[i, d])

    @ti.kernel
    def divide():
        """Divide accumulated sums by counts to get new centroid positions."""
        for ki in range(k):
            if f_counts[ki] > 0:
                inv = 1.0 / float(f_counts[ki])
                for d in range(D):
                    f_centroids[ki, d] = f_new_sums[ki, d] * inv

    # ── Trigger JIT compilation before timing starts ──────────────────────────
    # First kernel call compiles to Vulkan SPIR-V — exclude from benchmark time.
    assignment()
    reset()
    update()
    divide()
    ti.sync()
    # Re-upload centroids since we just ran a dummy pass
    f_centroids.from_numpy(centroids)

    # ── START timing ──────────────────────────────────────────────────────────
    ti.sync()
    t_start = time.perf_counter()

    n_iter    = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1

        assignment()
        reset()
        update()
        divide()
        ti.sync()

        # Read back only centroids for convergence check — K*D*4 bytes
        new_centroids = f_centroids.to_numpy()

        shifts    = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
        max_shift = shifts.max()
        centroids = new_centroids

        if max_shift < tol:
            break

    # ── STOP timing ───────────────────────────────────────────────────────────
    ti.sync()
    t_end = time.perf_counter()

    labels = f_labels.to_numpy()

    inertia = 0.0
    for ki in range(k):
        mask = labels == ki
        if mask.any():
            diff     = X[mask] - centroids[ki]
            inertia += float(np.sum(diff * diff))

    return KMeansResult(
        platform     = "AMD-iGPU-780M-Taichi-Vulkan",
        n_samples    = N,
        n_features   = D,
        k            = k,
        time_seconds = t_end - t_start,
        iterations   = n_iter,
        inertia      = inertia,
        labels       = labels,
        centers      = centroids,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic

    init_taichi()
    print("=== Test kmeans_taichi.py — AMD Radeon 780M (Vulkan) ===\n")
    X = generate_synthetic(n_samples=50_000, n_features=8, k=10)
    result = run_kmeans_taichi(X, k=10)
    print(result.summary())
    print(f"  Labels[:5]   : {result.labels[:5]}")
    print(f"  Centers shape: {result.centers.shape}")
