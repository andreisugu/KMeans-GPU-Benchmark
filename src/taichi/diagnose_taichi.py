"""
diagnose_taichi.py
------------------
Measures individual kernel times and per-iteration times
to identify where time is spent in Large and High-Dim scenarios.
"""
import sys, os, time
import numpy as np
import taichi as ti

sys.path.insert(0, "../baseline")
from data_generator import generate_synthetic
from kmeans_taichi import init_taichi, _init_kmeans_plus_plus

init_taichi()

def diagnose(name, n, d, k, n_iter_max=5):
    print(f"\n=== {name} | N={n:,} D={d} K={k} ===")
    X = np.ascontiguousarray(generate_synthetic(n, d, k, 42), dtype=np.float32)
    rng = np.random.default_rng(42)
    centroids = _init_kmeans_plus_plus(X, k, rng)

    f_points    = ti.field(dtype=ti.f32, shape=(n, d))
    f_centroids = ti.field(dtype=ti.f32, shape=(k, d))
    f_labels    = ti.field(dtype=ti.i32, shape=n)
    f_new_sums  = ti.field(dtype=ti.f32, shape=(k, d))
    f_counts    = ti.field(dtype=ti.i32, shape=k)
    f_points.from_numpy(X)
    f_centroids.from_numpy(centroids)

    @ti.kernel
    def assignment():
        for i in range(n):
            best_dist = float('inf')
            best_k = 0
            for ki in range(k):
                dist = 0.0
                for dim in range(d):
                    diff = f_points[i, dim] - f_centroids[ki, dim]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best_k = ki
            f_labels[i] = best_k

    @ti.kernel
    def reset():
        for ki in range(k):
            f_counts[ki] = 0
            for dim in range(d):
                f_new_sums[ki, dim] = 0.0

    @ti.kernel
    def update():
        for i in range(n):
            ki = f_labels[i]
            ti.atomic_add(f_counts[ki], 1)
            for dim in range(d):
                ti.atomic_add(f_new_sums[ki, dim], f_points[i, dim])

    @ti.kernel
    def divide():
        for ki in range(k):
            if f_counts[ki] > 0:
                inv = 1.0 / float(f_counts[ki])
                for dim in range(d):
                    f_centroids[ki, dim] = f_new_sums[ki, dim] * inv

    # Warmup
    assignment(); reset(); update(); divide(); ti.sync()
    f_centroids.from_numpy(centroids)

    t_assign = t_reset = t_update = t_divide = t_readback = 0.0

    for _ in range(n_iter_max):
        ti.sync()
        t0 = time.perf_counter()
        assignment()
        ti.sync()
        t_assign += time.perf_counter() - t0

        t0 = time.perf_counter()
        reset()
        ti.sync()
        t_reset += time.perf_counter() - t0

        t0 = time.perf_counter()
        update()
        ti.sync()
        t_update += time.perf_counter() - t0

        t0 = time.perf_counter()
        divide()
        ti.sync()
        t_divide += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = f_centroids.to_numpy()
        t_readback += time.perf_counter() - t0

    iters = n_iter_max
    print(f"  assignment : {t_assign/iters*1000:8.2f} ms/iter")
    print(f"  reset      : {t_reset/iters*1000:8.2f} ms/iter")
    print(f"  update     : {t_update/iters*1000:8.2f} ms/iter  <-- suspect")
    print(f"  divide     : {t_divide/iters*1000:8.2f} ms/iter")
    print(f"  readback   : {t_readback/iters*1000:8.2f} ms/iter")
    total = (t_assign + t_reset + t_update + t_divide + t_readback) / iters
    print(f"  TOTAL/iter : {total*1000:8.2f} ms")

diagnose("Medium",   100_000,    16,  10)
diagnose("Large",  1_000_000,    64,  20)
diagnose("High-Dim", 100_000,   512,  10)
