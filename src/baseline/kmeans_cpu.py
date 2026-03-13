"""
kmeans_cpu.py
-------------
Wrapper over sklearn.cluster.KMeans.
Runs the algorithm and returns performance and correctness metrics.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import KMeans


@dataclass
class KMeansResult:
    """Result of a K-Means run."""
    platform:    str
    n_samples:   int
    n_features:  int
    k:           int
    time_seconds: float
    iterations:  int
    inertia:     float
    labels:      np.ndarray = field(repr=False)
    centers:     np.ndarray = field(repr=False)

    def summary(self) -> str:
        return (
            f"[{self.platform}] N={self.n_samples:,} D={self.n_features} K={self.k} | "
            f"Time={self.time_seconds:.4f}s | "
            f"Iter={self.iterations} | "
            f"Inertia={self.inertia:.2f}"
        )


def run_kmeans_cpu(X: np.ndarray,
                   k: int,
                   init: str = "k-means++",
                   max_iter: int = 300,
                   tol: float = 1e-4,
                   n_init: int = 1,
                   random_state: int = 42) -> KMeansResult:
    """
    Runs K-Means on CPU using scikit-learn.

    Parameters
    ----------
    X            : input data (n_samples, n_features)
    k            : number of clusters
    init         : initialization method ('k-means++' or 'random')
    max_iter     : maximum iterations
    tol          : convergence threshold
    n_init       : runs with different initializations (1 = consistent with CUDA)
    random_state : reproducible seed

    Returns
    -------
    KMeansResult with all metrics
    """
    model = KMeans(
        n_clusters=k,
        init=init,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",   # standard Lloyd algorithm (compatible with CUDA)
    )

    # ── Precise timing — fit() only, no data preparation ────────────
    t_start = time.perf_counter()
    model.fit(X)
    t_end = time.perf_counter()

    return KMeansResult(
        platform="CPU-sklearn",
        n_samples=X.shape[0],
        n_features=X.shape[1],
        k=k,
        time_seconds=t_end - t_start,
        iterations=model.n_iter_,
        inertia=model.inertia_,
        labels=model.labels_,
        centers=model.cluster_centers_,
    )


def validate_against_reference(result: KMeansResult,
                                reference: KMeansResult,
                                inertia_tol_pct: float = 5.0) -> bool:
    """
    Validates a result by comparing it against a reference result (scikit-learn).

    Checks if inertia is within an acceptable percentage.
    (Centroids may be in different order, so direct comparison is not done.)

    Parameters
    ----------
    result        : result to validate (e.g., C++, CUDA)
    reference     : reference result (scikit-learn)
    inertia_tol_pct: maximum acceptable difference (%) for inertia

    Returns
    -------
    True if validation passes
    """
    diff_pct = abs(result.inertia - reference.inertia) / reference.inertia * 100
    passed = diff_pct <= inertia_tol_pct

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Validation {result.platform} vs {reference.platform}: "
          f"Inertia diff={diff_pct:.2f}% (tolerance={inertia_tol_pct}%) → {status}")
    return passed


# ── Rulare directa ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic

    print("=== Quick test kmeans_cpu.py ===\n")
    X = generate_synthetic(n_samples=50_000, n_features=8, k=10)
    result = run_kmeans_cpu(X, k=10)
    print(result.summary())
    print(f"  First 5 labels: {result.labels[:5]}")
    print(f"  Centroids shape:    {result.centers.shape}")
