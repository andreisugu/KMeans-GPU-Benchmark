"""
kmeans_rapids.py
----------------
Implementare GPU K-Means folosind NVIDIA RAPIDS cuML.
Interfata identica cu kmeans_cpu.py pentru comparatie directa.

Cerinte:
    - NVIDIA GPU (Compute Capability 6.0+)
    - RAPIDS cuML instalat (disponibil in Google Colab cu GPU runtime)

Instalare rapida in Colab:
    !pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
"""

import time
import numpy as np

# cuML este importat conditionat — permite rularea fisierului fara GPU
try:
    from cuml.cluster import KMeans as cuKMeans
    import cudf
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    print("[WARNING] cuML nu este instalat. Acest modul necesita GPU + RAPIDS.")

# Reutilizam structura de rezultat din varianta CPU
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
from kmeans_cpu import KMeansResult


def run_kmeans_rapids(X: np.ndarray,
                      k: int,
                      init: str = "random",
                      max_iter: int = 300,
                      tol: float = 1e-4,
                      random_state: int = 42) -> KMeansResult:
    """
    Ruleaza K-Means pe GPU folosind RAPIDS cuML.

    Nota: cuML KMeans foloseste init='random' implicit si n_init=1.
    Folosim aceiasi parametri ca baseline-ul C++ pentru comparatie corecta.

    Parametri
    ---------
    X            : date de intrare (n_samples, n_features) — numpy float32
    k            : numarul de clustere
    init         : 'random' (recomandat pentru consistenta cu CUDA manual)
    max_iter     : iteratii maxime
    tol          : prag convergenta
    random_state : seed reproductibil

    Returneaza
    ----------
    KMeansResult cu toate metricile
    """
    if not CUML_AVAILABLE:
        raise RuntimeError("cuML nu este disponibil. Foloseste Google Colab cu GPU runtime.")

    # Conversie numpy → cuDF (memoria GPU)
    X_gpu = cudf.DataFrame(X.astype(np.float32))

    model = cuKMeans(
        n_clusters=k,
        init=init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        output_type="numpy",
    )

    # Cronometrare — doar fit(), fara transferul de date
    t_start = time.perf_counter()
    model.fit(X_gpu)
    t_end = time.perf_counter()

    labels  = model.labels_.to_numpy() if hasattr(model.labels_, "to_numpy") else np.array(model.labels_)
    centers = model.cluster_centers_.to_numpy() if hasattr(model.cluster_centers_, "to_numpy") else np.array(model.cluster_centers_)

    return KMeansResult(
        platform="GPU-cuML",
        n_samples=X.shape[0],
        n_features=X.shape[1],
        k=k,
        time_seconds=t_end - t_start,
        iterations=model.n_iter_,
        inertia=float(model.inertia_),
        labels=labels,
        centers=centers,
    )


# ── Rulare directa ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline"))
    from data_generator import generate_synthetic

    print("=== Test rapid kmeans_rapids.py ===\n")
    X = generate_synthetic(n_samples=100_000, n_features=16, k=10)
    result = run_kmeans_rapids(X, k=10)
    print(result.summary())
