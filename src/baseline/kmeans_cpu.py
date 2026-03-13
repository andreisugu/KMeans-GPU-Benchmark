"""
kmeans_cpu.py
-------------
Wrapper peste sklearn.cluster.KMeans.
Ruleaza algoritmul si returneaza metrici de performanta si corectitudine.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import KMeans


@dataclass
class KMeansResult:
    """Rezultatul unei rulari K-Means."""
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
            f"Timp={self.time_seconds:.4f}s | "
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
    Ruleaza K-Means pe CPU folosind scikit-learn.

    Parametri
    ---------
    X            : date de intrare (n_samples, n_features)
    k            : numarul de clustere
    init         : metoda de initializare ('k-means++' sau 'random')
    max_iter     : iteratii maxime
    tol          : pragul de convergenta
    n_init       : rulari cu initializari diferite (1 = consistent cu CUDA)
    random_state : seed reproductibil

    Returneaza
    ----------
    KMeansResult cu toate metricile
    """
    model = KMeans(
        n_clusters=k,
        init=init,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",   # algoritmul standard Lloyd (compatibil cu CUDA)
    )

    # ── Cronometrare precisa — doar fit(), fara pregatirea datelor ────────────
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
    Valideaza un rezultat comparandu-l cu un rezultat de referinta (scikit-learn).

    Verifica daca inertia este in limita unui procent acceptabil.
    (Centroizii pot fi in ordine diferita, deci nu se compara direct.)

    Parametri
    ---------
    result        : rezultatul de validat (ex: C++, CUDA)
    reference     : rezultatul de referinta (scikit-learn)
    inertia_tol_pct: diferenta maxima acceptata in % pentru inertia

    Returneaza
    ----------
    True daca validarea trece
    """
    diff_pct = abs(result.inertia - reference.inertia) / reference.inertia * 100
    passed = diff_pct <= inertia_tol_pct

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Validare {result.platform} vs {reference.platform}: "
          f"Inertia diff={diff_pct:.2f}% (toleranta={inertia_tol_pct}%) → {status}")
    return passed


# ── Rulare directa ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_synthetic

    print("=== Test rapid kmeans_cpu.py ===\n")
    X = generate_synthetic(n_samples=50_000, n_features=8, k=10)
    result = run_kmeans_cpu(X, k=10)
    print(result.summary())
    print(f"  Primele 5 etichete: {result.labels[:5]}")
    print(f"  Shape centroizi:    {result.centers.shape}")
