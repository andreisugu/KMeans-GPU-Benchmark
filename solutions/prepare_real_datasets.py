# ============================================================================
# prepare_real_datasets.py
# Prepares real-world datasets for K-Means benchmarking.
#
# Sources:
#   1. Mall Customer Segmentation  - Kaggle (200 rows, 3D, K=5) [proof of concept]
#   2. Forest Covertype            - UCI via sklearn (581K rows, 54D, K=7 / K=50)
#   3. KDD Cup 1999 (10% subset)   - UCI via sklearn (~490K rows, 38D, K=23)
#   4. KDD Cup 1999 (full)         - UCI via sklearn (~5M rows, 38D, K=23)
#
# All datasets are written to inputs_real/ in the same custom format
# expected by all K-Means implementations:
#   Header: N D K maxIter hasName
#   Rows:   space-separated float features
# ============================================================================

import os
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_kddcup99
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "inputs_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_kmeans_input(filename, X, k, max_iter=100):
    """Write a dataset in the custom K-Means input format."""
    n, d = X.shape
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"  Writing {n:,} pts × {d}D  K={k}  → {filename}", end="", flush=True)
    with open(filepath, "w") as f:
        f.write(f"{n} {d} {k} {max_iter} 0\n")
        for row in X:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  [{mb:.1f} MB]")


# ============================================================================
# 1. MALL CUSTOMER SEGMENTATION (Kaggle download required)
#    https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
#    200 rows × 3 features, K=5 customer segments
# ============================================================================
def prepare_mall_customers():
    print("\n[1/4] Mall Customer Segmentation (Kaggle)")
    downloads = os.path.expanduser("~/Downloads")
    csv_path = os.path.join(downloads, "Mall_Customers.csv")

    # Try to extract from zip if CSV not yet unpacked
    if not os.path.exists(csv_path):
        zip_path = os.path.join(downloads, "customer-segmentation-tutorial-in-python.zip")
        if os.path.exists(zip_path):
            print("  Extracting Kaggle zip...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(downloads)
        else:
            print("  [SKIP] Kaggle zip not found — run the curl download first.")
            return

    df = pd.read_csv(csv_path)
    # CustomerID is an ID, Genre/Gender is categorical — use only numerical columns
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values.astype(float)
    X = StandardScaler().fit_transform(X)
    write_kmeans_input("real_01_mall_customers_200.txt", X, k=5)


# ============================================================================
# 2. FOREST COVERTYPE (UCI via sklearn, no authentication required)
#    581,012 rows × 54 features
#    K=7 matches the 7 natural forest cover types (semantic clusters)
#    K=50 is used for stress-testing (longer benchmark runs)
# ============================================================================
def prepare_covtype():
    print("\n[2/4] Forest Covertype — UCI (sklearn)")
    print("  Downloading / loading from cache...")
    data = fetch_covtype(as_frame=False)
    X = StandardScaler().fit_transform(data.data.astype(float))

    # Semantic K: 7 forest cover types
    write_kmeans_input("real_02_covtype_581k_k7.txt", X, k=7, max_iter=100)

    # Stress-test K: 50 clusters — more computation per iteration
    write_kmeans_input("real_03_covtype_581k_k50.txt", X, k=50, max_iter=100)


# ============================================================================
# 3. KDD CUP 1999 — NETWORK INTRUSION DATA (UCI via sklearn)
#    Real-world network traffic data used as a classic large-scale ML benchmark.
#    Only numerical columns are kept (3 categorical columns are dropped).
#    K=23 matches the 23 known attack categories + normal traffic.
#
#    10% subset  → ~490K rows (medium run, ~1 min sequential)
#    Full dataset → ~5M rows  (extreme run, ~5-8 min sequential)
# ============================================================================
def prepare_kddcup(full=False):
    label = "full (~5M rows)" if full else "10% subset (~490K rows)"
    print(f"\n[{'4' if full else '3'}/4] KDD Cup 1999 {label} — UCI (sklearn)")
    print("  Downloading / loading from cache...")

    data = fetch_kddcup99(subset=None, percent10=(not full), as_frame=False)
    # KDD has 41 features; columns 1, 2, 3 are categorical (protocol_type, service, flag).
    # We keep column 0 and columns 4–40 (38 numerical features total).
    raw = data.data
    numerical_indices = [0] + list(range(4, raw.shape[1]))
    X_raw = np.array(raw[:, numerical_indices], dtype=float)
    X = StandardScaler().fit_transform(X_raw)

    tag = "5m" if full else "490k"
    write_kmeans_input(f"real_0{'5' if full else '4'}_kddcup99_{tag}_k23.txt", X, k=23, max_iter=100)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Preparing Real-World Datasets for K-Means Benchmark")
    print("=" * 60)

    prepare_mall_customers()
    prepare_covtype()
    prepare_kddcup(full=False)   # ~490K rows
    prepare_kddcup(full=True)    # ~5M rows (takes several minutes to write)

    print("\n" + "=" * 60)
    print(f"  Done. All datasets saved to: {OUTPUT_DIR}/")
    print("=" * 60)
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
        print(f"  {f:50s}  {mb:7.1f} MB")
