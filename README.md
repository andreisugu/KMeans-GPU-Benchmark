# KMeans-GPU-Benchmark

A high-performance benchmark of the K-Means clustering algorithm across **4 parallel implementations**: a sequential C++ baseline, a scikit-learn vectorized variant, a HIP/GPU kernel for AMD GPUs, and an MPI distributed variant.

## 📌 Project Overview

The goal of this project is to demonstrate the speedup that parallelism achieves over a sequential CPU baseline in highly parallelizable machine learning tasks. K-Means clustering is an **Embarrassingly Parallel** problem — every point's cluster assignment in the Expectation step is completely independent, making it the ideal candidate for GPU and distributed computing.

This repository benchmarks the algorithm across 4 implementations, all running on the **same datasets** and measuring only the core algorithmic loop (I/O excluded).

---

## 🗂️ Implementations

| # | Variant | File | Parallelism Strategy |
|---|---------|------|---------------------|
| 1 | **Sequential C++** | `kmeans_seq.cpp` | Single-threaded, CPU baseline |
| 2 | **scikit-learn (Python)** | `kmeans_sklearn.py` | Vectorized BLAS (NumPy/OpenBLAS), multi-threaded |
| 3 | **HIP/GPU (AMD)** | `kmeans_hip.cpp` | GPU kernels via `hipcc`, AMD Radeon 780M (gfx1103) |
| 4 | **MPI (Distributed CPU)** | `kmeans_mpi.cpp` | Domain decomposition + `MPI_Allreduce`, 4 processes |

All implementations share:
- **Same algorithm** — Lloyd's algorithm, Forgy random initialization
- **Same optimization** — Squared Euclidean distance (no `sqrt()`)
- **Same design** — Data-Oriented Design (flat contiguous arrays)
- **Same output format** — iterations, execution time (ms), centroids, point counts

---

## 📊 Benchmark Results

All tests were run on an **AMD Ryzen 7 7840HS** with **AMD Radeon 780M iGPU (gfx1103)**, compiled with `-O3`.

### Execution Times (milliseconds)

| Dataset | $N$ | $D$ | $K$ | Sequential | sklearn | HIP/GPU | MPI (4P) |
|---------|-----|-----|-----|------------|---------|---------|----------|
| **1. Very Small** | 250,000 | 16 | 32 | 4,965 ms | **396 ms** | 2,122 ms | 1,484 ms |
| **2. Small** | 700,000 | 32 | 32 | 39,605 ms | **1,722 ms** | 15,031 ms | 9,502 ms |
| **3. Medium** | 1,400,000 | 32 | 64 | 212,183 ms | **8,109 ms** | 48,300 ms | 53,818 ms |
| **4. Large** | 3,500,000 | 32 | 64 | 306,346 ms | **21,833 ms** | 77,633 ms | 160,434 ms |
| **5. Extreme** | 3,500,000 | 32 | 128 | 613,309 ms | **31,115 ms** | 142,613 ms | 156,254 ms |

### Speedup vs Sequential Baseline

| Dataset | sklearn | HIP/GPU | MPI (4P) |
|---------|---------|---------|----------|
| **1. Very Small** | 12.5× | 2.3× | 3.3× |
| **2. Small** | 23.0× | 2.6× | 4.2× |
| **3. Medium** | 26.2× | 4.4× | 3.9× |
| **4. Large** | 14.0× | 3.9× | 1.9× |
| **5. Extreme** | 19.7× | 4.3× | 3.9× |

> **Key observations:**
> - **sklearn** is the fastest across all tests — vectorized BLAS routines (computing all N×K distances as a single matrix multiplication) bypass the explicit O(N·K·D) inner loop entirely.
> - **HIP/GPU** scales better than MPI as dataset size grows, overtaking MPI at dataset 3+. The GPU's massive parallelism benefits from larger N.
> - **MPI** is competitive on small/medium datasets but its advantage is limited by the `MPI_Allreduce` communication overhead, which grows with K (especially on dataset 5 with K=128).
> - The **iGPU** (Radeon 780M) uses shared system memory, limiting its memory bandwidth compared to a discrete GPU — a dedicated GPU would show much larger HIP speedups.

---

## 🗃️ Dataset Scaling

Datasets are synthetically generated using `datasets.py` (`sklearn.datasets.make_blobs`). They were mathematically designed to stress the sequential implementation to specific time targets:

| Test Level | $N$ (Points) | $D$ (Dims) | $K$ (Clusters) | Max Iter | Sequential Time |
|------------|-------------|-----------|---------------|----------|----------------|
| **1. Very Small** | 250,000 | 16 | 32 | 100 | ~5 sec |
| **2. Small** | 700,000 | 32 | 32 | 100 | ~40 sec |
| **3. Medium** | 1,400,000 | 32 | 64 | 100 | ~3.5 min |
| **4. Large** | 3,500,000 | 32 | 64 | 100 | ~5 min |
| **5. Extreme** | 3,500,000 | 32 | 128 | 100 | ~10 min |

---

## 🛠️ How to Build and Run

### Prerequisites

```bash
# Python + sklearn (for dataset generation and sklearn variant)
pip install numpy scikit-learn

# C++ compilers
sudo apt install g++ mpic++   # Sequential + MPI
sudo apt install hipcc         # HIP/GPU (AMD ROCm)
```

### 1. Generate Datasets

```bash
cd solutions/
python3 datasets.py
# Creates inputs/01_input_5sec.txt through inputs/05_input_10min.txt
```

### 2. Run All Variants

```bash
cd solutions/

# Sequential baseline
bash run_all_sequential.sh

# scikit-learn (activate venv first if using one)
bash run_all_sklearn.sh

# HIP/GPU — compiles with hipcc --offload-arch=gfx1103
bash run_all_hip.sh

# MPI — compiles with mpic++, runs with 4 processes
bash run_all_mpi.sh
```

Each script compiles (where applicable) and runs the implementation against all 5 datasets in order, printing results to stdout. Output is also saved to `results_seq.txt`, `results_sklearn.txt`, `results_hip.txt`, and `results_mpi.txt`.

### 3. Manual Compilation

```bash
# Sequential
g++ -O3 kmeans_seq.cpp -o kmeans_seq
./kmeans_seq < inputs/01_input_5sec.txt

# sklearn
python3 kmeans_sklearn.py < inputs/01_input_5sec.txt

# HIP/GPU (AMD Radeon 780M = gfx1103)
hipcc --offload-arch=gfx1103 -O3 kmeans_hip.cpp -o kmeans_hip
./kmeans_hip < inputs/01_input_5sec.txt

# MPI
mpic++ -O3 kmeans_mpi.cpp -o kmeans_mpi
mpirun --oversubscribe -np 4 ./kmeans_mpi < inputs/01_input_5sec.txt
```

---

## 📁 Project Structure

```
solutions/
├── kmeans_seq.cpp          # Sequential C++ (CPU baseline)
├── kmeans_sklearn.py       # scikit-learn (vectorized BLAS)
├── kmeans_hip.cpp          # HIP/GPU (AMD, hipcc)
├── kmeans_mpi.cpp          # MPI distributed (4 processes)
├── datasets.py             # Synthetic dataset generator
├── run_all_sequential.sh   # Runner for sequential variant
├── run_all_sklearn.sh      # Runner for sklearn variant
├── run_all_hip.sh          # Runner for HIP/GPU variant
├── run_all_mpi.sh          # Runner for MPI variant
├── results_seq.txt         # Benchmark output — sequential
├── results_sklearn.txt     # Benchmark output — sklearn
├── results_hip.txt         # Benchmark output — HIP/GPU
├── results_mpi.txt         # Benchmark output — MPI
└── inputs/
    ├── 01_input_5sec.txt
    ├── 02_input_30sec.txt
    ├── 03_input_2min.txt
    ├── 04_input_5min.txt
    └── 05_input_10min.txt
```

---

## ⚙️ Input Format

All implementations read the same custom format from `stdin`:

```
N D K maxIter hasName
x1_1 x1_2 ... x1_D [name1]
x2_1 x2_2 ... x2_D [name2]
...
```

- `N` — number of points
- `D` — number of dimensions (features)
- `K` — number of clusters
- `maxIter` — maximum number of Lloyd's iterations
- `hasName` — `1` if each row has a string label at the end, `0` otherwise