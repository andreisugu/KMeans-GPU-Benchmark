# KMeans-GPU-Benchmark

A high-performance benchmark of the K-Means clustering algorithm, comparing CPU execution against GPU acceleration using RAPIDS cuML and a custom C++/CUDA kernel.

## 📌 Project Overview

The goal of this project is to demonstrate the exponential speedup GPUs can achieve over CPUs in highly parallelizable machine learning tasks. K-Means clustering is an **Embarrassingly Parallel** problem, making it the perfect candidate for CUDA architecture.

This repository tracks the development of the algorithm from a foundational, sequential C++ implementation up to a highly optimized CUDA kernel.

## 🚀 Current Status: Phase 1 (CPU Baseline) Completed

We have successfully implemented and benchmarked the **Sequential C++ Baseline**.

To ensure a fair comparison against the GPU later, the CPU implementation abandons traditional Object-Oriented Programming (which causes memory thrashing) in favor of a **Data-Oriented Design (flat arrays)**. It relies on squared Euclidean distances to avoid expensive CPU `sqrt()` operations, keeping the arithmetic logic unit fed and maximizing cache locality.

### Dataset Scaling

To properly stress-test the CPU and establish an asymptotic execution curve, we built a Python dataset generator (`generate_dataset.py`) that scales $N$ (Points), $D$ (Dimensions), and $K$ (Clusters). The inputs were mathematically designed to hit specific execution time targets on a modern CPU:

| Test Level | Points ($N$) | Dimensions ($D$) | Clusters ($K$) | Target Time | Actual CPU Time | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **1. Very Small** | 250,000 | 16 | 32 | \~5 seconds | **\~5 sec** (0.08 min) | 
| **2. Small** | 700,000 | 32 | 32 | \~30 seconds | **\~31 sec** (0.52 min) | 
| **3. Medium** | 1,400,000 | 32 | 64 | \~2 minutes | **\~2.02 min** | 
| **4. Large** | 3,500,000 | 32 | 64 | \~5 minutes | **\~5.10 min** | 
| **5. Extreme** | 3,500,000 | 32 | 128 | \~10 minutes | **\~10.29 min** | 

*Note: The actual CPU times were recorded using a standard sequential execution (1 thread) with `-O3` compiler optimizations.*

We also include a script (`plot_benchmark.py`) to visually graph this execution curve.

## 🛠️ How to Build and Run (Sequential Baseline)

**1. Generate the Datasets:**

```bash
pip install numpy scikit-learn matplotlib
python generate_dataset.py