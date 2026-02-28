# K-Means Performance Benchmark: CPU vs. GPU

This repository provides a comprehensive performance analysis of the K-Means clustering algorithm across different hardware architectures and frameworks. It demonstrates the computational advantages of parallel processing for machine learning tasks.

## 🚀 Features

* **CPU Baseline:** Standard sequential implementation using `scikit-learn`.
* **GPU High-Level:** Accelerated implementation using NVIDIA's `RAPIDS cuML`.
* **Custom CUDA Kernel:** A low-level, hand-written C++/CUDA implementation to demonstrate memory management and thread-level parallelism optimization.
* **Performance Profiling:** Built-in benchmarking to compare execution time and scalability across varying dataset sizes.

## 🛠️ Tech Stack & Prerequisites

* **Languages:** Python 3.x, C++14/17
* **Frameworks/Libraries:** scikit-learn, RAPIDS cuML, NumPy, pandas, matplotlib
* **Hardware Requirements:** NVIDIA GPU (Compute Capability 6.0+). 
* *Note: The project is fully compatible with Google Colab for cloud-based execution if a local NVIDIA GPU is unavailable.*

## 📁 Project Structure
```text
cuda-kmeans-benchmark/
├── data/               # Local directory for Kaggle datasets (ignored by git)
├── notebooks/          # Google Colab / Jupyter Notebooks for testing
├── src/
│   ├── baseline/       # Python scripts for scikit-learn CPU evaluation
│   ├── rapids/         # Python scripts for cuML GPU evaluation
│   └── cuda/           # C++ and .cu source files for custom kernels
├── requirements.txt    # Python dependencies
└── README.md
```

## 📊 Benchmark Results

*(Note: Update this section with a visual plot once the benchmarking is complete)*

Initial tests indicate that the GPU-accelerated methods significantly outperform the CPU baseline as the number of data points and dimensions increase. Detailed metrics and speedup graphs will be added here.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
