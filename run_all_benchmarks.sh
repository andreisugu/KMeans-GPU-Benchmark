#!/usr/bin/env bash
# run_all_benchmarks.sh
# Easily run all KMeans benchmarks and generate plots in the correct order.
# Usage: bash run_all_benchmarks.sh

set -e

# 1. Python CPU (scikit-learn)
echo "[1/4] Running Python CPU benchmark (scikit-learn)..."
cd src/baseline
python3 benchmark_cpu.py
cd ../..

# 2. C++ Sequential benchmark
echo "[2/4] Building and running C++ sequential benchmark..."
make -C src/baseline
./src/baseline/kmeans_seq --all

# 3. Taichi (AMD iGPU, Vulkan backend)
echo "[3/4] Running Taichi (AMD iGPU, Vulkan backend) benchmark..."
cd src/taichi
python3 benchmark_taichi.py
cd ../..

# 4. RAPIDS cuML (NVIDIA GPU, optional, requires Colab or RAPIDS setup)
echo "[4/4] If on Colab or with RAPIDS/cuML, run the RAPIDS benchmark manually:"
echo "    cd src/rapids && python3 benchmark_rapids.py"
echo "(This step is not run automatically unless RAPIDS/cuML is installed.)"

# 5. Plot results
echo "[5/5] Generating plots from results..."
cd results
python3 plot_results.py
cd ..

echo "\nAll benchmarks complete! Check the results/ directory for CSVs and plots."
