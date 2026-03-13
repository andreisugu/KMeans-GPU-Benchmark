#!/usr/bin/env bash
# install_all_dependencies.sh
# Install all dependencies for KMeans-GPU-Benchmark (Python, C++, Taichi, plotting)
# Usage: bash install_all_dependencies.sh

set -e

# 1. Python dependencies (main)
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. Python: Taichi (for AMD iGPU backend)
echo "Installing Taichi (for AMD iGPU/Vulkan backend)..."
pip install taichi

# 3. Python: RAPIDS cuML (optional, for NVIDIA GPU, only if on Colab or RAPIDS supported system)
echo "If you want RAPIDS cuML (NVIDIA GPU), install manually as per README or Colab instructions."
echo "  Example: pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com"

# 4. C++ build tools (user must have g++/make)
echo "Ensure you have g++ and make installed for C++ benchmarks."
echo "  On Ubuntu/Debian: sudo apt install build-essential"
echo "  On Fedora: sudo dnf groupinstall 'Development Tools'"

# 5. (Optional) Jupyter for notebooks
echo "If you want to run the Colab notebook locally, install Jupyter: pip install notebook"

echo "\nAll main dependencies installed!"
