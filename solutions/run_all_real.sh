#!/bin/bash

# ============================================================================
# Runs all 4 K-Means implementations on every real-world dataset in inputs_real/
# Results are saved to results_real/ for comparison.
# ============================================================================

INPUT_DIR="inputs_real"
OUTPUT_DIR="results_real"
NUM_MPI_PROCS=4

mkdir -p "$OUTPUT_DIR"

# --- Make sure all binaries are compiled ---
echo "=== Compilare binare ==="
g++ -O3 kmeans_seq.cpp -o kmeans_seq
hipcc --offload-arch=gfx1103 -O3 kmeans_hip.cpp -o kmeans_hip
mpic++ -O3 kmeans_mpi.cpp -o kmeans_mpi

if [ $? -ne 0 ]; then
    echo "Compilare eșuată! Mă opresc."
    exit 1
fi
echo "Compilare reușită!"
echo ""

# Activate venv for sklearn
source ../.venv/bin/activate

# --- Run all implementations on each dataset ---
for input_file in $(ls "$INPUT_DIR"/*.txt 2>/dev/null | sort); do
    dataset=$(basename "$input_file")
    echo "============================================================"
    echo "▶ DATASET: $dataset"
    echo "============================================================"

    echo "--- Sequential ---"
    if [ -s "$OUTPUT_DIR/${dataset%.txt}_seq.txt" ]; then
        echo "Skipping: already run"
    else
        ./kmeans_seq < "$input_file" | tee "$OUTPUT_DIR/${dataset%.txt}_seq.txt"
    fi
    echo ""

    echo "--- scikit-learn ---"
    if [ -s "$OUTPUT_DIR/${dataset%.txt}_sklearn.txt" ]; then
        echo "Skipping: already run"
    else
        python3 kmeans_sklearn.py < "$input_file" | tee "$OUTPUT_DIR/${dataset%.txt}_sklearn.txt"
    fi
    echo ""

    echo "--- HIP/GPU ---"
    if [ -s "$OUTPUT_DIR/${dataset%.txt}_hip.txt" ]; then
        echo "Skipping: already run"
    else
        ./kmeans_hip < "$input_file" | tee "$OUTPUT_DIR/${dataset%.txt}_hip.txt"
    fi
    echo ""

    echo "--- MPI ($NUM_MPI_PROCS procese) ---"
    if [ -s "$OUTPUT_DIR/${dataset%.txt}_mpi.txt" ]; then
        echo "Skipping: already run"
    else
        mpirun --oversubscribe -np $NUM_MPI_PROCS ./kmeans_mpi < "$input_file" | tee "$OUTPUT_DIR/${dataset%.txt}_mpi.txt"
    fi
    echo ""
done

echo "=== TOATE TESTELE PE DATE REALE AU FOST FINALIZATE ==="
echo "Rezultate salvate în: $OUTPUT_DIR/"
