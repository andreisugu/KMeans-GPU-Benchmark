#!/bin/bash

# Numărul de procese MPI (ajustați după nevoie — implicit 4)
NUM_PROCS=4

# 1. Compilăm codul MPI cu optimizare maximă (-O3)
echo "=== Compilare kmeans_mpi.cpp cu mpic++ ==="
mpic++ -O3 kmeans_mpi.cpp -o kmeans_mpi

if [ $? -ne 0 ]; then
    echo "Eroare la compilare! Mă opresc aici."
    exit 1
fi

echo "Compilare reușită! Încep rularea testelor cu $NUM_PROCS procese MPI..."
echo ""

# 2. Rulăm executabilul pe fiecare fișier text din folderul inputs/
for input_file in $(ls inputs/*.txt | sort); do
    echo "============================================================"
    echo "▶ RULEZ TESTUL: $input_file"
    echo "============================================================"

    mpirun --allow-run-as-root --oversubscribe -np $NUM_PROCS ./kmeans_mpi < "$input_file"

    echo ""
done

echo "=== TOATE TESTELE AU FOST FINALIZATE ==="
