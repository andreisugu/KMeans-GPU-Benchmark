#!/bin/bash

# Activăm mediul virtual Python (dacă există)
if [ -d "../.venv" ]; then
    echo "=== Activare .venv ==="
    source ../.venv/bin/activate
fi

echo "Încep rularea testelor cu RAPIDS cuML (GPU k-NN)..."
echo ""

# Rulăm scriptul Python pe fiecare fișier text din folderul inputs/
for input_file in $(ls inputs/*.txt | sort); do
    echo "============================================================"
    echo "▶ RULEZ TESTUL: $input_file"
    echo "============================================================"

    python3 knn_cuml.py < "$input_file"

    echo ""
done

echo "=== TOATE TESTELE AU FOST FINALIZATE ==="
