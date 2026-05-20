#!/bin/bash

# 1. Activăm mediul virtual Python
echo "=== Activare .venv ==="
source ../.venv/bin/activate

echo "Încep rularea testelor cu scikit-learn..."
echo ""

# 2. Rulăm scriptul Python pe fiecare fișier text din folderul inputs/
for input_file in $(ls inputs/*.txt | sort); do
    echo "============================================================"
    echo "▶ RULEZ TESTUL: $input_file"
    echo "============================================================"

    python3 kmeans_sklearn.py < "$input_file"

    echo ""
done

echo "=== TOATE TESTELE AU FOST FINALIZATE ==="
