#!/bin/bash

# 1. Compilăm codul HIP cu optimizare maximă (-O3) pentru GPU AMD
echo "=== Compilare kmeans_hip.cpp cu hipcc ==="
hipcc --offload-arch=gfx1103 -O3 kmeans_hip.cpp -o kmeans_hip

if [ $? -ne 0 ]; then
    echo "Eroare la compilare! Mă opresc aici."
    exit 1
fi

echo "Compilare reușită! Încep rularea testelor pe GPU..."
echo ""

# 2. Rulăm executabilul pe fiecare fișier text din folderul inputs/
for input_file in $(ls inputs/*.txt | sort); do
    echo "============================================================"
    echo "▶ RULEZ TESTUL: $input_file"
    echo "============================================================"

    ./kmeans_hip < "$input_file"

    echo ""
done

echo "=== TOATE TESTELE AU FOST FINALIZATE ==="
