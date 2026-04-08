#!/bin/bash

# 1. Compilăm codul cu optimizare maximă (-O3)
echo "=== Compilare kmeans_seq.cpp ==="
g++ -O3 kmeans_seq.cpp -o kmeans_seq

if [ $? -ne 0 ]; then
    echo "Eroare la compilare! Mă opresc aici."
    exit 1
fi

echo "Compilare reușită! Încep rularea testelor..."
echo ""

# 2. Rulăm executabilul pe fiecare fișier text din folderul inputs/
# Sortăm alfabetic pentru a rula în ordine (01, 02, 03, etc.)
for input_file in $(ls inputs/*.txt | sort); do
    echo "============================================================"
    echo "▶ RULEZ TESTUL: $input_file"
    echo "============================================================"
    
    # Rulăm algoritmul
    ./kmeans_seq < "$input_file"
    
    echo ""
done

echo "=== TOATE TESTELE AU FOST FINALIZATE ==="