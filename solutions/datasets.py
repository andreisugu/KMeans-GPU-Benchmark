import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os

def generate_kmeans_dataset(filename, num_points, num_features, num_clusters, max_iter=100):
    """
    Generează un set de date sintetic și îl salvează în formatul specific
    așteptat de implementarea C++ a algoritmului K-Means.
    """
    print(f"Generez {num_points} puncte cu {num_features} dimensiuni și {num_clusters} clustere...")
    
    # make_blobs generează clustere perfect grupate pentru testare
    X, true_labels = make_blobs(n_samples=num_points, 
                                centers=num_clusters, 
                                n_features=num_features, 
                                cluster_std=1.5, 
                                random_state=42)

    # Scriere în fișier în formatul așteptat de codul C++
    with open(filename, 'w') as f:
        # Header: Puncte Atribute Clustere Max_Iteratii Are_Nume (0 pentru fara nume)
        f.write(f"{num_points} {num_features} {num_clusters} {max_iter} 0\n")
        
        # Scriere coordonate puncte
        for point in X:
            line = " ".join([f"{val:.4f}" for val in point])
            f.write(f"{line}\n")
            
    print(f"Set de date salvat cu succes în {filename}")

if __name__ == "__main__":
    # Creăm folderul pentru inputuri dacă nu există
    INPUT_DIR = "inputs"
    os.makedirs(INPUT_DIR, exist_ok=True)

    print("=== START GENERARE INPUTURI PENTRU BENCHMARK ===")
    print("ATENȚIE: Fișierele mari pot dura câteva minute pentru a fi generate și salvate pe disc.\n")

    # --- 1. INPUT MIC (Testare rapidă: sub 1 secundă) ---
    generate_kmeans_dataset(f"{INPUT_DIR}/01_input_mic.txt", 
                            num_points=10000, num_features=4, num_clusters=8)

    # --- 2. INPUT MEDIU-MIC (Aproximativ 1-2 minute, în funcție de CPU) ---
    generate_kmeans_dataset(f"{INPUT_DIR}/02_input_mediu1.txt", 
                            num_points=500000, num_features=16, num_clusters=32)

    # --- 3. INPUT MEDIU-MARE (Țintă: ~5 minute) ---
    # Face 1.000.000 * 32 * 64 calcule de distanță PER iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/03_input_mediu2.txt", 
                            num_points=1000000, num_features=32, num_clusters=64)

    # --- 4. INPUT MARE (Țintă: ~7 minute) ---
    # Face 2.000.000 * 64 * 128 calcule de distanță PER iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/04_input_mare.txt", 
                            num_points=2000000, num_features=64, num_clusters=128)

    # --- 5. INPUT FOARTE MARE (Țintă: ~10 minute MAX) ---
    # Face 4.000.000 * 64 * 256 calcule de distanță PER iterație. 
    # Va genera un fișier de aproximativ 2.5 GB.
    generate_kmeans_dataset(f"{INPUT_DIR}/05_input_extrem.txt", 
                            num_points=4000000, num_features=64, num_clusters=256)

    print("\n=== TOATE CELE 5 INPUTURI AU FOST GENERATE ÎN FOLDERUL 'inputs/' ===")