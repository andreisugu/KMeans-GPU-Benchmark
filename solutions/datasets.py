import numpy as np
from sklearn.datasets import make_blobs
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
    print("Parametrii au fost calculați matematic pentru a atinge ținte specifice de timp pe CPU-ul curent.\n")

    # --- 1. INPUT 1 (Țintă: ~5 secunde) ---
    # Volum calculat: 250,000 * 16 * 32 = 128 milioane de operații per iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/01_input_5sec.txt", 
                            num_points=250000, num_features=16, num_clusters=32)

    # --- 2. INPUT 2 (Țintă: ~30 secunde) ---
    # Volum calculat: 700,000 * 32 * 32 = 716 milioane de operații per iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/02_input_30sec.txt", 
                            num_points=700000, num_features=32, num_clusters=32)

    # --- 3. INPUT 3 (Țintă: ~2 minute / 120 sec) ---
    # Volum calculat: 1,400,000 * 32 * 64 = 2.86 miliarde de operații per iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/03_input_2min.txt", 
                            num_points=1400000, num_features=32, num_clusters=64)

    # --- 4. INPUT 4 (Țintă: ~5 minute / 300 sec) ---
    # Volum calculat: 3,500,000 * 32 * 64 = 7.16 miliarde de operații per iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/04_input_5min.txt", 
                            num_points=3500000, num_features=32, num_clusters=64)

    # --- 5. INPUT 5 (Țintă: ~10 minute / 600 sec) ---
    # Păstrăm numărul de puncte pentru a nu exploda memoria RAM, dar dublăm clusterele.
    # Volum calculat: 3,500,000 * 32 * 128 = 14.33 miliarde de operații per iterație
    generate_kmeans_dataset(f"{INPUT_DIR}/05_input_10min.txt", 
                            num_points=3500000, num_features=32, num_clusters=128)

    print("\n=== TOATE CELE 5 INPUTURI AU FOST REGENERATE ÎN FOLDERUL 'inputs/' ===")