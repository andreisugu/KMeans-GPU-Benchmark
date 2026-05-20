import matplotlib.pyplot as plt
import numpy as np

def plot_all_benchmarks():
    # Dataset names for the X axis
    datasets = [
        "1. Foarte Mic\n(250K pct, 16D, 32C)", 
        "2. Mic\n(700K pct, 32D, 32C)", 
        "3. Mediu\n(1.4M pct, 32D, 64C)", 
        "4. Mare\n(3.5M pct, 32D, 64C)", 
        "5. Extrem\n(3.5M pct, 32D, 128C)"
    ]

    # Execution times in milliseconds (raw data from run logs)
    times_ms = {
        "Sequential C++": [4965.37, 39604.7, 212183.0, 306346.0, 613309.0],
        "scikit-learn (Python)": [395.9959, 1722.0033, 8109.1219, 21833.2732, 31114.8611],
        "HIP/GPU (AMD 780M)": [2122.04, 15030.6, 48300.2, 77632.9, 142613.0],
        "MPI (4 Procese)": [1483.56, 9501.78, 53817.8, 160434.0, 156254.0]
    }

    # Convert milliseconds to seconds for readability
    times_sec = {key: [t / 1000.0 for t in val] for key, val in times_ms.items()}

    # Setup plotting style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

    # =========================================================================
    # GRAPH 1: EXECUTION TIMES (LOGARITHMIC SCALE)
    # Why log scale? Because the times range from 0.4s to 613s (3 orders of magnitude).
    # A linear scale would compress the fast parallel lines to the bottom.
    # =========================================================================
    plt.figure(figsize=(11, 6.5))
    
    colors = {
        "Sequential C++": "#e74c3c",      # Red
        "scikit-learn (Python)": "#2ecc71", # Green
        "HIP/GPU (AMD 780M)": "#3498db",    # Blue
        "MPI (4 Procese)": "#f39c12"       # Yellow/Orange
    }
    
    markers = {
        "Sequential C++": "o",
        "scikit-learn (Python)": "s",
        "HIP/GPU (AMD 780M)": "^",
        "MPI (4 Procese)": "D"
    }

    for label, data in times_sec.items():
        plt.plot(datasets, data, marker=markers[label], linestyle='-', 
                 color=colors[label], linewidth=2.5, markersize=8, label=label)

    # Add raw text annotations for key values (dataset 1 and 5)
    for label, data in times_sec.items():
        # First point
        plt.text(0, data[0] * 1.15 if label != "MPI (4 Procese)" else data[0] * 0.75, 
                 f"{data[0]:.2f}s", color=colors[label], fontweight='bold', ha='center', fontsize=9)
        # Last point
        plt.text(4, data[4] * 1.15 if label != "MPI (4 Procese)" else data[4] * 0.75, 
                 f"{data[4]:.1f}s", color=colors[label], fontweight='bold', ha='center', fontsize=9)

    plt.yscale('log')
    plt.title('Comparativ K-Means: Timp de Execuție (Scală Logaritmică)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Set de Date (N Puncte, Dimensiuni, Clustere)', fontsize=11, labelpad=10)
    plt.ylabel('Timp de Execuție (Secunde - Log)', fontsize=11, labelpad=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True, shadow=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    
    plt.savefig('grafic_compara_timpi.png', dpi=300)
    print("Graficul comparativ al timpilor a fost salvat ca 'grafic_compara_timpi.png'")

    # =========================================================================
    # GRAPH 2: SPEEDUP FACTOR (LINEAR GROUPED BAR CHART)
    # Speedup = Seq_Time / Parallel_Time
    # Shows exactly how many times faster each parallel implementation is.
    # =========================================================================
    plt.figure(figsize=(11, 6.5))
    
    seq_times = np.array(times_ms["Sequential C++"])
    speedups = {
        "scikit-learn (Python)": seq_times / np.array(times_ms["scikit-learn (Python)"]),
        "HIP/GPU (AMD 780M)": seq_times / np.array(times_ms["HIP/GPU (AMD 780M)"]),
        "MPI (4 Procese)": seq_times / np.array(times_ms["MPI (4 Procese)"])
    }

    x = np.arange(len(datasets))
    width = 0.25  # Width of each bar

    plt.bar(x - width, speedups["scikit-learn (Python)"], width, label='scikit-learn (Python)', color=colors["scikit-learn (Python)"], alpha=0.9, edgecolor='grey', linewidth=0.5)
    plt.bar(x, speedups["HIP/GPU (AMD 780M)"], width, label='HIP/GPU (AMD 780M)', color=colors["HIP/GPU (AMD 780M)"], alpha=0.9, edgecolor='grey', linewidth=0.5)
    plt.bar(x + width, speedups["MPI (4 Procese)"], width, label='MPI (4 Procese)', color=colors["MPI (4 Procese)"], alpha=0.9, edgecolor='grey', linewidth=0.5)

    # Add values on top of the bars
    for i in range(len(datasets)):
        # sklearn
        val_py = speedups["scikit-learn (Python)"][i]
        plt.text(i - width, val_py + 0.5, f"{val_py:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
        # HIP
        val_hip = speedups["HIP/GPU (AMD 780M)"][i]
        plt.text(i, val_hip + 0.5, f"{val_hip:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2980b9')
        # MPI
        val_mpi = speedups["MPI (4 Procese)"][i]
        plt.text(i + width, val_mpi + 0.5, f"{val_mpi:.1f}x", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d35400')

    plt.title('Comparativ K-Means: Factor de Accelerare (Speedup vs. Secvențial)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, datasets, fontsize=9)
    plt.xlabel('Set de Date (N Puncte, Dimensiuni, Clustere)', fontsize=11, labelpad=10)
    plt.ylabel('Multiplicator de Viteză (ex: 10x = de 10 ori mai rapid)', fontsize=11, labelpad=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Set y-limit with extra room for text labels
    plt.ylim(0, max([max(v) for v in speedups.values()]) + 3.0)
    
    plt.legend(loc="upper left", frameon=True, shadow=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    
    plt.savefig('grafic_compara_speedup.png', dpi=300)
    print("Graficul comparativ al speedup-ului a fost salvat ca 'grafic_compara_speedup.png'")

if __name__ == "__main__":
    plot_all_benchmarks()
