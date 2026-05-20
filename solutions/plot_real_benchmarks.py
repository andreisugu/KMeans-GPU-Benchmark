import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_execution_time(filepath):
    """Extracts the execution time in ms from a benchmark result file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Search for "Execution Time (Core Loop): X ms"
            match = re.search(r"Execution Time \(Core Loop\):\s*([\d\.]+)\s*ms", content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def plot_real_benchmarks():
    results_dir = "results_real"
    
    # Datasets keys and their pretty labels
    datasets = [
        ("real_01_mall_customers_200", "Mall Customers\n(200 pct, 3D, K=5)"),
        ("real_02_covtype_581k_k7", "Covertype K=7\n(581K pct, 54D)"),
        ("real_03_covtype_581k_k50", "Covertype K=50\n(581K pct, 54D)"),
        ("real_04_kddcup99_490k_k23", "KDD Cup 10%\n(494K pct, 38D)"),
        ("real_05_kddcup99_5m_k23", "KDD Cup 100%\n(4.9M pct, 38D)")
    ]

    labels = [d[1] for d in datasets]
    keys = [d[0] for d in datasets]

    paradigms = {
        "Sequential C++": "seq",
        "scikit-learn (Python)": "sklearn",
        "HIP/GPU (AMD 780M)": "hip",
        "MPI (4 Procese)": "mpi"
    }

    colors = {
        "Sequential C++": "#e74c3c",       # Red
        "scikit-learn (Python)": "#2ecc71",  # Green
        "HIP/GPU (AMD 780M)": "#3498db",     # Blue
        "MPI (4 Procese)": "#f39c12"        # Orange/Yellow
    }

    markers = {
        "Sequential C++": "o",
        "scikit-learn (Python)": "s",
        "HIP/GPU (AMD 780M)": "^",
        "MPI (4 Procese)": "D"
    }

    # Gather times in ms
    times_ms = {para: [] for para in paradigms}
    for para_name, suffix in paradigms.items():
        for key in keys:
            filepath = os.path.join(results_dir, f"{key}_{suffix}.txt")
            t = extract_execution_time(filepath)
            if t is not None:
                times_ms[para_name].append(t)
            else:
                # Use a default fallback or NaN
                print(f"Warning: Result file not found or invalid: {filepath}")
                times_ms[para_name].append(0.0)

    # Convert to seconds for plotting
    times_sec = {para: [t / 1000.0 for t in val] for para, val in times_ms.items()}

    # Setup plotting style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

    # =========================================================================
    # GRAPH 1: TIMPI DE EXECUȚIE (LOG SCALE)
    # =========================================================================
    plt.figure(figsize=(12, 7))
    for para_name, data in times_sec.items():
        plt.plot(labels, data, marker=markers[para_name], linestyle='-', 
                 color=colors[para_name], linewidth=2.5, markersize=8, label=para_name)

    # Add text annotations for first and last datasets (for visibility)
    for para_name, data in times_sec.items():
        # First point (Mall)
        t_val = data[0]
        if t_val > 0:
            if t_val < 0.001:
                label_str = f"{t_val*1000:.3f}ms"
            else:
                label_str = f"{t_val:.2f}s"
            plt.text(0, t_val * 1.3 if para_name != "MPI (4 Procese)" else t_val * 0.7, 
                     label_str, color=colors[para_name], fontweight='bold', ha='center', fontsize=8)
        
        # Last point (KDD 5M)
        t_val_last = data[4]
        if t_val_last > 0:
            plt.text(4, t_val_last * 1.3 if para_name != "MPI (4 Procese)" else t_val_last * 0.7, 
                     f"{t_val_last:.1f}s", color=colors[para_name], fontweight='bold', ha='center', fontsize=8)

    plt.yscale('log')
    plt.title('K-Means pe Date Reale: Timp de Execuție (Scală Logaritmică)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Set de Date Reale (N Puncte, Dimensiuni, Clustere)', fontsize=11, labelpad=10)
    plt.ylabel('Timp de Execuție (Secunde - Log)', fontsize=11, labelpad=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True, shadow=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    
    plt.savefig('grafic_real_timpi.png', dpi=300)
    print("Graficul timpilor pe date reale a fost salvat ca 'grafic_real_timpi.png'")

    # =========================================================================
    # GRAPH 2: SPEEDUP vs SEQUENTIAL
    # =========================================================================
    plt.figure(figsize=(12, 7))
    seq_times = np.array(times_ms["Sequential C++"])
    # Avoid division by zero
    seq_times_safe = np.where(seq_times == 0, 1.0, seq_times)

    speedups = {
        "scikit-learn (Python)": seq_times_safe / np.array(times_ms["scikit-learn (Python)"]),
        "HIP/GPU (AMD 780M)": seq_times_safe / np.array(times_ms["HIP/GPU (AMD 780M)"]),
        "MPI (4 Procese)": seq_times_safe / np.array(times_ms["MPI (4 Procese)"])
    }

    x = np.arange(len(labels))
    width = 0.25  # Width of each bar

    plt.bar(x - width, speedups["scikit-learn (Python)"], width, label='scikit-learn (Python)', color=colors["scikit-learn (Python)"], alpha=0.9, edgecolor='grey', linewidth=0.5)
    plt.bar(x, speedups["HIP/GPU (AMD 780M)"], width, label='HIP/GPU (AMD 780M)', color=colors["HIP/GPU (AMD 780M)"], alpha=0.9, edgecolor='grey', linewidth=0.5)
    plt.bar(x + width, speedups["MPI (4 Procese)"], width, label='MPI (4 Procese)', color=colors["MPI (4 Procese)"], alpha=0.9, edgecolor='grey', linewidth=0.5)

    # Add text labels on top of the bars
    for i in range(len(labels)):
        # sklearn
        val_py = speedups["scikit-learn (Python)"][i]
        plt.text(i - width, val_py + 0.5, f"{val_py:.1f}x", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#27ae60')
        # HIP
        val_hip = speedups["HIP/GPU (AMD 780M)"][i]
        plt.text(i, val_hip + 0.5, f"{val_hip:.1f}x", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2980b9')
        # MPI
        val_mpi = speedups["MPI (4 Procese)"][i]
        plt.text(i + width, val_mpi + 0.5, f"{val_mpi:.1f}x", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#d35400')

    plt.title('K-Means pe Date Reale: Factor de Accelerare (Speedup vs. Secvențial)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, labels, fontsize=9)
    plt.xlabel('Set de Date Reale (N Puncte, Dimensiuni, Clustere)', fontsize=11, labelpad=10)
    plt.ylabel('Multiplicator de Viteză (ex: 10x = de 10 ori mai rapid)', fontsize=11, labelpad=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Set y-limit with extra room for labels
    max_speedup = max([max(v) for v in speedups.values()])
    plt.ylim(0, max_speedup + 3.0)
    
    plt.legend(loc="upper left", frameon=True, shadow=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    
    plt.savefig('grafic_real_speedup.png', dpi=300)
    print("Graficul speedup-ului pe date reale a fost salvat ca 'grafic_real_speedup.png'")

    # Print markdown table of results
    print("\n=== TABEL REZULTATE TIMP DE EXECUȚIE (ms) ===")
    print("| Set de Date | Secvențial C++ | scikit-learn (Python) | HIP/GPU (AMD 780M) | MPI (4 Procese) |")
    print("|---|---|---|---|---|")
    for i, key in enumerate(keys):
        print(f"| {labels[i].replace(chr(10), ' ')} | {times_ms['Sequential C++'][i]:.2f} ms | {times_ms['scikit-learn (Python)'][i]:.2f} ms | {times_ms['HIP/GPU (AMD 780M)'][i]:.2f} ms | {times_ms['MPI (4 Procese)'][i]:.2f} ms |")

    print("\n=== TABEL SPEEDUP FACTOR ===")
    print("| Set de Date | scikit-learn | HIP/GPU | MPI |")
    print("|---|---|---|---|")
    for i, key in enumerate(keys):
        print(f"| {labels[i].replace(chr(10), ' ')} | {speedups['scikit-learn (Python)'][i]:.2f}x | {speedups['HIP/GPU (AMD 780M)'][i]:.2f}x | {speedups['MPI (4 Procese)'][i]:.2f}x |")

if __name__ == "__main__":
    plot_real_benchmarks()
