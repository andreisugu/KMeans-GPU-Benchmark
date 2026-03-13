"""
plot_results.py
---------------
Incarca rezultatele din toate implementarile si genereaza grafice comparative.

Rulare:
    python results/plot_results.py

Output:
    results/time_comparison.png   — 4 subploturi independente (unul per scenariu)
    results/speedup_chart.png     — speedup vs C++ sequential cu adnotari
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


RESULT_FILES = {
    "CPU sklearn":     "results_cpu.csv",
    "C++ sequential":  "results_cpp.csv",
    "GPU cuML (T4)":   "results_rapids.csv",
    "AMD iGPU Taichi": "results_taichi.csv",
    # "GPU CUDA custom": "results_cuda.csv",
}

COLORS = {
    "CPU sklearn":     "#4C72B0",
    "C++ sequential":  "#55A868",
    "GPU cuML (T4)":   "#C44E52",
    "AMD iGPU Taichi": "#F39C12",
    "GPU CUDA custom": "#DD8452",
}

SCENARIO_NAMES = {
    (10_000,    2,   5): "Small\nN=10K D=2 K=5",
    (100_000,   16,  10): "Medium\nN=100K D=16 K=10",
    (1_000_000, 64,  20): "Large\nN=1M D=64 K=20",
    (100_000,   512, 10): "High-Dim\nN=100K D=512 K=10",
}


def load_all_results(base_dir: str = ".") -> pd.DataFrame:
    frames = []
    for label, filename in RESULT_FILES.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Label"] = label
            frames.append(df)
            print(f"  ✅ Incarcat: {filename} ({len(df)} randuri)")
        else:
            print(f"  ⚠️  Lipsa: {filename}")
    if not frames:
        raise FileNotFoundError("Niciun fisier CSV gasit in " + base_dir)
    return pd.concat(frames, ignore_index=True)


def get_scenario_key(row):
    return (int(row["N_Samples"]), int(row["D_Features"]), int(row["K_Clusters"]))


def plot_time_comparison(df: pd.DataFrame, output_path: str):
    """
    4 subploturi independente — unul per scenariu.
    Fiecare subplot are scala proprie, deci Small/Medium sunt lizibile.
    Deasupra fiecarei bare: timpul exact in ms/s.
    """
    scenarios = df.groupby(["N_Samples", "D_Features", "K_Clusters"]) \
                  .size().reset_index()[["N_Samples", "D_Features", "K_Clusters"]]
    scenarios = sorted(scenarios.itertuples(index=False),
                       key=lambda r: r.N_Samples * r.D_Features)

    labels_present = list(dict.fromkeys(df["Label"].tolist()))  # preserve order
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 6))
    if n_scenarios == 1:
        axes = [axes]

    fig.suptitle("K-Means: Timp executie per scenariu (scala independenta)",
                 fontsize=14, fontweight="bold", y=1.01)

    x = np.arange(len(labels_present))
    width = 0.65

    for ax, sc in zip(axes, scenarios):
        key = (int(sc.N_Samples), int(sc.D_Features), int(sc.K_Clusters))
        title = SCENARIO_NAMES.get(key,
                    f"N={sc.N_Samples//1000}K\nD={sc.D_Features} K={sc.K_Clusters}")

        times = []
        for label in labels_present:
            match = df[
                (df["Label"] == label) &
                (df["N_Samples"] == sc.N_Samples) &
                (df["D_Features"] == sc.D_Features) &
                (df["K_Clusters"] == sc.K_Clusters)
            ]
            times.append(match["Time_Seconds"].values[0] if len(match) > 0 else None)

        # filter out missing
        present_labels = [l for l, t in zip(labels_present, times) if t is not None]
        present_times  = [t for t in times if t is not None]
        colors         = [COLORS.get(l, "#888888") for l in present_labels]
        xi             = np.arange(len(present_labels))

        bars = ax.bar(xi, present_times, width, color=colors, alpha=0.88, zorder=3)

        # Label on top of each bar: time value
        for bar, t in zip(bars, present_times):
            label_str = f"{t*1000:.1f}ms" if t < 1.0 else f"{t:.2f}s"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    label_str,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(xi)
        ax.set_xticklabels(present_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Timp (secunde)", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v*1000:.0f}ms" if v < 1 else f"{v:.1f}s"
        ))
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.set_ylim(0, max(present_times) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Salvat: {output_path}")
    plt.close()


def plot_speedup(df: pd.DataFrame, output_path: str,
                 baseline_label: str = "C++ sequential"):
    """
    Speedup vs C++ sequential.
    Fiecare bara e adnotata cu '6.2x' deasupra — speedup exact.
    Barele sub 1x (mai lent decat baseline) sunt rosii.
    """
    if baseline_label not in df["Label"].values:
        print(f"  ⚠️  Baseline '{baseline_label}' nu e disponibil.")
        return

    scenarios = df.groupby(["N_Samples", "D_Features", "K_Clusters"]) \
                  .size().reset_index()[["N_Samples", "D_Features", "K_Clusters"]]
    scenarios = sorted(scenarios.itertuples(index=False),
                       key=lambda r: r.N_Samples * r.D_Features)

    compare_labels = [l for l in RESULT_FILES.keys()
                      if l != baseline_label and l in df["Label"].values]

    scenario_labels = []
    for sc in scenarios:
        key = (int(sc.N_Samples), int(sc.D_Features), int(sc.K_Clusters))
        scenario_labels.append(SCENARIO_NAMES.get(key,
            f"N={sc.N_Samples//1000}K D={sc.D_Features}"))

    x = np.arange(len(scenarios))
    width = 0.8 / len(compare_labels)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(y=1.0, color="#333333", linestyle="--",
               linewidth=1.2, label=f"{baseline_label} (1×)", zorder=2)

    for i, label in enumerate(compare_labels):
        speedups = []
        for sc in scenarios:
            base = df[
                (df["Label"] == baseline_label) &
                (df["N_Samples"] == sc.N_Samples) &
                (df["D_Features"] == sc.D_Features) &
                (df["K_Clusters"] == sc.K_Clusters)
            ]["Time_Seconds"]
            impl = df[
                (df["Label"] == label) &
                (df["N_Samples"] == sc.N_Samples) &
                (df["D_Features"] == sc.D_Features) &
                (df["K_Clusters"] == sc.K_Clusters)
            ]["Time_Seconds"]
            if len(base) > 0 and len(impl) > 0 and impl.values[0] > 0:
                speedups.append(base.values[0] / impl.values[0])
            else:
                speedups.append(None)

        offset = (i - len(compare_labels) / 2 + 0.5) * width
        # For values < 1x (slower than baseline): flip to show slowdown magnitude
        # e.g. speedup=0.08 → display as 12.6× SLOWER (inverted, red bar)
        display_vals = []
        bar_colors   = []
        is_slowdown  = []
        for s in speedups:
            if s is None:
                display_vals.append(0)
                bar_colors.append("#cccccc")
                is_slowdown.append(False)
            elif s >= 1.0:
                display_vals.append(s)
                bar_colors.append(COLORS.get(label, "#888888"))
                is_slowdown.append(False)
            else:
                display_vals.append(1.0 / s)   # flip: 0.08 → 12.6
                bar_colors.append("#e74c3c")
                is_slowdown.append(True)

        bars = ax.bar(x + offset, display_vals, width,
                      label=label, color=bar_colors, alpha=0.88, zorder=3)

        # Annotate each bar
        for bar, s, slowdown in zip(bars, speedups, is_slowdown):
            if s is None:
                continue
            h = bar.get_height()
            if slowdown:
                txt = f"{1/s:.1f}× slower"
                col = "#c0392b"
            else:
                txt = f"{s:.1f}×"
                col = "#222222"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + ax.get_ylim()[1] * 0.01,
                    txt, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=col, zorder=4)

    ax.set_xlabel("Scenariu", fontsize=12)
    ax.set_ylabel("Speedup  /  Slowdown (× față de baseline)", fontsize=12)
    ax.set_title(f"K-Means: Speedup față de {baseline_label}",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)

    # Add a manual patch to legend explaining red bars
    from matplotlib.patches import Patch
    handles, leg_labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#e74c3c", alpha=0.88, label="Mai lent (roșu = slowdown)"))
    ax.legend(handles=handles, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Add log scale if range is huge (e.g. 0.08x to 13x)
    all_vals = [s for sc_speeds in [[
        df[(df["Label"] == l) &
           (df["N_Samples"] == sc.N_Samples) &
           (df["D_Features"] == sc.D_Features) &
           (df["K_Clusters"] == sc.K_Clusters)]["Time_Seconds"]
        for sc in scenarios] for l in compare_labels]
        for base in [df[(df["Label"] == baseline_label) &
                        (df["N_Samples"] == sc.N_Samples) &
                        (df["D_Features"] == sc.D_Features) &
                        (df["K_Clusters"] == sc.K_Clusters)]["Time_Seconds"]
                     for sc in scenarios]
        for s in ([base.values[0] / sc_speeds[0].values[0]]
                  if len(base) > 0 and len(sc_speeds[0]) > 0 else [])]

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Salvat: {output_path}")
    plt.close()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== Plot Results — K-Means Benchmark ===\n")

    df = load_all_results(base_dir)
    print(f"\nTotal randuri incarcate: {len(df)}")
    print(df[["Label", "N_Samples", "D_Features",
              "K_Clusters", "Time_Seconds", "Inertia"]].to_string(index=False))

    print("\nGenerare grafice...")
    plot_time_comparison(df, os.path.join(base_dir, "time_comparison.png"))

    # Prefer C++ sequential as baseline; fall back to CPU sklearn if missing
    if "C++ sequential" in df["Label"].values:
        baseline = "C++ sequential"
    elif "CPU sklearn" in df["Label"].values:
        baseline = "CPU sklearn"
        print("  ℹ️  C++ sequential lipsa — folosesc CPU sklearn ca baseline pentru speedup.")
    else:
        baseline = None

    if baseline:
        plot_speedup(df, os.path.join(base_dir, "speedup_chart.png"),
                     baseline_label=baseline)

    print("\nGata! Graficele sunt in folderul results/")
