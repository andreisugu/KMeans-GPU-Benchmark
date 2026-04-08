import matplotlib.pyplot as plt

def plot_execution_times():
    # Numele celor 5 inputuri pentru axa X
    inputs = [
        "1. Foarte Mic\n(250K pct, 16D)", 
        "2. Mic\n(700K pct, 32D)", 
        "3. Mediu\n(1.4M pct, 32D)", 
        "4. Mare\n(3.5M pct, 32D, 64C)", 
        "5. Extrem\n(3.5M pct, 32D, 128C)"
    ]

    # Timpii REALI extrași din rularea ta (convertiți în MINUTE)
    # T1: 5059ms -> 0.08 min (~5 sec)
    # T2: 30941ms -> 0.52 min (~31 sec)
    # T3: 121393ms -> 2.02 min (~2 min)
    # T4: 305976ms -> 5.10 min (~5 min)
    # T5: 617157ms -> 10.29 min (~10 min)
    timpi_executie_minute = [0.08, 0.52, 2.02, 5.10, 10.29] 

    plt.figure(figsize=(10, 6))
    
    # Desenăm linia cu puncte
    plt.plot(inputs, timpi_executie_minute, marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, markersize=10)
    
    # Adăugăm timpii deasupra fiecărui punct pentru claritate
    for i, timp in enumerate(timpi_executie_minute):
        # Dacă timpul e sub 1 minut, îl afișăm direct în secunde pentru a fi mai estetic
        if timp < 1.0:
            secunde = int(timp * 60)
            plt.text(i, timp + 0.5, f"~{secunde} sec", ha='center', fontweight='bold')
        else:
            plt.text(i, timp + 0.5, f"{timp:.2f} min", ha='center', fontweight='bold')

    # Titluri și etichete
    plt.title('K-Means CPU (Secvențial): Timp de execuție vs. Complexitate', fontsize=14, fontweight='bold')
    plt.xlabel('Dimensiunea Setului de Date (N Puncte, Dimensiuni, Clustere)', fontsize=12)
    plt.ylabel('Timp de execuție (Minute)', fontsize=12)
    
    # Setăm limita superioară a axei Y puțin mai sus pentru a face loc textului
    plt.ylim(0, max(timpi_executie_minute) + 1.5)
    
    # Estetică
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(inputs, timpi_executie_minute, color='#e74c3c', alpha=0.1) # Umbră subtilă sub grafic
    plt.tight_layout()
    
    # Salvăm graficul ca imagine
    plt.savefig('grafic_benchmark_cpu.png', dpi=300)
    print("Graficul a fost salvat cu succes ca 'grafic_benchmark_cpu.png'")

if __name__ == "__main__":
    plot_execution_times()