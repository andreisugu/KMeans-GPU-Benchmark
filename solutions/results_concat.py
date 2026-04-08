import matplotlib.pyplot as plt

def plot_execution_times():
    # Numele celor 5 inputuri pentru axa X
    inputs = [
        "1. Mic\n(10K pct)", 
        "2. Mediu 1\n(500K pct)", 
        "3. Mediu 2\n(1M pct)", 
        "4. Mare\n(2M pct)", 
        "5. Extrem\n(4M pct)"
    ]

    # TODO: După ce rulezi C++ pe cele 5 inputuri, pune timpii REALI (în MINUTE) aici!
    # Exemplu estimativ: [0.01, 1.5, 4.8, 7.2, 9.8] minute
    timpi_executie_minute = [0.05, 2.1, 5.3, 7.5, 10.2] 

    plt.figure(figsize=(10, 6))
    
    # Desenăm linia cu puncte
    plt.plot(inputs, timpi_executie_minute, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Adăugăm timpii deasupra fiecărui punct pentru claritate
    for i, timp in enumerate(timpi_executie_minute):
        plt.text(i, timp + 0.3, f"{timp} min", ha='center', fontweight='bold')

    # Titluri și etichete
    plt.title('K-Means C++ Secvențial: Timp de execuție în funcție de complexitatea setului de date', fontsize=14)
    plt.xlabel('Complexitatea Setului de Date (N Puncte, Dimensiuni, Clustere)', fontsize=12)
    plt.ylabel('Timp de execuție (Minute)', fontsize=12)
    
    # Setăm limita superioară a axei Y puțin mai sus pentru a face loc textului
    plt.ylim(0, max(timpi_executie_minute) + 1.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvăm graficul ca imagine
    plt.savefig('grafic_timpi_executie.png', dpi=300)
    print("Graficul a fost salvat ca 'grafic_timpi_executie.png'")
    # plt.show() # Decomentează dacă vrei să îl vezi direct în fereastră

if __name__ == "__main__":
    plot_execution_times()