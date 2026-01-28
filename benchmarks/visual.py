import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_benchmarks(csv_file):
    df = pd.read_csv(csv_file)
    
    # Criando um painel com 2 gráficos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. SER vs SNR
    sns.lineplot(ax=axes[0], data=df, x='snr_db', y='ser', hue='algo', marker='s')
    axes[0].set_title('Symbol Error Rate (SER) por Algoritmo')
    axes[0].set_yscale('log')
    axes[0].grid(True)

    # 2. Eficiência Computacional
    sns.barplot(ax=axes[1], data=df, x='algo', y='samples_per_s')
    axes[1].set_title('Throughput (Amostras por Segundo)')
    axes[1].set_ylabel('Amostras/s (Maior é melhor)')

    plt.tight_layout()
    plt.show()

plot_benchmarks('benchmarks/results_equalization.csv')