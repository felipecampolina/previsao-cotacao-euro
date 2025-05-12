import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter  # Para suavização opcional das curvas

def plot_real_vs_predicted(y_true, y_pred, title: str):
    """
    Plota valores reais vs previstos usando pontos.
    
    Args:
        y_true (array-like): Valores reais.
        y_pred (array-like): Valores previstos.
        title (str): Título do gráfico.
    """
    # Cria uma figura com tamanho definido
    plt.figure(figsize=(10, 5))
    
    # Plota os valores reais e previstos como pontos
    plt.scatter(range(len(y_true)), y_true, label="Real", alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label="Previsto", alpha=0.7)
    
    # Adiciona legenda e título
    plt.legend()
    plt.title(title)
    
    # Exibe o gráfico
    plt.show()


def plot_last_5_days(y_true, y_pred, title: str):
    """
    Plota os últimos 5 valores reais vs previstos por dia.
    
    Args:
        y_true (array-like): Valores reais.
        y_pred (array-like): Valores previstos.
        title (str): Título do gráfico.
    """
    # Seleciona os últimos 5 valores reais e previstos
    y_true_last_5 = y_true[-5:]
    y_pred_last_5 = y_pred[-5:]
    days = range(1, 6)  # Dias 1 a 5

    # Cria uma figura com tamanho definido
    plt.figure(figsize=(8, 5))
    
    # Plota os valores reais e previstos como pontos
    plt.scatter(days, y_true_last_5, marker='o', label="Real")
    plt.scatter(days, y_pred_last_5, marker='o', label="Previsto")
    
    # Configurações do eixo x e rótulos
    plt.xticks(days)
    plt.xlabel("Dia")
    plt.ylabel("Valor")
    
    # Adiciona título, legenda e grade
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Exibe o gráfico
    plt.show()


def plot_real_vs_predicted(
    y_true,
    y_pred,
    title: str = "",
    xlabel: str = "Amostras",
    ylabel: str = "Valores",
    smooth: bool = True,
    window: int = 11,        # Deve ser ímpar e maior que a ordem do polinômio
    polyorder: int = 3,
    shade_error: bool = True,
    style: str = "seaborn-v0_8-whitegrid"  # Estilo global do Matplotlib
):
    """
    Plota curvas reais vs previstas de forma mais legível.
    
    - Suaviza opcionalmente com Savitzky–Golay.
    - Destaca a região de erro entre as curvas.
    - Permite trocar facilmente o estilo global.
    
    Args:
        y_true (array-like): Valores reais.
        y_pred (array-like): Valores previstos.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo x.
        ylabel (str): Rótulo do eixo y.
        smooth (bool): Se True, aplica suavização nas curvas.
        window (int): Tamanho da janela para suavização (deve ser ímpar).
        polyorder (int): Ordem do polinômio para suavização.
        shade_error (bool): Se True, destaca a região de erro entre as curvas.
        style (str): Estilo global do Matplotlib.
    """
    # 1. Pré-processamento -----------------------------------------------------
    x = np.arange(len(y_true))  # Índices das amostras
    y_t = np.asarray(y_true)   # Converte valores reais para array
    y_p = np.asarray(y_pred)   # Converte valores previstos para array

    # Aplica suavização opcional nas curvas
    if smooth and len(y_t) >= window:
        y_t = savgol_filter(y_t, window_length=window, polyorder=polyorder)
        y_p = savgol_filter(y_p, window_length=window, polyorder=polyorder)

    # 2. Estilo e figura -------------------------------------------------------
    plt.style.use(style)  # Define o estilo global
    fig, ax = plt.subplots(figsize=(12, 6))  # Cria a figura e os eixos

    # Plota as curvas reais e previstas
    ax.plot(x, y_t, label="Real", linewidth=2)
    ax.plot(x, y_p, label="Previsto", linewidth=2, linestyle="--")

    # 3. Região do erro (opcional) --------------------------------------------
    if shade_error:
        ax.fill_between(x, y_t, y_p, alpha=0.15, label="Erro")

    # 4. Detalhes finais -------------------------------------------------------
    ax.set_title(title, fontsize=14, pad=10)  # Define o título
    ax.set_xlabel(xlabel, fontsize=12)        # Define o rótulo do eixo x
    ax.set_ylabel(ylabel, fontsize=12)        # Define o rótulo do eixo y
    ax.grid(True, linestyle="--", alpha=0.4)  # Adiciona grade ao gráfico
    ax.legend(fontsize=10, frameon=False)     # Adiciona legenda
    fig.tight_layout()                        # Ajusta o layout para evitar cortes
    
    # Exibe o gráfico
    plt.show()
