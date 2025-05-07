import matplotlib.pyplot as plt

def plot_real_vs_predicted(y_true, y_pred, title: str):
    """Plota valores reais vs previstos usando pontos."""
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_true)), y_true, label="Real", alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label="Previsto", alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.show()

def plot_last_5_days(y_true, y_pred, title: str):
    """Plota os últimos 5 valores reais vs previstos por dia."""
    y_true_last_5 = y_true[-5:]
    y_pred_last_5 = y_pred[-5:]
    days = range(1, 6)  # Dias 1 a 5

    plt.figure(figsize=(8, 5))
    plt.scatter(days, y_true_last_5, marker='o', label="Real")
    plt.scatter(days, y_pred_last_5, marker='o', label="Previsto")
    plt.xticks(days)
    plt.xlabel("Dia")
    plt.ylabel("Valor")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()