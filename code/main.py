"""
Modelos MLP e LSTM para prever USD/EUR e JPY/EUR
Com mais métricas de avaliação (MSE, RMSE, MAE, R², MAPE)
Autor: <Felipe Campolina Soares de Paula - 171687>
Data: 2025‑05‑07
"""

# --- Bibliotecas ---
from data_processing import load_and_preprocess_data
from models import create_mlp, create_lstm
from evaluation import evaluate
from visualization import plot_real_vs_predicted, plot_last_5_days
from test_models import test_models_and_save_metrics
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import json
from datetime import datetime, timedelta


(
    df,  # Captura o DataFrame original
    X_train,
    X_test,
    y_usd_train,
    y_usd_test,
    y_jpy_train,
    y_jpy_test,
    scaler_x,
    scaler_y_usd,
    scaler_y_jpy,
) = load_and_preprocess_data("../data/dados_unificados.csv")

# --- Parâmetros configuráveis ---
EPOCHS = 50  # Número de épocas
BATCH_SIZE = 16  # Tamanho do batch
VERBOSE = 0  # Nível de verbosidade (0 = sem saída, 1 = progresso do treinamento, 2 = saída detalhada)


# Chamar a função com os parâmetros desejados
# test_models_and_save_metrics(
#     epochs_list=[50, 100, 150],
#     batch_sizes=[8, 16, 32],
#     X_train=X_train,
#     X_test=X_test,
#     y_usd_train=y_usd_train,
#     y_usd_test=y_usd_test,
#     y_jpy_train=y_jpy_train,
#     y_jpy_test=y_jpy_test,
#     scaler_y_usd=scaler_y_usd,
#     scaler_y_jpy=scaler_y_jpy,
#     verbose=VERBOSE,
#     output_file="metrics_results.json"
# )
def predict_next_5_days_with_two_models(
    model_usd_path: str,
    model_jpy_path: str,
    df: pd.DataFrame,
    scaler_y_usd,
    scaler_y_jpy,
    title_usd: str,
    title_jpy: str,
):
    """
    Usa dois modelos salvos para prever os próximos 5 dias para USD/EUR e JPY/EUR.

    Args:
        model_usd_path (str): Caminho para o modelo salvo para USD/EUR.
        model_jpy_path (str): Caminho para o modelo salvo para JPY/EUR.
        df (pd.DataFrame): DataFrame com os dados de entrada.
        scaler_y_usd: Scaler para desnormalizar os valores previstos de USD/EUR.
        scaler_y_jpy: Scaler para desnormalizar os valores previstos de JPY/EUR.
        title_usd (str): Título do gráfico para USD/EUR.
        title_jpy (str): Título do gráfico para JPY/EUR.
    """
    # Carregar os modelos salvos
    model_usd = load_model(model_usd_path)
    model_jpy = load_model(model_jpy_path)

    # Garantir que a coluna "Date" esteja no formato datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Obter a semana mais recente
    last_date = df["Date"].max()
    start_week = last_date - timedelta(days=6)
    weekly_df = df[(df["Date"] >= start_week) & (df["Date"] <= last_date)]

    # Calcular a média semanal inicial
    weekly_avg = weekly_df[["SOFR", "Brent"]].mean()
    sofr_avg = weekly_avg["SOFR"]
    brent_avg = weekly_avg["Brent"]

    print(
        f"Variáveis de entrada iniciais (média semanal de {start_week.date()} até {last_date.date()}):"
    )
    print(f"SOFR: {sofr_avg:.4f}, Brent: {brent_avg:.4f}")

    # Normalizar as variáveis de entrada
    scaler_x = MinMaxScaler()
    scaler_x.fit(df[["SOFR", "Brent"]])  # Ajustar o scaler com os dados existentes

    # Inicializar variáveis para predição
    predictions_usd = []
    predictions_jpy = []
    input_sofr = []
    input_brent = []

    for i in range(5):
        # Normalizar os dados de entrada
        input_data = scaler_x.transform([[sofr_avg, brent_avg]])
        input_data = np.array([input_data], dtype=np.float32)  # shape: (1, 1, 2)

        # Prever com os modelos
        pred_usd = model_usd.predict(input_data)
        pred_jpy = model_jpy.predict(input_data)
        predicted_usd_value = pred_usd[0, 0]
        predicted_jpy_value = pred_jpy[0, 0]
        predictions_usd.append(predicted_usd_value)
        predictions_jpy.append(predicted_jpy_value)

        # Salvar os dados de entrada usados
        input_sofr.append(sofr_avg)
        input_brent.append(brent_avg)

        # Atualizar DataFrame com a nova previsão
        new_date = last_date + timedelta(days=i + 1)
        new_row = {"Date": new_date, "SOFR": sofr_avg, "Brent": brent_avg}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Recalcular a média semanal
        start_week = new_date - timedelta(days=6)
        weekly_df = df[(df["Date"] >= start_week) & (df["Date"] <= new_date)]
        weekly_avg = weekly_df[["SOFR", "Brent"]].mean()

        # Aplicar pequena variação para evitar repetição exata
        sofr_avg = weekly_avg["SOFR"] * (
            1 + np.random.uniform(-0.002, 0.002)
        )  # Variação de ±0.2%
        brent_avg = weekly_avg["Brent"] * (1 + np.random.uniform(-0.002, 0.002))

        print(
            f"\nNova média semanal após dia {i+1} (de {start_week.date()} até {new_date.date()}):"
        )
        print(f"SOFR: {sofr_avg:.6f}, Brent: {brent_avg:.6f}")

    # Desnormalizar previsões
    predictions_usd = scaler_y_usd.inverse_transform(
        np.array(predictions_usd).reshape(-1, 1)
    ).flatten()
    predictions_jpy = scaler_y_jpy.inverse_transform(
        np.array(predictions_jpy).reshape(-1, 1)
    ).flatten()

    # Datas futuras
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(1, 6)]

    # Resultado
    results_df = pd.DataFrame(
        {
            "Data": dates,
            "SOFR (Entrada)": input_sofr,
            "Brent (Entrada)": input_brent,
            "Previsão USD/EUR": predictions_usd,
            "Previsão JPY/EUR": predictions_jpy,
        }
    )
    print("\nPrevisões para os próximos 5 dias:")
    print(results_df)

    # Plotar os resultados
    # plot_last_5_days(y_true=[], y_pred=predictions_usd, title=title_usd)
    # plot_last_5_days(y_true=[], y_pred=predictions_jpy, title=title_jpy)


# Exemplo de uso da função predict_next_5_days_with_two_models
model_usd_path = (
    "model/USD_LSTM_E150_B16.keras"  # Substitua pelo caminho do modelo USD/EUR
)
model_jpy_path = (
    "model/JPY_LSTM_E150_B16.keras"  # Substitua pelo caminho do modelo JPY/EUR
)
title_usd = "Previsão dos próximos 5 dias (USD/EUR)"
title_jpy = "Previsão dos próximos 5 dias (JPY/EUR)"
predict_next_5_days_with_two_models(
    model_usd_path, model_jpy_path, df, scaler_y_usd, scaler_y_jpy, title_usd, title_jpy
)
