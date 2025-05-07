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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import json

# --- 1. Carregar e pré-processar os dados ---
(
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
test_models_and_save_metrics(
    epochs_list=[50, 100, 150],
    batch_sizes=[8, 16, 32],
    X_train=X_train,
    X_test=X_test,
    y_usd_train=y_usd_train,
    y_usd_test=y_usd_test,
    y_jpy_train=y_jpy_train,
    y_jpy_test=y_jpy_test,
    scaler_y_usd=scaler_y_usd,
    scaler_y_jpy=scaler_y_jpy,
    verbose=VERBOSE,
    output_file="metrics_results.json"
)
