import os
import json
from models import create_mlp, create_lstm
from evaluation import evaluate
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from visualization import plot_real_vs_predicted  # Importar a função de plotagem


def test_models_and_save_metrics(
    epochs_list,
    batch_sizes,
    X_train,
    X_test,
    y_usd_train,
    y_usd_test,
    y_jpy_train,
    y_jpy_test,
    scaler_y_usd,
    scaler_y_jpy,
    verbose=0,
    output_file="metrics_results.json",
):
    """
    Testa modelos MLP e LSTM em diferentes configurações, salva os melhores resultados
    e exporta as métricas para um arquivo JSON.
    """

    # Garante que a pasta 'model' exista para armazenar os modelos treinados
    os.makedirs("model", exist_ok=True)

    results = []  # Armazena todas as métricas coletadas
    best_model = None  # Nome do melhor modelo encontrado
    best_metrics = float(
        "inf"
    )  # Melhor RMSE inicializado com um valor alto (pior caso)
    best_y_true = None  # Valores reais do melhor modelo
    best_y_pred = None  # Valores previstos do melhor modelo
    best_title = ""  # Título do gráfico do melhor modelo

    # Loop sobre cada combinação de épocas e batch sizes
    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f"Treinando modelos com {epochs} épocas e batch size {batch_size}...")

            # -------------------- MLP para USD/EUR --------------------
            model_mlp_usd = create_mlp(X_train.shape[1])
            model_mlp_usd.fit(
                X_train,
                y_usd_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
            y_pred_usd_mlp = model_mlp_usd.predict(X_test)

            # Avaliação do modelo e registro das métricas
            metrics_usd_mlp = evaluate(
                y_usd_test,
                y_pred_usd_mlp,
                scaler_y_usd,
                label=f"USD_MLP_E{epochs}_B{batch_size}",
            )
            results.append(metrics_usd_mlp)
            model_mlp_usd.save(f"model/USD_MLP_E{epochs}_B{batch_size}.keras")

            # Verifica se é o melhor RMSE encontrado até agora
            y_usd_test_original = scaler_y_usd.inverse_transform(y_usd_test)
            y_pred_usd_mlp_original = scaler_y_usd.inverse_transform(y_pred_usd_mlp)
            if metrics_usd_mlp["RMSE"] < best_metrics:
                best_metrics = metrics_usd_mlp["RMSE"]
                best_model = "USD/EUR - MLP"
                best_y_true = y_usd_test_original.flatten()
                best_y_pred = y_pred_usd_mlp_original.flatten()
                best_title = f"USD/EUR - MLP (Épocas: {epochs}, Batch: {batch_size})"

            # -------------------- MLP para JPY/EUR --------------------
            model_mlp_jpy = create_mlp(X_train.shape[1])
            model_mlp_jpy.fit(
                X_train,
                y_jpy_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
            y_pred_jpy_mlp = model_mlp_jpy.predict(X_test)

            metrics_jpy_mlp = evaluate(
                y_jpy_test,
                y_pred_jpy_mlp,
                scaler_y_jpy,
                label=f"JPY_MLP_E{epochs}_B{batch_size}",
            )
            results.append(metrics_jpy_mlp)
            model_mlp_jpy.save(f"model/JPY_MLP_E{epochs}_B{batch_size}.keras")

            y_jpy_test_original = scaler_y_jpy.inverse_transform(y_jpy_test)
            y_pred_jpy_mlp_original = scaler_y_jpy.inverse_transform(y_pred_jpy_mlp)
            if metrics_jpy_mlp["RMSE"] < best_metrics:
                best_metrics = metrics_jpy_mlp["RMSE"]
                best_model = "JPY/EUR - MLP"
                best_y_true = y_jpy_test_original.flatten()
                best_y_pred = y_pred_jpy_mlp_original.flatten()
                best_title = f"JPY/EUR - MLP (Épocas: {epochs}, Batch: {batch_size})"

            # -------------------- Preparação para LSTM --------------------
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            # -------------------- LSTM para USD/EUR --------------------
            model_lstm_usd = create_lstm(X_train.shape[1])
            model_lstm_usd.fit(
                X_train_lstm,
                y_usd_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
            y_pred_usd_lstm = model_lstm_usd.predict(X_test_lstm)

            metrics_usd_lstm = evaluate(
                y_usd_test,
                y_pred_usd_lstm,
                scaler_y_usd,
                label=f"USD_LSTM_E{epochs}_B{batch_size}",
            )
            results.append(metrics_usd_lstm)
            model_lstm_usd.save(f"model/USD_LSTM_E{epochs}_B{batch_size}.keras")

            y_pred_usd_lstm_original = scaler_y_usd.inverse_transform(y_pred_usd_lstm)
            if metrics_usd_lstm["RMSE"] < best_metrics:
                best_metrics = metrics_usd_lstm["RMSE"]
                best_model = "USD/EUR - LSTM"
                best_y_true = y_usd_test_original.flatten()
                best_y_pred = y_pred_usd_lstm_original.flatten()
                best_title = f"USD/EUR - LSTM (Épocas: {epochs}, Batch: {batch_size})"

            # -------------------- LSTM para JPY/EUR --------------------
            model_lstm_jpy = create_lstm(X_train.shape[1])
            model_lstm_jpy.fit(
                X_train_lstm,
                y_jpy_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
            y_pred_jpy_lstm = model_lstm_jpy.predict(X_test_lstm)

            metrics_jpy_lstm = evaluate(
                y_jpy_test,
                y_pred_jpy_lstm,
                scaler_y_jpy,
                label=f"JPY_LSTM_E{epochs}_B{batch_size}",
            )
            results.append(metrics_jpy_lstm)
            model_lstm_jpy.save(f"model/JPY_LSTM_E{epochs}_B{batch_size}.keras")

            y_pred_jpy_lstm_original = scaler_y_jpy.inverse_transform(y_pred_jpy_lstm)
            if metrics_jpy_lstm["RMSE"] < best_metrics:
                best_metrics = metrics_jpy_lstm["RMSE"]
                best_model = "JPY/EUR - LSTM"
                best_y_true = y_jpy_test_original.flatten()
                best_y_pred = y_pred_jpy_lstm_original.flatten()
                best_title = f"JPY/EUR - LSTM (Épocas: {epochs}, Batch: {batch_size})"

    # -------------------- Plot do Melhor Modelo --------------------
    if best_model:
        print(f"Melhor modelo: {best_model} com RMSE = {best_metrics:.4f}")
        plot_real_vs_predicted(y_true=best_y_true, y_pred=best_y_pred, title=best_title)

    # -------------------- Salvar Métricas --------------------
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Métricas salvas em {output_file}")


def predict_next_5_days_with_two_models(
    model_usd_path: str,  # Caminho para o modelo salvo para USD/EUR
    model_jpy_path: str,  # Caminho para o modelo salvo para JPY/EUR
    df: pd.DataFrame,  # DataFrame com os dados de entrada
    scaler_y_usd,  # Scaler para desnormalizar os valores previstos de USD/EUR
    scaler_y_jpy,  # Scaler para desnormalizar os valores previstos de JPY/EUR
    title_usd: str,  # Título do gráfico para USD/EUR
    title_jpy: str,  # Título do gráfico para JPY/EUR
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
    weekly_avg = weekly_df[
        ["SOFR", "Brent"]
    ].mean()  # Variáveis de entrada: SOFR e Brent
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
        input_data = scaler_x.transform(
            [[sofr_avg, brent_avg]]
        )  # Normaliza SOFR e Brent
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


def predict_next_5_days_manual_input(
    model_usd_path: str,
    model_jpy_path: str,
    sofr_values: list,
    brent_values: list,
    scaler_y_usd,
    scaler_y_jpy,
):
    """
    Recebe listas de valores SOFR e Brent para simular os próximos 5 dias.
    """
    assert (
        len(sofr_values) == 5 and len(brent_values) == 5
    ), "Forneça exatamente 5 valores para SOFR e Brent"

    model_usd = load_model(model_usd_path)
    model_jpy = load_model(model_jpy_path)

    # Preparar scaler X com faixas fixas (você pode ajustar isso conforme seu contexto real)
    scaler_x = MinMaxScaler()
    combined = np.array(list(zip(sofr_values, brent_values)))
    scaler_x.fit(combined)  # Simula o ajuste com as próprias entradas

    predictions_usd = []
    predictions_jpy = []

    for sofr, brent in zip(sofr_values, brent_values):
        input_data = scaler_x.transform([[sofr, brent]])
        input_data = np.array([input_data], dtype=np.float32)

        pred_usd = model_usd.predict(input_data)
        pred_jpy = model_jpy.predict(input_data)

        predictions_usd.append(pred_usd[0, 0])
        predictions_jpy.append(pred_jpy[0, 0])

    # Desnormalizar previsões
    predictions_usd = scaler_y_usd.inverse_transform(
        np.array(predictions_usd).reshape(-1, 1)
    ).flatten()
    predictions_jpy = scaler_y_jpy.inverse_transform(
        np.array(predictions_jpy).reshape(-1, 1)
    ).flatten()

    results_df = pd.DataFrame(
        {
            "SOFR (Entrada)": sofr_values,
            "Brent (Entrada)": brent_values,
            "Previsão USD/EUR": predictions_usd,
            "Previsão JPY/EUR": predictions_jpy,
        }
    )

    print("\nPrevisões para os próximos 5 dias (Entrada Manual):")
    print(results_df)
    return results_df


def predict_one_day(
    model_usd_path: str,
    model_jpy_path: str,
    sofr_value: float,
    brent_value: float,
    scaler_y_usd,
    scaler_y_jpy,
    scaler_x: MinMaxScaler,
):
    model_usd = load_model(model_usd_path)
    model_jpy = load_model(model_jpy_path)

    # Usar o scaler_x já ajustado nos dados originais
    input_data = scaler_x.transform([[sofr_value, brent_value]])

    # Detectar o formato esperado pelo modelo
    expected_input_shape = model_usd.input_shape
    if len(expected_input_shape) == 3:
        input_data = np.array([input_data], dtype=np.float32)  # Shape (1, 1, 2)
    else:
        input_data = np.array(input_data, dtype=np.float32)  # Shape (1, 2)

    # Previsões
    pred_usd = model_usd.predict(input_data)[0, 0]
    pred_jpy = model_jpy.predict(input_data)[0, 0]

    # Desnormalizar
    pred_usd = scaler_y_usd.inverse_transform(np.array([[pred_usd]])).flatten()[0]
    pred_jpy = scaler_y_jpy.inverse_transform(np.array([[pred_jpy]])).flatten()[0]

    print(f"Previsão USD/EUR: {pred_usd:.6f}")
    print(f"Previsão JPY/EUR: {pred_jpy:.6f}")

    return {"USD/EUR": pred_usd, "JPY/EUR": pred_jpy}
