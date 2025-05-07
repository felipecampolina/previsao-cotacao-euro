import os
import json
from models import create_mlp, create_lstm
from evaluation import evaluate

def test_models_and_save_metrics(
    epochs_list, batch_sizes, X_train, X_test, y_usd_train, y_usd_test, y_jpy_train, y_jpy_test, 
    scaler_y_usd, scaler_y_jpy, verbose=0, output_file="metrics_results.json"
):
    """
    Testa os modelos MLP e LSTM para diferentes combinações de épocas e tamanhos de batch.
    Salva as métricas em um arquivo JSON e os modelos treinados na pasta 'model'.

    Args:
        epochs_list (list): Lista de valores de épocas para testar.
        batch_sizes (list): Lista de tamanhos de batch para testar.
        X_train, X_test: Dados de entrada para treino e teste.
        y_usd_train, y_usd_test, y_jpy_train, y_jpy_test: Dados de saída para treino e teste.
        scaler_y_usd, scaler_y_jpy: Scalers para desnormalizar os dados.
        verbose (int): Nível de verbosidade.
        output_file (str): Nome do arquivo JSON para salvar as métricas.
    """
    # Criar a pasta 'model' se não existir
    os.makedirs("model", exist_ok=True)

    results = []

    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f"Treinando modelos com {epochs} épocas e batch size {batch_size}...")

            # Treinar MLP para USD/EUR
            model_mlp_usd = create_mlp(X_train.shape[1])
            model_mlp_usd.fit(X_train, y_usd_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            y_pred_usd_mlp = model_mlp_usd.predict(X_test)
            metrics_usd_mlp = evaluate(y_usd_test, y_pred_usd_mlp, scaler_y_usd, label=f"USD_MLP_E{epochs}_B{batch_size}")
            results.append(metrics_usd_mlp)
            model_mlp_usd.save(f"model/USD_MLP_E{epochs}_B{batch_size}.keras")  # Salvar modelo

            # Treinar MLP para JPY/EUR
            model_mlp_jpy = create_mlp(X_train.shape[1])
            model_mlp_jpy.fit(X_train, y_jpy_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            y_pred_jpy_mlp = model_mlp_jpy.predict(X_test)
            metrics_jpy_mlp = evaluate(y_jpy_test, y_pred_jpy_mlp, scaler_y_jpy, label=f"JPY_MLP_E{epochs}_B{batch_size}")
            results.append(metrics_jpy_mlp)
            model_mlp_jpy.save(f"model/JPY_MLP_E{epochs}_B{batch_size}.keras")  # Salvar modelo

            # Preparar dados para LSTM
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            # Treinar LSTM para USD/EUR
            model_lstm_usd = create_lstm(X_train.shape[1])
            model_lstm_usd.fit(X_train_lstm, y_usd_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            y_pred_usd_lstm = model_lstm_usd.predict(X_test_lstm)
            metrics_usd_lstm = evaluate(y_usd_test, y_pred_usd_lstm, scaler_y_usd, label=f"USD_LSTM_E{epochs}_B{batch_size}")
            results.append(metrics_usd_lstm)
            model_lstm_usd.save(f"model/USD_LSTM_E{epochs}_B{batch_size}.keras")  # Salvar modelo

            # Treinar LSTM para JPY/EUR
            model_lstm_jpy = create_lstm(X_train.shape[1])
            model_lstm_jpy.fit(X_train_lstm, y_jpy_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            y_pred_jpy_lstm = model_lstm_jpy.predict(X_test_lstm)
            metrics_jpy_lstm = evaluate(y_jpy_test, y_pred_jpy_lstm, scaler_y_jpy, label=f"JPY_LSTM_E{epochs}_B{batch_size}")
            results.append(metrics_jpy_lstm)
            model_lstm_jpy.save(f"model/JPY_LSTM_E{epochs}_B{batch_size}.keras")  # Salvar modelo

    # Salvar métricas em um arquivo JSON
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Métricas salvas em {output_file}")