"""
Modelos MLP e LSTM para prever USD/EUR e JPY/EUR
Autor: <Felipe Campolina Soares de Paula - 1711687>
Data: 2025‑05‑07
"""

# --- Bibliotecas ---
from data_processing import load_and_preprocess_data  # Função para carregar e processar os dados
from test_models import test_models_and_save_metrics, predict_next_5_days_with_two_models, predict_next_5_days_manual_input, predict_one_day  # Teste e salvamento de métricas e função movida

# --- Carregamento e pré-processamento dos dados ---
(
    df,  # DataFrame original com os dados carregados de 'dados_unificados.csv'
    X_train,  # Dados de entrada para treino
    X_test,  # Dados de entrada para teste
    y_usd_train,  # Saída (USD/EUR) para treino
    y_usd_test,  # Saída (USD/EUR) para teste
    y_jpy_train,  # Saída (JPY/EUR) para treino
    y_jpy_test,  # Saída (JPY/EUR) para teste
    scaler_x,  # Scaler para normalizar os dados de entrada
    scaler_y_usd,  # Scaler para normalizar os dados de saída (USD/EUR)
    scaler_y_jpy,  # Scaler para normalizar os dados de saída (JPY/EUR)
) = load_and_preprocess_data("../data/dados_unificados.csv")  # Carrega os dados do arquivo CSV


VERBOSE = 2  # Nível de verbosidade (0 = sem saída, 1 = progresso do treinamento, 2 = saída detalhada)


# Chamar a função com os parâmetros desejados
test_models_and_save_metrics(
    epochs_list=[50,100,150],
    batch_sizes=[8,16,32],
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


# Predição dos próximos 5 dias com valores de  média semanal  além de variação de 2%
# model_usd_path = (
#     "model/USD_LSTM_E150_B16.keras"  # Substitua pelo caminho do modelo USD/EUR
# )
# model_jpy_path = (
#     "model/JPY_LSTM_E150_B16.keras"  # Substitua pelo caminho do modelo JPY/EUR
# )
# title_usd = "Previsão dos próximos 5 dias (USD/EUR)"
# title_jpy = "Previsão dos próximos 5 dias (JPY/EUR)"
# predict_next_5_days_with_two_models(
#     model_usd_path, model_jpy_path, df, scaler_y_usd, scaler_y_jpy, title_usd, title_jpy
# )


# Caminhos para os modelos salvos
# model_usd_path = "model/USD_LSTM_E150_B16.keras"
# model_jpy_path = "model/JPY_LSTM_E150_B16.keras"


# # Previsão para os dias específicos input manual

# print("\nPrevisão para 11/05/2025:")
# resultado_11 = predict_one_day(
#     model_usd_path=model_usd_path,
#     model_jpy_path=model_jpy_path,
#     sofr_value=4.29,
#     brent_value=64.20,
#     scaler_y_usd=scaler_y_usd,
#     scaler_y_jpy=scaler_y_jpy,
#     scaler_x=scaler_x,
# )

# # Para o dia 12/05/2025
# print("\nPrevisão para 12/05/2025:")
# resultado_12 = predict_one_day(
#     model_usd_path=model_usd_path,
#     model_jpy_path=model_jpy_path,
#     sofr_value=4.30,
#     brent_value=65.17,
#     scaler_y_usd=scaler_y_usd,
#     scaler_y_jpy=scaler_y_jpy,
#     scaler_x=scaler_x,
# )