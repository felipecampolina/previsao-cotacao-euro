import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.

    Returns:
        float: Erro percentual absoluto médio.
    """
    # Remove dimensões extras dos arrays, se existirem.
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    
    # Calcula o MAPE como a média do erro percentual absoluto.
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(y_true_scaled, y_pred_scaled, scaler, label: str):
    """
    Calcula métricas de avaliação no espaço original (desnormalizado).
    
    Args:
        y_true_scaled (np.ndarray): Valores reais normalizados.
        y_pred_scaled (np.ndarray): Valores previstos normalizados.
        scaler: Objeto MinMaxScaler usado para desnormalizar os valores.
        label (str): Nome ou descrição do modelo avaliado.

    Returns:
        dict: Dicionário contendo as métricas calculadas (MSE, RMSE, MAE, R2, MAPE).
    """
    # Desnormaliza os valores reais e previstos usando o scaler fornecido.
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Calcula o Mean Squared Error (MSE).
    mse = mean_squared_error(y_true, y_pred)
    
    # Calcula a raiz do Mean Squared Error (RMSE).
    rmse = np.sqrt(mse)
    
    # Calcula o Mean Absolute Error (MAE).
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calcula o coeficiente de determinação (R²).
    r2 = r2_score(y_true, y_pred)
    
    # Calcula o Mean Absolute Percentage Error (MAPE).
    mape_val = mape(y_true, y_pred)

    # Retorna as métricas em um dicionário.
    return {
        "Modelo": label,  # Nome ou descrição do modelo.
        "MSE": mse,       # Erro quadrático médio.
        "RMSE": rmse,     # Raiz do erro quadrático médio.
        "MAE": mae,       # Erro absoluto médio.
        "R2": r2,         # Coeficiente de determinação.
        "MAPE (%)": mape_val,  # Erro percentual absoluto médio.
    }