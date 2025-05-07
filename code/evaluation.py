import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(y_true_scaled, y_pred_scaled, scaler, label: str):
    """Calcula métricas no espaço original (desnormalizado)."""
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    return {
        "Modelo": label,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE (%)": mape_val,
    }