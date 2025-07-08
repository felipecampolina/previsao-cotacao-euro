# Currency Exchange Rate Prediction (USD/EUR and JPY/EUR)

## Overview

This project focuses on predicting currency exchange rates for USD/EUR and JPY/EUR using machine learning models. It leverages historical data, including financial indicators such as SOFR (Secured Overnight Financing Rate) and Brent Oil prices, to train and evaluate models like MLP (Multi-Layer Perceptron) and LSTM (Long Short-Term Memory). The project also provides tools for visualizing predictions and evaluating model performance.

## Features

- **Data Preprocessing**: Consolidates and normalizes data from multiple sources (SOFR, Brent Oil, and currency exchange rates).
- **Model Training**: Supports training MLP and LSTM models with configurable hyperparameters (epochs, batch sizes).
- **Evaluation**: Calculates metrics such as RMSE, MAE, R², and MAPE to evaluate model performance.
- **Visualization**: Provides detailed plots comparing real vs. predicted values, including error shading and smoothing options.
- **Prediction**: Predicts exchange rates for the next 5 days using trained models and allows manual input for specific scenarios.
- **Data Consolidation**: Merges and cleans data from multiple CSV files into a unified dataset.

## Project Structure

```
previsao-cotacao-euro/
├── code/
│   ├── cria_arquivo_dados_finais.py  # Consolidates raw data into a unified dataset.
│   ├── data_processing.py            # Handles data loading, preprocessing, and normalization.
│   ├── evaluation.py                 # Provides evaluation metrics for model performance.
│   ├── main.py                       # Main script for running predictions and visualizations.
│   ├── test_models.py                # Trains models, evaluates them, and saves metrics.
│   ├── visualization.py              # Contains functions for plotting real vs. predicted values.
├── data/
│   ├── sofr.csv                      # Historical SOFR data.
│   ├── curency.csv                   # Historical currency exchange rates.
│   ├── brentOil.csv                  # Historical Brent Oil prices.
│   ├── dados_unificados.csv          # Unified dataset generated from raw data.
├── model/
│   ├── USD_LSTM_E150_B16.keras       # Example trained model for USD/EUR.
│   ├── JPY_LSTM_E150_B16.keras       # Example trained model for JPY/EUR.
├── metrics_results.json              # JSON file containing evaluation metrics for trained models.
└── README.md                         # Project documentation.
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/previsao-cotacao-euro.git
   cd previsao-cotacao-euro
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `data/` folder contains the necessary CSV files (`sofr.csv`, `curency.csv`, `brentOil.csv`).

## Usage

### 1. Data Consolidation
Run the script to merge and clean the raw data:
```bash
python code/cria_arquivo_dados_finais.py
```
This generates the `dados_unificados.csv` file in the `data/` folder.

### 2. Model Training and Evaluation
Train MLP and LSTM models with different hyperparameters and save evaluation metrics:
```bash
python code/main.py
```
Uncomment the `test_models_and_save_metrics` function in `main.py` to enable training.

### 3. Predict Next 5 Days
Use trained models to predict exchange rates for the next 5 days:
```bash
python code/main.py
```
Ensure the `predict_next_5_days_with_two_models` function is active in `main.py`.

### 4. Manual Input Prediction
Predict exchange rates for specific inputs by providing SOFR and Brent values:
```python
# Example in main.py
predict_next_5_days_manual_input(
    model_usd_path="model/USD_LSTM_E150_B16.keras",
    model_jpy_path="model/JPY_LSTM_E150_B16.keras",
    sofr_values=[4.3, 4.35, 4.4, 4.45, 4.5],
    brent_values=[65.1, 65.2, 65.3, 65.4, 65.5],
    scaler_y_usd=scaler_y_usd,
    scaler_y_jpy=scaler_y_jpy,
)
```

### 5. Visualization
Generate plots comparing real vs. predicted values:
```python
# Example in main.py
plot_real_vs_predicted(
    y_true=best_y_true,
    y_pred=best_y_pred,
    title="USD/EUR Prediction",
    smooth=True,
    shade_error=True
)
```

## Evaluation Metrics

The project evaluates models using the following metrics:
- **MSE (Mean Squared Error)**: Measures the average squared difference between real and predicted values.
- **RMSE (Root Mean Squared Error)**: Square root of MSE, providing error in the same unit as the target variable.
- **MAE (Mean Absolute Error)**: Average absolute difference between real and predicted values.
- **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in the target variable.
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error between real and predicted values.

## Example Results

Sample metrics for trained models (stored in `metrics_results.json`):
```json
[
    {
        "Modelo": "USD_MLP_E10_B8",
        "MSE": 0.00044863995132196166,
        "RMSE": 0.02118112252270785,
        "MAE": 0.016578233533644543,
        "R2": 0.7255461235654005,
        "MAPE (%)": 1.826227825000038
    },
    {
        "Modelo": "JPY_LSTM_E10_B8",
        "MSE": 9.035102738089296e-08,
        "RMSE": 0.0003005844762806173,
        "MAE": 0.00024060531308969815,
        "R2": 0.8608445288716737,
        "MAPE (%)": 3.3643233819081497
    }
]
```

## Dependencies

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Future Improvements

- Add support for additional financial indicators.
- Implement hyperparameter optimization for model training.
- Extend predictions to other currency pairs.
- Automate data fetching from APIs for real-time predictions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Felipe Campolina Soares de Paula  
Contact: [felipecampolinacc@gmail.com]
