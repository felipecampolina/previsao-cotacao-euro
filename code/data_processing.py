import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath: str):
    """Carrega e pré-processa os dados."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()

    # Normalização
    features = ["SOFR", "Brent"]
    scaler_x = MinMaxScaler()
    X = scaler_x.fit_transform(df[features])

    scaler_y_usd = MinMaxScaler()
    y_usd = scaler_y_usd.fit_transform(df[["USD_EUR"]])

    scaler_y_jpy = MinMaxScaler()
    y_jpy = scaler_y_jpy.fit_transform(df[["JPY_EUR"]])

    # Divisão treino/teste
    X_train, X_test, y_usd_train, y_usd_test, y_jpy_train, y_jpy_test = train_test_split(
        X, y_usd, y_jpy, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_usd_train,
        y_usd_test,
        y_jpy_train,
        y_jpy_test,
        scaler_x,
        scaler_y_usd,
        scaler_y_jpy,
    )