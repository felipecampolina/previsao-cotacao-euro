import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath: str):
    """Carrega e pré-processa os dados."""
    
    # 1) Carregar os dados ---------------------------------------------------
    # Lê o arquivo CSV especificado no caminho `filepath`.
    df = pd.read_csv(filepath)
    
    # 2) Conversão e ordenação de datas --------------------------------------
    # Converte a coluna "Date" para o formato datetime.
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Ordena o DataFrame pela coluna "Date" e reseta os índices.
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Remove linhas com valores ausentes (NaN).
    df = df.dropna()

    # 3) Normalização dos dados ----------------------------------------------
    # Define as colunas de entrada (features) que serão normalizadas.
    features = ["SOFR", "Brent"]
    
    # Cria um escalador MinMaxScaler para normalizar as features.
    scaler_x = MinMaxScaler()
    
    # Aplica a normalização às colunas "SOFR" e "Brent".
    X = scaler_x.fit_transform(df[features])

    # Cria escaladores MinMaxScaler para as variáveis de saída (targets).
    scaler_y_usd = MinMaxScaler()
    scaler_y_jpy = MinMaxScaler()
    
    # Normaliza a coluna "USD_EUR" (taxa de câmbio USD/EUR).
    y_usd = scaler_y_usd.fit_transform(df[["USD_EUR"]])
    
    # Normaliza a coluna "JPY_EUR" (taxa de câmbio JPY/EUR).
    y_jpy = scaler_y_jpy.fit_transform(df[["JPY_EUR"]])

    # 4) Divisão dos dados em treino e teste ---------------------------------
    # Divide os dados em conjuntos de treino e teste.
    # - `X`: Dados de entrada normalizados.
    # - `y_usd`: Taxa de câmbio USD/EUR normalizada.
    # - `y_jpy`: Taxa de câmbio JPY/EUR normalizada.
    # O conjunto de teste corresponde a 20% dos dados, e o restante é usado para treino.
    # O parâmetro `random_state` garante reprodutibilidade.
    X_train, X_test, y_usd_train, y_usd_test, y_jpy_train, y_jpy_test = train_test_split(
        X, y_usd, y_jpy, test_size=0.2, random_state=42
    )

    # 5) Retornar os resultados ----------------------------------------------
    # Retorna o DataFrame original, os conjuntos de treino/teste e os escaladores.
    return (
        df,  # DataFrame original (não normalizado)
        X_train,  # Dados de entrada para treino
        X_test,   # Dados de entrada para teste
        y_usd_train,  # Taxa USD/EUR para treino
        y_usd_test,   # Taxa USD/EUR para teste
        y_jpy_train,  # Taxa JPY/EUR para treino
        y_jpy_test,   # Taxa JPY/EUR para teste
        scaler_x,     # Escalador usado para normalizar as features
        scaler_y_usd, # Escalador usado para normalizar USD/EUR
        scaler_y_jpy, # Escalador usado para normalizar JPY/EUR
    )