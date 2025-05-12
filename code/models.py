from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

def create_mlp(input_dim: int) -> Sequential:
    """
    Cria um modelo de rede neural do tipo MLP (Perceptron Multicamadas).
    
    - A arquitetura é composta por camadas densas totalmente conectadas.
    - Ideal para dados tabulares ou problemas de regressão simples.

    Args:
        input_dim (int): Dimensão da entrada (número de features).

    Returns:
        Sequential: Modelo MLP compilado.
    """
    model = Sequential(
        [
            # Camada de entrada com dimensão definida explicitamente
            Input(shape=(input_dim,)),  
            
            # Primeira camada oculta com 64 neurônios e função de ativação ReLU
            Dense(64, activation="relu"),
            
            # Segunda camada oculta com 64 neurônios e função de ativação ReLU
            Dense(64, activation="relu"),
            
            # Camada de saída com 1 neurônio (regressão para prever um único valor)
            Dense(1),
        ]
    )
    # Compilação do modelo com otimizador Adam e função de perda MSE (Erro Quadrático Médio)
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm(input_dim: int) -> Sequential:
    """
    Cria um modelo de rede neural do tipo LSTM (Long Short-Term Memory).
    
    - A arquitetura é projetada para lidar com dados sequenciais ou temporais.
    - LSTMs são eficazes para capturar dependências de longo prazo em séries temporais.

    Args:
        input_dim (int): Dimensão da entrada (número de features por timestep).

    Returns:
        Sequential: Modelo LSTM compilado.
    """
    model = Sequential(
        [
            # Camada de entrada para dados sequenciais com formato (timesteps=1, features=input_dim)
            Input(shape=(1, input_dim)),  
            
            # Camada LSTM com 50 unidades e função de ativação ReLU
            # - Captura padrões temporais nos dados
            LSTM(50, activation="relu"),
            
            # Camada de saída com 1 neurônio (regressão para prever um único valor)
            Dense(1),
        ]
    )
    # Compilação do modelo com otimizador Adam e função de perda MSE (Erro Quadrático Médio)
    model.compile(optimizer="adam", loss="mse")
    return model