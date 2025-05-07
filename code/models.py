from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def create_mlp(input_dim: int) -> Sequential:
    """Cria um modelo MLP."""
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm(input_dim: int) -> Sequential:
    """Cria um modelo LSTM."""
    model = Sequential(
        [
            LSTM(50, activation="relu", input_shape=(1, input_dim)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model