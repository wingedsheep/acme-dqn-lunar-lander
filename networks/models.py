from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense


class Models:

    @staticmethod
    def sequential_dropout_model(input_shape, num_outputs: int, dropout_factor: float = 0.2, layer_size: int = 60):
        model = Sequential()
        model.add(Dense(layer_size, input_shape=input_shape, activation="relu"))
        model.add(Dropout(dropout_factor))
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(dropout_factor))
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(dropout_factor))
        model.add(Dense(num_outputs, activation='softmax'))
        return model

    @staticmethod
    def sequential_model(input_shape, num_outputs: int, layer_size: int = 60, hidden_layers: int = 3):
        model = Sequential()
        model.add(Dense(layer_size, input_shape=input_shape, activation="relu"))
        for _ in range(hidden_layers - 1):
            model.add(Dense(layer_size, activation="relu"))
        model.add(Dense(num_outputs, activation='softmax'))
        return model
