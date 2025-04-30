from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.regularizers import l2
import numpy as np

def build(params):
    model_type = params["type"]
    architecture = params.get("architecture", "dense")  # default para 'dense'
    num_layers = int(params['num_layers'])
    num_neurons = int(params['num_neurons'])
    learning_rate = params['learning_rate']
    loss_function = params['loss_function']
    epochs = params['epochs']
    dir = params['dir']
    X_train = params['X_train']
    y_train = params['y_train']
    X_test = params['X_test']
    y_test = params['y_test']

    model = keras.Sequential()

    if architecture == "dense":
        model.add(Input(shape=(X_train.shape[1],)))
        for _ in range(num_layers):
            model.add(Dense(num_neurons, activation='relu', kernel_regularizer=l2(0.01)))
            num_neurons = max(1, num_neurons // 2)
        model.add(Dense(1, activation='linear'))

    elif architecture in ["lstm", "blstm"]:
        if len(X_train.shape) == 2:
            X_train = np.array(X_train)[..., None]
            X_test = np.array(X_test)[..., None]
        input_shape = (X_train.shape[1], X_train.shape[2])
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            units = max(1, num_neurons // (2 ** i))

            if i == 0:
                layer_input = Input(shape=input_shape)
                model.add(layer_input)

            lstm_layer = LSTM(units, return_sequences=return_seq)

            if architecture == "blstm":
                model.add(Bidirectional(lstm_layer))
            else:
                model.add(lstm_layer)

        model.add(Dense(1, activation='linear'))

    else:
        raise ValueError(f"Arquitetura desconhecida: {architecture}")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function,
                  metrics=['accuracy', MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()])

    if model_type == "optimization":
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        val_loss = history.history['val_loss'][-1]
        return val_loss
    else:
        model.summary()
        with open(f"{dir}/model-summary.txt", 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model
