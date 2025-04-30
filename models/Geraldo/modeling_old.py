from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.regularizers import l2

# Objective function to minimize
def build(params):
    type = params["type"]
    num_layers = int(params['num_layers'])  # Number of hidden layers
    num_neurons = int(params['num_neurons'])  # Neurons per layer
    learning_rate = params['learning_rate']  # Learning rate
    loss_function = params['loss_function']  # Loss function
    epochs = params['epochs']  # Epochs
    dir = params['dir']
    X_train = params['X_train']
    y_train = params['y_train']
    X_test = params['X_test']
    y_test = params['y_test']

    # Creating the dynamic neural network
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
    for _ in range(num_layers):  # Add the defined number of hidden layers
        model.add(layers.Dense(num_neurons, activation='relu', kernel_regularizer=l2(0.01)))
        #model.add(layers.Dropout(0.3))  # Regularization
        num_neurons = num_neurons // 2
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function,
                  metrics=['accuracy', MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()])
    
    if type == "optimization":
        # Train the model (quick training with only X epochs for optimization)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))

        # Get the last validation loss
        val_loss = history.history['val_loss'][-1]
    
        return val_loss   # Hyperopt minimizes this value
    else:
        model.summary()
        with open(f"{dir}/model-summary.txt", 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

# def build(dir):
#     model = Sequential()
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.7))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.5))
#     #model.add(Dense(16, activation='relu'))
#     #model.add(Dropout(0.3))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(1, activation='linear'))
#     model.compile(optimizer='adam', loss='mean_squared_error',  metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()])
#     model.summary()
#     with open(f"{dir}/model-summary.txt", 'w') as f:
#         with redirect_stdout(f):
#             model.summary()
#     return model

