from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from contextlib import redirect_stdout
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError

def build(dir):
    model = Sequential()
    model.add(Dense(64, input_dim=15, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()])
    model.summary()
    with open(f"{dir}/model-summary.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary()
    return model

