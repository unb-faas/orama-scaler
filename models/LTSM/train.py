import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
data = pd.read_csv('../../dataset/outputs/dataset.csv')

label_encoder = LabelEncoder()
data['provider_encoded'] = label_encoder.fit_transform(data['provider'])
data['usecase_encoded'] = label_encoder.fit_transform(data['usecase'])
data['Latency'] = label_encoder.fit_transform(data['Latency'])

# Separar recursos e rótulo
X = data.drop(columns=['Latency', 'timeStamp', 'label', 'usecase', 'provider', 'usecase'], axis=1)
y = data['Latency']

# Codificar colunas categóricas, se houver
# label_encoder = LabelEncoder()
# X['provider_encoded'] = label_encoder.fit_transform(X['provider'])
# X['usecase_encoded'] = label_encoder.fit_transform(X['usecase'])

# Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Redimensionar os dados para que possam ser usados na LSTM
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Construir o modelo
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Salvar o modelo
model.save('lstm_model.keras')

# 5. Avaliar o Modelo
loss, mse = model.evaluate(X_test, y_test, verbose=1)
print(f'Mean Squared Error: {mse}')

# Função de métrica personalizada para R^2
def r2_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

# Fazer previsões
predictions = model.predict(X_test)

# Calcular métricas adicionais
r2 = r2_score(y_test, predictions)
print(f'R^2 Score: {r2}')

# Exibir algumas previsões
for i in range(5):
    print(f"Predicted: {predictions[i][0]}, Actual: {y_test.iloc[i]}")