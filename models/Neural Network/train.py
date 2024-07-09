import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carregar os Dados
data = pd.read_csv('../../dataset/outputs/dataset.csv')

label_encoder = LabelEncoder()
data['provider'] = label_encoder.fit_transform(data['provider'])
data['usecase'] = label_encoder.fit_transform(data['usecase'])
data['Latency'] = label_encoder.fit_transform(data['Latency'])

# 2. Pré-processamento
X = data.drop(columns=['Latency', 'timeStamp', 'label', 'usecase', 'success'], axis=1)
y = data['Latency']

# Codificação de variáveis categóricas (se houver)
# X = pd.get_dummies(X)  # Utilize esta linha se houver variáveis categóricas

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados (importante para Redes Neurais)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Construir o Modelo de Rede Neural
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Saída de regressão (valor contínuo)

# Função de métrica personalizada para R^2
def r2_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))


# Compilar o modelo
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2_metric])

# 4. Treinar o Modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# 5. Avaliar o Modelo
loss, mse = model.evaluate(X_test, y_test, verbose=1)
print(f'Mean Squared Error: {mse}')

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas adicionais
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# Plotar a perda de treinamento e validação
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()
