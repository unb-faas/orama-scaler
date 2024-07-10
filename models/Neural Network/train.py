import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


# 1. Carregar os Dados
data = pd.read_csv('../../dataset/outputs/dataset.csv')

pd.set_option('display.max_rows', 100)  # Número máximo de linhas a serem exibidas
pd.set_option('display.max_columns', 20)  # Número máximo de colunas a serem exibidas


label_encoder = LabelEncoder()
data['provider_encoded'] = label_encoder.fit_transform(data['provider'])
data['usecase_encoded'] = label_encoder.fit_transform(data['usecase'])
data['Latency'] = label_encoder.fit_transform(data['Latency'])

# - Provider Encodes
# 0 - AFC
# 1 - AZF
# 2 - GCF
# 3 - Lamba

#data = data.query('usecase_encoded == 2')
#data = data.query('provider_encoded == 3')

data_temp = data.drop(columns=['Latency', 'timeStamp', 'label', 'success','concurrency','total_operands','distinct_operands','total_operators','distinct_operators','time','bugs','effort','volume','difficulty','vocabulary','length'], axis=1)
#mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print(data_temp)

# providers = ['afc','azf','gcf','lambda']
# concurrences = [1,2,128]
# usecases = ['calc','api4dbaas','api4os']
# for usecase in usecases:
#     for concurrence in concurrences:
#         for provider in providers:
#             filtered_df = data[(data['provider'] == provider) & (data['concurrency'] == concurrence) & (data['usecase'] == usecase)]
#             mean_latency = filtered_df['Latency'].mean()
#             print(str(usecase) + ' ' + str(provider) + ' ' + str(concurrence) + ' ' + str(mean_latency))

#exit()

# 2. Pré-processamento
X = data.drop(columns=['Latency', 'timeStamp', 'label', 'usecase', 'provider', 'usecase'], axis=1)
y = data['Latency']

#print(X.columns)

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
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))  # Saída de regressão (valor contínuo)

# Função de métrica personalizada para R^2
def r2_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

adam = Adam(learning_rate=0.0001)
sgd = SGD(learning_rate=0.001, momentum=0.0, nesterov=False)

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2_metric])

# 4. Treinar o Modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

model.save('model.keras')


# 5. Avaliar o Modelo
loss, mse = model.evaluate(X_test, y_test, verbose=1)
print(f'Mean Squared Error: {mse}')

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas adicionais
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# print("Valores Reais vs. Valores Previstos:")
# print("-------------------------------")
# print("| Valores Reais | Valores Previstos |")
# print("-------------------------------")
# errorn = 0
# counter = 0 
# print(errorn)
# for i in range(len(y_test)):
#     if (y_test.iloc[i]):
#         error_actual = float((100 * abs(y_pred[i] - y_test.iloc[i]))/y_test.iloc[i])
#         #print(error_actual)
#         errorn = float(errorn + error_actual)
#         #print(errorn)
#         counter = counter + 1
    
#     print(f"| {y_test.iloc[i]} | {y_pred[i]} | {y_pred[i] - y_test.iloc[i]} | {(100 * abs(y_pred[i] - y_test.iloc[i]))/y_test.iloc[i]} |")
# print("-------------------------------")
# print(errorn)
# print(f"Media de Erros: {errorn/counter}")

# Plotar a perda de treinamento e validação
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()


