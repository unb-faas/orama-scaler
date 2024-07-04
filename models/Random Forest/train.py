import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from joblib import dump

print("Carregar os dados do CSV")
df = pd.read_csv('../../dataset/outputs/dataset.csv')

df = df.loc[df['concurrency'] == 100]
df = df.loc[df['usecase'] == "api4dbaas"]

df.dropna(inplace=True)
label_encoder = LabelEncoder()

# Aplicar o LabelEncoder ao campo 'Provider'
df['provider'] = label_encoder.fit_transform(df['provider'])
df['usecase'] = label_encoder.fit_transform(df['usecase'])
df['Latency'] = label_encoder.fit_transform(df['Latency'])

print("Separar features e target")
X = df.drop(['timeStamp', 'Latency','provider', 'usecase', 'label', 'concurrency', 'success'], axis=1)
y = df['Latency']

# Tratar variáveis categóricas (codificação one-hot)
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


print("Padronizar os dados")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Instanciar e treinar o modelo")
model = RandomForestRegressor(n_estimators=100000, random_state=42)

print("Fit")

model.fit(X_train_scaled, y_train)
dump(model, 'modelo_random_forest.joblib')

print("Predict")

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

print(y_test)

print(y_pred)


print("Evaluate")

print("Valores Reais vs. Valores Previstos:")
print("-------------------------------")
print("| Valores Reais | Valores Previstos |")
print("-------------------------------")
for i in range(len(y_test)):
    print(f"| {y_test.iloc[i]} | {y_pred[i]} |")
print("-------------------------------")



# Avaliar o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)

print("R2 Score")


r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

y_pred_rounded = np.round(y_pred).astype(int)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred_rounded)

print(f'Acurácia: {accuracy}')
