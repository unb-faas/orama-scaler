import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

# Carregar os dados do CSV
df = pd.read_csv('../../dataset/outputs/casestopredict.csv') 
df.dropna(inplace=True)

label_encoder = LabelEncoder()

# Aplicar o LabelEncoder ao campo 'Provider'
df['provider'] = label_encoder.fit_transform(df['provider'])
df['usecase'] = label_encoder.fit_transform(df['usecase'])
df['Latency'] = label_encoder.fit_transform(df['Latency'])


# Separar features
X_new = df.drop(['timeStamp', 'provider', 'usecase', 'label', 'concurrency', 'success'], axis=1)
y_new = df['Latency']

# Tratar variáveis categóricas (codificação one-hot)
X_new = pd.get_dummies(X_new, drop_first=True)

# Padronizar os dados
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Carregar o modelo salvo
model = load('modelo_random_forest.joblib')  # Substitua 'modelo_random_forest.joblib' pelo nome do seu modelo salvo

# Fazer previsões com o novo conjunto de dados
predictions = model.predict(X_new_scaled)

# Exibir as previsões
print("Previsões:")
print(predictions)

print("Campo 'Latency' vs. Previsões:")
print("--------------------------------")
print("| Latency | Previsões |")
print("--------------------------------")
for i in range(len(y_new)):
    print(f"| {y_new.iloc[i]} | {predictions[i]} |")
print("--------------------------------")

# Avaliar o desempenho do modelo
mse = mean_squared_error(y_new, predictions)

print("R2 Score")


r2 = r2_score(y_new, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# Calcular a acurácia
# Definir intervalo de tolerância
tolerance = 0.05  # ±5%

# Calcular o limite inferior e superior do intervalo
lower_limit = y_new * (1 - tolerance)
upper_limit = y_new * (1 + tolerance)

# Verificar se as previsões estão dentro do intervalo de tolerância
within_tolerance = (predictions >= lower_limit) & (predictions <= upper_limit)

# Calcular a porcentagem de previsões dentro do intervalo de tolerância
accuracy = within_tolerance.mean() * 100

print(f'Acurácia: {accuracy:.2f}%')
