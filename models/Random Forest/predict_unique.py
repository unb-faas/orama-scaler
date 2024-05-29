import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

label_encoder = LabelEncoder()

# Dados a serem previstos
data_to_predict = {
    'Latency': [100000],
    'length': [3.166666666666664],
    'vocabulary': [24.0],
    'difficulty': [4.20969345969346],
    'volume': [234.73266874750584],
    'effort': [1758.8517899129445],
    'bugs': [0.07824422291583527],
    'time': [97.7139883284969],
    'distinct_operators': [6.666666666666667],
    'total_operators': [21.833333333333332],
    'distinct_operands': [17.333333333333332],
    'total_operands': [22.333333333333332]
}


# Converter os dados para um DataFrame
df_to_predict = pd.DataFrame(data_to_predict)

# Separar features
#X_new = df_to_predict.drop(['timeStamp', 'provider', 'usecase'], axis=1)

# Padronizar os dados
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(df_to_predict)

# Carregar o modelo salvo
model = load('modelo_random_forest.joblib')

# Fazer previsões com os novos dados
predictions = model.predict(X_new_scaled)

# Exibir as previsões
print("Previsões:")
print(predictions)
