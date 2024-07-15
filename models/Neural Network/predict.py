import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Função de métrica personalizada para R^2 (necessária para carregar o modelo)
def r2_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

# Carregar o modelo salvo
model = load_model('model.keras', custom_objects={'r2_metric': r2_metric})

# Carregar e pré-processar os dados para predição
data = pd.read_csv('to-predict.csv')

# Supõe que o arquivo de predição não contém a coluna 'Latency'
X = data.drop(columns=['target'])
y_true = data['target']

# Normalizar os dados (importante para Redes Neurais)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Fazer previsões
predictions = model.predict(X)


data['Predicted_Latency'] = predictions
data['target'] = y_true
data['Difference'] = data['Predicted_Latency'] - data['target']
data['Percentual_Diferenca'] = (abs(data['Difference']) / data['target']) * 100

data = data.drop(columns=['success','total_operands','distinct_operands','total_operators','distinct_operators','time','bugs','effort','volume','difficulty','vocabulary','length'])


somatorio = sum(data['Percentual_Diferenca'])
print("Somatorio do percentual de diferenca: " + str(somatorio))

# Se você quiser salvar as previsões em um arquivo CSV
#output = pd.DataFrame(predictions, columns=['Predicted_Latency'])
#output.to_csv('predictions.csv', index=False)

data.to_csv('predictions_with_data.csv', index=False)

#data.to_csv('predictions_with_data.csv', index=False)

print('Predictions saved to predictions.csv')
