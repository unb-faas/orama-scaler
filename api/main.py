from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Inicializar o app Flask
app = Flask(__name__)

# Carregar o modelo Keras
model = tf.keras.models.load_model("model.keras")

# Carregar os encorers salvos
encoders = joblib.load("encoders.pkl")

# Carregar o scaler salvo
scaler = joblib.load("scaler.pkl")

def categorize(data, encoders):
    for column in ['provider', 'usecase']:
        data[column] = encoders[column].fit_transform(data[column])
    return data
        
# Rota para previsão
@app.route('/predict_latency', methods=['POST'])
def predict_latency():
    try:
        # Extrair dados do corpo da requisição
        data = request.json
        
        # Campos esperados
        expected_fields = [
            "success",
            "concurrency", 
            "provider", 
            "usecase",
            "total_operands", 
            "distinct_operands",
            "total_operators", 
            "distinct_operators", 
            "time", 
            "bugs", 
            "effort",
            "volume", 
            "difficulty", 
            "vocabulary", 
            "length"
        ]
        
        # Validar presença dos campos
        if not all(field in data for field in expected_fields):
            return jsonify({"error": "Missing fields in input data"}), 400
        
        df = pd.DataFrame([data])

        
        data = categorize(df, encoders)
        
        return data.to_json()

        #transformed_fields = scaler.transform([[data["usecase"], data["provider"]]])

        #print(transformed_fields)

        #data["provider"] = encoders["provider"].fit_transform(data["provider"])
        #data["usecase"] = encoders["usecase"].fit_transforms(data["usecase"])


        #data["usecase"], data["provider"] = encoded_data[0]
        
        #for column, label_encoder in encoders.items():
        #    print("FIZ")
            #data[column] = label_encoder.transform(data[column])    
            
       
        # Converter dados para array NumPy (certifique-se de que os dados sejam numéricos)
        #input_data = np.array([[data[field] for field in expected_fields]], dtype=float)
        
        # Fazer a previsão
        #latency_prediction = model.predict(input_data)[0][0]
        
        # Retornar o resultado
        #return jsonify({"predicted_latency": str(latency_prediction)})
        return data
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Inicializar o servidor
if __name__ == '__main__':
    app.run(debug=True)
