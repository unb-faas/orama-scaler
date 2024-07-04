import joblib
import pandas as pd

# Função para carregar o modelo e fazer uma predição
def predict_with_saved_model(model_path, new_data):
    # Carregar o modelo treinado
    loaded_model = joblib.load(model_path)

    # Realizar a predição com o modelo carregado
    predictions = loaded_model.predict(new_data)

    return predictions

# Caminho onde o modelo foi salvo
model_path = 'random_forest_model.pkl'

data_list = []

concurrences = [1,2,8,16,32,64,128,256,512]

for i in concurrences:
    # Dados para fazer a predição (exemplo)
    new_data = pd.DataFrame({
        # 'success': [True, True, True],
        'concurrency': [i, i, i],
        'provider': [0, 1, 3],
        # 'usecase': [1, 1, 1],
        'total_operands': [9, 9, 9],
        'distinct_operands': [9, 9, 9],
        'total_operators': [12, 12, 12],
        'distinct_operators': [5, 5, 5],
        'time': [0.026651484454403226, 0.026651484454403226, 0.026651484454403226],
        'bugs': [199.8861334080242, 199.8861334080242, 199.8861334080242],
        'effort': [79.95445336320968, 79.95445336320968, 79.95445336320968],
        'volume': [2.5, 2.5, 2.5],
        'difficulty': [14, 14, 14],
        'vocabulary': [21, 21, 21],
        'length': [807, 807, 807]
    })
    data_list.append(new_data)


# new_data = pd.DataFrame({
#     # 'success': [True, True, True],
#     # 'concurrency': [512, 512, 512],
#     'provider': [0, 1, 2],
#     # 'usecase': [2, 2, 2],
#     'total_operands': [111, 111, 111],
#     'distinct_operands': [29, 29, 29],
#     'total_operators': [101, 101, 101],
#     'distinct_operators': [23, 23, 23],
#     'time': [2955.252099, 2955.252099, 2955.252099],
#     'bugs': [0.402831, 0.402831, 0.402831],
#     'effort': [53194.537781, 53194.537781, 53194.537781],
#     'volume': [1208.493220, 1208.493220, 1208.493220],
#     'difficulty': [44.017241, 44.017241, 44.017241],
#     'vocabulary': [52, 52, 52],
#     'length': [212, 212, 212]
# })


# Fazer a predição com o modelo carregado
try:
    for data in data_list:
        print(data)
        predictions = predict_with_saved_model(model_path, data)
        print("Predições:")
        print(predictions)
except Exception as e:
    print(f"Ocorreu um erro durante a predição: {e}")