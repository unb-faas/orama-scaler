import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Função para treinar e avaliar o modelo até alcançar o MSE desejado ou o número máximo de execuções
def train_until_target_mse(dataset_path, target_mse, max_executions):
    # Carregar o dataset a partir do arquivo CSV
    dataset = pd.read_csv(dataset_path)

    label_encoder = LabelEncoder()
    dataset['provider'] = label_encoder.fit_transform(dataset['provider'])
    dataset['usecase'] = label_encoder.fit_transform(dataset['usecase'])
    dataset['Latency'] = label_encoder.fit_transform(dataset['Latency'])

    # dataset = dataset.query('provider == 0')
    dataset = dataset.query('concurrency == 128')

    # Dividir o dataset em features (X) e target (y)
    X = dataset.drop(columns=['Latency', 'timeStamp', 'label', 'usecase', 'success'])
    y = dataset['Latency']

    #print(X)
   
    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definir a grade de parâmetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['rbf', 'linear']
    }

    # Inicializar e ajustar o GridSearchCV com paralelização
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    print(f'Best parameters: {best_params}')


    execution_count = 0
    mse = float('inf')

    while mse > target_mse and execution_count < max_executions:
        # Inicializar e treinar o modelo de regressão Random Forest
        #rf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model = SVR(**best_params)
        rf.fit(X_train, y_train)

        # Fazer previsões e avaliar o modelo
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        joblib.dump(rf, 'random_forest_model.pkl')

        execution_count += 1
        print(f'Execution {execution_count}: MSE = {mse}, R² = {r2}')

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
        #         print(error_actual)
        #         errorn = float(errorn + error_actual)
        #         print(errorn)
        #         counter = counter + 1
            
        #     print(f"| {y_test.iloc[i]} | {y_pred[i]} | {y_pred[i] - y_test.iloc[i]} | {(100 * abs(y_pred[i] - y_test.iloc[i]))/y_test.iloc[i]} |")
        # print("-------------------------------")
        # print(errorn)
        # print(f"Media de Erros: {errorn/counter}")


    return rf, mse, execution_count

# Parâmetros
dataset_path = '../../dataset/outputs/dataset.csv'

target_mse = 10.0  # Valor desejado de MSE
max_executions = 1  # Número máximo de execuções


# Treinar o modelo até atingir o MSE desejado ou o número máximo de execuções
final_model, final_mse, total_executions = train_until_target_mse(dataset_path, target_mse, max_executions)

print(f'\nFinal MSE: {final_mse}')
print(f'Total Executions: {total_executions}')

