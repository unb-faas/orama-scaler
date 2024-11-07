import os
from datetime import datetime
import preprocessing
import division
import modeling
import train
import evaluation

####
# Creates results structure
####
main_dir = "results"
if not os.path.exists(main_dir):
    os.makedirs(main_dir)
    print(f"Dir '{main_dir}' created.")
else:
    print(f"Dir '{main_dir}' exists.")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
sub_dir = os.path.join(main_dir, timestamp)
os.makedirs(sub_dir, exist_ok=True)
print(f"Subdir '{sub_dir}' created.")

print("----------------#####################################----------------")
print("----------------#      Init - preprocessing         #----------------")
print("----------------#####################################----------------")

####
# Loads Dataset
####
data = preprocessing.load_dataset('../../dataset/outputs/dataset.csv')
print("#####################################")
print("#           Initial Dataset         #")
print("#####################################")
print(data)
print(preprocessing.list_columns(data))

####
# Removes irrelevant collumns
####
data = preprocessing.remove_unused_collumns(data)

print("#####################################")
print("#             Categorize            #")
print("#####################################")
# SUGESTÃO 04 ##########################################################
#  Codificação de Variáveis Categóricas:
#  - As colunas "provider_encoded" e "usecase_encoded" já estão codificadas, 
#  mas se existirem outras variáveis categóricas, como a coluna success, 
#  verifique se faz sentido codificá-las numericamente ou usar técnicas 
#  como One-Hot Encoding.
########################################################################
data_categorized, encoders = preprocessing.categorize(data)
print(data_categorized)
print(encoders)

print("#####################################")
print("#          Clean duplicates         #")
print("#####################################")
# SUGESTÃO 01 ##########################################################
#  Verificação e Remoção de Duplicatas:
#  - O dataset parece conter muitas linhas repetidas. 
#  Verifique e remova registros duplicados para evitar que o modelo 
#  de aprendizado de máquina aprenda padrões irrelevantes.
########################################################################
data_cleaned = preprocessing.remove_duplicates(data_categorized)

print("#####################################")
print("#        Remove Missing values      #")
print("#####################################")
# SUGESTÃO 02 ##########################################################
#  Tratamento de Valores Ausentes ou Inválidos:
#  - Verifique se há valores ausentes ou inválidos em colunas importantes. 
#  Se existirem, decida se deseja preenchê-los com a média, 
#  mediana, moda ou simplesmente removê-los. 
#  Obs: Eu não notei isso, mas é bom verificar.
########################################################################
data_without_missing = preprocessing.check_and_remove_missing_values(data_cleaned)

print("#####################################")
print("#       Normalize with z-score      #")
print("#####################################")
# SUGESTÃO 03 ##########################################################
#  Normalização/Escalonamento:
#  - As colunas como "time", "bugs", "effort", "volume", entre outras, 
#  possuem valores muito grandes. Escalone esses valores usando técnicas 
#  como Min-Max Scaling ou Z-score para normalizar a amplitude dos dados, 
#  o que pode melhorar a performance de alguns algoritmos.
########################################################################
#data_normalized_z_score = preprocessing.normalize_z_score(data_without_missing)
#print(data_normalized_z_score)

data_normalized_z_score, scaler = preprocessing.normalize(data_without_missing)
print(data_normalized_z_score)


# SUGESTÃO 05 ##########################################################
#  Verificação de Outliers:
#  - Identifique e trate possíveis outliers nas colunas numéricas, 
#  especialmente nas colunas que têm valores muito elevados, 
#  pois isso pode distorcer o treinamento do modelo.
########################################################################
print("#####################################")
print("#         Indetify outliers         #")
print("#####################################")
preprocessing.identify_outliers(data_normalized_z_score)
print("#####################################")
print("#    Replace outliers with median   #")
print("#####################################")
#data_without_outliers = preprocessing.replace_outliers_with_median(data_normalized_z_score)
data_without_outliers = preprocessing.remove_outliers(data_normalized_z_score)
print(data_without_outliers)

# SUGESTÃO 06 ##########################################################
#  Análise de Correlação:
#  - Avalie a correlação entre as variáveis preditoras 
#  ("total_operands", "distinct_operands", "total_operators", etc.) 
#  para identificar redundâncias. 
#  Variáveis altamente correlacionadas podem ser removidas ou combinadas 
#  para reduzir a dimensionalidade.
########################################################################
preprocessing.correlation_analysis(data_without_outliers, sub_dir)

print("----------------#####################################----------------")
print("----------------#      Preprocessing finished       #----------------")
print("----------------#####################################----------------")

# SUGESTÃO 07 ##########################################################
#  Divisão de Dados de Treinamento e Teste:
#  - Após o pré-processamento, divida o dataset em conjuntos de treinamento 
#  e teste para avaliar a performance do modelo de forma justa. Use o famoso 
#  70% para treino e 30% para teste.
########################################################################
print("----------------#####################################----------------")
print("----------------#          Init - division          #----------------")
print("----------------#####################################----------------")
X_train, X_test, y_train, y_test = division.divide(data_without_outliers)
print("----------------#####################################----------------")
print("----------------#          Division finished        #----------------")
print("----------------#####################################----------------")

print("----------------#####################################----------------")
print("----------------#          Init - Modeling          #----------------")
print("----------------#####################################----------------")
model = modeling.build(sub_dir)
print("----------------#####################################----------------")
print("----------------#          Modeling finished        #----------------")
print("----------------#####################################----------------")

print("----------------#####################################----------------")
print("----------------#          Init - Training          #----------------")
print("----------------#####################################----------------")
train_results, model = train.fit(X_train, y_train, model, sub_dir)
print("----------------#####################################----------------")
print("----------------#          Training finished        #----------------")
print("----------------#####################################----------------")

print("----------------#####################################----------------")
print("----------------#         Init - Evaluation         #----------------")
print("----------------#####################################----------------")
evaluation.evaluate(train_results, model, X_test, y_test, sub_dir, scaler, encoders)
print("----------------#####################################----------------")
print("----------------#         Evaluation finished       #----------------")
print("----------------#####################################----------------")