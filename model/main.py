try:
    import os
    from datetime import datetime
    import preprocessing
    import division
    import modeling
    import optimization
    import train
    import evaluation
    import joblib
    from contextlib import redirect_stdout

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    ## Force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    ## Flags
    optimization_enabled = int(input("Do you want to use optimizator (0/1)? "))
    optimization_enabled = True if optimization_enabled==1 else False
    epoches_optimization = 5
    if optimization_enabled:
        epoches_optimization = int(input("Epoches in optimization: "))
        attempts_optimization = int(input("Attempts in optimization: "))
    epoches_training = int(input("Epoches in training: "))

    ####
    # Creates results structure
    ####
    main_dir = "results"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Dir '{main_dir}' created.")
    else:
        print(f"Dir '{main_dir}' exists.")
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-opt_{optimization_enabled}-opt_ep_{epoches_optimization}-train_ep_{epoches_training}"
    sub_dir = os.path.join(main_dir, timestamp)
    os.makedirs(sub_dir, exist_ok=True)
    print(f"Subdir '{sub_dir}' created.")

    print("----------------#####################################----------------")
    print("----------------#      Init - preprocessing         #----------------")
    print("----------------#####################################----------------")

    ####
    # Loads Dataset
    ####
    data = preprocessing.load_dataset('../dataset/outputs/dataset.csv')
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

    joblib.dump(encoders, f'{sub_dir}/encoders.pkl')

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
    print("#        Remove Spikes with         #")
    print("#   WMA - Weighted Moving Average   #")
    print("#####################################")
    data_without_spikes = preprocessing.remove_spikes(data_without_missing)

    #data_without_spikes = data_without_missing

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

    data_normalized_z_score, scaler = preprocessing.normalize(data_without_spikes)
    print(data_normalized_z_score)
    joblib.dump(scaler, f'{sub_dir}/scaler.pkl')

    #data_normalized_z_score_temp = preprocessing.remove_spikes(data_normalized_z_score)

    #data_normalized_z_score = data_normalized_z_score_temp

    print("#####################################")
    print("#          winsorize                #")
    print("#####################################")
    data_winsored = preprocessing.winsorization(data_normalized_z_score)

    # SUGESTÃO 05 ##########################################################
    #  Verificação de Outliers:
    #  - Identifique e trate possíveis outliers nas colunas numéricas, 
    #  especialmente nas colunas que têm valores muito elevados, 
    #  pois isso pode distorcer o treinamento do modelo.
    ########################################################################
    print("#####################################")
    print("#         Identify outliers         #")
    print("#####################################")
    preprocessing.identify_outliers(data_winsored)

    print("#####################################")
    print("#    Replace outliers with median   #")
    print("#####################################")
    #data_without_outliers = preprocessing.replace_outliers_with_median(data_winsored)
    data_without_outliers = preprocessing.remove_outliers(data_winsored)
    #data_without_outliers = data_winsored
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

    print("#####################################")
    print("#     Reducing Scale with PCA       #")
    print("#####################################")
    data_with_pca = preprocessing.reduce_scale_pca(data_without_outliers)
    preprocessing.correlation_analysis(data_with_pca, sub_dir, True, "reduced")
    data_reduced = preprocessing.remove_reduced_collumns(data_with_pca)
    preprocessing.correlation_analysis(data_reduced, sub_dir, True, "reduced_cleaned")

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
    X_train, X_test, y_train, y_test = division.divide(data_reduced)
    print("----------------#####################################----------------")
    print("----------------#          Division finished        #----------------")
    print("----------------#####################################----------------")

    best_params = {
        'Dense':{   
            'epochs': 3, 
            'learning_rate': 0.0030487328393528374, 
            'loss_function': 'mean_squared_error', 
            'num_layers': 1.0, 
            'num_neurons': 120.0
        },
        'LSTM':{   
            'epochs': 3, 
            'learning_rate': 0.0030487328393528374, 
            'loss_function': 'mean_squared_error', 
            'num_layers': 1.0, 
            'num_neurons': 120.0
        },
        'BLSTM':{   
            'epochs': 3, 
            'learning_rate': 0.0030487328393528374, 
            'loss_function': 'mean_squared_error', 
            'num_layers': 1.0, 
            'num_neurons': 120.0
        }
    }

    arch_result = {}
    archs = ['Dense', 'LSTM', 'BLSTM']
    batch_size = 32

    for arch in archs:

        print("----------------#####################################----------------")
        print(f"----------------#           Init - {arch}        #----------------")
        print("----------------#####################################----------------")

        if optimization_enabled == True:
            print("----------------#####################################----------------")
            print("----------------#        Init - Optimization        #----------------")
            print("----------------#####################################----------------")
            best_params[arch] = optimization.optimize(sub_dir, X_train, y_train, X_test, y_test, arch, batch_size, epochs=epoches_optimization, attempts=attempts_optimization)
            print("----------------#####################################----------------")
            print("----------------#       Optimization finished       #----------------")
            print("----------------#####################################----------------")

        print("----------------#####################################----------------")
        print("----------------#          Init - Modeling          #----------------")
        print("----------------#####################################----------------")
        params = best_params[arch]
        params['X_train'] = X_train
        params['y_train'] = y_train
        params['X_test'] = X_test
        params['y_test'] = y_test
        params['dir'] = sub_dir
        params['type'] = "train"
        params['epochs'] = epoches_training
        params['architecture'] = arch
        params['batch_size'] = batch_size
        
        model = modeling.build(params)
        print("----------------#####################################----------------")
        print("----------------#          Modeling finished        #----------------")
        print("----------------#####################################----------------")

        print("----------------#####################################----------------")
        print("----------------#          Init - Training          #----------------")
        print("----------------#####################################----------------")
        train_results, model = train.fit(sub_dir, X_train, y_train, X_test, y_test, model, arch, batch_size, int(params['epochs']))
        print("----------------#####################################----------------")
        print("----------------#          Training finished        #----------------")
        print("----------------#####################################----------------")

        arch_result[arch] = {
                "params": params, 
                "model": model,
                "train_results": train_results,
                "X_test":X_test,
                "y_test": y_test,
                "scaler": scaler, 
                "encoders": encoders
        }

        print("----------------#####################################----------------")
        print("----------------#         Init - Evaluation         #----------------")
        print("----------------#####################################----------------")
        evaluation.evaluate({arch:arch_result[arch]}, X_test, y_test, scaler, encoders, arch, sub_dir)
        print("----------------#####################################----------------")
        print("----------------#         Evaluation finished       #----------------")
        print("----------------#####################################----------------")     

        with open(f"{sub_dir}/results_{arch}.txt", 'w') as f:
            with redirect_stdout(f):
                print(f"{arch} results:", arch_result[arch])

        print("----------------#####################################----------------")
        print(f"----------------#         {arch} - finished      #----------------")
        print("----------------#####################################----------------")

    with open(f"{sub_dir}/results_consolidated.txt", 'w') as f:
            with redirect_stdout(f):
                print(f"Results:", arch_result)

    print("----------------##################################################----------------")
    print("----------------#         Init - Consolidated Evaluation         #----------------")
    print("----------------##################################################----------------")
    evaluation.evaluate(arch_result, X_test, y_test, scaler, encoders, arch, sub_dir)
    print("----------------##################################################----------------")
    print("----------------#         Consolidated Evaluation finished       #----------------")
    print("----------------##################################################----------------") 
    
    print("----------------#####################################----------------")
    print(f"----------------#           All finished            #----------------")
    print("----------------#####################################----------------")

except Exception as e:
    print(f"Erro: {e}")