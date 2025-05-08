from hyperopt import fmin, tpe, hp, Trials
import modeling
from contextlib import redirect_stdout

def optimize(dir, X_train, y_train, X_test, y_test, arch, epochs=5, attempts=10):
    loss_functions = ['mean_squared_error','mean_absolute_error','huber']
    epochs_list = [epochs]
    # Define the search space for hyperparameters
    space = {
        "type": "optimization",
        'num_layers': hp.quniform('num_layers', 1, 20, 1),  # From 1 to 20 hidden layers
        'num_neurons': hp.quniform('num_neurons', 16, 256, 8),  # From 16 to 256 neurons per layer
        'learning_rate': hp.loguniform('learning_rate', -5, 0),  # Between 10^-5 and 1
        'loss_function': hp.choice('loss_function', loss_functions),  # Different loss functions
        'epochs': hp.choice('epochs', epochs_list),
        'dir':dir,
        'X_train': X_train, 
        'y_train': y_train, 
        'X_test': X_test,
        'y_test': y_test,
        'architecture': arch
    }

    # Create Trials to store results
    trials = Trials()

    # Run the optimization
    best = fmin(fn=modeling.build, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=attempts,  # Number of attempts
                trials=trials)
    
    best["epochs"] = epochs_list[best["epochs"]]
    best["loss_function"] = loss_functions[best["loss_function"]]
    print("------------*******----------------")
    print("Best Hyperparameters Found:", best)
    print("------------*******----------------")
    with open(f"{dir}/model-optimization-{arch}.txt", 'w') as f:
            with redirect_stdout(f):
                print("Best Hyperparameters Found:", best)
    return best
