from hyperopt import fmin, tpe, hp, Trials
import modeling
from contextlib import redirect_stdout

def optimize(dir, X_train, y_train, X_test, y_test):
    attempts = 20
    loss_functions = ['mean_squared_error','mean_absolute_error','huber']
    epochs = [5]
    # Define the search space for hyperparameters
    space = {
        "type": "optimization",
        'num_layers': hp.quniform('num_layers', 1, 50, 1),  # From 1 to 5 hidden layers
        'num_neurons': hp.quniform('num_neurons', 16, 512, 8),  # From 32 to 256 neurons per layer
        'learning_rate': hp.loguniform('learning_rate', -5, 0),  # Between 10^-5 and 1
        'loss_function': hp.choice('loss_function', loss_functions),  # Different loss functions
        'epochs': hp.choice('epochs', epochs),
        'dir':dir,
        'X_train': X_train, 
        'y_train': y_train, 
        'X_test': X_test,
        'y_test': y_test
    }

    # Create Trials to store results
    trials = Trials()

    # Run the optimization
    best = fmin(fn=modeling.build, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=attempts,  # Number of attempts
                trials=trials)
    
    best["epochs"] = epochs[best["epochs"]]
    best["loss_function"] = loss_functions[best["loss_function"]]
    print("------------*******----------------")
    print("Best Hyperparameters Found:", best)
    print("------------*******----------------")
    with open(f"{dir}/model-optimization.txt", 'w') as f:
            with redirect_stdout(f):
                print("Best Hyperparameters Found:", best)
    return best
