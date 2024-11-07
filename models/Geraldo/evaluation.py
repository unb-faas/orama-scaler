from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import preprocessing
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate(train_results, model, X_test, y_test, dir, scaler, encoders, plot=True, test=True):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2}')

    if plot:
        plt.plot(train_results.history['loss'], label='Training Loss')
        plt.plot(train_results.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(f"{dir}/graph-loss.png")
        plt.close()

    if test:
        y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
        y_test_reshaped = y_test.values.reshape(-1, 1)
        y_test_df = pd.DataFrame(y_test_reshaped, columns=['Original'])        
        X_test_reset = X_test.reset_index(drop=True)
        predictions = pd.concat([X_test_reset, y_pred_df], axis=1)
        originals = pd.concat([X_test_reset, y_test_df], axis=1)
        predictions_denormalized = preprocessing.denormalize(predictions, scaler)
        predictions_decategorized = preprocessing.decategorize(predictions_denormalized, encoders)
        originals_denormalized = preprocessing.denormalize(originals, scaler)
        originals_decategorized = preprocessing.decategorize(originals_denormalized, encoders)
        full = pd.concat([predictions_decategorized, originals_decategorized], axis=1)
        full = full.loc[:, ~full.columns.duplicated()]
        predictions_sorted = full.sort_values(by=['concurrency', 'provider', 'usecase'], ascending=[True, True, True])        
        predictions_sorted['difference'] = predictions_sorted['Original'] - predictions_sorted['Prediction']
        predictions_sorted.to_csv(f"{dir}/predictions.csv", index=False)
        print(predictions_sorted)