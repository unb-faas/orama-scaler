from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

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

        y_test = np.squeeze(y_test)
        y_pred = np.squeeze(y_pred)
        
        # RMSE - BOXPLOT
        rmse_values = np.sqrt((y_test - y_pred) ** 2)
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=rmse_values, color='skyblue')
        plt.ylabel("RMSE")
        plt.title("Testset RMSE Boxplot")
        plt.savefig(f"{dir}/graph-boxplot-rmse.png")

        # MSE - BOXPLOT
        mse_values = (y_test - y_pred) ** 2
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=mse_values, color='lightcoral')
        plt.title("Testset MSE Boxplot")
        plt.ylabel("MSE")
        plt.savefig(f"{dir}/graph-boxplot-mse.png")

        # R2 - BOXPLOT
        X_test_r2 = np.asarray(X_test)
        y_test_r2 = np.squeeze(np.asarray(y_test))
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds
        r2_values = []
        for train_index, test_index in kf.split(X_test_r2):
            X_fold, y_fold = X_test_r2[test_index], y_test_r2[test_index]  # Seleção correta dos dados
            y_pred_fold = model.predict(X_fold)  
            y_pred_fold = np.squeeze(y_pred_fold)  
            r2 = r2_score(y_fold, y_pred_fold)
            r2_values.append(r2)
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=r2_values, color='lightgreen')
        plt.title("Testset R^2 Boxplot")
        plt.ylabel("R^2")
        plt.savefig(f"{dir}/graph-boxplot-r2.png")

        # MAPE - BOXPLOT
        X_test_mape = np.array(X_test)
        y_test_mape = np.squeeze(np.array(y_test))
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds
        mape_values = []
        for train_index, test_index in kf.split(X_test_mape):
            X_fold, y_fold = X_test_mape[test_index], y_test_mape[test_index]  # Seleção dos dados
            y_pred_fold = model.predict(X_fold)
            y_pred_fold = np.squeeze(y_pred_fold)
            mape = mean_absolute_percentage_error(y_fold, y_pred_fold)
            mape_values.append(mape)
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=mape_values, color='lightblue')
        plt.title("Testset MAPE Boxplot")
        plt.ylabel("MAPE (%)")
        plt.savefig(f"{dir}/graph-boxplot-mape.png")

        # OBSERVATIONS vs predictions
        observations = y_test
        predictions = y_pred

        plt.figure(figsize=(10, 6))

        # Plotting both the observations and predictions as horizontal lines
        plt.plot(range(len(observations)), observations, label='Observations', color='blue', marker='o')
        plt.plot(range(len(predictions)), predictions, label='Predictions', color='orange', marker='x')

        # Adjusting X Axis 
        plt.xlim(150, 250)

        # Adding titles and labels
        plt.title('Observation vs Prediction', fontsize=14)
        plt.xlabel('Time / Data Point', fontsize=12)
        plt.ylabel('Value', fontsize=12)

        # Adding a legend
        plt.legend()

        plt.savefig(f"{dir}/graph-obs-vs-pred.png")


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
        predictions["provider_cat"] = predictions["provider"]
        full = pd.concat([predictions_decategorized, originals_decategorized, predictions], axis=1)
        full = full.loc[:, ~full.columns.duplicated()]
        predictions_sorted = full.sort_values(by=['concurrency', 'provider'], ascending=[True, True])        
        predictions_sorted['difference'] = predictions_sorted['Original'] - predictions_sorted['Prediction']
        predictions_sorted.to_csv(f"{dir}/predictions.csv", index=False)
        print(predictions_sorted)