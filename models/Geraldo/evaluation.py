from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import random

def calculate_metrics(models, X_test, y_test):
    # Assume X_test and y_test are the same for all models
    X_test_r2 = np.asarray(X_test)
    y_test_r2 = np.squeeze(np.asarray(y_test))
    results = {}
    for i, (arch, model) in enumerate(models.items()):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        r2_values = []
        rmse_values = []
        mse_values = []
        mape_values = []
        mape2_values = []
        mae_values = []
        observations = []
        predictions = []

        # Perform 10-fold evaluation on the test set
        for train_index, test_index in kf.split(X_test_r2):
            X_fold = X_test_r2[test_index]
            y_fold = y_test_r2[test_index]
            y_pred_fold = model["model"].predict(X_fold)
            y_pred_fold = np.squeeze(y_pred_fold)
            r2 = r2_score(y_fold, y_pred_fold)
            r2_values.append(r2)
            rmse = np.sqrt(np.mean((y_fold - y_pred_fold) ** 2))
            rmse_values.append(rmse)
            mse = np.mean((y_fold - y_pred_fold) ** 2)
            mse_values.append(mse)
            mape = mean_absolute_percentage_error(y_fold, y_pred_fold)
            mape_values.append(mape)
            mask = y_fold != 0
            mape2 = np.mean(np.abs((y_fold[mask] - y_pred_fold[mask]) / y_fold[mask])) * 100
            mape2_values.append(mape2)
            mae = np.mean(np.abs(y_fold - y_pred_fold))
            mae_values.append(mae)
        
        observations.append(y_test)
        predictions.append(model["model"].predict(X_test))
                
        results[arch] = {
            "r2": r2_values,
            "rmse": rmse_values,
            "mse": mse_values,
            "mape": mape_values,
            "mape2": mape_values,
            "mae": mae_values,
            "observations": observations,
            "predictions": predictions,
        }
    return results

def plot_metric_boxplot(metrics, name, label, color, dir):
    plt.clf()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5), sharey=True)
    str_arch = ""
    if len(metrics)==1:
        axes = [axes]
    for i, arch in enumerate(metrics):
        values = metrics[arch][name]
        str_arch = f"{str_arch}_{arch}"
        sns.boxplot(y=values, color=color, ax=axes[i])
        axes[i].set_title(arch)
        axes[i].set_ylabel(label if i == 0 else "")  # Only show label on the first plot
    # Adjust layout and save figure
    plt.suptitle(f"{name.upper()} Boxplot", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{dir}/graph-{name}-boxplot-{str_arch}.png")
    plt.close()

def plot_loss(models, dir):
    plt.clf()
    fig, axes = plt.subplots(1, len(models), figsize=(10*len(models), 5), sharey=True)
    str_arch = ""
    if len(models)==1:
        axes = [axes]
    for i, (arch, model) in enumerate(models.items()):
        str_arch = f"{str_arch}_{arch}"
        ax = axes[i]
        ax.plot(model["train_results"].history['loss'], label='Training Loss')
        ax.plot(model["train_results"].history['val_loss'], label='Validation Loss')
        ax.set_title(f"{arch}")
        ax.set_xlabel("Epochs")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend()

    # Set a common title
    fig.suptitle("Training and Validation Loss", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
    plt.savefig(f"{dir}/graph-loss-{str_arch}.png")
    plt.close()

def plot_obs_preds(metrics, dir):
    plt.clf()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)), sharex=True)
    str_arch = ""

    # Ensure axes is always iterable
    if len(metrics) == 1:
        axes = [axes]

    for i, (arch, model) in enumerate(metrics.items()):
        str_arch += f"_{arch}"
        ax = axes[i]
        observations = np.squeeze(model["observations"])
        predictions = np.squeeze(model["predictions"])
        ax.plot(range(len(observations)), observations, label='Observations', color='blue', marker='o')
        ax.plot(range(len(predictions)), predictions, label='Predictions', color='orange', marker='x')
        ax.set_xlim(150, 200)
        ax.set_title(f"{arch}", fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        if i == len(metrics) - 1:
            ax.set_xlabel('Time / Data Point', fontsize=12)
        ax.legend()

    # Add a common title and adjust layout
    fig.suptitle("Observations vs Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{dir}/graph-obs-preds-{str_arch}.png")
    plt.close()


def evaluate(results, X_test, y_test, scaler, encoders, arch, dir, plot=True, test=False):
    if isinstance(results, dict):
        metrics = calculate_metrics(results, X_test, y_test)
        plot_metric_boxplot(metrics, "r2", "R²", "#%06x" % random.randint(0, 0xFFFFFF), dir)
        plot_metric_boxplot(metrics, "rmse", "RMSE", "#%06x" % random.randint(0, 0xFFFFFF), dir)
        plot_metric_boxplot(metrics, "mse", "MSE", "#%06x" % random.randint(0, 0xFFFFFF), dir)
        plot_metric_boxplot(metrics, "mape", "MAPE", "#%06x" % random.randint(0, 0xFFFFFF), dir)
        plot_metric_boxplot(metrics, "mae", "MAE", "#%06x" % random.randint(0, 0xFFFFFF), dir)
        plot_obs_preds(metrics, dir)
        plot_loss(results, dir) 

    if isinstance(results, list):
        for result in results:
            train_results = result["train_results"]
            model = result["model"]
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"{arch} MAE: {mae}")
            mse = mean_squared_error(y_test, y_pred)
            print(f"{arch} MSE: {mse}")
            rmse = np.sqrt(mse)
            print(f"{arch} RMSE: {rmse}")
            r2 = r2_score(y_test, y_pred)
            print(f'{arch} R^2 Score: {r2}')

            if plot:
                plt.clf()
                plt.plot(train_results.history['loss'], label='Training Loss')
                plt.plot(train_results.history['val_loss'], label='Validation Loss')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title(f"{arch}: Training and Validation Loss")
                plt.savefig(f"{dir}/{arch}_graph-loss.png")
                plt.close()

                y_test = np.squeeze(y_test)
                y_pred = np.squeeze(y_pred)
                
                # RMSE - BOXPLOT
                plt.clf()
                rmse_values = np.sqrt((y_test - y_pred) ** 2)
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=rmse_values, color='skyblue')
                plt.ylabel("RMSE")
                plt.title(f"{arch}: Testset RMSE Boxplot")
                plt.savefig(f"{dir}/{arch}_graph-boxplot-rmse.png")
                print("RMSE - BOXPLOT generated")

                # MSE - BOXPLOT
                plt.clf()
                mse_values = (y_test - y_pred) ** 2
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=mse_values, color='lightcoral')
                plt.title(f"{arch}: Testset MSE Boxplot")
                plt.ylabel("MSE")
                plt.savefig(f"{dir}/{arch}_graph-boxplot-mse.png")
                print("MSE - BOXPLOT generated")

                # R2 - BOXPLOT
                plt.clf()
                X_test_r2 = np.asarray(X_test)
                y_test_r2 = np.squeeze(np.asarray(y_test))
                kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10 folds
                r2_values = []
                for train_index, test_index in kf.split(X_test_r2):
                    X_fold, y_fold = X_test_r2[test_index], y_test_r2[test_index] 
                    y_pred_fold = model.predict(X_fold)  
                    y_pred_fold = np.squeeze(y_pred_fold)  
                    r2 = r2_score(y_fold, y_pred_fold)
                    r2_values.append(r2)
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=r2_values, color='lightgreen')
                plt.title(f"{arch}: Testset R^2 Boxplot")
                plt.ylabel("R^2")
                plt.savefig(f"{dir}/{arch}_graph-boxplot-r2.png")
                print("R² - BOXPLOT generated")

                # MAPE - BOXPLOT
                plt.clf()
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
                plt.title(f"{arch}: Testset MAPE Boxplot")
                plt.ylabel("MAPE (%)")
                plt.savefig(f"{dir}/{arch}_graph-boxplot-mape.png")
                print("MAPE - BOXPLOT generated")

                # OBSERVATIONS vs predictions
                plt.clf()
                observations = y_test
                predictions = y_pred
                plt.figure(figsize=(10, 6))
                # Plotting both the observations and predictions as horizontal lines
                plt.plot(range(len(observations)), observations, label='Observations', color='blue', marker='o')
                plt.plot(range(len(predictions)), predictions, label='Predictions', color='orange', marker='x')
                # Adjusting X Axis 
                plt.xlim(150, 250)
                # Adding titles and labels
                plt.title(f"{arch}: Observation vs Prediction", fontsize=14)
                plt.xlabel('Time / Data Point', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                # Adding a legend
                plt.legend()
                plt.savefig(f"{dir}/{arch}_graph-obs-vs-pred.png")
                print("OBSERVATIONS vs predictions generated")


            #if test:
            #    y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
            #    y_test_reshaped = y_test.values.reshape(-1, 1)
            #    y_test_df = pd.DataFrame(y_test_reshaped, columns=['Original'])        
            #    X_test_reset = X_test.reset_index(drop=True)
            #    predictions = pd.concat([X_test_reset, y_pred_df], axis=1)
            #    originals = pd.concat([X_test_reset, y_test_df], axis=1)
            #    predictions["provider_cat"] = predictions["provider"]
            #    full = predictions
            #    full = full.loc[:, ~full.columns.duplicated()]
            #    predictions_sorted = full.sort_values(by=['concurrency', 'provider'], ascending=[True, True])        
            #    predictions_sorted['difference'] = predictions_sorted['Original'] - predictions_sorted['Prediction']
            #    predictions_sorted.to_csv(f"{dir}/{arch}_predictions.csv", index=False)
            #    print(predictions_sorted)