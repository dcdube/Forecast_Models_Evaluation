import pandas as pd  
import os
import time
import logging
import warnings
import gc
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import pmdarima as pm
from dataset_config import DatasetBelgium1D, DatasetLondonZonnedael1D
from utils import split_train_test, calculate_metrics, forecast_plot_and_csv, plot_model_metrics

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "london_zonnedael"  # Options: "belgium" or "london_zonnedael"

dataset_belgium = DatasetBelgium1D()
dataset_london_zonnedael = DatasetLondonZonnedael1D()
# =========================================================================================

# Set of deterministic models
deterministic_models = {"ARIMA", "NaiveMovingAverage"}
 
# Setup logger per run
def setup_model_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logger initialized at {log_file}")

# Generalized model training function for both univariate and multivariate models
def generic_model(X, y, model_name, save_dir, model_type, run_num, sampling_rate, forecast_horizon):
    current_seed = int(time.time() * 1000) % (2**32 - 1)  # Ensure higher variability

    is_univariate = X is None

    logging.info(f"Run {run_num} - {model_name} - {model_type}")
    logging.info(f"Using random seed: {current_seed}")
    
    if is_univariate:
        # Univariate setup
        y = y.iloc[::int(100/sampling_rate)]
        y_train = y.iloc[:-forecast_horizon]
        y_test = y.iloc[-forecast_horizon:]

        logging.info(f"Training {model_type} model for {model_name} on {len(y_train)} samples (univariate)...")

        d_order = pm.arima.ndiffs(y_train, alpha=0.05, test='kpss', max_d=2)

        # If you MUST use seasonality, pre-calculate D as well.
        # Note: With m=48, this can still be slow.
        D_order = pm.arima.nsdiffs(y_train, m=48, max_D=1, test='ch')

        if model_type == "ARIMA":
            model = pm.auto_arima(
                y_train,
                start_p=0, start_q=0,
                max_p=1, max_q=1,
                start_P=0, start_Q=0,
                max_P=1, max_Q=1,
                d=d_order,                # Use pre-calculated d
                D=D_order,                # Use pre-calculated D
                m=48,                     # Acknowledge this is the main bottleneck
                seasonal=True,
                stepwise=True,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                information_criterion='aic',
                max_order=4,
                maxiter=10,               # Allow the model to converge reasonably
                # Use n_jobs=-1 for parallel processing if stepwise=False,
                # but with stepwise=True, this has limited effect.
            )
            preds = model.predict(n_periods=forecast_horizon)

        elif model_type == "NaiveMovingAverage":
            k = int(forecast_horizon / 1)
            hist = y_train.tolist().copy()
            for i in range(forecast_horizon):
                hist.append(sum(hist[-k:]) / k)
            preds = hist[-forecast_horizon:]
            model = "NaiveMovingAverage"
        else:
            raise NotImplementedError(f"{model_type} not supported for univariate.")

        y_true = y_test.values
        mae, rmse, mape, r2 = calculate_metrics(preds, y_true)
        logging.info(f"{model_name} ({model_type}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

        df_plot = pd.DataFrame({"datetime": y_test.index, "Actual": y_true, "Forecast": preds}).set_index("datetime")
        forecast_plot_and_csv(df_plot, model_name, save_dir)
        return model, mae, rmse, mape, r2

    else:
        # Multivariate setup
        X = X.iloc[::int(100/sampling_rate)]
        y = y.iloc[::int(100/sampling_rate)]
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=forecast_horizon)

        logging.info(f"Training {model_type} model for {model_name} on {X_train.shape[0]} samples ({sampling_rate}% of original data)...")

        params = {
            'random_state': current_seed,
            'bagging_fraction': 0.5,
            'feature_fraction': 0.5,
            'bagging_freq': 1,
            'verbosity': -1,
            'n_jobs': -1,
            'deterministic': False
        }
        if model_type == "LightGBM":
            model = lgb.LGBMRegressor(**params).fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_type == "KNNRegression":
            model = KNeighborsRegressor().fit(X_train, y_train)
            preds = model.predict(X_test)

        else:
            raise NotImplementedError(f"{model_type} not supported for multivariate models.")

        mae, rmse, mape, r2 = calculate_metrics(preds, y_test)
        logging.info(f"{model_name} ({model_type}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

        df_plot = pd.DataFrame({"datetime": y_test.index, "Actual": y_test, "Forecast": preds}).set_index("datetime")
        forecast_plot_and_csv(df_plot, model_name, save_dir)
        return model, mae, rmse, mape, r2

# ==================================================DatasetBelgium1D============================================================
def train_load_model(start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset_belgium.get_load_data(start_dt, end_dt)[1]
        return generic_model(None, y, "load", save_dir, model_type, run_num, sampling_rate, forecast_horizon)
    else:
        X, y = dataset_belgium.get_load_data(start_dt, end_dt)
        return generic_model(X, y, "load", save_dir, model_type, run_num, sampling_rate, forecast_horizon)

def train_pv_model(start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset_belgium.get_pv_data(house, start_dt, end_dt)[1]
        return generic_model(None, y, f"pv_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon)
    else:
        X, y = dataset_belgium.get_pv_data(house, start_dt, end_dt)
        return generic_model(X, y, f"pv_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon)

def train_battery_model(start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset_belgium.get_battery_data(house, start_dt, end_dt)[1]
        return generic_model(None, y, f"bess_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon)
    else:
        X, y = dataset_belgium.get_battery_data(house, start_dt, end_dt)
        return generic_model(X, y, f"bess_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon)

# ================================================DatasetLondonZonnedael1D=========================================================
def train_london_consumption_model(save_dir, model_type, run_num, sampling_rate, forecast_horizon):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset_london_zonnedael.get_inputs_for_london_consumption()[1]
        return generic_model(None, y, "london_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon)
    else:
        X, y = dataset_london_zonnedael.get_inputs_for_london_consumption()
        return generic_model(X, y, "london_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon)

def train_zonnedael_consumption_model(save_dir, model_type, run_num, sampling_rate, forecast_horizon):
    metrics = []
    customer_ids = [8, 9, 43]

    for customer_number in customer_ids:
        if model_type in ["NaiveMovingAverage", "ARIMA"]:
            _, y = dataset_london_zonnedael.get_inputs_for_zonnedael_consumption(customer_number)
            model, mae, rmse, mape, r2 = generic_model(
                None, y, f"zonnedael_customer_{customer_number}",
                save_dir, model_type, run_num, sampling_rate, forecast_horizon
            )
        else:
            X, y = dataset_london_zonnedael.get_inputs_for_zonnedael_consumption(customer_number)
            model, mae, rmse, mape, r2 = generic_model(
                X, y, f"zonnedael_customer_{customer_number}",
                save_dir, model_type, run_num, sampling_rate, forecast_horizon
            )

        metrics.append({"model": f"zonnedael_load_{customer_number}", "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2})
    return metrics

# Unified training pipeline
def train_all_models(start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info(f"Run {run_num}: Start training {model_type} models...")

    if selected_dataset == "belgium":
        load_model, load_mae, load_rmse, load_mape, load_r2 = train_load_model(start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon)
        metrics.append({"model": "load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        for house in [1, 2, 3, 4]:
            pv_model, pv_mae, pv_rmse, pv_mape, pv_r2 = train_pv_model(start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse, "MAPE": pv_mape, "R2": pv_r2})

        for house in [1, 2, 3, 4]:
            battery_model, battery_mae, battery_rmse, battery_mape, battery_r2 = train_battery_model(start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse, "MAPE": battery_mape, "R2": battery_r2})

    elif selected_dataset == "london_zonnedael":
        london_model, mae, rmse, mape, r2 = train_london_consumption_model(save_dir, model_type, run_num, sampling_rate, forecast_horizon)
        metrics.append({"model": "london_load", "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2})
        
        zonnedael_metrics = train_zonnedael_consumption_model(save_dir, model_type, run_num, sampling_rate, forecast_horizon)
        metrics.extend(zonnedael_metrics)

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    end_time = time.time()
    logging.info(f"Run {run_num}: End training {model_type} models.")
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, model_type, sampling_rate):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15*(100/sampling_rate))
    train_freq = f"{train_freq}min"
    forecast_horizon = int(192/(100/sampling_rate))

    save_dir = f"results/results_{selected_dataset}/{model_type}/Sampling_{sampling_rate:.0f}/Run_{run_num}"
    train_all_models(start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon)

# Main script
if __name__ == "__main__":
    model_types = [
        "KNNRegression",
        "LightGBM",
        "ARIMA",
        "NaiveMovingAverage",
    ]

    for sampling_rate in [25, 50, 100/3, 100]:
        for model_type in model_types:
            n_runs = 1 if model_type in deterministic_models else 10
            for run_num in range(1, n_runs + 1):
                try:
                    gc.collect()
                    paper_forecasting_train(run_num, model_type, sampling_rate)
                    gc.collect()
                except Exception as e:
                    logging.error(f"{model_type} Run {run_num} (Sampling {sampling_rate}) failed: {str(e)}", exc_info=True)
