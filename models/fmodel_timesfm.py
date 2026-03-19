import pandas as pd  
import os
import warnings
import logging
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import (
    DatasetBelgiumNF,
    DatasetGermanyNF,
    DatasetLondonNF,
    DatasetZonnedaelNF,
)
import time
import gc
import timesfm

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "belgium"  # Options: "belgium" or "germany" or "london" or "zonnedael"

dataset_belgium = DatasetBelgiumNF()
dataset_germany = DatasetGermanyNF()
dataset_london = DatasetLondonNF()
dataset_zonnedael = DatasetZonnedaelNF()
# =========================================================================================

# Logger setup
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

# Core function for TimesFM
def timesfm_forecast_model(y_df, model_name, save_dir, forecast_horizon, tfm, sampling_rate):
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    logging.info(f"Running TimesFM {model_name} for {len(train_df)} samples ({sampling_rate:.0f}% of training set)...")

    train_values = train_df["y"].values.astype("float32")
    freq_input = [0]

    forecast_outputs = tfm.forecast([train_values], freq=freq_input)
    mean_forecast = forecast_outputs[0]
    y_pred = mean_forecast[0]

    y_true = test_df["y"].values

    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return mae, rmse

def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, tfm, sampling_rate):
    return timesfm_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, tfm, sampling_rate)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, tfm, sampling_rate):
    return timesfm_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, tfm, sampling_rate)

def train_london_model(load_data, save_dir, freq, forecast_horizon, tfm, sampling_rate):
    return timesfm_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, tfm, sampling_rate)

def train_germany_model(load_data, save_dir, freq, forecast_horizon, tfm, sampling_rate):
    return timesfm_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, tfm, sampling_rate)

def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, tfm, sampling_rate):
    return timesfm_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, tfm, sampling_rate)

# Full pipeline for TimesFM with dataset toggle
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, sampling_rate):
    setup_model_logger(save_dir)
    metrics = []

    tfm = timesfm.TimesFm( 
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=forecast_horizon,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

    start_time = time.time()
    logging.info("...Start TimesFM forecasting...")

    if selected_dataset == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})

        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})

    elif selected_dataset == "germany":
        logging.info("Forecasting Germany load")
        load_data = dataset_germany.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = train_germany_model(load_data, save_dir, freq, forecast_horizon, tfm, sampling_rate)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "london":
        logging.info("Forecasting London load")
        load_data = dataset_london.get_inputs_for_load()
        load_mae, load_rmse = train_london_model(load_data, save_dir, freq, forecast_horizon, tfm, sampling_rate)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End TimesFM forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, sampling_rate_local):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate_local))
    freq_str = f"{train_freq}MIN"
    forecast_horizon = int(192 / (100 / sampling_rate_local))

    try:
        gc.collect()
        save_dir = f"results/results_{selected_dataset}/TimesFM/Sampling_{sampling_rate_local:.0f}/Run_{run_num}"
        train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate_local)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping TimesFM due to error: {str(e)}", exc_info=True)

# Run all sampling rates and seeds
if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 2):
            paper_forecasting_train(run_num, sampling_rate)
