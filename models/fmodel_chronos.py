import pandas as pd 
import os
import warnings
import logging
from utils import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from dataset_config import DatasetBelgiumNeuralForecast, DatasetLondonZonnedaelNeuralForecast
import time
import gc
import torch
from chronos import BaseChronosPipeline

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "london_zonnedael"  # Options: "belgium" or "london_zonnedael" 

dataset_belgium = DatasetBelgiumNeuralForecast()
dataset_london_zonnedael = DatasetLondonZonnedaelNeuralForecast()
# =========================================================================================

# Load Chronos pipeline pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",  # Change to "amazon/chronos-t5-small" if desired
    device_map=device,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)

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

# Core function for Chronos forecasting
def chronos_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon):
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    logging.info(f"Running Chronos {model_name} for {len(train_df)} samples ({sampling_rate:.0f}% of original training set)...")

    # Prepare context tensor for Chronos model
    context_values = torch.tensor(train_df["y"].values, dtype=torch.float32).to(device)

    # Predict quantiles and mean forecast
    quantile_levels = [0.1, 0.5, 0.9]  # 90% confidence interval
    quantiles, mean = pipeline.predict_quantiles(
        context=context_values,
        prediction_length=forecast_horizon,
        quantile_levels=quantile_levels,
    )

    # Use median prediction for evaluation and plotting
    y_pred = quantiles[:, :, quantile_levels.index(0.5)].cpu().numpy().flatten()
    y_true = test_df["y"].values

    mae, rmse, mape, r2 = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return mae, rmse, mape, r2

# Train Belgium models
def train_load_model(load_data, save_dir, freq, forecast_horizon):
    return chronos_forecast_model(load_data, "Load", save_dir, freq, forecast_horizon)

def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon):
    return chronos_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon):
    return chronos_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon)

# Train London-Zonnedael models
def train_london_model(load_data, save_dir, freq, forecast_horizon):
    return chronos_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon)

def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon):
    return chronos_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon)

# Full pipeline for Chronos
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info("...Start Chronos forecasting...")

    if selected_dataset == "belgium":
        logging.info("Forecasting load consumption")
        load_data = dataset_belgium.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse, load_mape, load_r2 = train_load_model(load_data, save_dir, freq, forecast_horizon)
        metrics.append({"model": "load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse, pv_mape, pv_r2 = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse, "MAPE": pv_mape, "R2": pv_r2})

        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse, battery_mape, battery_r2 = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse, "MAPE": battery_mape, "R2": battery_r2})

    elif selected_dataset == "london_zonnedael":
        logging.info("Forecasting London load")
        load_data = dataset_london_zonnedael.get_inputs_for_london_consumption()
        load_mae, load_rmse, load_mape, load_r2 = train_london_model(load_data, save_dir, freq, forecast_horizon)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset_london_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse, cust_mape, cust_r2 = train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse, "MAPE": cust_mape, "R2": cust_r2})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End Chronos forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, sampling_rate_local):
    global sampling_rate
    sampling_rate = sampling_rate_local
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate))
    freq_str = f"{train_freq}T"
    forecast_horizon = int(192 / (100 / sampling_rate))

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_dir = f"results/results_{selected_dataset}/Chronos/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Skipping Chronos due to error: {str(e)}", exc_info=True)

# Run all sampling rates and seeds
if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 2):  # Loop for run_num
            paper_forecasting_train(run_num, sampling_rate)
