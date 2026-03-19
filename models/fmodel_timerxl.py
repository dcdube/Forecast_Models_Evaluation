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
import torch
from transformers import AutoModelForCausalLM
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "london"  # Options: "belgium" or "germany" or "london" or "zonnedael"

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

# Core forecasting function
def hf_causal_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, hf_model_name):
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    logging.info(f"Running {hf_model_name} ({model_name}) for {len(train_df)} samples ({sampling_rate:.0f}% of original training set)...")

    context_values = train_df["y"].values.astype("float32")
    context_tensor = torch.tensor(context_values).unsqueeze(0).to(device)

    model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True).to(device)

    with torch.no_grad():
        output = model.generate(context_tensor, max_new_tokens=forecast_horizon)

    y_pred = output[0, -forecast_horizon:].detach().cpu().numpy()
    y_true = test_df["y"].values

    if "sundial" in hf_model_name.lower():
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return mae, rmse

# Train each target based on selected dataset
def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, model_id):
    return hf_causal_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, model_id)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, model_id):
    return hf_causal_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, model_id)

def train_london_load_model(load_data, save_dir, freq, forecast_horizon, model_id):
    return hf_causal_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, model_id)

def train_germany_load_model(load_data, save_dir, freq, forecast_horizon, model_id):
    return hf_causal_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, model_id)

def train_zonnedael_customer_model(customer_id, data_df, save_dir, freq, forecast_horizon, model_id):
    return hf_causal_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, model_id)

# Full pipeline with dataset toggle
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, model_id):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info(f"...Start {model_id} forecasting on {selected_dataset} dataset...")

    if selected_dataset == "belgium":
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, model_id)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})

        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, model_id)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})

    elif selected_dataset == "germany":
        load_data = dataset_germany.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = train_germany_load_model(load_data, save_dir, freq, forecast_horizon, model_id)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "london":
        load_data = dataset_london.get_inputs_for_load()
        load_mae, load_rmse = train_london_load_model(load_data, save_dir, freq, forecast_horizon, model_id)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "zonnedael":
        for customer_id in [8, 9, 43]:
            data_df = dataset_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = train_zonnedael_customer_model(customer_id, data_df, save_dir, freq, forecast_horizon, model_id)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info(f"...End {model_id} forecasting on {selected_dataset} dataset...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

# Entry point for training
def paper_forecasting_train(run_num, sampling_rate_local, model_id, save_folder):
    global sampling_rate, device
    sampling_rate = sampling_rate_local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15*(100/sampling_rate))
    freq_str = f"{train_freq}T"

    forecast_horizon = int(192/(100/sampling_rate))

    try:
        gc.collect()
        save_dir = f"results/results_{selected_dataset}/{save_folder}/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, model_id)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping {model_id} on {selected_dataset} due to error: {str(e)}", exc_info=True)

# Run all sampling rates and models
if __name__ == "__main__":
    model_configs = {
        'thuml/timer-base-84m': 'TimerXL'
    }
    for model_id, save_folder in model_configs.items():
        for sampling_rate in [25, 100/3, 50, 100]:
            for run_num in range(1, 2):  # Loop for run_num
                paper_forecasting_train(run_num, sampling_rate, model_id, save_folder)
