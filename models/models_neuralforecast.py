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
import torch
import time
import gc 
from neuralforecast import NeuralForecast
from neuralforecast.models import *

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "belgium"  # Options: "belgium" or "germany" or "london" or "zonnedael"

dataset_belgium = DatasetBelgiumNF()
dataset_germany = DatasetGermanyNF()
dataset_london = DatasetLondonNF()
dataset_zonnedael = DatasetZonnedaelNF()
n_epochs = 100
# =========================================================================================

# List of models to iterate through
model_classes = {
    "BiTCN": BiTCN,
    "DeepNPTS": DeepNPTS,
    "Informer": Informer,
    "NBEATS": NBEATS,
    "NHITS": NHITS,
    "NLinear": NLinear,
    "PatchTST": PatchTST,
    "TCN": TCN,
    "TiDE": TiDE,
    "TimesNet": TimesNet,
    "TimeXer": TimeXer,
    "iTransformer": iTransformer,
    "VanillaTransformer": VanillaTransformer
}

# Custom logger setup per model
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

# Core training function for any model type
def neural_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, nf_model_name="TimesNet", NFmodel=None):
    current_seed = int(time.time()) % (2**32 - 1)  # Use current time for true randomness
    logging.info(f"Using random seed: {current_seed}")

    # Split data
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    # Sample % of training data
    # train_df = full_train_df.iloc[::int(100/sampling_rate)]

    # plt.plot(train_df['ds'], train_df['y'])
    # plt.show()

    logging.info(f"Training {nf_model_name} {model_name} model on {len(train_df)} samples ({sampling_rate}% of original training set)...")

    if nf_model_name in ["TimeXer", "iTransformer"]:
        nf = NeuralForecast(
            models=[NFmodel(h=forecast_horizon, input_size=forecast_horizon * 2, n_series=1, max_steps=n_epochs, random_seed=current_seed)],
            freq=freq)
    else:
        nf = NeuralForecast(
            models=[NFmodel(h=forecast_horizon, input_size=forecast_horizon * 2, max_steps=n_epochs, random_seed=current_seed)],
            freq=freq)

    nf.fit(df=train_df, val_size=forecast_horizon)

    forecast_df = nf.predict()
    y_pred = forecast_df[nf_model_name].values
    y_true = test_df["y"].values

    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return nf, mae, rmse

# Train PV model
def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel):
    return neural_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, nf_model_name=nf_model_name, NFmodel=NFmodel)

# Train battery model
def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel):
    return neural_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, nf_model_name=nf_model_name, NFmodel=NFmodel)

# Train london load model
def train_london_model(load_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return neural_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, nf_model_name=nf_model_name, NFmodel=NFmodel)

def train_germany_model(load_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return neural_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, nf_model_name=nf_model_name, NFmodel=NFmodel)

# Train zonnedael customers
def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return neural_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, nf_model_name=nf_model_name, NFmodel=NFmodel)

# Main pipeline for training all models
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    setup_model_logger(save_dir)        # Ensure logging is set up first
    metrics = []

    start_time = time.time()
    logging.info(f"...Start training for {n_epochs} epochs using {nf_model_name}...")

    if selected_dataset == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            _, pv_mae, pv_rmse = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})

        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            _, battery_mae, battery_rmse = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})

    elif selected_dataset == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset_germany.get_inputs_for_load(start_dt, end_dt)
        _, load_mae, load_rmse = train_germany_model(germany_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "london":
        logging.info("Training London load model")
        london_data = dataset_london.get_inputs_for_load()
        _, load_mae, load_rmse = train_london_model(london_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            _, cust_mae, cust_rmse = train_zonnedael_model(customer_id, customer_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, sampling_rate):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate))  # Adjust frequency based on sampling rate
    freq_str = f"{train_freq}min"  # Convert to string format for frequency
    forecast_horizon = int(192 / (100 / sampling_rate))  # 2 days of 15-minute intervals

    for model_name, model_class in model_classes.items():
        try:
            # Clear memory before training a new model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_dir = f"results/results_{selected_dataset}/{model_name}/Sampling_{sampling_rate:.0f}/Epochs_{n_epochs}_{run_num}"
            train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, nf_model_name=model_name, NFmodel=model_class)

            # Clear memory after training a model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Skipping model {model_name} due to error: {str(e)}", exc_info=True)
            continue

# Run if script is executed directly
if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 11):  # Loop from run_num = 1 to 10
            paper_forecasting_train(run_num, sampling_rate)
