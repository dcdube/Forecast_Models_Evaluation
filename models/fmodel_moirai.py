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
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "belgium"  # Options: "belgium" or "germany" or "london" or "zonnedael"

dataset_belgium = DatasetBelgiumNF()
dataset_germany = DatasetGermanyNF()
dataset_london = DatasetLondonNF()
dataset_zonnedael = DatasetZonnedaelNF()
# =========================================================================================

# Moirai Model Parameters
MODEL = "moirai"      # Options: "moirai" or "moirai-moe"
SIZE = "small"         # Options: "small", "base", "large"
# CTX = forecast_horizon  # Context length
PSZ = "auto"          # Patch size
BSZ = 32              # Batch size

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

# Core function for Moirai
def moirai_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate):
    y_df = y_df.iloc[::int(100 / sampling_rate)].copy()                        # Avoid SettingWithCopyWarning
    y_df["ds"] = pd.to_datetime(y_df["ds"]).dt.tz_localize(None)              # Ensure datetime format and strip timezone
    y_df.set_index("ds", inplace=True)                                        # Set datetime index

    # Split into train and test
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    logging.info(f"Running Moirai {model_name} for {len(train_df)} samples ({sampling_rate:.0f}% of original training set)...")

    # Combine train and test for uniform indexing check
    ts_df = pd.concat([train_df, test_df])

    # Check frequency and reindex if needed
    inferred_freq = pd.infer_freq(ts_df.index)
    if inferred_freq is None or inferred_freq != freq:
        logging.warning(f"Frequency inferred as {inferred_freq}. Reindexing to uniform frequency {freq}.")
        full_index = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq=freq)
        ts_df = ts_df.reindex(full_index)
        ts_df["y"] = ts_df["y"].interpolate(method="time")  # Interpolate missing values

    # Re-split after reindexing
    train_df = ts_df.iloc[:-forecast_horizon]
    test_df = ts_df.iloc[-forecast_horizon:]

    ds = PandasDataset({"target": ts_df["y"]}, freq=freq)

    train, test_template = split(ds, offset=-forecast_horizon)
    test_data = test_template.generate_instances(
        prediction_length=forecast_horizon,
        windows=1,
        distance=forecast_horizon
    )

    CTX = 2 * forecast_horizon  # Context length

    if MODEL == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
            prediction_length=forecast_horizon,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    else:
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
            prediction_length=forecast_horizon,
            context_length=CTX,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = list(predictor.predict(test_data.input))
    forecast = forecasts[0].mean
    y_pred = forecast.tolist()
    y_true = test_df["y"].values.tolist()

    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df.index, "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return mae, rmse

# Train each target (load, PV, battery)
def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, sampling_rate):
    return moirai_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, sampling_rate):
    return moirai_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)

def train_london_model(load_data, save_dir, freq, forecast_horizon, sampling_rate):
    return moirai_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate)

def train_germany_model(load_data, save_dir, freq, forecast_horizon, sampling_rate):
    return moirai_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate)

def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, sampling_rate):
    return moirai_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, sampling_rate)

# Full pipeline for Moirai
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, sampling_rate):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info("...Start Uni2TS/Moirai forecasting...")

    if selected_dataset == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})

        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})

    elif selected_dataset == "germany":
        logging.info("Forecasting Germany load")
        load_data = dataset_germany.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = train_germany_model(load_data, save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "london":
        logging.info("Forecasting London load")
        load_data = dataset_london.get_inputs_for_load()
        load_mae, load_rmse = train_london_model(load_data, save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})

    elif selected_dataset == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End Uni2TS/Moirai forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, sampling_rate):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate))
    freq_str = f"{train_freq}T"

    forecast_horizon = int(192 / (100 / sampling_rate))

    try:
        gc.collect()
        save_dir = f"results/results_{selected_dataset}/MOIRAI/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping Moirai due to error: {str(e)}", exc_info=True)

# Run all sampling rates and seeds
if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 2):  # Loop for run_num
            paper_forecasting_train(run_num, sampling_rate)
