import pandas as pd
import os
import warnings
import logging
import time
import gc  
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import mxnet as mx
from dataset_config import DatasetBelgiumNeuralForecast, DatasetLondonZonnedaelNeuralForecast
from utils import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from gluonts.model.seq2seq import MQRNNEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.model.deepar import DeepAREstimator

# ============================
# Dataset Selection Toggle
selected_dataset = "belgium"  # Options: "belgium" or "london_zonnedael"
dataset_belgium = DatasetBelgiumNeuralForecast()
dataset_london_zonnedael = DatasetLondonZonnedaelNeuralForecast()
n_epochs = 100  # Number of epochs for training
# ============================

# ========================
# Supported GluonTS Models
# ========================
model_classes = {
    "DeepAR": DeepAREstimator,
    "DeepFactor": DeepFactorEstimator,
    "MQCNN": MQRNNEstimator,
    "MQRNN": MQCNNEstimator,
    "TemporalFusionTransformer": TemporalFusionTransformerEstimator, # TFT
    "WaveNet": WaveNetEstimator
}

# ========================
# Logger Setup
# ========================
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

# ========================
# Helper to Convert to ListDataset
# ========================
def to_gluonts_dataset(df, freq, start):
    return ListDataset([{"start": start, "target": df["y"].values}], freq=freq)

# ========================
# Core GluonTS Model Training
# ========================
def train_gluonts_model(y_df, model_name, save_dir, model_class, sampling_rate, forecast_horizon, freq):
    current_seed = int(time.time()) % (2**32 - 1)
    logging.info(f"Using random seed: {current_seed}")

    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df

    start_idx = pd.Timestamp(train_df["ds"].iloc[0])
    train_ds = to_gluonts_dataset(train_df, freq, start_idx)
    test_ds = to_gluonts_dataset(test_df, freq, start_idx)

    logging.info(f"Training {model_name} model on {len(train_df)} samples...")

    if model_name in ["MQRNN", "MQCNN"]:
        estimator = model_class(
            freq=freq,
            prediction_length=forecast_horizon,
            trainer=Trainer(epochs=n_epochs, num_batches_per_epoch=1, ctx=mx.cpu())
        )
    else:
        estimator = model_class(
            freq=freq,
            prediction_length=forecast_horizon,
            trainer=Trainer(epochs=n_epochs, num_batches_per_epoch=1, ctx=mx.cpu())
        )

    predictor = estimator.train(training_data=train_ds)

    forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    y_pred = forecasts[0].mean[:forecast_horizon]
    y_true = y_df["y"].values[-forecast_horizon:]

    mae, rmse, mape, r2 = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": y_df["ds"].values[-forecast_horizon:], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return predictor, mae, rmse, mape, r2

# ========================
# Wrapper Functions (Belgium)
# ========================
def train_load_model(load_data, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq):
    return train_gluonts_model(load_data, "Load", save_dir, model_class, sampling_rate, forecast_horizon, freq)

def train_pv_model(pv_data, save_dir, house, model_name, model_class, sampling_rate, forecast_horizon, freq):
    return train_gluonts_model(pv_data, f"PV_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq)

def train_battery_model(battery_data, save_dir, house, model_name, model_class, sampling_rate, forecast_horizon, freq):
    return train_gluonts_model(battery_data, f"BESS_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq)

# ========================
# Wrapper Functions (London Zonnedael)
# ========================
def train_london_model(load_data, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq):
    return train_gluonts_model(load_data, "london_load", save_dir, model_class, sampling_rate, forecast_horizon, freq)

def train_zonnedael_model(customer_id, data_df, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq):
    return train_gluonts_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, model_class, sampling_rate, forecast_horizon, freq)

# ========================
# Full Model Training Routine
# ========================
def train_all_models(start_dt, end_dt, save_dir, model_name, model_class, sampling_rate):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info(f"...Start training using {model_name} with {sampling_rate}% sampling rate...")

    forecast_horizon = int(192 / (100 / sampling_rate))
    freq = f"{int(15 * (100 / sampling_rate))}min"

    if selected_dataset == "belgium":

        logging.info("Training load model")
        load_data = dataset_belgium.get_inputs_for_load(start_dt, end_dt)
        _, load_mae, load_rmse, load_mape, load_r2 = train_load_model(load_data, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq)
        metrics.append({"model": "load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            _, pv_mae, pv_rmse, pv_mape, pv_r2 = train_pv_model(pv_data, save_dir, house, model_name, model_class, sampling_rate, forecast_horizon, freq)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse, "MAPE": pv_mape, "R2": pv_r2})

        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            _, b_mae, b_rmse, b_mape, b_r2 = train_battery_model(battery_data, save_dir, house, model_name, model_class, sampling_rate, forecast_horizon, freq)
            metrics.append({"model": f"bess_house_{house}", "MAE": b_mae, "RMSE": b_rmse, "MAPE": b_mape, "R2": b_r2})

    elif selected_dataset == "london_zonnedael":

        logging.info("Training London load model")
        london_data = dataset_london_zonnedael.get_inputs_for_london_consumption()
        _, load_mae, load_rmse, load_mape, load_r2 = train_london_model(london_data, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            cust_data = dataset_london_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            _, c_mae, c_rmse, c_mape, c_r2 = train_zonnedael_model(customer_id, cust_data, save_dir, model_name, model_class, sampling_rate, forecast_horizon, freq)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": c_mae, "RMSE": c_rmse, "MAPE": c_mape, "R2": c_r2})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

# ========================
# Run All Selected Models
# ========================
def paper_forecasting_train():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    for sampling_rate in [100/3, 50, 100]:
        for model_name, model_class in model_classes.items():
            for run_num in range(1, 11):  # Run 10 times
                try:
                    gc.collect()
                    if mx.context.num_gpus() > 0:
                        mx.nd.waitall()

                    save_dir = f"results/results_{selected_dataset}/{model_name}/Sampling_{sampling_rate:.0f}/Epochs_100_{run_num}"
                    train_all_models(start_dt, end_dt, save_dir, model_name, model_class, sampling_rate)

                    gc.collect()
                    if mx.context.num_gpus() > 0:
                        mx.nd.waitall()

                except Exception as e:
                    logging.error(f"Skipping model {model_name} run {run_num} at sampling {sampling_rate}% due to error: {str(e)}", exc_info=True)
                    continue

# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    paper_forecasting_train()
