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
import numpy as np  

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "belgium"  # Options: "belgium" or "germany" or "london" or "zonnedael"

dataset_belgium = DatasetBelgiumNF()
dataset_germany = DatasetGermanyNF()
dataset_london = DatasetLondonNF()
dataset_zonnedael = DatasetZonnedaelNF()
# =========================================================================================

# --- Naive Random Walk with Drift + Noise Model Implementation ---
class NaiveDrift:
    def __init__(self):
        self.last_value = None
        self.drift = 0
        self.noise_std = 0
        self.last_ds = None
        self.freq = None

    def fit(self, df):
        y = df['y'].values
        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 observations to estimate drift")
        self.last_value = y[-1]
        self.drift = (y[-1] - y[0]) / (n - 1)
        self.last_ds = df['ds'].iloc[-1]
        # Estimate noise std as std of first differences residuals around drift
        diffs = np.diff(y)
        drift_per_step = self.drift
        residuals = diffs - drift_per_step
        self.noise_std = np.std(residuals) if len(residuals) > 1 else 0

    def predict(self, h):
        if self.last_ds is None or self.freq is None:
            raise ValueError("Must call fit and set frequency before predict")

        future_dates = pd.date_range(start=self.last_ds + pd.Timedelta(self.freq), periods=h, freq=self.freq)
        
        # Generate forecast values: deterministic drift + Gaussian noise
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=h)
        forecast_values = [self.last_value + self.drift * (i + 1) + noise[i] for i in range(h)]

        return pd.DataFrame({'ds': future_dates, 'NaiveDrift': forecast_values})

    def fit_predict(self, df, h, freq='1min'):
        self.freq = freq
        self.fit(df)
        return self.predict(h)

# Use this class instead of external NaiveDrift
model_classes = {
    "NaiveDrift": NaiveDrift
}

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

def generic_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, nf_model_name="NaiveDrift", NFmodel=None):
    logging.info(f"Using {nf_model_name} model")

    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    # Instantiate your custom model and forecast
    model_instance = NFmodel()

    # For your NaiveDrift, use fit_predict, otherwise fallback (if needed)
    if nf_model_name == "NaiveDrift":
        forecast_df = model_instance.fit_predict(train_df, forecast_horizon, freq=freq)
    else:
        # Placeholder for other models if needed
        # (e.g., your original StatsForecast usage)
        raise NotImplementedError("Only NaiveDrift is implemented in this example.")

    logging.info(f"Forecast columns: {forecast_df.columns.tolist()}")  # DEBUG: show columns

    # Pick forecast column dynamically
    forecast_cols = [col for col in forecast_df.columns if col != "ds"]
    if not forecast_cols:
        raise ValueError("No forecast columns found in forecast_df")
    y_pred = forecast_df[forecast_cols[0]].values

    y_true = test_df["y"].values

    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} ({nf_model_name}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        f"{model_name}",
        save_dir
    )
    return None, mae, rmse

def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel):
    return generic_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, nf_model_name, NFmodel)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, nf_model_name, NFmodel):
    return generic_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, nf_model_name, NFmodel)

def train_london_model(load_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return generic_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, nf_model_name, NFmodel)

def train_germany_model(load_data, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return generic_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, nf_model_name, NFmodel)

def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    return generic_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, nf_model_name, NFmodel)

def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, nf_model_name, NFmodel):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info(f"...Start training using {nf_model_name}...")

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

def paper_forecasting_train(run_num, sampling_rate):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    sampling_rate = sampling_rate

    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate))
    freq_str = f"{train_freq}min"
    forecast_horizon = int(192 / (100 / sampling_rate))

    for model_name, model_class in model_classes.items():
        try:
            gc.collect()
            save_dir = f"results/results_{selected_dataset}/{model_name}/Sampling_{sampling_rate:.0f}/Run_{run_num}"
            train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, nf_model_name=model_name, NFmodel=model_class)
            gc.collect()
        except Exception as e:
            logging.error(f"Skipping model {model_name} due to error: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 2):
            paper_forecasting_train(run_num, sampling_rate)
