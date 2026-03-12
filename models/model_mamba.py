import os
import warnings
import logging
import time
import gc
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
from dataset_config import DatasetBelgiumNeuralForecast, DatasetLondonZonnedaelNeuralForecast
from utils import calculate_metrics, forecast_plot_and_csv, plot_model_metrics

# ============================ Dataset Selection Toggle ===================================
selected_dataset = "london_zonnedael"  # Options: "belgium" or "london_zonnedael"

dataset_belgium = DatasetBelgiumNeuralForecast()
dataset_london_zonnedael = DatasetLondonZonnedaelNeuralForecast()
n_epochs = 50
# =========================================================================================

# ============================ Mamba Model Wrapper ========================================
class MambaForecaster(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, forecast_horizon):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        # x: (batch, length, d_model)
        y = self.mamba(x)
        last_state = y[:, -1, :]
        return self.head(last_state)

# ============================ Logger Setup ===============================================
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

def build_windows(series, context_length, forecast_horizon):
    values = series.values.astype("float32")
    total_length = context_length + forecast_horizon
    if len(values) < total_length:
        return None, None

    X = []
    y = []
    for i in range(len(values) - total_length + 1):
        window = values[i:i + total_length]
        X.append(window[:context_length])
        y.append(window[context_length:])

    X = torch.tensor(X).unsqueeze(-1)  # (batch, length, 1)
    y = torch.tensor(y)  # (batch, horizon)
    return X, y

# Core function for Mamba forecasting
def mamba_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate):
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    context_length = min(forecast_horizon * 2, max(1, len(train_df) - forecast_horizon))
    logging.info(
        f"Running Mamba {model_name} for {len(train_df)} samples "
        f"(sampling {sampling_rate:.0f}%, context {context_length}, horizon {forecast_horizon})"
    )

    X_train, y_train = build_windows(train_df["y"], context_length, forecast_horizon)
    if X_train is None:
        raise ValueError("Not enough data to build training windows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaForecaster(
        d_model=1,
        d_state=16,
        d_conv=4,
        expand=2,
        forecast_horizon=forecast_horizon
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch}/{n_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")

    model.eval()
    with torch.no_grad():
        context_values = train_df["y"].values.astype("float32")
        context_values = context_values[-context_length:]
        context_tensor = torch.tensor(context_values).unsqueeze(0).unsqueeze(-1).to(device)
        forecast = model(context_tensor).cpu().numpy().flatten()

    y_true = test_df["y"].values
    y_pred = forecast[:forecast_horizon]

    mae, rmse, mape, r2 = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name, save_dir
    )
    return mae, rmse, mape, r2

# Train each target (load, PV, battery)
def train_load_model(load_data, save_dir, freq, forecast_horizon, sampling_rate):
    return mamba_forecast_model(load_data, "Load", save_dir, freq, forecast_horizon, sampling_rate)

def train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, sampling_rate):
    return mamba_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)

def train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, sampling_rate):
    return mamba_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)

# London-Zonnedael training functions
def train_london_model(load_data, save_dir, freq, forecast_horizon, sampling_rate):
    return mamba_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate)

def train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, sampling_rate):
    return mamba_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, sampling_rate)

# Full pipeline for Mamba
def train_all_models(start_dt, end_dt, save_dir, freq, forecast_horizon, sampling_rate):
    setup_model_logger(save_dir)
    metrics = []

    start_time = time.time()
    logging.info("...Start Mamba forecasting...")

    if selected_dataset == "belgium":
        logging.info("Forecasting load consumption")
        load_data = dataset_belgium.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse, load_mape, load_r2 = train_load_model(load_data, save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset_belgium.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse, pv_mape, pv_r2 = train_pv_model(pv_data, save_dir, house, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse, "MAPE": pv_mape, "R2": pv_r2})

        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset_belgium.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse, battery_mape, battery_r2 = train_battery_model(battery_data, save_dir, house, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse, "MAPE": battery_mape, "R2": battery_r2})

    elif selected_dataset == "london_zonnedael":
        logging.info("Forecasting London load")
        load_data = dataset_london_zonnedael.get_inputs_for_london_consumption()
        load_mae, load_rmse, load_mape, load_r2 = train_london_model(load_data, save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse, "MAPE": load_mape, "R2": load_r2})

        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset_london_zonnedael.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse, cust_mape, cust_r2 = train_zonnedael_model(customer_id, data_df, save_dir, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse, "MAPE": cust_mape, "R2": cust_r2})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False)
    plot_model_metrics(metrics, save_dir)

    elapsed_time = time.time() - start_time
    logging.info("...End Mamba forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

# Entry point
def paper_forecasting_train(run_num, sampling_rate):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    train_freq = int(15 * (100 / sampling_rate))
    freq_str = f"{train_freq}min"
    forecast_horizon = int(192 / (100 / sampling_rate))

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_dir = f"results/results_{selected_dataset}/Mamba/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Skipping Mamba due to error: {str(e)}", exc_info=True)

# Run all sampling rates and seeds
if __name__ == "__main__":
    for sampling_rate in [25, 100/3, 50, 100]:
        for run_num in range(1, 11):
            paper_forecasting_train(run_num, sampling_rate)
