import pandas as pd
import numpy as np
from utils import min_max_normalize
from scipy import stats

class DatasetBelgiumNeuralForecast:
    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_load(self, start_dt, end_dt):
        load_file = "data/germany_wpuq_dataset/common_env/SFH19_2023_2024_15min_3_month.csv"
        load_df = pd.read_csv(load_file)
        load_df["datetime"] = pd.to_datetime(load_df["datetime"])

        # Filter rows within time range
        df = load_df[
            (load_df["datetime"] >= start_dt) & (load_df["datetime"] < end_dt)
        ][["datetime", "Consumer_0_electric (kW)"]]

        df = df.rename(columns={"datetime": "ds", "Consumer_0_electric (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = "series_load"
        return df[["unique_id", "ds", "y"]]

    # Load PV data for specified range and reformat for NeuralForecast
    def get_inputs_for_pv(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"

        df = pd.read_csv(pv_file, parse_dates=["datetime"])

        # Filter rows within time range
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "SolarPv_0 (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = f"series_pv_house_{house}"
        return df[["unique_id", "ds", "y"]]

    def get_inputs_for_battery(self, house, start_dt, end_dt, freq="15min"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"

        df = pd.read_csv(battery_file, parse_dates=["datetime"])

        # Filter and rename
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "Battery_0 (kW)": "y"})

        # Create full datetime range
        full_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(freq), freq=freq)
        df = df.set_index("ds").reindex(full_range)
        df.index.name = "ds"

        # Detect and replace anomalies with NaN
        if df["y"].notna().sum() > 0:  # Avoid zscore on all-NaN column
            z_scores = np.abs(stats.zscore(df["y"], nan_policy='omit'))
            df.loc[z_scores >= 3, "y"] = np.nan

        # Interpolate
        df["y"] = df["y"].interpolate(method='linear', limit_direction='both')

        # Normalize
        df["y"] = min_max_normalize(df[["y"]])

        # Add unique_id and reset index
        df["unique_id"] = f"series_battery_house_{house}"
        df = df.reset_index()  # 'ds' is already the index name

        return df[["unique_id", "ds", "y"]]

class DatasetBelgium1D:
    # Load electric consumption data for specified range and reformat for LightGBM
    def get_load_data(self, start_dt, end_dt):
        load_file = "data/belgium_dataset/common_env/SFH19_2023_2024_15min_3_month.csv"
        df = pd.read_csv(load_file)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Filter rows within time range
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]

        # Normalize target
        df["y"] = min_max_normalize(df[["Consumer_0_electric (kW)"]])

        # Time-based features
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y

    # Load PV data for specified range and reformat for LightGBM
    def get_pv_data(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"

        df = pd.read_csv(pv_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]

        # Normalize target
        df["y"] = min_max_normalize(df[["SolarPv_0 (kW)"]])

        # Time-based features
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        
        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y

    # Load battery data for specified range and reformat for LightGBM
    def get_battery_data(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"

        df = pd.read_csv(battery_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]

        # Detect anomalies using Z-score on 'y'
        df["y"] = df["Battery_0 (kW)"]
        z_scores = np.abs(stats.zscore(df["y"], nan_policy='omit'))
        df.loc[z_scores >= 3, "y"] = np.nan

        # Impute missing values (NaNs and replaced anomalies) with linear interpolation
        df["y"] = df["y"].interpolate(method='linear', limit_direction='both')

        # Normalize target
        df["y"] = min_max_normalize(df[["y"]])

        # Time-based features
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y

class DatasetLondonZonnedaelNeuralForecast:
    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_london_consumption(self):
        london_consumption_file = "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])

        df = df.rename(columns={"datetime": "ds", "KWH/hh (per half hour)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = "series_load"
        return df[["unique_id", "ds", "y"]]

    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_zonnedael_consumption(self, customer_number: int):
        zonnedael_consumption_file = "data/liander_zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)

        # Correct datetime parsing format (day-first)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")

        client_no = f"Klant {customer_number}"
        df = df.rename(columns={"datetime": "ds", client_no: "y"})
        df["y"] = min_max_normalize(df["y"])
        df["unique_id"] = f"series_load_{customer_number}"

        return df[["unique_id", "ds", "y"]]

class DatasetLondonZonnedael1D:
    # Load electric consumption data for specified range and reformat for LightGBM
    def get_inputs_for_london_consumption(self):
        london_consumption_file = "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])

        # Normalize target
        df["y"] = min_max_normalize(df[["KWH/hh (per half hour)"]])

        # Time-based features
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y

    def get_inputs_for_zonnedael_consumption(self, customer_number: int):
        zonnedael_consumption_file = "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Construct dynamic customer column name
        customer_column = f"Klant {customer_number}"

        # Normalize target
        df["y"] = min_max_normalize(df[[customer_column]])

        # Time-based features
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour", "is_weekend"]]
        y = df["y"]
        return X, y

