import numpy as np
import pandas as pd


def add_time_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def add_lag_features(df):
    df["lag_1"] = df["demand"].shift(1)
    df["lag_24"] = df["demand"].shift(24)

    df["rolling_24"] = df["demand"].rolling(24).mean()

    return df.dropna()
