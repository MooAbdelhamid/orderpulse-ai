"""
Production pipeline for demand forecasting
"""

import joblib
import numpy as np
import pandas as pd


class DemandForecastPipeline:
    """
    Forecasting Pipeline

    Handles:
    - Data validation
    - Feature engineering
    - Model interface
    """

    def __init__(self, model_path: str, features_path: str):
        """
        Initialize pipeline with trained model and feature names

        Args:
            model_path: Path to model pickle
            features_path: Path to feature names pickle
        """
        try:
            self.model = joblib.load(model_path)
            self.features = joblib.load(features_path)
        except Exception as e:
            print(f"Error while initializing class: {e}")

    def validate_input(self, df: pd.DataFrame):
        """
        Validate input data

        Requirements:
        - Must have timestamp column
        - Must have demand column
        - Must be parsable

        Args:
            df: input data

        Returns:
            (is_valid, message)
        """
        if "timestamp" not in df.columns:
            return False, "Missing timestamp column"

        if "demand" not in df.columns:
            return False, "Missing demand column"

        if df[["timestamp", "demand"]].isnull().any().any():
            return False, "Found null values in timestamp or demand"

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            return False, f"Cannot parse timestamp: {e}"

        try:
            df["demand"] = pd.to_numeric(df["demand"])
        except Exception as e:
            return False, f"Cannot parse demand: {e}"

        return True, "Validation passed"

    def add_time_features(self, df: pd.DataFrame):
        """
        Add time features for time series forecasting
        """
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        return df

    def add_lag_features(self, df: pd.DataFrame):
        """
        Add lag and rolling features
        """
        df["lag_1"] = df["demand"].shift(1)
        df["lag_24"] = df["demand"].shift(24)

        df["rolling_24"] = df["demand"].rolling(24).mean()

        return df.dropna()

    def engineer_features(self, df: pd.DataFrame):
        """
        Execute full feature engineering pipeline
        """
        try:
            df = self.add_time_features(df)
            df = self.add_lag_features(df)
            return df
        except Exception as e:
            print(f"Feature engineering failed: {e}")
            return df

    def predict(self, df: pd.DataFrame):
        """
        Predict with engineered data
        """

        # Validate Input
        is_valid, message = self.validate_input(df)
        if not is_valid:
            return {"error": message, "success": False}

        # Engineer Features
        df = self.engineer_features(df)

        X = df[self.features]

        # Make Prediction
        y_pred = self.model.predict(X)

        y_actual = df["demand"].values
        timestamps = df["timestamp"].values

        residuals = y_actual - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / (y_actual + 1e-6))) * 100

        return {
            "success": True,
            "predictions": y_pred.tolist(),
            "actual": y_actual.tolist(),
            "timestamps": timestamps.astype(str).tolist(),
            "residuals": residuals.tolist(),
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "count": len(y_pred),
            },
        }
