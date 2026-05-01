import joblib

from features import add_lag_features, add_time_features

model_path = "../models/xgboost_hourly_model.pkl"
features_path = "../models/features.pkl"

xgb_model = joblib.load(model_path)
features = joblib.load(features_path)


def predict(df):
    df = add_time_features(df)
    df = add_lag_features(df)

    X = df[features]
    preds = xgb_model.predict(X)
    df["prediction"] = preds
    return df
