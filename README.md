# OrderPulse AI

Production demand forecasting system for food delivery using XGBoost, FastAPI, and Streamlit.

## What it does

Predicts hourly order demand by analyzing temporal patterns (hour, day of week) and historical trends (lag features). Takes CSV file with timestamps and demand values, outputs predictions with performance metrics (RMSE, MAE, MAPE).

## Tech Stack

- **ML Model**: XGBoost (time series regression)
- **Backend**: FastAPI with Pydantic validation
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Containarization**: Docker


## How it works

1. Upload CSV with `timestamp` and `demand` columns (min 25 rows)
2. Pipeline extracts time features (hour, day_of_week, is_weekend, sin/cos encoding)
3. Calculates lag features (lag_1, lag_24, rolling_24)
4. XGBoost predicts demand
5. Returns predictions with RMSE/MAE/MAPE metrics

## Project Structure

```
src/
  pipeline.py        # Feature engineering + inference
api/
  main.py           # FastAPI endpoints
  schemas.py        # Pydantic models
ui/
  app.py            # Streamlit dashboard
models/
  xgboost_hourly_model.pkl
  features.pkl
```

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - File validation
- `POST /batch-predict` - Upload and predict

## Status

Phase 3 - Backend & Frontend development. Local testing ready. Docker and deployment coming next.