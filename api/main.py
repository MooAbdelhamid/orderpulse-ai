import io
from datetime import datetime
from pathlib import Path

# from fastapi.responses import JSONResponse
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import FileUploadResponse
from src.pipeline import DemandForecastPipeline

# Initialize FastAPI
app = FastAPI(title="Demand Forecasting API")


# Add CORS middleware for Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
BASE_DIR = Path.cwd()
model_path = BASE_DIR / "models" / "xgboost_hourly_model.pkl"
features_path = BASE_DIR / "models" / "features.pkl"

pipeline = DemandForecastPipeline(model_path, features_path)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": pipeline is not None,
    }


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel/JSON file for predictions.
    """

    try:
        filename = file.filename.lower()
        content = await file.read()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use CSV, Excel, or JSON.",
            )
        # Validate data
        is_valid, message = pipeline.validate_input(df)

        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "is_valid": is_valid,
            "message": message,
            "data_preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Pass
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        filename = file.filename.lower()
        content = await file.read()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        result = pipeline.predict(df)

        if "error" in result and not result.get("success"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        raise e
