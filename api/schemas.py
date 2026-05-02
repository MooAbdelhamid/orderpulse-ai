from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """Single data point for prediction"""

    timestamp: str = Field(..., description="ISO format datetime: YYYY-MM-DD HH:MM:SS")
    demand: float = Field(..., description="Historical demand value")


class PredictionRequest(BaseModel):
    """Request for making predictions"""

    data: List[DataPoint] = Field(..., description="List of timestamp and demand data")


class MetricsData(BaseModel):
    """Prediction metrics"""

    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    count: int = Field(..., description="Number of predictions")


class PredictionResponse(BaseModel):
    """Response with predictions"""

    success: bool
    predictions: List[float] = Field(..., description="Predicted demand values")
    metrics: MetricsData


class FileUploadResponse(BaseModel):
    """Response from file upload"""

    filename: str
    rows: int
    columns: List[str]
    is_valid: bool
    message: str
    data_preview: List[Dict[str, Any]]
