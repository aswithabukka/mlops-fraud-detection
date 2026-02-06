"""
FastAPI application for fraud detection predictions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for credit card transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class TransactionRequest(BaseModel):
    """Single transaction prediction request."""
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str
    merchant_country: str
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_online: bool
    is_weekend: bool
    transactions_last_24h: int = Field(..., ge=0)
    total_amount_last_24h: float = Field(..., ge=0)
    transactions_last_1h: int = Field(..., ge=0)
    distance_from_home: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Prediction response."""
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


# Global model (loaded on startup)
model = None
preprocessor = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, preprocessor
    try:
        # In production, load from MLflow
        # model = load_production_model()
        logger.info("Model loading would happen here (MLflow integration)")
        logger.info("✅ API started successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """
    Predict if a transaction is fraudulent.

    Returns fraud probability and risk level.
    """
    try:
        # Generate transaction ID
        txn_id = f"TXN{np.random.randint(10000000, 99999999)}"

        # Convert to DataFrame
        data = pd.DataFrame([transaction.dict()])

        # In production: preprocess and predict
        # data_processed = preprocessor.transform(data)
        # fraud_prob = model.predict_proba(data_processed)[0, 1]

        # Mock prediction for demo
        fraud_prob = np.random.random()
        is_fraud = int(fraud_prob > settings.fraud_threshold)

        # Determine risk level
        if fraud_prob < 0.3:
            risk_level = "low"
        elif fraud_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        logger.info(f"Prediction: {txn_id}, fraud_prob={fraud_prob:.3f}")

        return PredictionResponse(
            transaction_id=txn_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(transactions: List[TransactionRequest]):
    """Batch prediction endpoint."""
    predictions = []
    for txn in transactions:
        pred = await predict(txn)
        predictions.append(pred)
    return predictions


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # In production: return Prometheus metrics
    return {"predictions_total": 0, "predictions_fraud": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
