"""Pydantic schemas for regime prediction endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime


class RegimeProbabilities(BaseModel):
    """Probability distribution across regimes."""
    bearish: float = Field(..., ge=0.0, le=1.0, description="Probability of bearish regime")
    bullish: float = Field(..., ge=0.0, le=1.0, description="Probability of bullish regime")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral regime")


class RegimeResponse(BaseModel):
    """Response model for regime prediction."""
    symbol: str = Field(..., description="Cryptocurrency symbol")
    regime: str = Field(..., description="Predicted regime (Bearish, Bullish, or Neutral)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for prediction")
    probabilities: RegimeProbabilities
    timestamp: datetime = Field(..., description="Prediction timestamp (UTC)")
    features_used: int = Field(..., description="Number of features used in prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC",
                "regime": "Bullish",
                "confidence": 0.8523,
                "probabilities": {
                    "bearish": 0.0823,
                    "bullish": 0.8523,
                    "neutral": 0.0654
                },
                "timestamp": "2025-11-19T12:00:00Z",
                "features_used": 17
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Model version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Check timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
