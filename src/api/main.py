"""FastAPI application for Crypto Market Regime Classification."""
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from src.utils.config import settings
from src.utils.logging import setup_logging, get_logger
from src.models.predictor import RegimePredictor
from src.api.schemas.regime import RegimeResponse, HealthResponse, ErrorResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Market Regime Classification API",
    description="AI-powered cryptocurrency market regime prediction using LSTM neural networks",
    version=settings.model_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded once at startup)
predictor: RegimePredictor = None
model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    global predictor, model_loaded

    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Model version: {settings.model_version}")

    # Try to load model
    try:
        model_path = Path(settings.model_path)
        scaler_path = Path(settings.scaler_path)

        if model_path.exists() and scaler_path.exists():
            logger.info("Loading model and scaler...")
            predictor = RegimePredictor(
                model_path=str(model_path),
                scaler_path=str(scaler_path)
            )
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model or scaler not found. Prediction endpoints will not work.")
            logger.warning(f"Model path: {model_path} (exists: {model_path.exists()})")
            logger.warning(f"Scaler path: {scaler_path} (exists: {scaler_path.exists()})")
            model_loaded = False

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.app_name}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.model_version,
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "regime_current": "/v1/regime/current?symbol=BTC",
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
async def health_check():
    """
    Health check endpoint.

    Returns service health status and model information.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.model_version,
        model_loaded=model_loaded,
        timestamp=datetime.utcnow()
    )


@app.get(
    "/v1/regime/current",
    response_model=RegimeResponse,
    tags=["Regime Prediction"],
    summary="Get current regime for cryptocurrency",
    description="Predict the current market regime (Bearish, Bullish, or Neutral) for a cryptocurrency"
)
async def get_current_regime(
    symbol: str = Query(..., description="Cryptocurrency symbol (e.g., BTC, ETH)", example="BTC"),
    data_file: str = Query(
        None,
        description="Optional: Path to CSV file with OHLCV data. If not provided, will try to load from data/processed/",
        example="data/processed/filtered_crypto_data.csv"
    )
):
    """
    Get current market regime prediction for a cryptocurrency.

    This endpoint:
    1. Loads historical OHLCV data for the symbol
    2. Applies feature engineering (technical indicators)
    3. Runs LSTM model prediction
    4. Returns regime classification with confidence scores

    **Note**: For this demo, you need to provide historical data via CSV.
    In production, this would fetch real-time data from APIs.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model and scaler files are available."
        )

    try:
        # Determine data source
        if data_file:
            data_path = Path(data_file)
        else:
            # Try default processed data location
            data_path = Path("data/processed/merged_data_final.csv")
            if not data_path.exists():
                data_path = Path("merged_data_final.csv")
            if not data_path.exists():
                data_path = Path("filtered_crypto_data.csv")

        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found. Please provide data_file parameter or place data at: {data_path}"
            )

        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pd.read_csv(data_path)

        # Filter for requested symbol
        if 'symbol' in df.columns:
            symbol_data = df[df['symbol'] == symbol.upper()].copy()
        else:
            # If no symbol column, assume single asset data
            symbol_data = df.copy()

        if len(symbol_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}. Available symbols: {df['symbol'].unique().tolist() if 'symbol' in df.columns else 'N/A'}"
            )

        # Ensure data is sorted by date
        if 'date' in symbol_data.columns:
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.sort_values('date')

        # Check minimum data requirements
        if len(symbol_data) < 30:  # Need enough for technical indicators + sequence
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}. Need at least 30 days, found {len(symbol_data)}"
            )

        # Run prediction
        logger.info(f"Running prediction for {symbol}")
        prediction = predictor.predict(symbol_data, symbol=symbol.upper())

        # Convert to response model
        response = RegimeResponse(
            symbol=prediction.symbol,
            regime=prediction.regime,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            timestamp=prediction.timestamp,
            features_used=prediction.features_used
        )

        return response

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
