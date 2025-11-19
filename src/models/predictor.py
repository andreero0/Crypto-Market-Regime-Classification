"""Regime prediction engine using trained LSTM model."""
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Import TensorFlow/Keras
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Model prediction will not work.")

from src.features.technical_indicators import TechnicalIndicators
from src.features.lag_features import LagFeatureBuilder
from src.features.rolling_stats import RollingStatistics
from src.utils.config import settings

logger = logging.getLogger(__name__)


class RegimePrediction:
    """Container for regime prediction results."""

    def __init__(
        self,
        symbol: str,
        regime: str,
        confidence: float,
        probabilities: Dict[str, float],
        timestamp: datetime,
        features_used: int = 17
    ):
        self.symbol = symbol
        self.regime = regime
        self.confidence = confidence
        self.probabilities = probabilities
        self.timestamp = timestamp
        self.features_used = features_used

    def to_dict(self) -> dict:
        """Convert prediction to dictionary format."""
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                k: round(v, 4) for k, v in self.probabilities.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "features_used": self.features_used
        }

    def __repr__(self) -> str:
        return f"RegimePrediction(symbol={self.symbol}, regime={self.regime}, confidence={self.confidence:.2%})"


class RegimePredictor:
    """
    Cryptocurrency market regime predictor.

    Uses trained Bidirectional LSTM model to classify market conditions
    into Bearish, Bullish, or Neutral regimes.
    """

    REGIME_LABELS = ["Bearish", "Bullish", "Neutral"]

    # Feature columns expected by the model (17 features)
    FEATURE_COLUMNS = [
        "open", "high", "low", "close", "volume", "market",
        "close_ratio", "spread", "SMA_7", "RSI_14", "EMA_12",
        "EMA_26", "MACD", "SMA_20", "STD_20", "Upper_Band", "Lower_Band"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        sequence_length: int = 10
    ):
        """
        Initialize the regime predictor.

        Args:
            model_path: Path to saved Keras model (.h5 file)
            scaler_path: Path to saved scaler (.pkl file)
            sequence_length: Number of timesteps for LSTM input
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for model prediction. Install with: pip install tensorflow")

        self.model_path = Path(model_path or settings.model_path)
        self.scaler_path = Path(scaler_path or settings.scaler_path)
        self.sequence_length = sequence_length

        # Initialize feature engineering components
        self.technical_indicators = TechnicalIndicators()
        self.lag_builder = LagFeatureBuilder()
        self.rolling_stats = RollingStatistics()

        # Load model and scaler
        self._load_model()
        self._load_scaler()

    def _load_model(self):
        """Load the trained Keras model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please ensure the trained model is available at this path."
            )

        logger.info(f"Loading model from {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        logger.info("Model loaded successfully")

    def _load_scaler(self):
        """Load the fitted scaler for feature normalization."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {self.scaler_path}\n"
                f"Please ensure the fitted scaler is available at this path."
            )

        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(str(self.scaler_path))
        logger.info("Scaler loaded successfully")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering pipeline to raw OHLCV data.

        Args:
            df: DataFrame with raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")

        # Calculate technical indicators
        df = self.technical_indicators.calculate_all(df)

        # Note: Lag features and rolling stats are optional for prediction
        # They're more useful for training but may cause data loss at boundaries
        # For production, we use only the 17 core features

        logger.info("Feature engineering complete")
        return df

    def prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare data sequence for LSTM model input.

        Args:
            df: DataFrame with engineered features

        Returns:
            Numpy array of shape (sequence_length, n_features)

        Raises:
            ValueError: If not enough data for sequence
        """
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} days of data, "
                f"but only {len(df)} rows provided"
            )

        # Ensure we have all required features
        missing_features = set(self.FEATURE_COLUMNS) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Get feature columns in correct order
        features = df[self.FEATURE_COLUMNS].values

        # Take last sequence_length rows
        sequence = features[-self.sequence_length:]

        return sequence

    def predict(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> RegimePrediction:
        """
        Predict market regime for given cryptocurrency data.

        Args:
            df: DataFrame with OHLCV data (at least sequence_length rows)
            symbol: Symbol identifier (e.g., 'BTC', 'ETH')

        Returns:
            RegimePrediction object with regime, confidence, and probabilities

        Raises:
            ValueError: If insufficient data or missing features
        """
        try:
            # Determine symbol
            if symbol is None:
                symbol = df['symbol'].iloc[-1] if 'symbol' in df.columns else 'UNKNOWN'

            logger.info(f"Predicting regime for {symbol}")

            # Apply feature engineering
            df_engineered = self.engineer_features(df.copy())

            # Prepare sequence
            X = self.prepare_sequence(df_engineered)

            # Scale features
            X_scaled = self.scaler.transform(X.reshape(-1, len(self.FEATURE_COLUMNS)))
            X_scaled = X_scaled.reshape(1, self.sequence_length, len(self.FEATURE_COLUMNS))

            # Predict probabilities
            probabilities = self.model.predict(X_scaled, verbose=0)[0]

            # Get regime and confidence
            regime_idx = np.argmax(probabilities)
            regime = self.REGIME_LABELS[regime_idx]
            confidence = float(probabilities[regime_idx])

            # Create prediction result
            prediction = RegimePrediction(
                symbol=symbol,
                regime=regime,
                confidence=confidence,
                probabilities={
                    "bearish": float(probabilities[0]),
                    "bullish": float(probabilities[1]),
                    "neutral": float(probabilities[2]),
                },
                timestamp=datetime.utcnow(),
                features_used=len(self.FEATURE_COLUMNS)
            )

            logger.info(
                f"Prediction for {symbol}: {prediction.regime} "
                f"(confidence: {prediction.confidence:.2%})"
            )

            return prediction

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            raise

    def predict_batch(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, RegimePrediction]:
        """
        Predict regimes for multiple cryptocurrencies.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame

        Returns:
            Dictionary mapping symbol to RegimePrediction
        """
        logger.info(f"Batch prediction for {len(data_dict)} symbols")

        predictions = {}
        for symbol, df in data_dict.items():
            try:
                predictions[symbol] = self.predict(df, symbol=symbol)
            except Exception as e:
                logger.error(f"Failed to predict for {symbol}: {e}")
                continue

        logger.info(f"Successfully predicted {len(predictions)}/{len(data_dict)} symbols")
        return predictions
