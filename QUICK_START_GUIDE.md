# Quick Start Implementation Guide

## Getting Started with Phase 1 Development

This guide will help you set up your development environment and start implementing Phase 1 of the Crypto Market Regime Classification platform.

---

## Prerequisites

### Required Software
- **Python**: 3.11 or higher
- **Git**: Latest version
- **Docker**: For containerization (optional but recommended)
- **PostgreSQL**: 14+ with TimescaleDB extension
- **Redis**: 7+ for caching

### Recommended Tools
- **VS Code** with Python extension
- **Postman** or **Insomnia** for API testing
- **DBeaver** or **pgAdmin** for database management
- **Poetry** for Python dependency management (alternative to pip)

---

## Step 1: Environment Setup (Day 1)

### 1.1 Clone and Setup Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd Crypto-Market-Regime-Classification

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 1.2 Install Dependencies

Create `requirements.txt`:
```bash
# requirements.txt
# Core dependencies
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# ML/Data Science
tensorflow==2.15.0
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.3
shap==0.44.1

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
asyncpg==0.29.0

# Caching
redis==5.0.1
hiredis==2.3.2

# API & Auth
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
passlib[bcrypt]==1.7.4

# Data Collection
requests==2.31.0
websockets==12.0
python-binance==1.0.19

# Monitoring
prometheus-client==0.19.0
sentry-sdk==1.40.0

# ML Ops
mlflow==2.10.0

# Task Queue
celery==5.3.6
flower==2.0.1

# Utils
python-dotenv==1.0.1
PyYAML==6.0.1
click==8.1.7
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 1.3 Development Dependencies

Create `requirements-dev.txt`:
```bash
# Testing
pytest==8.0.0
pytest-cov==4.1.0
pytest-asyncio==0.23.3
pytest-mock==3.12.0
httpx==0.26.0

# Code Quality
black==24.1.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2
pylint==3.0.3

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.6
```

Install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

### 1.4 Environment Configuration

Create `.env` file:
```bash
# .env
# Application
APP_NAME=crypto-regime-classifier
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/crypto_regime
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# External APIs
COINGECKO_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=

# ML Model
MODEL_PATH=/models/best_crypto_model.h5
SCALER_PATH=/models/scaler.pkl
MODEL_VERSION=1.0.0

# Monitoring
SENTRY_DSN=
PROMETHEUS_ENABLED=True
```

Add to `.gitignore`:
```bash
# Add to .gitignore
.env
*.pyc
__pycache__/
venv/
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.DS_Store
```

---

## Step 2: Project Structure Setup (Days 2-3)

### 2.1 Create Directory Structure

```bash
mkdir -p src/{data,models,features,evaluation,api,utils}
mkdir -p src/data/{collectors,preprocessors,loaders}
mkdir -p tests/{unit,integration}
mkdir -p configs
mkdir -p scripts
mkdir -p notebooks
mkdir -p models
mkdir -p logs
```

### 2.2 Project Structure Overview

```
crypto-regime-classifier/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data pipeline
│   │   ├── __init__.py
│   │   ├── collectors/           # Data collection
│   │   │   ├── __init__.py
│   │   │   ├── coingecko.py
│   │   │   ├── binance.py
│   │   │   └── economic.py
│   │   ├── preprocessors/        # Data cleaning
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py
│   │   │   └── validator.py
│   │   └── loaders/              # Data loading
│   │       ├── __init__.py
│   │       └── data_loader.py
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── lag_features.py
│   │   └── rolling_stats.py
│   ├── models/                   # ML models
│   │   ├── __init__.py
│   │   ├── lstm_classifier.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   ├── evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualizations.py
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── regime.py
│   │   │   └── auth.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── regime.py
│   │   │   └── user.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── rate_limit.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── database.py
├── tests/                        # Tests
│   ├── __init__.py
│   ├── unit/
│   └── integration/
├── configs/                      # Configuration files
│   ├── model_config.yaml
│   └── feature_config.yaml
├── scripts/                      # Utility scripts
│   ├── train_model.py
│   └── migrate_data.py
├── notebooks/                    # Jupyter notebooks
│   └── main.ipynb               # Original notebook
├── models/                       # Saved models
│   └── best_crypto_model.h5
├── data/                         # Data storage
├── logs/                         # Application logs
├── requirements.txt
├── requirements-dev.txt
├── .env
├── .gitignore
├── README.md
└── Dockerfile
```

---

## Step 3: Core Implementation (Days 3-7)

### 3.1 Configuration Management

Create `src/utils/config.py`:

```python
"""Configuration management using Pydantic."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "crypto-regime-classifier"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Database
    database_url: str
    database_pool_size: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440

    # Model
    model_path: str = "/models/best_crypto_model.h5"
    scaler_path: str = "/models/scaler.pkl"
    model_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
```

### 3.2 Logging Setup

Create `src/utils/logging.py`:

```python
"""Logging configuration."""
import logging
import sys
from pathlib import Path
from src.utils.config import settings


def setup_logging():
    """Configure application logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log"),
        ],
    )

    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger for module."""
    return logging.getLogger(name)
```

### 3.3 Feature Engineering Module

Create `src/features/technical_indicators.py`:

```python
"""Technical indicators calculation."""
import pandas as pd
import numpy as np
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for cryptocurrency data."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = "close", window: int = 7) -> pd.Series:
        """Calculate Simple Moving Average."""
        return df[column].rolling(window=window).mean()

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = "close", span: int = 12) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return df[column].ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, column: str = "close", fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD."""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, column: str = "close", window: int = 20, num_std: int = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return sma, upper_band, lower_band

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        logger.info(f"Calculating technical indicators for {len(df)} rows")

        df = df.copy()

        # Moving averages
        df["SMA_7"] = self.calculate_sma(df, window=7)
        df["SMA_20"] = self.calculate_sma(df, window=20)
        df["EMA_12"] = self.calculate_ema(df, span=12)
        df["EMA_26"] = self.calculate_ema(df, span=26)

        # Momentum
        df["RSI_14"] = self.calculate_rsi(df, period=14)
        df["MACD"] = self.calculate_macd(df)

        # Volatility
        sma_20, upper_band, lower_band = self.calculate_bollinger_bands(df)
        df["Upper_Band"] = upper_band
        df["Lower_Band"] = lower_band
        df["STD_20"] = df["close"].rolling(window=20).std()

        logger.info(f"Technical indicators calculated successfully")
        return df
```

### 3.4 Model Predictor

Create `src/models/predictor.py`:

```python
"""Regime prediction engine."""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List
from tensorflow import keras
from src.features.technical_indicators import TechnicalIndicators
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegimePrediction:
    """Regime prediction result."""

    def __init__(
        self,
        symbol: str,
        regime: str,
        confidence: float,
        probabilities: Dict[str, float],
        timestamp: datetime,
    ):
        self.symbol = symbol
        self.regime = regime
        self.confidence = confidence
        self.probabilities = probabilities
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp.isoformat(),
        }


class RegimePredictor:
    """Cryptocurrency market regime predictor."""

    REGIME_LABELS = ["Bearish", "Bullish", "Neutral"]
    FEATURE_COLUMNS = [
        "open", "high", "low", "close", "volume", "market",
        "close_ratio", "spread", "SMA_7", "RSI_14", "EMA_12",
        "EMA_26", "MACD", "SMA_20", "STD_20", "Upper_Band", "Lower_Band"
    ]

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """Initialize predictor."""
        self.model_path = model_path or settings.model_path
        self.scaler_path = scaler_path or settings.scaler_path

        logger.info(f"Loading model from {self.model_path}")
        self.model = keras.models.load_model(self.model_path)

        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

        self.feature_engineer = TechnicalIndicators()

    def prepare_sequence(self, df: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
        """Prepare data sequence for prediction."""
        # Ensure we have enough data
        if len(df) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} days of data")

        # Get feature columns
        features = df[self.FEATURE_COLUMNS].values

        # Take last sequence_length rows
        sequence = features[-sequence_length:]

        return sequence

    def predict(self, df: pd.DataFrame) -> RegimePrediction:
        """Predict market regime."""
        try:
            # Apply feature engineering
            df = self.feature_engineer.calculate_all(df)

            # Prepare sequence
            X = self.prepare_sequence(df)

            # Scale features
            X_scaled = self.scaler.transform(X.reshape(-1, len(self.FEATURE_COLUMNS)))
            X_scaled = X_scaled.reshape(1, 10, len(self.FEATURE_COLUMNS))

            # Predict
            probabilities = self.model.predict(X_scaled, verbose=0)[0]

            # Get regime
            regime_idx = np.argmax(probabilities)
            regime = self.REGIME_LABELS[regime_idx]
            confidence = float(probabilities[regime_idx])

            # Create result
            prediction = RegimePrediction(
                symbol=df["symbol"].iloc[-1] if "symbol" in df.columns else "UNKNOWN",
                regime=regime,
                confidence=confidence,
                probabilities={
                    "bearish": float(probabilities[0]),
                    "bullish": float(probabilities[1]),
                    "neutral": float(probabilities[2]),
                },
                timestamp=datetime.utcnow(),
            )

            logger.info(
                f"Prediction: {prediction.regime} with {prediction.confidence:.2%} confidence"
            )

            return prediction

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
```

### 3.5 FastAPI Application

Create `src/api/main.py`:

```python
"""FastAPI application."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils.config import settings
from src.utils.logging import setup_logging, get_logger
from src.models.predictor import RegimePredictor

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = RegimePredictor()


@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.app_env}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.model_version,
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
```

---

## Step 4: Testing Setup (Day 7)

### 4.1 Create Test Configuration

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=src --cov-report=html --cov-report=term
```

### 4.2 Sample Unit Test

Create `tests/unit/test_technical_indicators.py`:

```python
"""Tests for technical indicators."""
import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators


@pytest.fixture
def sample_data():
    """Create sample cryptocurrency data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "symbol": ["BTC"] * 100,
        "close": np.random.uniform(40000, 50000, 100),
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(1e9, 1e10, 100),
    }
    return pd.DataFrame(data)


def test_calculate_sma(sample_data):
    """Test SMA calculation."""
    indicators = TechnicalIndicators()
    sma = indicators.calculate_sma(sample_data, window=7)

    assert len(sma) == len(sample_data)
    assert not sma.iloc[-1] == np.nan


def test_calculate_rsi(sample_data):
    """Test RSI calculation."""
    indicators = TechnicalIndicators()
    rsi = indicators.calculate_rsi(sample_data, period=14)

    assert len(rsi) == len(sample_data)
    # RSI should be between 0 and 100
    assert all((rsi >= 0) & (rsi <= 100) | rsi.isna())
```

---

## Step 5: Running the Application

### 5.1 Run API Server

```bash
# Development mode with auto-reload
python src/api/main.py

# Or using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5.2 Test API

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

### 5.3 Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_technical_indicators.py
```

---

## Step 6: Docker Setup (Optional)

### 6.1 Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY .env .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Create docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/crypto_regime
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  db:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=crypto_regime
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 6.3 Run with Docker

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Next Steps

After completing this setup:

1. **Migrate existing notebook code** into the modular structure
2. **Implement data collection** from CoinGecko/Binance APIs
3. **Build API endpoints** for regime prediction
4. **Add comprehensive tests** for all modules
5. **Set up CI/CD pipeline** with GitHub Actions
6. **Deploy to staging environment**

---

## Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: TensorFlow not loading model
```bash
# Solution: Check TensorFlow version matches training version
pip install tensorflow==2.15.0
```

**Issue**: Database connection error
```bash
# Solution: Verify PostgreSQL is running
docker-compose ps
# Check connection string in .env
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [TimescaleDB Docs](https://docs.timescale.com/)
- [Project Roadmap](./PROJECT_SCOPE_AND_ROADMAP.md)
- [Architecture Diagram](./ARCHITECTURE.md)

---

**Happy Coding! 🚀**
