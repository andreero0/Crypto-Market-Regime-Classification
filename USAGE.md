# Usage Guide - Crypto Market Regime Classification

This guide shows you how to use the implemented regime classification system.

---

## Prerequisites

Before starting, ensure you have:

1. **Python 3.11+** installed
2. **Trained model files** in `models/` directory:
   - `best_crypto_model.h5` (LSTM model)
   - `scaler.pkl` (Fitted MinMaxScaler)
3. **Data files** (CSV with OHLCV data):
   - `merged_data_final.csv` or `filtered_crypto_data.csv`

---

## Installation

### 1. Set up environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if needed (defaults should work for local development)
```

---

## Usage Options

### Option 1: Test Prediction Script (Easiest)

Run the test script to see predictions for multiple cryptocurrencies:

```bash
python scripts/test_prediction.py
```

**Expected output:**
```
===========================================================
Crypto Market Regime Classification - Prediction Test
===========================================================

✅ Model path: models/best_crypto_model.h5
✅ Scaler path: models/scaler.pkl
✅ Data path: merged_data_final.csv

Loading model and scaler...
✅ Model loaded successfully

Loading cryptocurrency data...
✅ Data loaded successfully
Available symbols: ['BTC', 'ETH', 'LTC', ...]

Running predictions...
-----------------------------------------------------------

📊 BTC
   Regime:     Bullish
   Confidence: 85.23%
   Probabilities:
      Bearish: 8.23%
      Bullish: 85.23%
      Neutral: 6.54%

📊 ETH
   Regime:     Neutral
   Confidence: 72.45%
   ...
```

---

### Option 2: Python API

Use the predictor directly in your Python code:

```python
from src.models.predictor import RegimePredictor
from src.data.loaders.data_loader import CryptoDataLoader

# Load model
predictor = RegimePredictor(
    model_path="models/best_crypto_model.h5",
    scaler_path="models/scaler.pkl"
)

# Load data
loader = CryptoDataLoader("merged_data_final.csv")
btc_data = loader.get_latest_data("BTC", days=50)

# Predict regime
prediction = predictor.predict(btc_data, symbol="BTC")

# Access results
print(f"Regime: {prediction.regime}")
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Probabilities: {prediction.probabilities}")
```

**Prediction object attributes:**
- `symbol`: Cryptocurrency symbol (e.g., "BTC")
- `regime`: Predicted regime ("Bearish", "Bullish", or "Neutral")
- `confidence`: Confidence score (0.0 to 1.0)
- `probabilities`: Dict with probabilities for each regime
- `timestamp`: Prediction timestamp (UTC)

---

### Option 3: REST API

Start the FastAPI server and make HTTP requests:

#### Start the API server

```bash
python scripts/run_api.py
```

The API will start at `http://localhost:8000`

**Access interactive docs:** http://localhost:8000/docs

#### Make API requests

**1. Health check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2025-11-19T12:00:00Z"
}
```

**2. Get current regime:**
```bash
curl "http://localhost:8000/v1/regime/current?symbol=BTC"
```

**Response:**
```json
{
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
```

**3. Specify custom data file:**
```bash
curl "http://localhost:8000/v1/regime/current?symbol=ETH&data_file=path/to/data.csv"
```

---

### Option 4: Unit Tests

Run the test suite to validate all components:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_technical_indicators.py

# Run with verbose output
pytest -v
```

**Expected output:**
```
==================== test session starts ====================
tests/unit/test_technical_indicators.py::TestTechnicalIndicators::test_calculate_sma PASSED
tests/unit/test_technical_indicators.py::TestTechnicalIndicators::test_calculate_ema PASSED
tests/unit/test_technical_indicators.py::TestTechnicalIndicators::test_calculate_rsi PASSED
...
==================== 15 passed in 2.34s ====================
```

---

## Feature Engineering Example

See how features are engineered from raw OHLCV data:

```python
import pandas as pd
from src.features.technical_indicators import TechnicalIndicators

# Load your data
df = pd.read_csv("data.csv")

# Calculate technical indicators
indicators = TechnicalIndicators()
df_with_features = indicators.calculate_all(df)

# View added features
print(df_with_features.columns)
# ['open', 'high', 'low', 'close', 'volume', 'market',
#  'close_ratio', 'spread', 'SMA_7', 'RSI_14', 'EMA_12',
#  'EMA_26', 'MACD', 'SMA_20', 'STD_20', 'Upper_Band', 'Lower_Band']
```

---

## Docker Deployment

Run the API in a container:

```bash
# Build image
docker-compose build

# Start all services (API + PostgreSQL + Redis)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

The API will be available at http://localhost:8000

---

## Interpreting Results

### Regime Classifications

**🟢 Bullish Regime:**
- Both 1-day and 3-day price changes are positive
- At least one exceeds +1.5%
- **Trading Strategy:** Momentum trading, trend-following, long positions
- **Risk Level:** Medium (watch for reversal)

**🔴 Bearish Regime:**
- Both 1-day and 3-day price changes are negative
- At least one exceeds -1.5%
- **Trading Strategy:** Short selling, defensive positioning, reduce exposure
- **Risk Level:** High (consider hedging)

**🟡 Neutral Regime:**
- Price changes within ±1% (1-day) and ±2% (3-day)
- Mixed signals or sideways movement
- **Trading Strategy:** Mean reversion, range trading, wait for clarity
- **Risk Level:** Low (consolidation phase)

### Confidence Scores

- **> 80%**: High confidence - Strong signal
- **60-80%**: Medium confidence - Moderate signal
- **< 60%**: Low confidence - Weak signal, be cautious

---

## Common Issues

### Issue: Model file not found
```
FileNotFoundError: Model file not found: models/best_crypto_model.h5
```

**Solution:**
Ensure your trained model is in the `models/` directory. The model file should be named `best_crypto_model.h5`.

---

### Issue: TensorFlow not installed
```
ImportError: TensorFlow is required for model prediction
```

**Solution:**
```bash
pip install tensorflow==2.15.0
```

---

### Issue: No data for symbol
```
ValueError: No data found for symbol: XRP
```

**Solution:**
Check available symbols in your data file:
```python
from src.data.loaders.data_loader import CryptoDataLoader
loader = CryptoDataLoader("your_data.csv")
loader.load_data()
print(loader.get_available_symbols())
```

---

### Issue: Insufficient data
```
ValueError: Need at least 10 days of data, but only 5 rows provided
```

**Solution:**
Ensure your data has at least 30 rows for the symbol (to account for technical indicators + sequence length).

---

## Next Steps

1. **Integrate real-time data:**
   - Implement CoinGecko or Binance API collectors
   - See `src/data/collectors/` (to be implemented)

2. **Add more features:**
   - Customize technical indicators
   - Add sentiment data
   - Include macroeconomic factors

3. **Retrain model:**
   - Use updated data (2015-2024)
   - Experiment with hyperparameters
   - See training scripts (to be added)

4. **Deploy to production:**
   - Set up Kubernetes cluster
   - Configure monitoring (Prometheus + Grafana)
   - Add authentication and rate limiting

---

## Support

For issues or questions:
- Check [QUICK_START_GUIDE.md](./QUICK_START_GUIDE.md) for detailed setup
- Review [ARCHITECTURE.md](./ARCHITECTURE.md) for technical details
- See [PROJECT_SCOPE_AND_ROADMAP.md](./PROJECT_SCOPE_AND_ROADMAP.md) for future plans

---

**Happy regime classifying! 🚀**
