"""Tests for technical indicators module."""
import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators


@pytest.fixture
def sample_data():
    """Create sample cryptocurrency data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    data = {
        "date": dates,
        "symbol": ["BTC"] * 100,
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(40000, 50000, 100),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(1e9, 1e10, 100),
        "market": np.random.uniform(1e12, 1e13, 100),
    }

    df = pd.DataFrame(data)
    # Ensure high/low are correct
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 1000, 100)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 1000, 100)

    # Add close_ratio and spread (features expected by model)
    df['close_ratio'] = df['close'] / df['open']
    df['spread'] = df['high'] - df['low']

    return df


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators class."""

    def test_calculate_sma(self, sample_data):
        """Test SMA calculation."""
        indicators = TechnicalIndicators()
        sma = indicators.calculate_sma(sample_data, window=7)

        assert len(sma) == len(sample_data)
        assert not sma.iloc[-1] == np.nan
        # First 6 values should be NaN (window=7)
        assert pd.isna(sma.iloc[0:6]).all()

    def test_calculate_ema(self, sample_data):
        """Test EMA calculation."""
        indicators = TechnicalIndicators()
        ema = indicators.calculate_ema(sample_data, span=12)

        assert len(ema) == len(sample_data)
        assert not pd.isna(ema.iloc[-1])

    def test_calculate_rsi(self, sample_data):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()
        rsi = indicators.calculate_rsi(sample_data, period=14)

        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100 (excluding NaN)
        valid_rsi = rsi.dropna()
        assert all((valid_rsi >= 0) & (valid_rsi <= 100))

    def test_calculate_macd(self, sample_data):
        """Test MACD calculation."""
        indicators = TechnicalIndicators()
        macd, signal, histogram = indicators.calculate_macd(sample_data)

        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)

    def test_calculate_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()
        middle, upper, lower = indicators.calculate_bollinger_bands(sample_data)

        assert len(middle) == len(sample_data)
        assert len(upper) == len(sample_data)
        assert len(lower) == len(sample_data)

        # Upper band should be above middle, middle above lower (where not NaN)
        valid_idx = ~middle.isna()
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])

    def test_calculate_all(self, sample_data):
        """Test calculation of all indicators."""
        indicators = TechnicalIndicators()
        result = indicators.calculate_all(sample_data)

        # Check all expected columns are present
        expected_cols = [
            'SMA_7', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'Upper_Band', 'Lower_Band', 'STD_20'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check no NaN in final output (should be forward filled)
        assert result.isnull().sum().sum() == 0
