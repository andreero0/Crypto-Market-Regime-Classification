"""Tests for data loader module."""
import pytest
import pandas as pd
from pathlib import Path
from src.data.loaders.data_loader import CryptoDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    data = {
        'date': pd.date_range('2024-01-01', periods=50, freq='D'),
        'symbol': ['BTC'] * 25 + ['ETH'] * 25,
        'open': range(50),
        'high': range(50),
        'low': range(50),
        'close': range(50),
        'volume': range(50),
        'market': range(50),
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


class TestCryptoDataLoader:
    """Test suite for CryptoDataLoader class."""

    def test_init_with_valid_file(self, sample_csv):
        """Test initialization with valid file."""
        loader = CryptoDataLoader(str(sample_csv))
        assert loader.data_path.exists()

    def test_init_with_invalid_file(self):
        """Test initialization with invalid file raises error."""
        with pytest.raises(FileNotFoundError):
            CryptoDataLoader("nonexistent_file.csv")

    def test_load_data(self, sample_csv):
        """Test loading data."""
        loader = CryptoDataLoader(str(sample_csv))
        df = loader.load_data()

        assert len(df) == 50
        assert 'date' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['date'])

    def test_load_data_with_symbol_filter(self, sample_csv):
        """Test loading data with symbol filter."""
        loader = CryptoDataLoader(str(sample_csv))
        df = loader.load_data(symbols=['BTC'])

        assert len(df) == 25
        assert all(df['symbol'] == 'BTC')

    def test_get_symbol_data(self, sample_csv):
        """Test getting data for specific symbol."""
        loader = CryptoDataLoader(str(sample_csv))
        loader.load_data()

        btc_data = loader.get_symbol_data('BTC')
        assert len(btc_data) == 25
        assert all(btc_data['symbol'] == 'BTC')

    def test_get_symbol_data_invalid_symbol(self, sample_csv):
        """Test getting data for invalid symbol raises error."""
        loader = CryptoDataLoader(str(sample_csv))
        loader.load_data()

        with pytest.raises(ValueError):
            loader.get_symbol_data('XRP')

    def test_get_latest_data(self, sample_csv):
        """Test getting latest N days of data."""
        loader = CryptoDataLoader(str(sample_csv))
        loader.load_data()

        latest = loader.get_latest_data('BTC', days=10)
        assert len(latest) == 10

    def test_get_available_symbols(self, sample_csv):
        """Test getting available symbols."""
        loader = CryptoDataLoader(str(sample_csv))
        loader.load_data()

        symbols = loader.get_available_symbols()
        assert set(symbols) == {'BTC', 'ETH'}

    def test_validate_data(self, sample_csv):
        """Test data validation."""
        loader = CryptoDataLoader(str(sample_csv))
        df = loader.load_data()

        assert loader.validate_data(df) is True

    def test_validate_data_missing_columns(self):
        """Test validation with missing columns."""
        loader = CryptoDataLoader.__new__(CryptoDataLoader)
        df = pd.DataFrame({'date': [1, 2, 3]})

        assert loader.validate_data(df) is False
