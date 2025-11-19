"""Data loading utilities for cryptocurrency data."""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class CryptoDataLoader:
    """Load and prepare cryptocurrency data for regime classification."""

    def __init__(self, data_path: str):
        """
        Initialize data loader.

        Args:
            data_path: Path to CSV file with crypto data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = None

    def load_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load cryptocurrency data from CSV.

        Args:
            symbols: List of symbols to load (None = all)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {self.data_path}")

        # Load CSV
        df = pd.read_csv(self.data_path)

        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        # Filter by symbols
        if symbols and 'symbol' in df.columns:
            df = df[df['symbol'].isin(symbols)]
            logger.info(f"Filtered to symbols: {symbols}")

        # Filter by date range
        if start_date and 'date' in df.columns:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            logger.info(f"Filtered from: {start_date}")

        if end_date and 'date' in df.columns:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            logger.info(f"Filtered to: {end_date}")

        logger.info(f"Loaded {len(df)} rows")

        self.df = df
        return df

    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get data for a specific symbol.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            DataFrame for the symbol
        """
        if self.df is None:
            self.load_data()

        if 'symbol' in self.df.columns:
            symbol_data = self.df[self.df['symbol'] == symbol.upper()].copy()
        else:
            symbol_data = self.df.copy()

        if len(symbol_data) == 0:
            available = self.df['symbol'].unique().tolist() if 'symbol' in self.df.columns else []
            raise ValueError(
                f"No data found for symbol: {symbol}. "
                f"Available: {available}"
            )

        return symbol_data.sort_values('date') if 'date' in symbol_data.columns else symbol_data

    def get_latest_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get latest N days of data for a symbol.

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to retrieve

        Returns:
            DataFrame with latest N days
        """
        symbol_data = self.get_symbol_data(symbol)
        return symbol_data.tail(days)

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available cryptocurrency symbols.

        Returns:
            List of symbols
        """
        if self.df is None:
            self.load_data()

        if 'symbol' in self.df.columns:
            return sorted(self.df['symbol'].unique().tolist())
        else:
            return []

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data has required columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        # Check for missing values in required columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")

        return True
