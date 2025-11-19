"""Technical indicators calculation for cryptocurrency data."""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for cryptocurrency market data."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = "close", window: int = 7) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with price data
            column: Column to calculate SMA on
            window: Window size for moving average

        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=window).mean()

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = "close", span: int = 12) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price data
            column: Column to calculate EMA on
            span: Span for exponential weighting

        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures momentum on a scale of 0-100.
        > 70 = overbought, < 30 = oversold

        Args:
            df: DataFrame with price data
            column: Column to calculate RSI on
            period: Lookback period

        Returns:
            Series with RSI values
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        column: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD = Fast EMA - Slow EMA
        Signal = EMA of MACD
        Histogram = MACD - Signal

        Args:
            df: DataFrame with price data
            column: Column to calculate MACD on
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        column: str = "close",
        window: int = 20,
        num_std: int = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Middle Band = SMA
        Upper Band = SMA + (std * num_std)
        Lower Band = SMA - (std * num_std)

        Args:
            df: DataFrame with price data
            column: Column to calculate bands on
            window: Window size for SMA
            num_std: Number of standard deviations

        Returns:
            Tuple of (Middle band, Upper band, Lower band)
        """
        sma = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return sma, upper_band, lower_band

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Adds the following columns to the DataFrame:
        - SMA_7, SMA_20: Simple moving averages
        - EMA_12, EMA_26: Exponential moving averages
        - RSI_14: Relative Strength Index
        - MACD: MACD line
        - Upper_Band, Lower_Band: Bollinger Bands
        - STD_20: 20-day standard deviation

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicator columns
        """
        logger.info(f"Calculating technical indicators for {len(df)} rows")

        df = df.copy()

        # Moving averages
        df["SMA_7"] = self.calculate_sma(df, window=7)
        df["SMA_20"] = self.calculate_sma(df, window=20)
        df["EMA_12"] = self.calculate_ema(df, span=12)
        df["EMA_26"] = self.calculate_ema(df, span=26)

        # Momentum indicators
        df["RSI_14"] = self.calculate_rsi(df, period=14)

        # MACD
        macd, signal_line, histogram = self.calculate_macd(df)
        df["MACD"] = macd
        # Optionally add signal and histogram
        # df["MACD_signal"] = signal_line
        # df["MACD_histogram"] = histogram

        # Volatility indicators
        sma_20, upper_band, lower_band = self.calculate_bollinger_bands(df)
        df["Upper_Band"] = upper_band
        df["Lower_Band"] = lower_band
        df["STD_20"] = df["close"].rolling(window=20).std()

        # Fill any remaining NaN values with forward fill
        df = df.fillna(method='ffill')

        logger.info("Technical indicators calculated successfully")
        return df
