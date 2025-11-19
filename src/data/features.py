"""Feature engineering for cryptocurrency data

CRITICAL FIX: RSI calculation now includes epsilon to prevent division by zero
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for crypto data"""

    @staticmethod
    def calculate_sma(df, column='close', window=7):
        """
        Calculate Simple Moving Average.

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            column (str): Column to calculate SMA on
            window (int): Rolling window size

        Returns:
            pd.Series: SMA values
        """
        return df.groupby('symbol')[column].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    @staticmethod
    def calculate_rsi(df, column='close', window=14, epsilon=1e-10):
        """
        Calculate Relative Strength Index with division by zero protection.

        CRITICAL FIX: Added epsilon to prevent division by zero when avg_loss = 0

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            column (str): Column to calculate RSI on
            window (int): RSI period
            epsilon (float): Small value to prevent division by zero

        Returns:
            pd.Series: RSI values (0-100)
        """
        def rsi_for_group(group):
            delta = group.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            # CRITICAL FIX: Add epsilon to prevent division by zero
            rs = gain / (loss + epsilon)
            rsi = 100 - (100 / (1 + rs))

            return rsi

        return df.groupby('symbol')[column].transform(rsi_for_group)

    @staticmethod
    def calculate_ema(df, column='close', span=12):
        """
        Calculate Exponential Moving Average.

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            column (str): Column to calculate EMA on
            span (int): Span for EMA calculation

        Returns:
            pd.Series: EMA values
        """
        return df.groupby('symbol')[column].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )

    @staticmethod
    def calculate_macd(df, column='close', fast=12, slow=26):
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            column (str): Column to calculate MACD on
            fast (int): Fast EMA period
            slow (int): Slow EMA period

        Returns:
            pd.Series: MACD values
        """
        ema_fast = TechnicalIndicators.calculate_ema(df, column, fast)
        ema_slow = TechnicalIndicators.calculate_ema(df, column, slow)
        return ema_fast - ema_slow

    @staticmethod
    def calculate_bollinger_bands(df, column='close', window=20, num_std=2):
        """
        Calculate Bollinger Bands.

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            column (str): Column to calculate bands on
            window (int): Rolling window for SMA and STD
            num_std (int): Number of standard deviations

        Returns:
            tuple: (upper_band, lower_band)
        """
        sma = df.groupby('symbol')[column].transform(
            lambda x: x.rolling(window=window).mean()
        )
        std = df.groupby('symbol')[column].transform(
            lambda x: x.rolling(window=window).std()
        )
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    @classmethod
    def add_all_indicators(cls, df, config):
        """
        Add all technical indicators based on configuration.

        Args:
            df (pd.DataFrame): DataFrame with cryptocurrency data
            config (dict): Configuration dictionary with indicator parameters

        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        tech_config = config['features']['technical_indicators']

        print("Adding technical indicators...")

        # SMA
        df['SMA_7'] = cls.calculate_sma(df, window=tech_config['sma_window'])
        print("  ✓ SMA added")

        # RSI (with fix)
        df['RSI_14'] = cls.calculate_rsi(df, window=tech_config['rsi_window'])
        print("  ✓ RSI added (with division-by-zero fix)")

        # EMA
        df['EMA_12'] = cls.calculate_ema(df, span=tech_config['ema_windows'][0])
        df['EMA_26'] = cls.calculate_ema(df, span=tech_config['ema_windows'][1])
        print("  ✓ EMA added")

        # MACD
        df['MACD'] = cls.calculate_macd(df)
        print("  ✓ MACD added")

        # Bollinger Bands
        df['BB_upper'], df['BB_lower'] = cls.calculate_bollinger_bands(
            df, window=tech_config['bollinger_window']
        )
        print("  ✓ Bollinger Bands added")

        # Validate no inf/nan values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_count = df[numeric_cols].isnull().sum().sum()
        inf_count = np.isinf(df[numeric_cols]).sum().sum()

        if nan_count > 0:
            print(f"  ⚠️  WARNING: {nan_count} NaN values after feature engineering")
            print("  Columns with NaN:", df[numeric_cols].isnull().sum()[df[numeric_cols].isnull().sum() > 0])

        if inf_count > 0:
            print(f"  ⚠️  WARNING: {inf_count} Inf values after feature engineering")

        if nan_count == 0 and inf_count == 0:
            print("  ✓ No NaN/Inf values detected")

        return df


def add_lag_features(df, config):
    """
    Add lag features for specified columns.

    Args:
        df (pd.DataFrame): DataFrame with cryptocurrency data
        config (dict): Configuration with lag feature parameters

    Returns:
        pd.DataFrame: DataFrame with lag features added
    """
    lag_config = config['features']['lag_features']
    columns = lag_config['columns']
    lags = lag_config['lags']

    print(f"\nAdding lag features for {columns}...")

    for col in columns:
        if col in df.columns:
            for lag in lags:
                lag_col_name = f'{col}_lag_{lag}'
                df[lag_col_name] = df.groupby('symbol')[col].shift(lag)
                print(f"  ✓ {lag_col_name} added")

    return df


def add_rolling_features(df, config):
    """
    Add rolling statistical features.

    Args:
        df (pd.DataFrame): DataFrame with cryptocurrency data
        config (dict): Configuration with rolling feature parameters

    Returns:
        pd.DataFrame: DataFrame with rolling features added
    """
    rolling_config = config['features']['rolling_features']
    windows = rolling_config['windows']
    statistics = rolling_config['statistics']

    print(f"\nAdding rolling features (windows: {windows})...")

    for window in windows:
        for stat in statistics:
            if stat == 'mean':
                df[f'rolling_mean_{window}'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window=window).mean()
                )
            elif stat == 'std':
                df[f'rolling_std_{window}'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window=window).std()
                )
            print(f"  ✓ rolling_{stat}_{window} added")

    return df
