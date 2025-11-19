"""Lag features generation for time series data."""
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


class LagFeatureBuilder:
    """Create lag features for time series prediction."""

    @staticmethod
    def create_lag_features(
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int],
        group_by: str = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.

        Args:
            df: DataFrame with time series data
            columns: List of columns to create lags for
            lags: List of lag periods (e.g., [1, 3, 5, 7])
            group_by: Optional column to group by (e.g., 'symbol' for multiple assets)

        Returns:
            DataFrame with added lag feature columns
        """
        logger.info(f"Creating lag features for columns {columns} with lags {lags}")

        df = df.copy()

        for col in columns:
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"

                if group_by:
                    # Create lags per group to avoid data leakage between different assets
                    df[lag_col_name] = df.groupby(group_by)[col].shift(lag)
                else:
                    df[lag_col_name] = df[col].shift(lag)

                logger.debug(f"Created {lag_col_name}")

        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return df

    def create_lags(
        self,
        df: pd.DataFrame,
        group_by: str = "symbol"
    ) -> pd.DataFrame:
        """
        Create standard lag features for crypto regime classification.

        Creates lags for:
        - close price: 1, 3, 5, 7 days
        - volume: 1, 3, 5, 7 days

        Args:
            df: DataFrame with OHLCV data
            group_by: Column to group by (default: 'symbol')

        Returns:
            DataFrame with lag features added
        """
        lag_columns = ['close', 'volume']
        lag_periods = [1, 3, 5, 7]

        return self.create_lag_features(
            df=df,
            columns=lag_columns,
            lags=lag_periods,
            group_by=group_by if group_by in df.columns else None
        )
