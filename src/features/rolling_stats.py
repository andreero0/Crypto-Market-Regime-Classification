"""Rolling statistics calculation for time series data."""
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


class RollingStatistics:
    """Calculate rolling window statistics for time series features."""

    @staticmethod
    def calculate_rolling_stats(
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        metrics: List[str] = ['mean', 'std'],
        group_by: str = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for specified columns.

        Args:
            df: DataFrame with time series data
            columns: List of columns to calculate stats for
            windows: List of window sizes (e.g., [7, 14, 21])
            metrics: List of metrics to calculate ('mean', 'std', 'min', 'max')
            group_by: Optional column to group by

        Returns:
            DataFrame with added rolling statistic columns
        """
        logger.info(f"Calculating rolling stats for columns {columns} with windows {windows}")

        df = df.copy()

        for col in columns:
            for window in windows:
                for metric in metrics:
                    stat_col_name = f"{col}_rolling_{metric}_{window}"

                    if group_by and group_by in df.columns:
                        # Calculate per group
                        grouped = df.groupby(group_by)[col]
                        if metric == 'mean':
                            df[stat_col_name] = grouped.transform(lambda x: x.rolling(window).mean())
                        elif metric == 'std':
                            df[stat_col_name] = grouped.transform(lambda x: x.rolling(window).std())
                        elif metric == 'min':
                            df[stat_col_name] = grouped.transform(lambda x: x.rolling(window).min())
                        elif metric == 'max':
                            df[stat_col_name] = grouped.transform(lambda x: x.rolling(window).max())
                    else:
                        # Calculate globally
                        if metric == 'mean':
                            df[stat_col_name] = df[col].rolling(window).mean()
                        elif metric == 'std':
                            df[stat_col_name] = df[col].rolling(window).std()
                        elif metric == 'min':
                            df[stat_col_name] = df[col].rolling(window).min()
                        elif metric == 'max':
                            df[stat_col_name] = df[col].rolling(window).max()

                    logger.debug(f"Created {stat_col_name}")

        total_features = len(columns) * len(windows) * len(metrics)
        logger.info(f"Created {total_features} rolling statistic features")
        return df

    def calculate_rolling(
        self,
        df: pd.DataFrame,
        group_by: str = "symbol"
    ) -> pd.DataFrame:
        """
        Calculate standard rolling statistics for crypto regime classification.

        Creates rolling mean and std for:
        - close price: 7, 14, 21 day windows
        - volume: 7, 14, 21 day windows

        Args:
            df: DataFrame with OHLCV data
            group_by: Column to group by (default: 'symbol')

        Returns:
            DataFrame with rolling statistics added
        """
        stat_columns = ['close', 'volume']
        windows = [7, 14, 21]
        metrics = ['mean', 'std']

        return self.calculate_rolling_stats(
            df=df,
            columns=stat_columns,
            windows=windows,
            metrics=metrics,
            group_by=group_by if group_by in df.columns else None
        )
