"""Data validation utilities"""

import numpy as np
import pandas as pd


def validate_data_pipeline(df, stage_name):
    """
    Validate data at each pipeline stage.

    Args:
        df (pd.DataFrame): DataFrame to validate
        stage_name (str): Name of the pipeline stage

    Returns:
        pd.DataFrame: Same dataframe (for chaining)
    """
    print(f"\n{'='*60}")
    print(f"Validation: {stage_name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")

    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Check for null values
    null_count = df.isnull().sum().sum()
    print(f"Null values: {null_count}")
    if null_count > 0:
        print("Columns with nulls:")
        null_cols = df.isnull().sum()
        print(null_cols[null_cols > 0])

    # Check for duplicates
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")

    # Check for inf values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    total_inf = inf_counts.sum()

    if total_inf > 0:
        print(f"⚠️  WARNING: {total_inf} Inf values found!")
        print("Columns with Inf:")
        print(inf_counts[inf_counts > 0])
    else:
        print("✓ No Inf values detected")

    print(f"{'='*60}\n")

    return df


def validate_features(df, feature_ranges=None):
    """
    Validate feature value ranges.

    Args:
        df (pd.DataFrame): DataFrame with features
        feature_ranges (dict): Expected ranges for features
                              e.g., {'RSI_14': (0, 100)}
    """
    if feature_ranges is None:
        feature_ranges = {
            'RSI_14': (0, 100),  # RSI should be 0-100
        }

    print("\n=== Feature Range Validation ===")

    issues_found = False
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df.columns:
            actual_min = df[feature].min()
            actual_max = df[feature].max()

            if actual_min < min_val or actual_max > max_val:
                print(f"⚠️  {feature}: Expected [{min_val}, {max_val}], "
                      f"Got [{actual_min:.2f}, {actual_max:.2f}]")
                issues_found = True
            else:
                print(f"✓ {feature}: [{actual_min:.2f}, {actual_max:.2f}]")

    if not issues_found:
        print("✓ All features within expected ranges")

    print()


def check_no_nan_inf(df, stage_name=""):
    """
    Assert no NaN or Inf values exist.

    Args:
        df (pd.DataFrame): DataFrame to check
        stage_name (str): Stage name for error message

    Raises:
        AssertionError: If NaN or Inf values found
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Check NaN
    nan_count = df[numeric_cols].isnull().sum().sum()
    assert nan_count == 0, f"NaN values detected at {stage_name}: {nan_count} total"

    # Check Inf
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    assert inf_count == 0, f"Inf values detected at {stage_name}: {inf_count} total"

    print(f"✓ No NaN/Inf values at {stage_name}")
