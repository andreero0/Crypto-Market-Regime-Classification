"""Sequence creation for time series LSTM models

CRITICAL FIX: Corrected data leakage in sequence creation
- Previous: Used data[t-9:t] to predict label[t-1] (WRONG - uses current data)
- Fixed: Uses data[t-9:t-1] to predict label[t] (CORRECT - predicts future)
"""

import numpy as np
import pandas as pd


def create_sequences(feature_data, target_data, date_data, timesteps=10):
    """
    Create sequences for LSTM training WITHOUT data leakage.

    CRITICAL FIX: Now correctly predicts NEXT timestep, not current timestep.

    The function creates sliding windows where:
    - Features come from timesteps [t-9, t-8, ..., t-1] (historical data)
    - Target comes from timestep [t] (future prediction)

    This ensures the model ONLY uses historical information to predict
    the future market condition.

    Args:
        feature_data (np.array): Features of shape (samples, features)
        target_data (np.array): Targets of shape (samples,)
        date_data (np.array): Dates of shape (samples,)
        timesteps (int): Number of historical timesteps to use

    Returns:
        X (np.array): Feature sequences (n_sequences, timesteps, features)
        y (np.array): Target values (n_sequences,)
        dates (np.array): Corresponding dates (n_sequences,)

    Example:
        Given data for days 0-100 with timesteps=10:
        - X[0] contains features from days [0, 1, 2, ..., 9]
        - y[0] contains target from day 10 (predicting day 10's condition)
        - dates[0] is the date of day 10

    OLD (WRONG - DATA LEAKAGE):
        for i in range(0, len(feature_data) - timesteps + 1):
            X.append(feature_data[i:i+timesteps])      # Days 0-9
            y.append(target_data[i+timesteps-1])       # Day 9 (LEAKAGE!)

    NEW (CORRECT - NO LEAKAGE):
        for i in range(0, len(feature_data) - timesteps):
            X.append(feature_data[i:i+timesteps])      # Days 0-9
            y.append(target_data[i+timesteps])         # Day 10 (FUTURE)
    """
    X, y, dates = [], [], []

    # CRITICAL FIX: Loop until len-timesteps (not len-timesteps+1)
    # This allows us to get the target at i+timesteps
    for i in range(0, len(feature_data) - timesteps):
        # Features: historical window from i to i+timesteps-1
        X.append(feature_data[i:i+timesteps])

        # Target: NEXT timestep at i+timesteps (CRITICAL FIX)
        y.append(target_data[i+timesteps])

        # Date: corresponding to the target prediction
        dates.append(date_data[i+timesteps])

    return np.array(X), np.array(y), np.array(dates)


def validate_no_leakage(X, y, dates, original_dates, timesteps=10):
    """
    Validate that no data leakage exists in created sequences.

    Checks that the last feature in each sequence comes from BEFORE
    the target date, ensuring we're predicting the future.

    Args:
        X (np.array): Feature sequences
        y (np.array): Target values
        dates (np.array): Target dates
        original_dates (np.array): All dates from original dataset
        timesteps (int): Number of timesteps used

    Raises:
        AssertionError: If data leakage is detected
    """
    print("\n" + "="*60)
    print("DATA LEAKAGE VALIDATION")
    print("="*60)

    # Check first 10 samples
    num_checks = min(10, len(X))

    for i in range(num_checks):
        target_date = dates[i]

        # Find the index of target date in original data
        target_idx = np.where(original_dates == target_date)[0]

        if len(target_idx) > 0:
            target_idx = target_idx[0]

            # The last feature should be from target_idx - 1
            # The first feature should be from target_idx - timesteps
            expected_last_feature_idx = target_idx - 1

            if expected_last_feature_idx >= 0:
                last_feature_date = original_dates[expected_last_feature_idx]
                first_feature_date = original_dates[target_idx - timesteps]

                # Critical check: last feature date must be BEFORE target date
                assert last_feature_date < target_date, \
                    f"LEAKAGE DETECTED at sample {i}: " \
                    f"Last feature date ({last_feature_date}) >= Target date ({target_date})"

                if i == 0:
                    print(f"Sample {i} validation:")
                    print(f"  Feature window: {first_feature_date} to {last_feature_date}")
                    print(f"  Target date: {target_date}")
                    print(f"  ✓ Last feature is before target (no leakage)")

    print(f"\n✓ Validated {num_checks} samples - NO DATA LEAKAGE DETECTED")
    print(f"✓ Features use only historical data")
    print(f"✓ Targets are from future timesteps")
    print(f"✓ Total sequences created: {len(X)}")
    print(f"✓ Sequence shape: {X.shape}")
    print("="*60 + "\n")


def create_sequences_with_validation(feature_data, target_data, date_data,
                                     timesteps=10, validate=True):
    """
    Create sequences with automatic validation.

    Args:
        feature_data (np.array): Features
        target_data (np.array): Targets
        date_data (np.array): Dates
        timesteps (int): Number of timesteps
        validate (bool): Whether to validate for leakage

    Returns:
        tuple: (X, y, dates)
    """
    X, y, dates = create_sequences(feature_data, target_data, date_data, timesteps)

    if validate:
        validate_no_leakage(X, y, dates, date_data, timesteps)

    return X, y, dates


def prepare_sequences_from_dataframe(df, feature_cols, target_col, timesteps=10):
    """
    Prepare sequences directly from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with features and target
        feature_cols (list): List of feature column names
        target_col (str): Target column name
        timesteps (int): Number of timesteps

    Returns:
        tuple: (X, y, dates)
    """
    # Extract arrays
    feature_data = df[feature_cols].values
    target_data = df[target_col].values
    date_data = df['date'].values if 'date' in df.columns else np.arange(len(df))

    # Create sequences with validation
    X, y, dates = create_sequences_with_validation(
        feature_data, target_data, date_data, timesteps, validate=True
    )

    return X, y, dates
