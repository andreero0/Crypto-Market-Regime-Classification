"""
Corrected training script for Crypto Market Regime Classification

This script addresses all critical bugs identified in the original notebook:
1. Fixed data leakage in sequence creation
2. Fixed RSI division by zero
3. Fixed double scaling issue
4. Added comprehensive validation
5. Added baseline comparisons
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Import custom modules
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_random_seeds
from src.data.validation import validate_data_pipeline, check_no_nan_inf
from src.data.features import TechnicalIndicators, add_lag_features, add_rolling_features
from src.models.sequences import create_sequences_with_validation
from src.evaluation.baselines import BaselineModels
from src.evaluation.metrics import ModelEvaluator

import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(config):
    """
    Load and prepare cryptocurrency data.

    Args:
        config (dict): Configuration dictionary

    Returns:
        pd.DataFrame: Prepared dataframe
    """
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70 + "\n")

    # Load data (assuming it's already filtered)
    df = pd.read_csv('filtered_crypto_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate
    df = validate_data_pipeline(df, "After loading")

    return df


def engineer_features(df, config):
    """
    Engineer features with bug fixes.

    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration

    Returns:
        pd.DataFrame: DataFrame with features
    """
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING (WITH BUG FIXES)")
    print("="*70 + "\n")

    # Add technical indicators (includes RSI fix)
    df = TechnicalIndicators.add_all_indicators(df, config)

    # Add lag features
    df = add_lag_features(df, config)

    # Add rolling features
    df = add_rolling_features(df, config)

    # Drop NaN rows created by rolling windows
    df = df.dropna()

    # Validate no inf/nan
    check_no_nan_inf(df, "After feature engineering")

    print(f"\nFinal shape after feature engineering: {df.shape}")

    return df


def create_market_condition_labels(df, config):
    """
    Create market condition labels.

    Args:
        df (pd.DataFrame): DataFrame with price data
        config (dict): Configuration

    Returns:
        pd.DataFrame: DataFrame with market_condition column
    """
    print("\n" + "="*70)
    print("STEP 3: CREATE MARKET CONDITION LABELS")
    print("="*70 + "\n")

    mc_config = config['market_conditions']

    # Calculate percentage changes
    df['pct_change_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['pct_change_3d'] = df.groupby('symbol')['close'].pct_change(3)

    # Define conditions
    bullish_mask = (
        (df['pct_change_1d'] > 0) &
        (df['pct_change_3d'] > 0) &
        ((df['pct_change_1d'] > mc_config['bullish_threshold']) |
         (df['pct_change_3d'] > mc_config['bullish_threshold']))
    )

    bearish_mask = (
        (df['pct_change_1d'] < 0) &
        (df['pct_change_3d'] < 0) &
        ((df['pct_change_1d'] < mc_config['bearish_threshold']) |
         (df['pct_change_3d'] < mc_config['bearish_threshold']))
    )

    neutral_mask = (
        (np.abs(df['pct_change_1d']) < mc_config['neutral_threshold']) &
        (np.abs(df['pct_change_3d']) < 0.02)
    )

    # Assign labels
    df['market_condition'] = 'neutral'  # Default
    df.loc[bullish_mask, 'market_condition'] = 'bullish'
    df.loc[bearish_mask, 'market_condition'] = 'bearish'

    # Drop NaN from pct_change
    df = df.dropna()

    # Print distribution
    print("\nMarket condition distribution:")
    print(df['market_condition'].value_counts())
    print(f"\nPercentages:")
    print(df['market_condition'].value_counts(normalize=True) * 100)

    return df


def prepare_train_val_test_splits(df, config):
    """
    Prepare train, validation, and test splits.

    Args:
        df (pd.DataFrame): DataFrame with all features
        config (dict): Configuration

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*70)
    print("STEP 4: TRAIN/VAL/TEST SPLIT")
    print("="*70 + "\n")

    # Sort by date for time series
    df = df.sort_values(['symbol', 'date'])

    # Get feature columns (exclude metadata and target)
    exclude_cols = ['symbol', 'date', 'market_condition', 'pct_change_1d', 'pct_change_3d']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} features")

    # Encode target
    le = LabelEncoder()
    df['market_condition_encoded'] = le.fit_transform(df['market_condition'])

    # Split by time (70% train, 15% val, 15% test)
    train_size = 0.70
    val_size = 0.15

    total_samples = len(df)
    train_end = int(total_samples * train_size)
    val_end = int(total_samples * (train_size + val_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"Val:   {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"Test:  {len(test_df)} samples ({len(test_df)/total_samples*100:.1f}%)")

    return train_df, val_df, test_df, feature_cols, le


def scale_and_create_sequences(train_df, val_df, test_df, feature_cols, config):
    """
    Scale features and create sequences (SINGLE SCALING - no double scaling!).

    Args:
        train_df, val_df, test_df: DataFrames for each split
        feature_cols: List of feature columns
        config: Configuration

    Returns:
        tuple: Scaled sequences and labels
    """
    print("\n" + "="*70)
    print("STEP 5: SCALING AND SEQUENCE CREATION (FIXED)")
    print("="*70 + "\n")

    timesteps = config['model']['timesteps']

    # Extract features and targets
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df['market_condition_encoded'].values
    y_val = val_df['market_condition_encoded'].values
    y_test = test_df['market_condition_encoded'].values

    dates_train = train_df['date'].values
    dates_val = val_df['date'].values
    dates_test = test_df['date'].values

    # CRITICAL FIX: Single scaling step
    print("Fitting scaler on training data...")
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    print("Transforming all datasets...")
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("✓ Scaling complete (single pass, no double scaling)")

    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print("✓ Scaler saved to models/feature_scaler.pkl")

    # Create sequences with data leakage fix
    print(f"\nCreating sequences with timesteps={timesteps}...")
    print("CRITICAL: Using FIXED sequence creation (no data leakage)")

    X_train_seq, y_train_seq, dates_train_seq = create_sequences_with_validation(
        X_train_scaled, y_train, dates_train, timesteps=timesteps, validate=True
    )

    X_val_seq, y_val_seq, dates_val_seq = create_sequences_with_validation(
        X_val_scaled, y_val, dates_val, timesteps=timesteps, validate=False
    )

    X_test_seq, y_test_seq, dates_test_seq = create_sequences_with_validation(
        X_test_scaled, y_test, dates_test, timesteps=timesteps, validate=False
    )

    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train_seq.shape}")
    print(f"  X_val:   {X_val_seq.shape}")
    print(f"  X_test:  {X_test_seq.shape}")

    # One-hot encode targets
    y_train_cat = to_categorical(y_train_seq, num_classes=3)
    y_val_cat = to_categorical(y_val_seq, num_classes=3)
    y_test_cat = to_categorical(y_test_seq, num_classes=3)

    return (X_train_seq, y_train_cat, y_train_seq,
            X_val_seq, y_val_cat, y_val_seq,
            X_test_seq, y_test_cat, y_test_seq)


def build_model(config, input_shape):
    """
    Build LSTM model.

    Args:
        config (dict): Configuration
        input_shape (tuple): Input shape (timesteps, features)

    Returns:
        keras.Model: Compiled model
    """
    print("\n" + "="*70)
    print("STEP 6: MODEL BUILDING")
    print("="*70 + "\n")

    model_config = config['model']

    input_layer = Input(shape=input_shape)

    # First LSTM layer
    x = Bidirectional(LSTM(
        units=model_config['lstm_layers'][0]['units'],
        return_sequences=model_config['lstm_layers'][0]['return_sequences']
    ))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(model_config['lstm_layers'][0]['dropout'])(x)

    # Second LSTM layer
    x = Bidirectional(LSTM(
        units=model_config['lstm_layers'][1]['units'],
        return_sequences=model_config['lstm_layers'][1]['return_sequences']
    ))(x)
    x = BatchNormalization()(x)
    x = Dropout(model_config['lstm_layers'][1]['dropout'])(x)

    # Dense layer
    x = Dense(
        model_config['dense_layers'][0]['units'],
        activation=model_config['dense_layers'][0]['activation']
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(model_config['dense_layers'][0]['dropout'])(x)

    # Output layer
    output_layer = Dense(3, activation=model_config['output_activation'])(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=config['training']['optimizer']['learning_rate']
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    return model


def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train model with callbacks.

    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Configuration

    Returns:
        History: Training history
    """
    print("\n" + "="*70)
    print("STEP 7: MODEL TRAINING")
    print("="*70 + "\n")

    # Callbacks (FIXED: consistent monitoring)
    callbacks_config = config['training']['callbacks']

    early_stopping = EarlyStopping(
        monitor=callbacks_config['early_stopping']['monitor'],
        patience=callbacks_config['early_stopping']['patience'],
        restore_best_weights=callbacks_config['early_stopping']['restore_best_weights'],
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=callbacks_config['reduce_lr']['monitor'],
        factor=callbacks_config['reduce_lr']['factor'],
        patience=callbacks_config['reduce_lr']['patience'],
        min_lr=callbacks_config['reduce_lr']['min_lr'],
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        callbacks_config['model_checkpoint']['filepath'],
        monitor=callbacks_config['model_checkpoint']['monitor'],
        save_best_only=callbacks_config['model_checkpoint']['save_best_only'],
        verbose=1
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    print("\n✓ Training complete")

    return history


def evaluate_with_baselines(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model and compare with baselines.

    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data (integer labels, not one-hot)
    """
    print("\n" + "="*70)
    print("STEP 8: EVALUATION WITH BASELINES")
    print("="*70 + "\n")

    # Evaluate LSTM
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    lstm_accuracy = accuracy_score(y_test, y_pred)

    print(f"LSTM Test Accuracy: {lstm_accuracy:.4f} ({lstm_accuracy*100:.2f}%)\n")

    # Evaluate baselines
    baseline_results = BaselineModels.evaluate_all_baselines(
        X_train, y_train, X_test, y_test
    )

    # Compare
    evaluator = ModelEvaluator()
    evaluator.compare_with_baselines(lstm_accuracy, baseline_results)

    # Detailed metrics
    results = evaluator.evaluate_all_metrics(y_test, y_pred)
    evaluator.print_results(results)

    # Confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='results/figures/confusion_matrix_corrected.png',
        title='Confusion Matrix - Corrected Model'
    )

    # Per-class metrics
    evaluator.plot_per_class_metrics(
        results,
        save_path='results/figures/per_class_metrics_corrected.png'
    )

    return results


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("CRYPTO MARKET REGIME CLASSIFICATION - CORRECTED TRAINING")
    print("="*70)
    print("\nThis script fixes all critical bugs:")
    print("  ✓ Data leakage in sequence creation")
    print("  ✓ RSI division by zero")
    print("  ✓ Double scaling issue")
    print("  ✓ Inconsistent callback monitoring")
    print("="*70 + "\n")

    # Load config
    config = load_config()

    # Set random seeds
    set_random_seeds(config['random_seed'])

    # Load data
    df = load_and_prepare_data(config)

    # Engineer features
    df = engineer_features(df, config)

    # Create labels
    df = create_market_condition_labels(df, config)

    # Train/val/test split
    train_df, val_df, test_df, feature_cols, le = prepare_train_val_test_splits(df, config)

    # Scale and create sequences
    (X_train, y_train_cat, y_train,
     X_val, y_val_cat, y_val,
     X_test, y_test_cat, y_test) = scale_and_create_sequences(
        train_df, val_df, test_df, feature_cols, config
    )

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(config, input_shape)

    # Train
    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat, config)

    # Load best model
    model = keras.models.load_model(config['training']['callbacks']['model_checkpoint']['filepath'])

    # Evaluate
    results = evaluate_with_baselines(model, X_train, y_train, X_test, y_test)

    print("\n" + "="*70)
    print("✓ TRAINING PIPELINE COMPLETE")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Review evaluation metrics in results/metrics/")
    print("  2. Check plots in results/figures/")
    print("  3. Run SHAP analysis for interpretability")
    print("  4. Compare with original notebook results")


if __name__ == '__main__':
    main()
