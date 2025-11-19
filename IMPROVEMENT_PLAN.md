# Comprehensive Improvement Plan
## Crypto Market Regime Classification Project

**Created**: 2025-11-19
**Status**: Implementation Ready
**Priority**: Critical fixes first, then incremental improvements

---

## Overview

This document outlines the complete improvement plan for the Crypto Market Regime Classification project. Issues are categorized by severity and organized into implementation phases.

---

## Phase 1: CRITICAL FIXES (Must Do Immediately)

### 1.1 Fix Data Leakage in Sequence Creation
**File**: `main.ipynb` (sequence creation cell)
**Issue**: Model may be using current timestep data to predict current label
**Severity**: CRITICAL
**Impact**: Artificially inflated accuracy (81.2% may drop after fix)

**Current Implementation**:
```python
for i in range(0, len(feature_data) - timesteps + 1):
    X.append(feature_data[i:i+timesteps])      # t-9 to t-1
    y.append(target_data[i+timesteps-1])       # label at t-1 (WRONG!)
```

**Corrected Implementation**:
```python
for i in range(0, len(feature_data) - timesteps):
    X.append(feature_data[i:i+timesteps])      # t-9 to t-1
    y.append(target_data[i+timesteps])         # label at t (predicting future)
    dates.append(date_data[i+timesteps])
```

**Validation**:
- Verify no future information in features
- Re-evaluate model performance
- Compare before/after accuracy

---

### 1.2 Fix RSI Division by Zero
**File**: `main.ipynb` (RSI calculation in feature engineering)
**Issue**: When avg_loss = 0, division causes inf/nan values
**Severity**: HIGH
**Impact**: Invalid feature values, model instability

**Fix**:
```python
# Before
rs = avg_gain / avg_loss

# After
rs = avg_gain / (avg_loss + 1e-10)  # Add epsilon
```

**Additional**: Add NaN/Inf detection after all feature engineering:
```python
# Validation check
assert not df_filtered_copy.isnull().any().any(), "NaN values detected!"
assert not np.isinf(df_filtered_copy.select_dtypes(include=[np.number])).any().any(), "Inf values detected!"
```

---

### 1.3 Consolidate Feature Scaling
**File**: `main.ipynb` (multiple scaling cells)
**Issue**: Features scaled twice - once in Cell 28, again in Cell 32
**Severity**: HIGH
**Impact**: Inconsistent normalization, may affect model performance

**Solution**: Single scaling pipeline
```python
# Step 1: Split data (train/val/test)
X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(...)

# Step 2: Fit scaler ONLY on training data
scaler = MinMaxScaler()
scaler.fit(X_train)  # Fit once on train

# Step 3: Transform all sets
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 4: Create sequences from scaled data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps=10)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps=10)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, timesteps=10)

# Save scaler for deployment
import joblib
joblib.dump(scaler, 'models/feature_scaler.pkl')
```

---

## Phase 2: HIGH PRIORITY IMPROVEMENTS (Do Soon)

### 2.1 Add Data Validation
**Files**: New module `src/data/validation.py`

```python
def validate_data_pipeline(df, stage_name):
    """Validate data at each pipeline stage"""
    print(f"\n=== Validation: {stage_name} ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Null values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Check for inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    if inf_counts.sum() > 0:
        print(f"WARNING: Inf values found: {inf_counts[inf_counts > 0]}")

    return df
```

**Usage**:
```python
crypto_df = validate_data_pipeline(crypto_df, "After crypto load")
merged_df = validate_data_pipeline(merged_df, "After SP500 merge")
final_df = validate_data_pipeline(final_df, "After inflation merge")
```

---

### 2.2 Create requirements.txt
**File**: `requirements.txt`

```
# Core dependencies
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2

# Deep Learning
tensorflow==2.12.0
keras==2.12.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Data sources
kagglehub==0.2.0

# Model interpretation
shap==0.42.1

# Utilities
joblib==1.2.0
pyyaml==6.0

# Statistical
scipy==1.10.1
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

### 2.3 Add Configuration Management
**File**: `config/config.yaml`

```yaml
# Data Configuration
data:
  date_range:
    start: "2015-01-01"
    end: "2018-11-30"
  top_n_cryptos: 10
  outlier_percentiles: [1, 99]

# Feature Engineering
features:
  technical_indicators:
    sma_window: 7
    rsi_window: 14
    ema_windows: [12, 26]
    bollinger_window: 20
    macd_windows: [12, 26]

  lag_features:
    columns: ['close', 'volume']
    lags: [1, 3, 5, 7]

  rolling_features:
    windows: [7, 14, 21]
    statistics: ['mean', 'std']

# Market Condition Classification
market_conditions:
  bullish_threshold: 0.015
  bearish_threshold: -0.015
  neutral_threshold: 0.01
  lookback_days: [1, 3]

# Model Architecture
model:
  name: "BidirectionalLSTM"
  timesteps: 10
  lstm_layers:
    - units: 64
      return_sequences: true
      dropout: 0.3
    - units: 64
      return_sequences: false
      dropout: 0.3
  dense_layers:
    - units: 32
      activation: "relu"
      dropout: 0.5
  output_activation: "softmax"

# Training Configuration
training:
  epochs: 20
  batch_size: 64
  validation_split: 0.15
  test_split: 0.15

  optimizer:
    name: "adam"
    learning_rate: 0.001

  callbacks:
    early_stopping:
      monitor: "val_loss"
      patience: 5
      restore_best_weights: true

    reduce_lr:
      monitor: "val_loss"
      factor: 0.5
      patience: 3
      min_lr: 0.0001

    model_checkpoint:
      monitor: "val_loss"  # Changed from val_accuracy for consistency
      save_best_only: true
      filepath: "models/best_crypto_model.h5"

# Evaluation
evaluation:
  metrics: ['accuracy', 'precision', 'recall', 'f1']
  generate_confusion_matrix: true
  per_class_analysis: true
  per_crypto_analysis: true

# Reproducibility
random_seed: 42

# Paths
paths:
  data_dir: "data/"
  models_dir: "models/"
  results_dir: "results/"
  logs_dir: "logs/"
```

**Usage**:
```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access configuration
timesteps = config['model']['timesteps']
learning_rate = config['training']['optimizer']['learning_rate']
```

---

### 2.4 Fix Callback Monitoring Inconsistency
**File**: `main.ipynb` (callback definition)

**Issue**: EarlyStopping monitors `val_loss`, ModelCheckpoint monitors `val_accuracy`

**Fix**: Use consistent metric
```python
early_stopping = EarlyStopping(
    monitor='val_loss',  # Consistent
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'models/best_crypto_model.h5',
    monitor='val_loss',  # Changed from val_accuracy
    save_best_only=True,
    verbose=1
)
```

---

## Phase 3: CODE QUALITY & STRUCTURE (Refactoring)

### 3.1 Modular Package Structure
**New Directory Structure**:
```
Crypto-Market-Regime-Classification/
├── config/
│   └── config.yaml                    # Configuration file
├── data/
│   ├── raw/                           # Original downloaded data
│   ├── processed/                     # Cleaned data
│   └── external/                      # SP500, inflation data
├── models/
│   ├── best_crypto_model.h5          # Saved model
│   └── feature_scaler.pkl            # Saved scaler
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Load datasets
│   │   ├── preprocessing.py          # Clean, merge, outliers
│   │   ├── features.py               # Feature engineering
│   │   └── validation.py             # Data validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py             # Model architecture
│   │   ├── trainer.py                # Training logic
│   │   └── sequences.py              # Sequence creation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── baselines.py              # Baseline models
│   │   └── interpretability.py       # SHAP analysis
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # Load YAML config
│       └── logger.py                 # Logging setup
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_interpretability_analysis.ipynb
├── scripts/
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── predict.py                    # Prediction script
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
├── results/
│   ├── figures/                      # Plots
│   ├── metrics/                      # Metric reports
│   └── predictions/                  # Model predictions
├── requirements.txt
├── setup.py
├── README.md
├── IMPROVEMENT_PLAN.md               # This file
└── .gitignore
```

---

### 3.2 Create Core Modules

#### 3.2.1 `src/data/features.py`
```python
"""Feature engineering for cryptocurrency data"""

import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Calculate technical indicators for crypto data"""

    @staticmethod
    def calculate_sma(df, column='close', window=7):
        """Calculate Simple Moving Average"""
        return df.groupby('symbol')[column].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    @staticmethod
    def calculate_rsi(df, column='close', window=14):
        """Calculate Relative Strength Index with division by zero protection"""
        def rsi_for_group(group):
            delta = group.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)  # Prevent division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi

        return df.groupby('symbol')[column].transform(rsi_for_group)

    @staticmethod
    def calculate_ema(df, column='close', span=12):
        """Calculate Exponential Moving Average"""
        return df.groupby('symbol')[column].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )

    @staticmethod
    def calculate_macd(df, column='close', fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = TechnicalIndicators.calculate_ema(df, column, fast)
        ema_slow = TechnicalIndicators.calculate_ema(df, column, slow)
        return ema_fast - ema_slow

    @staticmethod
    def calculate_bollinger_bands(df, column='close', window=20, num_std=2):
        """Calculate Bollinger Bands"""
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
        """Add all technical indicators based on config"""
        # SMA
        df['SMA_7'] = cls.calculate_sma(df, window=config['sma_window'])

        # RSI
        df['RSI_14'] = cls.calculate_rsi(df, window=config['rsi_window'])

        # EMA
        df['EMA_12'] = cls.calculate_ema(df, span=config['ema_windows'][0])
        df['EMA_26'] = cls.calculate_ema(df, span=config['ema_windows'][1])

        # MACD
        df['MACD'] = cls.calculate_macd(df)

        # Bollinger Bands
        df['BB_upper'], df['BB_lower'] = cls.calculate_bollinger_bands(
            df, window=config['bollinger_window']
        )

        # Validate no inf/nan values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not df[numeric_cols].isnull().any().any(), "NaN detected after feature engineering"
        assert not np.isinf(df[numeric_cols]).any().any(), "Inf detected after feature engineering"

        return df
```

#### 3.2.2 `src/models/sequences.py`
```python
"""Sequence creation for time series LSTM models"""

import numpy as np

def create_sequences(feature_data, target_data, date_data, timesteps=10):
    """
    Create sequences for LSTM training without data leakage.

    Uses timesteps of historical data to predict the NEXT timestep.

    Args:
        feature_data (np.array): Features of shape (samples, features)
        target_data (np.array): Targets of shape (samples,)
        date_data (np.array): Dates of shape (samples,)
        timesteps (int): Number of historical timesteps

    Returns:
        X (np.array): Feature sequences (samples, timesteps, features)
        y (np.array): Target values (samples,)
        dates (np.array): Corresponding dates (samples,)

    Example:
        Given data for days 0-100:
        - X[0] will contain features from days 0-9
        - y[0] will contain target from day 10 (predicting future)
    """
    X, y, dates = [], [], []

    # Critical: Predict NEXT timestep, not current
    for i in range(0, len(feature_data) - timesteps):
        # Features: from i to i+timesteps-1 (historical window)
        X.append(feature_data[i:i+timesteps])

        # Target: at i+timesteps (future prediction)
        y.append(target_data[i+timesteps])

        # Date: of the prediction target
        dates.append(date_data[i+timesteps])

    return np.array(X), np.array(y), np.array(dates)


def validate_no_leakage(X, y, dates, feature_dates):
    """
    Validate that no data leakage exists in sequences.

    Checks that features come from before the target date.
    """
    print("\n=== Data Leakage Validation ===")

    # Check that last feature date < target date
    for i in range(min(10, len(X))):  # Check first 10 samples
        last_feature_idx = np.where(feature_dates == dates[i])[0][0] - 1
        if last_feature_idx >= 0:
            last_feature_date = feature_dates[last_feature_idx]
            target_date = dates[i]

            assert last_feature_date < target_date, \
                f"Data leakage detected: Feature date {last_feature_date} >= Target date {target_date}"

    print("✓ No data leakage detected")
    print(f"✓ Features use historical data only")
    print(f"✓ Total sequences: {len(X)}")
```

---

## Phase 4: ENHANCED EVALUATION (Better Metrics)

### 4.1 Baseline Models
**File**: `src/evaluation/baselines.py`

```python
"""Baseline models for comparison"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class BaselineModels:
    """Simple baseline models for comparison"""

    @staticmethod
    def random_classifier(y_test, num_classes=3):
        """Random predictions"""
        y_pred = np.random.randint(0, num_classes, size=len(y_test))
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    @staticmethod
    def majority_class_classifier(y_train, y_test):
        """Always predict majority class"""
        majority_class = np.bincount(y_train).argmax()
        y_pred = np.full(len(y_test), majority_class)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    @staticmethod
    def random_forest_baseline(X_train, y_train, X_test, y_test):
        """Random Forest baseline"""
        # Flatten sequences for RF
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train)
        y_pred = rf.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    @staticmethod
    def logistic_regression_baseline(X_train, y_train, X_test, y_test):
        """Logistic Regression baseline"""
        # Flatten sequences
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_flat, y_train)
        y_pred = lr.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
```

### 4.2 Enhanced Metrics
**File**: `src/evaluation/metrics.py`

```python
"""Comprehensive evaluation metrics"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, class_names=['bearish', 'bullish', 'neutral']):
        self.class_names = class_names

    def evaluate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all evaluation metrics"""
        results = {}

        # Overall accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )

        results['per_class'] = {
            self.class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
            for i in range(len(self.class_names))
        }

        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        results['weighted_avg'] = {
            'precision': precision_w,
            'recall': recall_w,
            'f1_score': f1_w
        }

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return results

    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Accuracy: {results['accuracy']:.4f}")

        print("\nPer-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-" * 60)

        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            print(f"{class_name:<12} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{int(metrics['support']):<12}")

        print("\nWeighted Averages:")
        print(f"Precision: {results['weighted_avg']['precision']:.4f}")
        print(f"Recall: {results['weighted_avg']['recall']:.4f}")
        print(f"F1-Score: {results['weighted_avg']['f1_score']:.4f}")

    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

## Phase 5: MODEL INTERPRETABILITY

### 5.1 SHAP Analysis
**File**: `src/evaluation/interpretability.py`

```python
"""Model interpretability using SHAP"""

import shap
import matplotlib.pyplot as plt
import numpy as np

class ModelInterpreter:
    """SHAP-based model interpretation"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def analyze_feature_importance(self, X_train, X_test, max_samples=100):
        """
        Analyze feature importance using SHAP.

        Args:
            X_train: Training data for background samples
            X_test: Test data to explain
            max_samples: Maximum samples for analysis (computational limit)
        """
        print(f"Creating SHAP explainer with {max_samples} background samples...")

        # Create explainer with subset of training data
        background = X_train[:max_samples]
        explainer = shap.DeepExplainer(self.model, background)

        # Calculate SHAP values for test set
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test[:max_samples])

        return shap_values

    def plot_summary(self, shap_values, X_test, max_samples=100):
        """Plot SHAP summary"""
        # For multi-class, shap_values is a list of arrays
        # Flatten temporal dimension for visualization
        X_test_flat = X_test[:max_samples].reshape(X_test[:max_samples].shape[0], -1)

        # Create feature names for flattened data
        temporal_features = []
        for t in range(X_test.shape[1]):  # timesteps
            for f in self.feature_names:
                temporal_features.append(f"{f}_t-{X_test.shape[1]-t-1}")

        # Plot for each class
        for class_idx, class_name in enumerate(['bearish', 'bullish', 'neutral']):
            shap_values_flat = shap_values[class_idx].reshape(
                shap_values[class_idx].shape[0], -1
            )

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values_flat,
                X_test_flat,
                feature_names=temporal_features,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Feature Importance - {class_name.capitalize()} Class')
            plt.tight_layout()
            plt.savefig(f'results/figures/shap_summary_{class_name}.png', dpi=300)
            plt.show()
```

---

## Phase 6: REPRODUCIBILITY & DEPLOYMENT

### 6.1 Set Random Seeds
**File**: `src/utils/reproducibility.py`

```python
"""Ensure reproducibility across runs"""

import random
import numpy as np
import tensorflow as tf
import os

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    print(f"✓ Random seeds set to {seed}")
    print(f"✓ Deterministic operations enabled")
```

### 6.2 Logging Configuration
**File**: `src/utils/logger.py`

```python
"""Centralized logging configuration"""

import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

---

## Implementation Timeline

### Week 1: Critical Fixes
- [ ] Day 1-2: Fix data leakage, RSI bug, scaling issues
- [ ] Day 3: Add requirements.txt and validation
- [ ] Day 4-5: Re-train model and verify accuracy

### Week 2: Code Structure
- [ ] Day 1-2: Create modular package structure
- [ ] Day 3-4: Refactor notebook into modules
- [ ] Day 5: Add configuration management

### Week 3: Enhanced Evaluation
- [ ] Day 1-2: Implement baseline models
- [ ] Day 3-4: Add comprehensive metrics
- [ ] Day 5: SHAP analysis implementation

### Week 4: Testing & Documentation
- [ ] Day 1-2: Add unit tests
- [ ] Day 3-4: Update documentation
- [ ] Day 5: Final review and deployment preparation

---

## Success Criteria

### Technical Metrics
- [ ] No data leakage detected
- [ ] All features free of NaN/Inf values
- [ ] Consistent scaling across train/val/test
- [ ] Test accuracy with proper validation
- [ ] Baseline comparisons documented

### Code Quality
- [ ] All code modularized
- [ ] >80% test coverage
- [ ] All functions documented
- [ ] Configuration externalized
- [ ] Logging implemented

### Reproducibility
- [ ] Random seeds set
- [ ] Dependencies pinned
- [ ] Results reproducible across runs
- [ ] Environment documented

---

## Notes

This plan prioritizes correctness and reproducibility over incremental improvements. The critical fixes in Phase 1 may result in decreased model accuracy, but this reflects the true performance of the model without data leakage.

All subsequent improvements should be measured against baseline models to ensure meaningful progress.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-19
