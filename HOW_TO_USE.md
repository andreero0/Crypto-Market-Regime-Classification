# How to Use the Corrected Codebase

This guide explains how to use the refactored and bug-fixed cryptocurrency market regime classification system.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Corrected Training Pipeline

```bash
python scripts/train_corrected.py
```

This single command will:
- Load and validate data
- Engineer features (with bug fixes)
- Create sequences (without data leakage)
- Train the LSTM model
- Evaluate against 4 baseline models
- Generate comprehensive metrics and plots

---

## What's Different from the Original Notebook?

### Critical Fixes Applied

1. **✅ No Data Leakage**
   - Old: Used data from time `t` to predict condition at time `t` (leakage!)
   - New: Uses data from `t-9` to `t-1` to predict condition at time `t` (correct)

2. **✅ RSI Bug Fixed**
   - Old: Division by zero when `avg_loss = 0` → `inf/nan` values
   - New: Added epsilon (`1e-10`) to prevent division by zero

3. **✅ Single Scaling**
   - Old: Features scaled twice with different scalers (inconsistent)
   - New: Features scaled once with single scaler (correct)

4. **✅ Baseline Comparisons**
   - Old: Only LSTM accuracy reported (81.2%)
   - New: Compared against random, majority class, Random Forest, Logistic Regression

5. **✅ Enhanced Metrics**
   - Old: Only accuracy
   - New: Precision, recall, F1, confusion matrix, per-class analysis

---

## Project Structure

```
Crypto-Market-Regime-Classification/
├── config/
│   └── config.yaml                    # Edit hyperparameters here
├── src/
│   ├── data/
│   │   ├── features.py               # Feature engineering (RSI fixed)
│   │   └── validation.py             # Data validation
│   ├── models/
│   │   └── sequences.py              # Sequence creation (no leakage)
│   ├── evaluation/
│   │   ├── baselines.py              # Baseline models
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── interpretability.py       # SHAP analysis
│   └── utils/
│       ├── config_loader.py          # Load config
│       └── reproducibility.py        # Set random seeds
├── scripts/
│   └── train_corrected.py            # Main training script
├── models/                           # Saved models and scalers
├── results/                          # Generated plots and metrics
└── requirements.txt                  # Dependencies
```

---

## Step-by-Step Usage

### Step 1: Configure Your Experiment

Edit `config/config.yaml` to change:

```yaml
# Model architecture
model:
  timesteps: 10              # Number of historical timesteps
  lstm_layers:
    - units: 64              # LSTM units
      dropout: 0.3           # Dropout rate

# Training parameters
training:
  epochs: 20
  batch_size: 64
  optimizer:
    learning_rate: 0.001

# Feature engineering
features:
  technical_indicators:
    sma_window: 7
    rsi_window: 14
    ema_windows: [12, 26]
```

### Step 2: Train the Model

```bash
python scripts/train_corrected.py
```

**Output**: You'll see:
```
=======================================================================
CRYPTO MARKET REGIME CLASSIFICATION - CORRECTED TRAINING
=======================================================================

This script fixes all critical bugs:
  ✓ Data leakage in sequence creation
  ✓ RSI division by zero
  ✓ Double scaling issue
  ✓ Inconsistent callback monitoring
=======================================================================

STEP 1: DATA LOADING
...
STEP 2: FEATURE ENGINEERING (WITH BUG FIXES)
  ✓ SMA added
  ✓ RSI added (with division-by-zero fix)
  ✓ EMA added
  ✓ MACD added
  ✓ Bollinger Bands added
  ✓ No NaN/Inf values detected
...
STEP 5: SCALING AND SEQUENCE CREATION (FIXED)
Fitting scaler on training data...
Transforming all datasets...
✓ Scaling complete (single pass, no double scaling)

Creating sequences with timesteps=10...
CRITICAL: Using FIXED sequence creation (no data leakage)

=== DATA LEAKAGE VALIDATION ===
Sample 0 validation:
  Feature window: 2015-01-08 to 2015-01-21
  Target date: 2015-01-22
  ✓ Last feature is before target (no leakage)

✓ Validated 10 samples - NO DATA LEAKAGE DETECTED
...
STEP 8: EVALUATION WITH BASELINES

LSTM Test Accuracy: 0.XXXX (XX.XX%)

BASELINE MODEL EVALUATION
1. Random Classifier
   Accuracy: 0.3333
2. Majority Class Classifier
   Accuracy: 0.XXXX
3. Random Forest
   Accuracy: 0.XXXX
4. Logistic Regression
   Accuracy: 0.XXXX

LSTM vs BASELINE COMPARISON
Model                          Accuracy        Improvement vs Random
LSTM                           0.XXXX           +XX.XX%
...
```

### Step 3: Review Results

**Plots** saved to `results/figures/`:
- `confusion_matrix_corrected.png` - Confusion matrix
- `per_class_metrics_corrected.png` - Precision/Recall/F1 per class

**Models** saved to `models/`:
- `best_crypto_model.h5` - Best trained model
- `feature_scaler.pkl` - Fitted scaler (for deployment)

---

## Using Individual Modules

### Feature Engineering

```python
from src.data.features import TechnicalIndicators
from src.utils.config_loader import load_config

config = load_config()

# Add all technical indicators (with fixes)
df = TechnicalIndicators.add_all_indicators(df, config)

# Manually calculate specific indicators
df['RSI_14'] = TechnicalIndicators.calculate_rsi(df, window=14)  # Fixed version
df['SMA_7'] = TechnicalIndicators.calculate_sma(df, window=7)
```

### Sequence Creation (No Data Leakage)

```python
from src.models.sequences import create_sequences_with_validation

# Create sequences with automatic validation
X, y, dates = create_sequences_with_validation(
    feature_data=scaled_features,
    target_data=labels,
    date_data=dates,
    timesteps=10,
    validate=True  # Will check for data leakage
)
```

### Baseline Evaluation

```python
from src.evaluation.baselines import BaselineModels

# Evaluate all baselines at once
baseline_results = BaselineModels.evaluate_all_baselines(
    X_train, y_train, X_test, y_test
)

# Or individual baselines
rf_acc, rf_pred, rf_model = BaselineModels.random_forest_baseline(
    X_train, y_train, X_test, y_test
)
```

### Enhanced Metrics

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(class_names=['bearish', 'bullish', 'neutral'])

# Get all metrics
results = evaluator.evaluate_all_metrics(y_true, y_pred)

# Print formatted results
evaluator.print_results(results)

# Plot confusion matrix
evaluator.plot_confusion_matrix(
    results['confusion_matrix'],
    save_path='results/figures/confusion_matrix.png'
)

# Compare with baselines
evaluator.compare_with_baselines(lstm_accuracy, baseline_results)
```

### SHAP Interpretability

```python
from src.evaluation.interpretability import run_full_interpretability_analysis

# Run complete SHAP analysis
interpreter = run_full_interpretability_analysis(
    model=trained_model,
    X_train=X_train,
    X_test=X_test,
    feature_names=['close', 'volume', 'RSI_14', ...],
    max_background=100,
    max_samples=100,
    save_dir='results/figures'
)
```

---

## Troubleshooting

### Issue: "Config file not found"
**Solution**: Make sure you're running from the project root directory:
```bash
cd Crypto-Market-Regime-Classification/
python scripts/train_corrected.py
```

### Issue: "Module not found"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

Or add project to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Crypto-Market-Regime-Classification"
```

### Issue: "CSV file not found"
**Solution**: Ensure `filtered_crypto_data.csv` exists in the project root. If you need to regenerate it, refer to the data preparation section of the original notebook.

### Issue: "Out of memory"
**Solution**: Reduce batch size or max samples in `config/config.yaml`:
```yaml
training:
  batch_size: 32  # Reduce from 64

# Or when using SHAP:
max_background: 50  # Reduce from 100
max_samples: 50
```

---

## Expected Performance

### Original Notebook (with bugs)
- Test Accuracy: **81.2%**
- Issues: Data leakage, RSI bugs, double scaling

### Corrected Code (bugs fixed)
- Test Accuracy: **Expected to be lower** (60-75% range)
- Why? Because we removed data leakage - the model now truly predicts the future

**This is correct!** Lower accuracy with proper validation is better than inflated accuracy with data leakage.

---

## Customization

### Change Model Architecture

Edit `config/config.yaml`:

```yaml
model:
  lstm_layers:
    - units: 128              # Increase units
      return_sequences: true
      dropout: 0.4            # Increase dropout
    - units: 128
      return_sequences: false
      dropout: 0.4
  dense_layers:
    - units: 64               # Increase dense units
      activation: "relu"
      dropout: 0.5
```

### Change Features

Edit `config/config.yaml`:

```yaml
features:
  technical_indicators:
    sma_window: 14           # Change window
    rsi_window: 21
    ema_windows: [5, 20]     # Different EMAs

  lag_features:
    columns: ['close', 'volume', 'high', 'low']  # More columns
    lags: [1, 2, 3, 5, 7, 10]                    # More lags
```

### Change Market Condition Thresholds

Edit `config/config.yaml`:

```yaml
market_conditions:
  bullish_threshold: 0.02    # More conservative (2% instead of 1.5%)
  bearish_threshold: -0.02
  neutral_threshold: 0.015
```

---

## Next Steps

1. **Train with corrected code** and note the true accuracy
2. **Compare with baselines** to ensure LSTM adds value
3. **Run SHAP analysis** to understand feature importance
4. **Experiment with hyperparameters** in `config/config.yaml`
5. **Try different architectures** (add layers, change units)
6. **Add more features** (volume indicators, market sentiment, etc.)

---

## Reference Documents

- **IMPROVEMENT_PLAN.md** - Detailed improvement roadmap
- **FIXES_APPLIED.md** - Summary of all bug fixes
- **config/config.yaml** - All configurable parameters
- **README.md** - Original project description

---

## Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review **FIXES_APPLIED.md** for details on specific fixes
3. Check inline code comments in modules
4. Review **IMPROVEMENT_PLAN.md** for design decisions

**Good luck with your improved cryptocurrency market regime classification system!** 🚀
