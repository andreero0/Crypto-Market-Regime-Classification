#!/usr/bin/env python3
"""
Test script for regime prediction.

This script demonstrates how to use the regime predictor
with your existing trained model.

Usage:
    python scripts/test_prediction.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.models.predictor import RegimePredictor
from src.data.loaders.data_loader import CryptoDataLoader


def main():
    """Run prediction test."""
    print("=" * 60)
    print("Crypto Market Regime Classification - Prediction Test")
    print("=" * 60)

    # Paths
    model_path = "models/best_crypto_model.h5"
    scaler_path = "models/scaler.pkl"
    data_path = "merged_data_final.csv"

    # Alternative data paths
    if not Path(data_path).exists():
        data_path = "data/processed/merged_data_final.csv"
    if not Path(data_path).exists():
        data_path = "filtered_crypto_data.csv"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the trained model is in the models/ directory")
        return

    if not Path(scaler_path).exists():
        print(f"❌ Scaler not found: {scaler_path}")
        print("Please ensure the fitted scaler is in the models/ directory")
        return

    if not Path(data_path).exists():
        print(f"❌ Data not found: {data_path}")
        print("Please ensure data is available")
        return

    print(f"\n✅ Model path: {model_path}")
    print(f"✅ Scaler path: {scaler_path}")
    print(f"✅ Data path: {data_path}")

    # Load predictor
    print("\n" + "-" * 60)
    print("Loading model and scaler...")
    try:
        predictor = RegimePredictor(
            model_path=model_path,
            scaler_path=scaler_path
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load data
    print("\n" + "-" * 60)
    print("Loading cryptocurrency data...")
    try:
        loader = CryptoDataLoader(data_path)
        loader.load_data()
        symbols = loader.get_available_symbols()
        print(f"✅ Data loaded successfully")
        print(f"Available symbols: {symbols[:10]}")  # Show first 10
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Test prediction for each available symbol
    print("\n" + "-" * 60)
    print("Running predictions...")
    print("-" * 60)

    test_symbols = symbols[:5] if len(symbols) >= 5 else symbols

    for symbol in test_symbols:
        try:
            # Get data for symbol
            symbol_data = loader.get_latest_data(symbol, days=50)

            # Run prediction
            prediction = predictor.predict(symbol_data, symbol=symbol)

            # Display results
            print(f"\n📊 {prediction.symbol}")
            print(f"   Regime:     {prediction.regime}")
            print(f"   Confidence: {prediction.confidence:.2%}")
            print(f"   Probabilities:")
            print(f"      Bearish: {prediction.probabilities['bearish']:.2%}")
            print(f"      Bullish: {prediction.probabilities['bullish']:.2%}")
            print(f"      Neutral: {prediction.probabilities['neutral']:.2%}")

        except Exception as e:
            print(f"\n❌ Prediction failed for {symbol}: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
