"""Model diagnostics and analysis tools."""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Import ML classes; tests will monkeypatch WaveHeightPredictor.load
from buoy_data.ml import WaveHeightPredictor, DataCollector, FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_model(model_path: str, db_path: str = "sqlite:///buoy_ml_data.db"):
    """
    Analyze a trained model and provide diagnostics.

    Args:
        model_path: Path to trained model
        db_path: Database connection string
    """
    logger.info("Loading model...")
    # Load the predictor (WaveHeightPredictor.load should handle model_path semantics)
    predictor = WaveHeightPredictor.load(model_path)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL DIAGNOSTICS")
    logger.info("=" * 70)

    # Basic info (safe attribute reads)
    logger.info(f"\nModel Type: {getattr(predictor, 'model_type', 'unknown')}")
    feature_cols = getattr(predictor, "feature_columns", None)
    if feature_cols is None:
        logger.info("Number of features: unknown")
    else:
        logger.info(f"Number of features: {len(feature_cols)}")

    # Performance metrics (defensive access)
    metrics = getattr(predictor, "metrics", {}) or {}

    logger.info(f"\nPerformance Metrics:")
    # Train RMSE
    train_rmse = metrics.get("train_rmse")
    if train_rmse is not None:
        logger.info(f"  Train RMSE: {train_rmse:.3f}m")
    else:
        logger.warning("  train_rmse missing from metrics")

    # Test RMSE
    test_rmse = metrics.get("test_rmse")
    if test_rmse is not None:
        logger.info(f"  Test RMSE:  {test_rmse:.3f}m")
    else:
        logger.warning("  test_rmse missing from metrics")

    # Test MAE
    test_mae = metrics.get("test_mae")
    if test_mae is not None:
        logger.info(f"  Test MAE:   {test_mae:.3f}m")
    else:
        logger.warning("  test_mae missing from metrics")

    # Test R²
    test_r2 = metrics.get("test_r2")
    if test_r2 is not None:
        logger.info(f"  Test R²:    {test_r2:.3f}")
    else:
        logger.warning("  test_r2 missing from metrics")

    # CV RMSE and std
    cv_rmse = metrics.get("cv_rmse")
    cv_rmse_std = metrics.get("cv_rmse_std")
    if cv_rmse is not None:
        if cv_rmse_std is not None:
            logger.info(f"  CV RMSE:    {cv_rmse:.3f} ± {cv_rmse_std:.3f}m")
        else:
            logger.info(f"  CV RMSE:    {cv_rmse:.3f} (std missing)")
    else:
        logger.warning("  cv_rmse missing from metrics")

    # Confidence analysis
    if test_r2 is None:
        logger.warning("\nCannot compute model confidence: test_r2 missing from metrics")
        confidence_pct = 0.0
    else:
        confidence_pct = max(0.0, test_r2 * 100.0)
        logger.info(f"\n✓ Model Confidence: {confidence_pct:.1f}%")
        logger.info(f"  (Based on R² = {test_r2:.3f})")

    # Feature importance (if available)
    if "feature_importance" in metrics and metrics.get("feature_importance") is not None:
        logger.info(f"\nTop 15 Most Important Features:")
        feature_imp = metrics["feature_importance"]
        # Expect feature_imp to be a DataFrame with columns ['feature', 'importance']
        if isinstance(feature_imp, pd.DataFrame):
            for idx, row in feature_imp.head(15).iterrows():
                # Align feature name and importance for readable output
                logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")
        else:
            logger.info("  feature_importance present but not a pandas.DataFrame - skipping detailed print")
    else:
        logger.info("\nFeature importance not available in metrics")

    # Recommendations
    logger.info(f"\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    if confidence_pct < 70:
        logger.warning("\n⚠ LOW CONFIDENCE - Action Required:")
        logger.warning("  1. Collect MORE historical data:")
        logger.warning(
            "     python train_model_advanced.py --days 21 --buoys 44017 44008 44013 44025 44065 44066"
        )
        logger.warning("  2. Use hyperparameter tuning:")
        logger.warning("     python train_model_advanced.py --tune --days 14")
        logger.warning("  3. Add more buoy stations for better spatial coverage")
    elif confidence_pct < 85:
        logger.info("\n✓ MODERATE CONFIDENCE - Improvements Possible:")
        logger.info("  1. Use --tune flag for hyperparameter optimization")
        logger.info("  2. Increase training data to 21+ days")
        logger.info("  3. Consider adding more buoy stations")
    else:
        # Fixed formatting: use f-string so confidence_pct is interpolated correctly
        logger.info("\n✓✓ HIGH CONFIDENCE MODEL - Great job!")
        logger.info(f"  Model is performing well with {confidence_pct:.1f}% confidence")

    # Error analysis: compute approximate feet-based RMSE and an approximate 95% bound
    # Note: Using 1.96 * RMSE as an approximate 95% interval (assumes approximate normal residuals).
    if test_rmse is not None:
        test_rmse_ft = test_rmse * 3.28  # convert meters to feet
        # approximate 95% interval (1.96 * std_est); using RMSE as a rough estimator of residual std
        approx_95_ft = 1.96 * test_rmse_ft
        logger.info(f"\nError Analysis:")
        logger.info(f"  Average prediction error: ±{test_rmse_ft:.2f} feet (approx RMSE)")
        logger.info(f"  Approximate 95% prediction interval: ±{approx_95_ft:.2f} feet")
        # Heuristic checks
        if test_rmse_ft > 1.5:
            logger.warning("  High error rate detected - consider retraining with more data")
        elif test_rmse_ft < 0.5:
            logger.info("  ✓ Excellent accuracy!")
    else:
        logger.info("\nError Analysis: test_rmse not available - cannot compute error summary")

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze and diagnose trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/wave_predictor.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///buoy_ml_data.db",
        help="Database connection string",
    )

    args = parser.parse_args()

    try:
        analyze_model(args.model, args.db)
    except FileNotFoundError:
        # FileNotFoundError is expected when the model file is missing
        logger.error(f"Model not found: {args.model}")
        logger.error("Train a model first using: python train_model.py")
        return 1
    except Exception as e:
        # Generic fallback: log the exception for debugging
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
