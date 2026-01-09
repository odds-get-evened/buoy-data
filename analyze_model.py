"""Model diagnostics and analysis tools."""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from buoy_data.ml import WaveHeightPredictor, DataCollector, FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_model(model_path: str, db_path: str = "sqlite:///buoy_ml_data.db"):
    """
    Analyze trained model and provide diagnostics.

    Args:
        model_path: Path to trained model
        db_path: Database connection string
    """
    logger.info("Loading model...")
    predictor = WaveHeightPredictor.load(model_path)

    logger.info("\n" + "="*70)
    logger.info("MODEL DIAGNOSTICS")
    logger.info("="*70)

    # Basic info
    logger.info(f"\nModel Type: {predictor.model_type}")
    logger.info(f"Number of features: {len(predictor.feature_columns)}")

    # Performance metrics
    metrics = predictor.metrics
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Train RMSE: {metrics['train_rmse']:.3f}m")
    logger.info(f"  Test RMSE:  {metrics['test_rmse']:.3f}m")
    logger.info(f"  Test MAE:   {metrics['test_mae']:.3f}m")
    logger.info(f"  Test R²:    {metrics['test_r2']:.3f}")
    logger.info(f"  CV RMSE:    {metrics['cv_rmse']:.3f} ± {metrics['cv_rmse_std']:.3f}m")

    # Confidence analysis
    r2 = metrics['test_r2']
    confidence_pct = max(0, r2 * 100)

    logger.info(f"\n✓ Model Confidence: {confidence_pct:.1f}%")
    logger.info(f"  (Based on R² = {r2:.3f})")

    # Feature importance
    if 'feature_importance' in metrics:
        logger.info(f"\nTop 15 Most Important Features:")
        feature_imp = metrics['feature_importance']
        for idx, row in feature_imp.head(15).iterrows():
            logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    # Recommendations
    logger.info(f"\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)

    if confidence_pct < 70:
        logger.warning("\n⚠ LOW CONFIDENCE - Action Required:")
        logger.warning("  1. Collect MORE historical data:")
        logger.warning("     python train_model_advanced.py --days 21 --buoys 44017 44008 44013 44025 44065 44066")
        logger.warning("  2. Use hyperparameter tuning:")
        logger.warning("     python train_model_advanced.py --tune --days 14")
        logger.warning("  3. Add more buoy stations for better spatial coverage")
    elif confidence_pct < 85:
        logger.info("\n✓ MODERATE CONFIDENCE - Improvements Possible:")
        logger.info("  1. Use --tune flag for hyperparameter optimization")
        logger.info("  2. Increase training data to 21+ days")
        logger.info("  3. Consider adding more buoy stations")
    else:
        logger.info("\n✓✓ HIGH CONFIDENCE MODEL - Great job!")
        logger.info("  Model is performing well with {confidence_pct:.1f}% confidence")

    # Error analysis
    test_rmse_ft = metrics['test_rmse'] * 3.28
    logger.info(f"\nError Analysis:")
    logger.info(f"  Average prediction error: ±{test_rmse_ft:.2f} feet")
    logger.info(f"  95% of predictions within: ±{test_rmse_ft*2:.2f} feet")

    if test_rmse_ft > 1.5:
        logger.warning("  High error rate detected - consider retraining with more data")
    elif test_rmse_ft < 0.5:
        logger.info("  ✓ Excellent accuracy!")

    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and diagnose trained model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/wave_predictor.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='sqlite:///buoy_ml_data.db',
        help='Database connection string'
    )

    args = parser.parse_args()

    try:
        analyze_model(args.model, args.db)
    except FileNotFoundError:
        logger.error(f"Model not found: {args.model}")
        logger.error("Train a model first using: python train_model.py")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
