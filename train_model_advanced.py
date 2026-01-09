"""Advanced model training with hyperparameter tuning and diagnostics."""

import argparse
import logging
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

from buoy_data.ml import BuoyForecaster
from buoy_data.ml.wave_predictor import WaveHeightPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tune_hyperparameters(X, y):
    """
    Perform hyperparameter tuning for Random Forest.

    Args:
        X: Feature matrix
        y: Target values

    Returns:
        Best hyperparameters
    """
    logger.info("Starting hyperparameter tuning...")

    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    from sklearn.ensemble import RandomForestRegressor

    # Create base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Random search with cross-validation
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=20,  # Number of parameter settings to try
        cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X, y)

    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV RMSE: {-random_search.best_score_:.3f}m")

    return random_search.best_params_


def main():
    parser = argparse.ArgumentParser(
        description='Advanced model training with hyperparameter tuning'
    )
    parser.add_argument(
        '--buoys',
        nargs='+',
        default=['44017', '44008', '44013', '44025', '44065', '44066'],
        help='List of buoy station IDs to train on (more buoys = better)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='Number of days of historical data to collect (more = better)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower but better results)'
    )
    parser.add_argument(
        '--model-type',
        choices=['random_forest', 'gradient_boosting'],
        default='random_forest',
        help='Type of ML model to train'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/wave_predictor_optimized.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='sqlite:///buoy_ml_data.db',
        help='Database connection string'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("ADVANCED BUOY WAVE HEIGHT FORECASTING - MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Buoy stations: {args.buoys} ({len(args.buoys)} buoys)")
    logger.info(f"Days of data: {args.days}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Hyperparameter tuning: {args.tune}")
    logger.info(f"Output path: {args.output}")
    logger.info("="*70)

    try:
        # Initialize forecaster
        with BuoyForecaster(db_path=args.db) as forecaster:

            # Collect data
            logger.info(f"\nCollecting training data from {len(args.buoys)} buoys...")
            raw_data = forecaster.data_collector.collect_training_dataset(
                args.buoys,
                days_back=args.days
            )

            if raw_data.empty:
                raise ValueError("No data collected for training")

            logger.info(f"✓ Collected {len(raw_data)} data points")

            # Prepare features
            logger.info("\nEngineering features...")
            prepared_data = forecaster.feature_engineer.prepare_features(
                raw_data,
                add_lags=True,
                add_rolling=True,
                add_inter_buoy=True
            )

            logger.info(f"✓ Created {len(prepared_data.columns)} features")

            # Split features and target
            X, y = forecaster.feature_engineer.split_features_target(
                prepared_data,
                target_col='wave_height_m'
            )

            # Remove rows with missing target
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]

            logger.info(f"✓ Training set: {len(X)} samples, {X.shape[1]} features")

            # Hyperparameter tuning if requested
            if args.tune:
                best_params = tune_hyperparameters(X, y)

                # Create predictor with tuned parameters
                from sklearn.ensemble import RandomForestRegressor
                predictor = WaveHeightPredictor(model_type=args.model_type)
                predictor.model = RandomForestRegressor(
                    **best_params,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Use default predictor with improved parameters
                predictor = WaveHeightPredictor(model_type=args.model_type)
                if args.model_type == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    predictor.model = RandomForestRegressor(
                        n_estimators=200,  # Increased from 100
                        max_depth=30,      # Deeper trees
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    )

            # Train the model
            logger.info("\nTraining model with improved parameters...")
            forecaster.predictor = predictor
            metrics = predictor.train(prepared_data, cv_folds=10)  # More CV folds

            # Save model
            predictor.save(args.output)

            # Print results
            logger.info("\n" + "="*70)
            logger.info("TRAINING COMPLETE - MODEL PERFORMANCE")
            logger.info("="*70)
            logger.info(f"Test RMSE:     {metrics['test_rmse']:.3f}m ({metrics['test_rmse']*3.28:.3f}ft)")
            logger.info(f"Test MAE:      {metrics['test_mae']:.3f}m ({metrics['test_mae']*3.28:.3f}ft)")
            logger.info(f"Test R²:       {metrics['test_r2']:.3f} ({metrics['test_r2']*100:.1f}% variance explained)")
            logger.info(f"CV RMSE:       {metrics['cv_rmse']:.3f} ± {metrics['cv_rmse_std']:.3f}m")

            # Calculate confidence metrics
            if 'test_r2' in metrics:
                confidence_pct = max(0, metrics['test_r2'] * 100)
                logger.info(f"\n✓ Model Confidence: {confidence_pct:.1f}%")

                if confidence_pct < 70:
                    logger.warning("\n⚠ Low confidence detected. Recommendations:")
                    logger.warning("  1. Increase --days (collect more historical data)")
                    logger.warning("  2. Add more --buoys (improve spatial coverage)")
                    logger.warning("  3. Use --tune flag (optimize hyperparameters)")
                elif confidence_pct < 85:
                    logger.info("\n✓ Moderate confidence. Consider:")
                    logger.info("  - Using --tune for better optimization")
                    logger.info("  - Collecting more training data")
                else:
                    logger.info("\n✓✓ High confidence model!")

            logger.info(f"\nModel saved to: {args.output}")
            logger.info("="*70)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
