#!/usr/bin/env python
"""Training script for buoy wave height forecasting model."""

import argparse
import logging

from buoy_data.ml import BuoyForecaster
from buoy_data.utils import get_available_stations, filter_stations_by_region

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train wave height forecasting model'
    )
    parser.add_argument(
        '--buoys',
        nargs='+',
        default=None,
        help='List of buoy station IDs to train on'
    )
    parser.add_argument(
        '--all-stations',
        action='store_true',
        help='Use all available stations from NOAA realtime2 directory'
    )
    parser.add_argument(
        '--region',
        type=str,
        choices=['northeast', 'southeast', 'caribbean', 'pacific', 'greatlakes', 'hawaii'],
        help='Filter stations by region (use with --all-stations)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data to collect'
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
        default='models/wave_predictor.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='sqlite:///buoy_ml_data.db',
        help='Database connection string'
    )

    args = parser.parse_args()

    # Determine which buoys to use
    if args.all_stations:
        try:
            buoys = get_available_stations()
            if args.region:
                buoys = filter_stations_by_region(buoys, args.region)
            logger.info(f"Using all available stations: {len(buoys)} buoys")
        except Exception as e:
            logger.error(f"Failed to fetch available stations: {e}")
            logger.info("Falling back to default buoys")
            buoys = ['44017', '44008', '44013', '44025']
    elif args.buoys:
        buoys = args.buoys
    else:
        # Default buoys if neither flag is specified
        buoys = ['44017', '44008', '44013', '44025']
        logger.info("Using default buoys (use --all-stations to train on all available stations)")

    args.buoys = buoys

    logger.info("="*60)
    logger.info("BUOY WAVE HEIGHT FORECASTING - MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Buoy stations: {args.buoys}")
    logger.info(f"Days of data: {args.days}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output path: {args.output}")
    logger.info("="*60)

    try:
        # Initialize forecaster
        with BuoyForecaster(db_path=args.db) as forecaster:
            # Train model
            metrics = forecaster.train(
                buoy_ids=args.buoys,
                days_back=args.days,
                model_type=args.model_type,
                save_path=args.output
            )

            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETE")
            logger.info("="*60)
            logger.info(f"Test RMSE: {metrics['test_rmse']:.3f}m")
            logger.info(f"Test MAE:  {metrics['test_mae']:.3f}m")
            logger.info(f"Test R²:   {metrics['test_r2']:.3f}")
            logger.info(f"CV RMSE:   {metrics['cv_rmse']:.3f} ± {metrics['cv_rmse_std']:.3f}m")
            logger.info(f"\nModel saved to: {args.output}")
            logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
