#!/usr/bin/env python
"""Prediction script for querying wave heights."""

import argparse
import json
import logging
from buoy_data.ml import BuoyForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Query current readings and predict wave heights'
    )
    parser.add_argument(
        '--buoys',
        nargs='+',
        required=True,
        help='List of buoy station IDs to query'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/wave_predictor.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--mode',
        choices=['current', 'forecast', 'summary'],
        default='forecast',
        help='Query mode: current (readings only), forecast (with predictions), summary (regional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for JSON results (default: print to console)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='sqlite:///buoy_ml_data.db',
        help='Database connection string'
    )

    args = parser.parse_args()

    try:
        # Initialize forecaster
        with BuoyForecaster(model_path=args.model, db_path=args.db) as forecaster:

            if args.mode == 'current':
                # Get current readings only
                logger.info(f"Fetching current readings for {len(args.buoys)} buoys...")
                results = forecaster.get_current_readings(args.buoys)
                output = results.to_dict('records')

            elif args.mode == 'forecast':
                # Get forecast with predictions
                logger.info(f"Generating forecast for {len(args.buoys)} buoys...")
                results = forecaster.forecast_between_buoys(args.buoys)
                output = results.to_dict('records')

            elif args.mode == 'summary':
                # Get regional summary
                logger.info(f"Generating regional summary for {len(args.buoys)} buoys...")
                output = forecaster.get_regional_summary(args.buoys)

            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(output, indent=2, default=str))

    except FileNotFoundError:
        logger.error(f"Model file not found: {args.model}")
        logger.error("Train a model first using: python train_model.py")
        return 1
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
