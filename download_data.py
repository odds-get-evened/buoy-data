#!/usr/bin/env python
"""Download and cache buoy data for offline training."""

import argparse
import logging

from buoy_data.ml import DataCollector
from buoy_data.utils import get_available_stations, filter_stations_by_region

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Download and cache buoy data for offline training'
    )
    parser.add_argument(
        '--buoys',
        nargs='+',
        default=None,
        help='List of buoy station IDs to download'
    )
    parser.add_argument(
        '--all-stations',
        action='store_true',
        help='Download data from all available stations'
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
        default=14,
        help='Number of days of historical data to download (default: 14)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='sqlite:///buoy_ml_data.db',
        help='Database connection string'
    )

    args = parser.parse_args()

    # Determine which buoys to download
    if args.all_stations:
        try:
            buoys = get_available_stations()
            if args.region:
                buoys = filter_stations_by_region(buoys, args.region)
            logger.info(f"Downloading data from all available stations: {len(buoys)} buoys")
        except Exception as e:
            logger.error(f"Failed to fetch available stations: {e}")
            return 1
    elif args.buoys:
        buoys = args.buoys
    else:
        # Default buoys if neither flag is specified
        buoys = ['44017', '44008', '44013', '44025', '44065', '44066']
        logger.info("Using default buoys (use --all-stations to download from all)")

    logger.info("="*70)
    logger.info("BUOY DATA DOWNLOAD AND CACHING")
    logger.info("="*70)
    logger.info(f"Buoy stations: {len(buoys)} buoys")
    logger.info(f"Days of data: {args.days}")
    logger.info(f"Database: {args.db}")
    logger.info("="*70)

    try:
        # Initialize data collector
        collector = DataCollector(db_path=args.db)

        # Download and cache data (use_cache=False forces download)
        logger.info("\nDownloading data from NOAA...")
        data = collector.collect_training_dataset(
            buoys,
            days_back=args.days,
            use_cache=False,  # Always download fresh data
            save_to_db=True   # Save to database
        )

        if data.empty:
            logger.warning("No data was collected")
            return 1

        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*70)
        logger.info(f"Total readings downloaded: {len(data)}")
        logger.info(f"Buoys with data: {data['buoy_id'].nunique()}")
        logger.info(f"\nData cached in: {args.db}")
        logger.info("\nYou can now train models with --use-cache flag:")
        logger.info(f"  python train_model.py --buoys {' '.join(buoys[:3])} --use-cache")
        logger.info("="*70)

        collector.close()
        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
