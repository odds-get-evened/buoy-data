"""Data collection module for historical buoy data."""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..buoy_real_time import BuoyRealTime
from ..buoy_hourly import BuoyHourly
from ..database import BuoyDataDB, BuoyReading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects historical and real-time buoy data for ML training.

    This class handles fetching data from multiple buoy stations,
    storing it in a database, and preparing it for ML model training.
    """

    def __init__(self, db_path: str = "sqlite:///buoy_ml_data.db"):
        """
        Initialize DataCollector.

        Args:
            db_path: Database connection string (default: SQLite)
        """
        self.db_path = db_path
        self.engine = create_engine(db_path, echo=False)
        BuoyDataDB.create_tables(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.db = BuoyDataDB(self.session)

    def collect_current_data(self, buoy_ids: List[str]) -> pd.DataFrame:
        """
        Collect current real-time data from multiple buoys.

        Args:
            buoy_ids: List of buoy station IDs

        Returns:
            DataFrame with current readings from all buoys
        """
        data_list = []

        for buoy_id in buoy_ids:
            try:
                logger.info(f"Fetching real-time data for buoy {buoy_id}")
                buoy = BuoyRealTime(buoy_id)
                data = buoy.get_data()

                # Extract relevant data
                record = {
                    'buoy_id': buoy_id,
                    'timestamp': data.get('timestamp', 0),
                    'wind_direction_deg': data.get('wind_direction', {}).get('degree', None),
                    'wind_speed': data.get('wind_speed', None),
                    'gusts': data.get('gusts', None),
                    'wave_height_m': data.get('wave_height', {}).get('metric', None),
                    'wave_height_ft': data.get('wave_height', {}).get('stnd', None),
                    'dominant_wave_period': data.get('dominant_wave_period', None),
                    'avg_wave_period': data.get('avg_wave_period', None),
                    'mean_wave_direction_deg': data.get('mean_wave_direction', {}).get('degree', None),
                    'barometer': data.get('barometer', None),
                    'air_temp_c': data.get('air_temp', {}).get('celsius', None),
                    'water_temp_c': data.get('water_temp', {}).get('celsius', None),
                    'dewpoint_c': data.get('dewpoint', {}).get('celsius', None),
                }

                data_list.append(record)

                # Store in database
                if data.get('timestamp', 0) > 0:
                    self.db.log_data(
                        buoy_id=buoy_id,
                        wind_dir=int(data.get('wind_direction', {}).get('degree') or 0),
                        wind_spd=float(data.get('wind_speed') or 0.0),
                        wave_height=float(data.get('wave_height', {}).get('metric') or 0.0),
                        water_temp=float(data.get('water_temp', {}).get('celsius') or 0.0),
                        reading_time=int(data.get('timestamp'))
                    )

                # Be nice to NOAA servers
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching data for buoy {buoy_id}: {e}")
                continue

        return pd.DataFrame(data_list)

    def collect_hourly_data(
        self,
        buoy_ids: List[str],
        hr_from: int = 0,
        hr_to: int = 23
    ) -> pd.DataFrame:
        """
        Collect hourly data from multiple buoys.

        Args:
            buoy_ids: List of buoy station IDs
            hr_from: Starting hour (0-23)
            hr_to: Ending hour (0-23)

        Returns:
            DataFrame with hourly readings from all buoys
        """
        data_list = []

        for buoy_id in buoy_ids:
            try:
                logger.info(f"Fetching hourly data for buoy {buoy_id} (hours {hr_from}-{hr_to})")
                buoy = BuoyHourly(buoy_id, hr_from, hr_to)
                readings = buoy.get_data()

                for reading in readings:
                    record = {
                        'buoy_id': buoy_id,
                        'timestamp': reading.get('timestamp', 0),
                        'year': reading.get('year'),
                        'month': reading.get('month'),
                        'day': reading.get('day'),
                        'hour': reading.get('hour'),
                        'minute': reading.get('min'),
                        'wind_direction_deg': reading.get('wind_direction', {}).get('degree', None),
                        'wind_speed': reading.get('wind_speed', None),
                        'gusts': reading.get('gusts', None),
                        'wave_height_m': reading.get('wave_height', {}).get('metric', None),
                        'wave_height_ft': reading.get('wave_height', {}).get('stnd', None),
                        'dominant_wave_period': reading.get('dominant_wave_period', None),
                        'avg_wave_period': reading.get('avg_wave_period', None),
                        'mean_wave_direction_deg': reading.get('mean_wave_direction', {}).get('degree', None),
                        'barometer': reading.get('barometer', None),
                        'air_temp_c': reading.get('air_temp', {}).get('celsius', None),
                        'water_temp_c': reading.get('water_temp', {}).get('celsius', None),
                        'dewpoint_c': reading.get('dewpoint', {}).get('celsius', None),
                    }
                    data_list.append(record)

                # Be nice to NOAA servers
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching hourly data for buoy {buoy_id}: {e}")
                continue

        return pd.DataFrame(data_list)

    def load_from_database(
        self,
        buoy_ids: Optional[List[str]] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load historical data from database.

        Args:
            buoy_ids: List of buoy IDs to filter (None = all)
            start_timestamp: Start time filter (Unix timestamp)
            end_timestamp: End time filter (Unix timestamp)

        Returns:
            DataFrame with historical readings
        """
        query = self.session.query(BuoyReading)

        if buoy_ids:
            query = query.filter(BuoyReading.buoy_id.in_(buoy_ids))

        if start_timestamp:
            query = query.filter(BuoyReading.reading_time >= start_timestamp)

        if end_timestamp:
            query = query.filter(BuoyReading.reading_time <= end_timestamp)

        results = query.all()

        data_list = []
        for result in results:
            data_list.append({
                'buoy_id': result.buoy_id,
                'timestamp': result.reading_time,
                'wind_direction_deg': result.wind_dir,
                'wind_speed': result.wind_spd,
                'wave_height_m': result.wave_height,
                'water_temp_c': result.water_temp,
            })

        return pd.DataFrame(data_list)

    def collect_training_dataset(
        self,
        buoy_ids: List[str],
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Collect a comprehensive training dataset.

        This collects both current real-time data and recent hourly data
        to build a training dataset for ML models.

        Args:
            buoy_ids: List of buoy station IDs
            days_back: Number of days of historical data to collect

        Returns:
            Combined DataFrame ready for training
        """
        all_data = []

        # Collect current real-time data
        logger.info("Collecting current real-time data...")
        current_data = self.collect_current_data(buoy_ids)
        all_data.append(current_data)

        # Collect hourly data for past days
        logger.info(f"Collecting hourly data for past {days_back} days...")
        for day in range(days_back):
            # Collect data for all 24 hours
            hourly_data = self.collect_hourly_data(buoy_ids, 1, 24)
            if not hourly_data.empty:
                all_data.append(hourly_data)

            # Sleep between days to be respectful
            if day < days_back - 1:
                time.sleep(2)

        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Remove duplicates based on buoy_id and timestamp
            combined = combined.drop_duplicates(subset=['buoy_id', 'timestamp'])

            # Sort by timestamp
            combined = combined.sort_values('timestamp')

            logger.info(f"Collected {len(combined)} total readings")
            return combined

        return pd.DataFrame()

    def close(self):
        """Close database connection."""
        self.session.close()
