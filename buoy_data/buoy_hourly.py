"""Retrieves hourly data from the National Buoy Data Center."""

import re
import time
from typing import List, Dict, Any
import requests

from .buoy_data import BuoyData
from .conversions import Conversions


class BuoyHourly(BuoyData):
    """
    Retrieves hourly data from the National Buoy Data Center.

    This class extends BuoyData to fetch historical hourly readings
    for a specified time range.
    """

    def __init__(self, buoy_id: str, hr_from: int = 0, hr_to: int = 23):
        """
        Initialize BuoyHourly with a station ID and hour range.

        Args:
            buoy_id: Buoy station ID number
            hr_from: Beginning hour in 24-hour time (0-23)
            hr_to: Ending hour in 24-hour time (0-23)

        Raises:
            ValueError: If hr_from >= hr_to
        """
        super().__init__(buoy_id)
        self._hour_from = 0
        self._hour_to = 0
        self._files: List[str] = []
        self._readings: List[Dict[str, Any]] = []

        self._set_time_range(hr_from, hr_to)
        self._set_files()
        self._set_data()
        self._set_readings()

    def _set_time_range(self, hr_from: int, hr_to: int):
        """
        Set the time range of hours to read for the day.

        Args:
            hr_from: Beginning hour (0-23)
            hr_to: Ending hour (0-23)

        Raises:
            ValueError: If hr_from >= hr_to
        """
        if hr_from >= hr_to:
            raise ValueError("hr_from must be less than hr_to")

        self._hour_from = hr_from - 1
        self._hour_to = hr_to - 1

    def _set_files(self):
        """Build list of file URLs for each hour in the range."""
        self._files = []
        for i in range(self._hour_from, self._hour_to + 1):
            hour = f"{i:02d}"
            file_url = f"{self._data_path}hourly2/hour_{hour}.txt"
            self._files.append(file_url)

    def _set_data(self):
        """
        Fetch and parse data from each hourly file.

        This method retrieves data from NOAA's hourly data files and
        extracts the relevant station data.
        """
        self._data = []

        for file_url in self._files:
            try:
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                lines = response.text.split('\n')

                # Skip first line (header) and search for station
                for line in lines[1:]:
                    if not line.strip():
                        continue

                    # Check if this line is for our station
                    station = line[:5].strip()

                    if station == self._buoy_id:
                        # Parse the line into data fields
                        parsed_line = re.sub(r'\s+', '|', line, 19)
                        parts = parsed_line.split('|')
                        self._data.append(parts)
                        break

            except requests.RequestException as e:
                # Log error but continue with other files
                print(f"Warning: Failed to fetch {file_url}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing {file_url}: {e}")
                continue

    def _set_readings(self):
        """Convert raw data into structured readings."""
        self._readings = []
        for data_line in self._data:
            reading = self._parse_reading(data_line)
            self._readings.append(reading)

    def _parse_reading(self, val: list) -> Dict[str, Any]:
        """
        Parse a single reading into structured format.

        Args:
            val: List of data values from one hourly reading

        Returns:
            Dictionary of parsed reading data
        """
        try:
            timestamp = time.mktime((
                int(val[1]),  # year
                int(val[2]),  # month
                int(val[3]),  # day
                int(val[4]),  # hour
                int(val[5]),  # minute
                0,            # second
                0,            # weekday (ignored by mktime)
                0,            # yearday (ignored by mktime)
                -1            # isdst (let mktime determine)
            ))
        except (ValueError, IndexError):
            timestamp = 0

        return {
            'year': val[1] if len(val) > 1 else '',
            'month': val[2] if len(val) > 2 else '',
            'day': val[3] if len(val) > 3 else '',
            'hour': val[4] if len(val) > 4 else '',
            'min': val[5] if len(val) > 5 else '',
            'timestamp': timestamp,
            'wind_direction': Conversions.get_wind_direction(val[6]) if len(val) > 6 else {'degree': '', 'compass': ''},
            'wind_speed': Conversions.get_wind_speed(val[7]) if len(val) > 7 else '',
            'gusts': Conversions.get_wind_speed(val[8]) if len(val) > 8 else '',
            'wave_height': Conversions.get_wave_height(val[9]) if len(val) > 9 else {'metric': '', 'stnd': ''},
            'dominant_wave_period': Conversions.get_wave_period(val[10]) if len(val) > 10 else '',
            'avg_wave_period': Conversions.get_wave_period(val[11]) if len(val) > 11 else '',
            'mean_wave_direction': Conversions.get_mean_wave_direction(val[12]) if len(val) > 12 else {'degree': '', 'compass': ''},
            'barometer': Conversions.get_barometer(val[13]) if len(val) > 13 else '',
            'air_temp': Conversions.get_temp(val[14]) if len(val) > 14 else {'celsius': '', 'fahr': ''},
            'water_temp': Conversions.get_temp(val[15]) if len(val) > 15 else {'celsius': '', 'fahr': ''},
            'dewpoint': Conversions.get_temp(val[16]) if len(val) > 16 else {'celsius': '', 'fahr': ''},
            'visibility': Conversions.get_visibility(val[17]) if len(val) > 17 else '',
            'pressure_tendency': Conversions.get_pressure_trend(val[18]) if len(val) > 18 else '',
            'tide': Conversions.get_tide(val[19].replace('|', '') if len(val) > 19 else 'MM'),
        }

    def get_data(self) -> List[Dict[str, Any]]:
        """
        Get the list of hourly readings.

        Returns:
            List of dictionaries, each containing one hour's data
        """
        return self._readings
