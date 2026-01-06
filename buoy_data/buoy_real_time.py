"""Retrieves real-time data from the National Buoy Data Center."""

import re
import time
from typing import Dict, Any
import requests

from .buoy_data import BuoyData
from .conversions import Conversions


class BuoyRealTime(BuoyData):
    """
    Retrieves data from the National Buoy Data Center's real-time data files.

    This class extends BuoyData to fetch and parse the latest buoy readings
    from NOAA's real-time data feeds.
    """

    def __init__(self, buoy_id: str):
        """
        Initialize BuoyRealTime with a station ID.

        Args:
            buoy_id: Buoy station ID number
        """
        super().__init__(buoy_id)
        self._set_file()
        self._set_data()

    def _set_file(self):
        """Set the URL path to the real-time data file."""
        self._data_file = f"{self._data_path}realtime2/{self._buoy_id}.txt"

    def _set_data(self):
        """
        Fetch and parse the data from the NOAA website.

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If data parsing fails
        """
        try:
            response = requests.get(self._data_file, timeout=30)
            response.raise_for_status()
            lines = response.text.split('\n')

            if len(lines) < 2:
                raise ValueError("Insufficient data in response")

            # Parse the second line (first line is header)
            line = re.sub(r'\s+', '|', lines[1])
            parts = line.split('|')

            self._data = self._parse_buoy_data(parts)

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch buoy data: {e}")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse buoy data: {e}")

    def _parse_buoy_data(self, val: list) -> Dict[str, Any]:
        """
        Parse and assign data file's information into readable format.

        Args:
            val: List of buoy data values

        Returns:
            Dictionary of parsed buoy data
        """
        try:
            timestamp = time.mktime((
                int(val[0]),  # year
                int(val[1]),  # month
                int(val[2]),  # day
                int(val[3]),  # hour
                int(val[4]),  # minute
                0,            # second
                0,            # weekday (ignored by mktime)
                0,            # yearday (ignored by mktime)
                -1            # isdst (let mktime determine)
            ))
        except (ValueError, IndexError):
            timestamp = 0

        return {
            'year': val[0] if len(val) > 0 else '',
            'month': val[1] if len(val) > 1 else '',
            'day': val[2] if len(val) > 2 else '',
            'hour': val[3] if len(val) > 3 else '',
            'min': val[4] if len(val) > 4 else '',
            'timestamp': timestamp,
            'wind_direction': Conversions.get_wind_direction(val[5]) if len(val) > 5 else {'degree': '', 'compass': ''},
            'wind_speed': Conversions.get_wind_speed(val[6]) if len(val) > 6 else '',
            'gusts': Conversions.get_wind_speed(val[7]) if len(val) > 7 else '',
            'wave_height': Conversions.get_wave_height(val[8]) if len(val) > 8 else {'metric': '', 'stnd': ''},
            'dominant_wave_period': Conversions.get_wave_period(val[9]) if len(val) > 9 else '',
            'avg_wave_period': Conversions.get_wave_period(val[10]) if len(val) > 10 else '',
            'mean_wave_direction': Conversions.get_mean_wave_direction(val[11]) if len(val) > 11 else {'degree': '', 'compass': ''},
            'barometer': Conversions.get_barometer(val[12]) if len(val) > 12 else '',
            'air_temp': Conversions.get_temp(val[13]) if len(val) > 13 else {'celsius': '', 'fahr': ''},
            'water_temp': Conversions.get_temp(val[14]) if len(val) > 14 else {'celsius': '', 'fahr': ''},
            'dewpoint': Conversions.get_temp(val[15]) if len(val) > 15 else {'celsius': '', 'fahr': ''},
            'visibility': Conversions.get_visibility(val[16]) if len(val) > 16 else '',
            'pressure_tendency': Conversions.get_pressure_trend(val[17]) if len(val) > 17 else '',
            'tide': Conversions.get_tide(val[18].replace('|', '') if len(val) > 18 else 'MM'),
        }

    def get_data(self) -> Dict[str, Any]:
        """
        Get the parsed buoy data.

        Returns:
            Dictionary containing all parsed buoy measurements
        """
        return self._data
