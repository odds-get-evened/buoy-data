"""Base class for reading buoy data from NOAA's National Buoy Data Center."""

import time
from typing import Optional
from .station import Station


class BuoyData:
    """
    Base class for reading buoy data stored in flat text files
    on NOAA's National Buoy Data Center website.

    This class provides core functionality for managing buoy station data,
    time calculations, and database operations.
    """

    def __init__(self, buoy_id: str):
        """
        Initialize BuoyData with a station ID.

        Args:
            buoy_id: Buoy station ID number
        """
        self._buoy_id = buoy_id
        self._data_path = 'https://www.ndbc.noaa.gov/data/'
        self._data_file: Optional[str] = None
        self._data = None
        self._db_obj = None
        self._utc: float = 0
        self._time_offset: int = 0
        self._local_time: float = 0

        # Initialize station object
        self._station_obj = Station(buoy_id)

        # Set time values
        self.set_utc()
        self.set_local_time()

    def set_buoy_id(self, buoy_id: str):
        """
        Set the buoy station ID.

        Args:
            buoy_id: Buoy station ID number
        """
        self._buoy_id = buoy_id

    def get_buoy_id(self) -> str:
        """Get the buoy station ID."""
        return self._buoy_id

    def set_utc(self):
        """Set the UTC timestamp and time offset."""
        self._utc = time.time()
        # Get local time offset in seconds
        # In Python, time.timezone gives offset for standard time
        # We need to check if DST is in effect
        if time.localtime().tm_isdst:
            self._time_offset = -time.altzone
        else:
            self._time_offset = -time.timezone

    def set_local_time(self):
        """Set the local time based on UTC and offset."""
        self._local_time = self._utc + self._time_offset

    def utc_to_local_time(self, timestamp: float) -> float:
        """
        Convert UTC timestamp to local time.

        Args:
            timestamp: UTC timestamp

        Returns:
            Local timestamp
        """
        return timestamp + self._time_offset

    def get_station(self) -> Station:
        """
        Get the Station object.

        Returns:
            Station object for this buoy
        """
        return self._station_obj

    def set_database(self, db_obj):
        """
        Set the database object.

        Args:
            db_obj: Database connection/session object
        """
        from .database import BuoyDataDB
        self._db_obj = BuoyDataDB(db_obj)

    def get_database(self):
        """Get the database object."""
        return self._db_obj
