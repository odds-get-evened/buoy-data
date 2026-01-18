"""
Buoy Data - NOAA National Data Buoy Center aggregator.

A Python package for retrieving and processing data from NOAA's National
Data Buoy Center (NDBC).
"""

from .buoy_data import BuoyData
from .buoy_real_time import BuoyRealTime
from .buoy_hourly import BuoyHourly
from .station import Station, BuoyDataStation
from .database import BuoyDataDB
from .utils import (
    get_available_stations,
    filter_stations_by_region,
    validate_station_ids,
    haversine_distance,
    find_stations_by_location,
)

__version__ = "1.0.0"
__author__ = "C.J. Walsh"
__all__ = [
    "BuoyData",
    "BuoyRealTime",
    "BuoyHourly",
    "Station",
    "BuoyDataStation",
    "BuoyDataDB",
    "get_available_stations",
    "filter_stations_by_region",
    "validate_station_ids",
    "haversine_distance",
    "find_stations_by_location",
]
