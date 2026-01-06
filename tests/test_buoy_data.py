"""Tests for buoy_data module."""

import pytest
from buoy_data.buoy_data import BuoyData
from buoy_data.station import Station


class TestBuoyData:
    """Test suite for BuoyData class."""

    def test_initialization(self):
        """Test BuoyData initialization."""
        buoy = BuoyData('44017')
        assert buoy.get_buoy_id() == '44017'
        assert buoy._data_path == 'https://www.ndbc.noaa.gov/data/'
        assert isinstance(buoy._station_obj, Station)

    def test_set_buoy_id(self):
        """Test setting buoy ID."""
        buoy = BuoyData('44017')
        buoy.set_buoy_id('44008')
        assert buoy.get_buoy_id() == '44008'

    def test_utc_setting(self):
        """Test UTC time setting."""
        buoy = BuoyData('44017')
        assert buoy._utc > 0
        assert buoy._local_time > 0

    def test_utc_to_local_time(self):
        """Test UTC to local time conversion."""
        buoy = BuoyData('44017')
        utc_time = 1000000000
        local_time = buoy.utc_to_local_time(utc_time)
        assert local_time == utc_time + buoy._time_offset

    def test_get_station(self):
        """Test getting station object."""
        buoy = BuoyData('44017')
        station = buoy.get_station()
        assert isinstance(station, Station)
        assert station.get_description() == 'Montauk Point'
