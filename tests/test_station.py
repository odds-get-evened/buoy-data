"""Tests for station module."""

import pytest
from buoy_data.station import Station, BuoyDataStation


class TestStation:
    """Test suite for Station class."""

    def test_station_initialization(self):
        """Test station initialization with valid ID."""
        station = Station('44017')
        assert station._station_id == '44017'
        assert station.get_latitude() == 40.70
        assert station.get_longitude() == -72.00

    def test_station_invalid_id(self):
        """Test station initialization with invalid ID."""
        with pytest.raises(ValueError, match="does not exist"):
            Station('99999')

    def test_get_description(self):
        """Test getting station description."""
        station = Station('44017')
        assert station.get_description() == 'Montauk Point'

    def test_get_depth(self):
        """Test getting water depth."""
        station = Station('44017')
        assert station.get_depth() == 45.0

    def test_get_shore_distance(self):
        """Test getting shore distance."""
        station = Station('44017')
        assert station.get_shore_distance() == 23

    def test_get_shore_direction(self):
        """Test getting shore direction."""
        station = Station('44017')
        assert station.get_shore_direction() == 180


class TestBuoyDataStation:
    """Test suite for BuoyDataStation class."""

    def test_initialization(self):
        """Test BuoyDataStation initialization."""
        station = BuoyDataStation()
        assert station.station_id is None
        assert station.owner_id is None

    def test_setters_and_getters(self):
        """Test setter and getter methods."""
        station = BuoyDataStation()

        station.set_station_id('44017')
        assert station.get_station_id() == '44017'

        station.set_name('Test Buoy')
        assert station.get_name() == 'Test Buoy'

        station.set_location((40.70, -72.00))
        assert station.get_location() == (40.70, -72.00)

        station.set_note('Test note')
        assert station.get_note() == 'Test note'
