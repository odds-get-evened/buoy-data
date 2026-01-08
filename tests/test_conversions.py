"""Tests for conversions module."""

import pytest
from buoy_data.conversions import Conversions


class TestConversions:
    """Test suite for Conversions class."""

    def test_get_compass_direction(self):
        """Test compass direction conversion."""
        assert Conversions.get_compass_direction(0) == 'North'
        assert Conversions.get_compass_direction(360) == 'North'
        assert Conversions.get_compass_direction(45) == 'Northeast'
        assert Conversions.get_compass_direction(90) == 'East'
        assert Conversions.get_compass_direction(135) == 'Southeast'
        assert Conversions.get_compass_direction(180) == 'South'
        assert Conversions.get_compass_direction(225) == 'Southwest'
        assert Conversions.get_compass_direction(270) == 'West'
        assert Conversions.get_compass_direction(315) == 'Northwest'

    def test_get_wave_height(self):
        """Test wave height conversion."""
        result = Conversions.get_wave_height('2.5')
        assert result['metric'] == 2.5
        assert result['stnd'] == 8.2  # 2.5 * 3.28 = 8.2

        result = Conversions.get_wave_height('MM')
        assert result['metric'] == ''
        assert result['stnd'] == ''

    def test_get_wind_speed(self):
        """Test wind speed parsing."""
        assert Conversions.get_wind_speed('15.5') == 15.5
        assert Conversions.get_wind_speed('MM') == ''

    def test_get_wind_direction(self):
        """Test wind direction parsing."""
        result = Conversions.get_wind_direction('180')
        assert result['degree'] == 180
        assert result['compass'] == 'South'

        result = Conversions.get_wind_direction('MM')
        assert result['degree'] == ''
        assert result['compass'] == ''

    def test_get_temp(self):
        """Test temperature conversion."""
        result = Conversions.get_temp('20')
        assert result['celsius'] == 20.0
        assert result['fahr'] == 68.0  # (20 * 1.8) + 32

        result = Conversions.get_temp('MM')
        assert result['celsius'] == ''
        assert result['fahr'] == ''

    def test_get_barometer(self):
        """Test barometer reading parsing."""
        assert Conversions.get_barometer('1013.5') == 1013.5
        assert Conversions.get_barometer('MM') == ''

    def test_get_visibility(self):
        """Test visibility parsing."""
        assert Conversions.get_visibility('10.5') == 10.5
        assert Conversions.get_visibility('MM') == ''

    def test_get_tide(self):
        """Test tide parsing."""
        assert Conversions.get_tide('2.5') == 2.5
        assert Conversions.get_tide('MM') == ''

    def test_get_pressure_trend(self):
        """Test pressure trend parsing."""
        assert Conversions.get_pressure_trend('0.5') == 0.5
        assert Conversions.get_pressure_trend('MM') == ''
