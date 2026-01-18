"""Tests for geospatial utility functions."""

import unittest
import math
from buoy_data.utils import haversine_distance, find_stations_by_location


class TestHaversineDistance(unittest.TestCase):
    """Test cases for haversine distance calculations."""

    def test_same_location(self):
        """Distance between same coordinates should be zero."""
        distance = haversine_distance(40.0, -74.0, 40.0, -74.0)
        self.assertAlmostEqual(distance, 0.0, places=1)

    def test_known_distance(self):
        """Test with known distance between New York and Boston."""
        # Approximate coordinates
        ny_lat, ny_lon = 40.7128, -74.0060
        boston_lat, boston_lon = 42.3601, -71.0589
        
        # Distance should be approximately 305-310 km
        distance = haversine_distance(ny_lat, ny_lon, boston_lat, boston_lon)
        distance_km = distance / 1000
        
        # Allow 5% margin of error
        self.assertGreater(distance_km, 290)
        self.assertLess(distance_km, 320)

    def test_antipodal_points(self):
        """Test distance between antipodal points (should be ~half Earth's circumference)."""
        # Point and its antipode
        distance = haversine_distance(0, 0, 0, 180)
        earth_circumference = 2 * math.pi * 6371000
        half_circumference = earth_circumference / 2
        
        # Should be approximately half the Earth's circumference
        self.assertAlmostEqual(distance, half_circumference, delta=1000)

    def test_equator_longitude(self):
        """Test distance along equator."""
        # 1 degree longitude at equator is approximately 111 km
        distance = haversine_distance(0, 0, 0, 1)
        distance_km = distance / 1000
        
        self.assertGreater(distance_km, 110)
        self.assertLess(distance_km, 112)


class TestFindStationsByLocation(unittest.TestCase):
    """Test cases for finding stations by location."""

    def test_find_northeast_stations(self):
        """Test finding stations near New York area."""
        # New York coordinates
        ny_lat, ny_lon = 40.7128, -74.0060
        
        # Find stations within 200km (200,000 meters)
        nearby = find_stations_by_location(ny_lat, ny_lon, 200000)
        
        # Should find some stations
        self.assertGreater(len(nearby), 0)
        
        # Results should be sorted by distance
        distances = [s['distance'] for s in nearby]
        self.assertEqual(distances, sorted(distances))
        
        # All distances should be within radius
        for station in nearby:
            self.assertLessEqual(station['distance'], 200000)
            self.assertIn('station_id', station)
            self.assertIn('latitude', station)
            self.assertIn('longitude', station)

    def test_no_stations_in_radius(self):
        """Test location with no stations nearby."""
        # Middle of Sahara Desert
        sahara_lat, sahara_lon = 23.0, 10.0
        
        # Very small radius
        nearby = find_stations_by_location(sahara_lat, sahara_lon, 1000)
        
        # Should find no stations
        self.assertEqual(len(nearby), 0)

    def test_invalid_coordinates(self):
        """Test with invalid latitude/longitude values."""
        with self.assertRaises(ValueError):
            find_stations_by_location(91.0, 0.0, 100000)  # Invalid latitude
        
        with self.assertRaises(ValueError):
            find_stations_by_location(0.0, 181.0, 100000)  # Invalid longitude
        
        with self.assertRaises(ValueError):
            find_stations_by_location(0.0, 0.0, -100)  # Negative radius

    def test_large_radius(self):
        """Test with very large radius."""
        # Boston area with 1000km radius
        boston_lat, boston_lon = 42.3601, -71.0589
        
        nearby = find_stations_by_location(boston_lat, boston_lon, 1000000)
        
        # Should find many stations
        self.assertGreater(len(nearby), 5)

    def test_station_filtering(self):
        """Test filtering specific stations by location."""
        # New York area
        ny_lat, ny_lon = 40.7128, -74.0060
        
        # Search within specific list of stations
        test_stations = ['44017', '44008', '44013', '44025']
        nearby = find_stations_by_location(ny_lat, ny_lon, 200000, station_ids=test_stations)
        
        # All returned stations should be from the test list
        for station in nearby:
            self.assertIn(station['station_id'], test_stations)


if __name__ == '__main__':
    unittest.main()
