"""Tests for gradient and energy differential calculations."""

import unittest
import numpy as np
from buoy_data.utils import (
    calculate_wave_energy_density,
    calculate_wave_power,
    calculate_spatial_gradient,
    identify_significant_gradients
)


class TestWaveEnergyCalculations(unittest.TestCase):
    """Test cases for wave energy and power calculations."""

    def test_wave_energy_density_basic(self):
        """Test basic wave energy density calculation."""
        # 2m wave should have energy density
        energy = calculate_wave_energy_density(2.0, 8.0)
        
        # E = (1/8) * ρ * g * H² = (1/8) * 1025 * 9.81 * 4 = ~5,028 J/m²
        expected = (1/8) * 1025 * 9.81 * 4
        self.assertAlmostEqual(energy, expected, places=1)
    
    def test_wave_energy_density_zero_height(self):
        """Test energy density with zero wave height."""
        energy = calculate_wave_energy_density(0.0, 8.0)
        self.assertEqual(energy, 0.0)
    
    def test_wave_energy_density_negative_height(self):
        """Test energy density with negative wave height."""
        energy = calculate_wave_energy_density(-2.0, 8.0)
        self.assertEqual(energy, 0.0)
    
    def test_wave_energy_density_none_values(self):
        """Test energy density with None values."""
        energy = calculate_wave_energy_density(None, 8.0)
        self.assertEqual(energy, 0.0)
        
        energy = calculate_wave_energy_density(2.0, None)
        self.assertEqual(energy, 0.0)
    
    def test_wave_power_basic(self):
        """Test basic wave power calculation."""
        # 2m wave with 8s period
        power = calculate_wave_power(2.0, 8.0)
        
        # Power should be positive and reasonable for ocean waves
        self.assertGreater(power, 0)
        # Typical ocean wave power is in the range of kW/m
        self.assertLess(power, 100000)  # Less than 100 kW/m
    
    def test_wave_power_larger_waves(self):
        """Test that larger waves produce more power."""
        power_small = calculate_wave_power(1.0, 8.0)
        power_large = calculate_wave_power(3.0, 8.0)
        
        # Larger waves should produce more power
        self.assertGreater(power_large, power_small)
    
    def test_wave_power_period_effect(self):
        """Test that longer period waves produce more power."""
        power_short = calculate_wave_power(2.0, 6.0)
        power_long = calculate_wave_power(2.0, 10.0)
        
        # Longer period should produce more power (higher group velocity)
        self.assertGreater(power_long, power_short)
    
    def test_wave_power_invalid_inputs(self):
        """Test wave power with invalid inputs."""
        self.assertEqual(calculate_wave_power(0.0, 8.0), 0.0)
        self.assertEqual(calculate_wave_power(2.0, 0.0), 0.0)
        self.assertEqual(calculate_wave_power(None, 8.0), 0.0)


class TestSpatialGradient(unittest.TestCase):
    """Test cases for spatial gradient calculations."""

    def test_gradient_increasing(self):
        """Test gradient calculation for increasing values."""
        # Wave height increases from 1m to 2m over ~111km
        gradient = calculate_spatial_gradient(
            1.0, 2.0,  # values
            0.0, 0.0,  # first point (equator, prime meridian)
            0.0, 1.0   # second point (1 degree east)
        )
        
        self.assertGreater(gradient['gradient'], 0)
        self.assertEqual(gradient['direction'], 'increasing')
        self.assertAlmostEqual(gradient['value_diff'], 1.0, places=5)
        # Distance should be approximately 111 km at equator
        self.assertGreater(gradient['distance_km'], 110)
        self.assertLess(gradient['distance_km'], 112)
    
    def test_gradient_decreasing(self):
        """Test gradient calculation for decreasing values."""
        gradient = calculate_spatial_gradient(
            2.0, 1.0,  # values
            0.0, 0.0,
            0.0, 1.0
        )
        
        self.assertLess(gradient['gradient'], 0)
        self.assertEqual(gradient['direction'], 'decreasing')
    
    def test_gradient_constant(self):
        """Test gradient with same values."""
        gradient = calculate_spatial_gradient(
            2.0, 2.0,  # same values
            0.0, 0.0,
            0.0, 1.0
        )
        
        self.assertEqual(gradient['gradient'], 0.0)
        self.assertEqual(gradient['direction'], 'constant')
        self.assertEqual(gradient['value_diff'], 0.0)
    
    def test_gradient_same_location(self):
        """Test gradient at same location."""
        gradient = calculate_spatial_gradient(
            1.0, 2.0,
            0.0, 0.0,
            0.0, 0.0  # same location
        )
        
        self.assertEqual(gradient['gradient'], 0.0)
        self.assertEqual(gradient['distance_km'], 0.0)
        self.assertEqual(gradient['direction'], 'none')
    
    def test_gradient_with_none_values(self):
        """Test gradient calculation with None values."""
        gradient = calculate_spatial_gradient(
            None, 2.0,
            0.0, 0.0,
            0.0, 1.0
        )
        
        self.assertEqual(gradient['gradient'], 0.0)
        self.assertEqual(gradient['direction'], 'none')


class TestIdentifySignificantGradients(unittest.TestCase):
    """Test cases for identifying significant gradients."""

    def test_identify_gradients_basic(self):
        """Test identifying gradients with basic station data."""
        stations = [
            {
                'station_id': 'A',
                'latitude': 40.0,
                'longitude': -74.0,
                'wave_height_m': 2.0,
                'wave_period': 8.0
            },
            {
                'station_id': 'B',
                'latitude': 40.5,
                'longitude': -74.0,
                'wave_height_m': 1.5,
                'wave_period': 7.0
            },
            {
                'station_id': 'C',
                'latitude': 41.0,
                'longitude': -74.0,
                'wave_height_m': 2.5,
                'wave_period': 9.0
            }
        ]
        
        gradients = identify_significant_gradients(stations, threshold_percentile=50)
        
        # Should find at least some gradients
        self.assertGreater(len(gradients), 0)
        
        # Each gradient should have required fields
        for grad in gradients:
            self.assertIn('station1', grad)
            self.assertIn('station2', grad)
            self.assertIn('gradient', grad)
            self.assertIn('distance_km', grad)
            self.assertIn('value_diff', grad)
            self.assertIn('direction', grad)
            self.assertIn('energy_differential', grad)
    
    def test_identify_gradients_sorted(self):
        """Test that gradients are sorted by magnitude."""
        stations = [
            {
                'station_id': 'A',
                'latitude': 40.0,
                'longitude': -74.0,
                'wave_height_m': 1.0,
                'wave_period': 8.0
            },
            {
                'station_id': 'B',
                'latitude': 40.1,
                'longitude': -74.0,
                'wave_height_m': 2.0,
                'wave_period': 8.0
            },
            {
                'station_id': 'C',
                'latitude': 40.2,
                'longitude': -74.0,
                'wave_height_m': 3.0,
                'wave_period': 8.0
            }
        ]
        
        gradients = identify_significant_gradients(stations, threshold_percentile=0)
        
        # Should be sorted by absolute gradient magnitude (descending)
        magnitudes = [g['gradient_abs'] for g in gradients]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
    
    def test_identify_gradients_threshold(self):
        """Test that threshold filtering works."""
        stations = [
            {'station_id': 'A', 'latitude': 40.0, 'longitude': -74.0, 'wave_height_m': 1.0},
            {'station_id': 'B', 'latitude': 40.1, 'longitude': -74.0, 'wave_height_m': 1.5},
            {'station_id': 'C', 'latitude': 40.2, 'longitude': -74.0, 'wave_height_m': 3.0}
        ]
        
        # High threshold should return fewer gradients
        gradients_high = identify_significant_gradients(stations, threshold_percentile=90)
        gradients_low = identify_significant_gradients(stations, threshold_percentile=50)
        
        self.assertLessEqual(len(gradients_high), len(gradients_low))
    
    def test_identify_gradients_insufficient_stations(self):
        """Test with insufficient stations."""
        stations = [
            {'station_id': 'A', 'latitude': 40.0, 'longitude': -74.0, 'wave_height_m': 2.0}
        ]
        
        gradients = identify_significant_gradients(stations)
        self.assertEqual(len(gradients), 0)
    
    def test_identify_gradients_missing_data(self):
        """Test handling of missing wave height data."""
        stations = [
            {'station_id': 'A', 'latitude': 40.0, 'longitude': -74.0, 'wave_height_m': 2.0},
            {'station_id': 'B', 'latitude': 40.5, 'longitude': -74.0, 'wave_height_m': None},
            {'station_id': 'C', 'latitude': 41.0, 'longitude': -74.0, 'wave_height_m': 1.5}
        ]
        
        gradients = identify_significant_gradients(stations, threshold_percentile=0)
        
        # Should only calculate gradient between A and C (both have valid data)
        # Station B should be skipped
        for grad in gradients:
            self.assertNotEqual(grad['station1'], 'B')
            self.assertNotEqual(grad['station2'], 'B')
    
    def test_identify_gradients_string_wave_height(self):
        """Test handling of string wave height values (regression test for type error)."""
        stations = [
            {'station_id': 'A', 'latitude': 40.0, 'longitude': -74.0, 'wave_height_m': '2.0'},
            {'station_id': 'B', 'latitude': 40.5, 'longitude': -74.0, 'wave_height_m': ''},
            {'station_id': 'C', 'latitude': 41.0, 'longitude': -74.0, 'wave_height_m': '1.5'}
        ]
        
        # This should not raise a TypeError when comparing with 0
        # String wave heights should be handled gracefully
        gradients = identify_significant_gradients(stations, threshold_percentile=0)
        
        # Empty string should be treated as invalid, so B should be skipped
        # We should get gradients only between A and C if strings are properly converted
        for grad in gradients:
            self.assertNotEqual(grad['station1'], 'B')
            self.assertNotEqual(grad['station2'], 'B')
    
    def test_identify_gradients_energy_flow(self):
        """Test energy flow direction calculation."""
        stations = [
            {
                'station_id': 'HIGH',
                'latitude': 40.0,
                'longitude': -74.0,
                'wave_height_m': 3.0,
                'wave_period': 10.0
            },
            {
                'station_id': 'LOW',
                'latitude': 40.5,
                'longitude': -74.0,
                'wave_height_m': 1.0,
                'wave_period': 6.0
            }
        ]
        
        gradients = identify_significant_gradients(stations, threshold_percentile=0)
        
        self.assertEqual(len(gradients), 1)
        grad = gradients[0]
        
        # Energy should flow from HIGH to LOW
        self.assertIn('energy_flow_direction', grad)
        self.assertEqual(grad['energy_flow_direction'], 'HIGH')


class TestGradientIntegration(unittest.TestCase):
    """Integration tests for gradient features."""

    def test_energy_differential_calculation(self):
        """Test that energy differentials are calculated correctly."""
        # Two stations with different wave conditions
        stations = [
            {
                'station_id': '44017',
                'latitude': 40.69,
                'longitude': -72.05,
                'wave_height_m': 2.5,
                'wave_period': 8.0
            },
            {
                'station_id': '44008',
                'latitude': 40.50,
                'longitude': -69.25,
                'wave_height_m': 1.8,
                'wave_period': 7.0
            }
        ]
        
        gradients = identify_significant_gradients(stations, threshold_percentile=0)
        
        self.assertEqual(len(gradients), 1)
        grad = gradients[0]
        
        # Verify energy differential is calculated
        self.assertIn('energy_differential', grad)
        self.assertGreater(grad['energy_differential'], 0)
        
        # Verify wave power values
        self.assertIn('wave_power_1', grad)
        self.assertIn('wave_power_2', grad)
        self.assertGreater(grad['wave_power_1'], 0)
        self.assertGreater(grad['wave_power_2'], 0)


if __name__ == '__main__':
    unittest.main()
