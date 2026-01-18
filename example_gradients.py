#!/usr/bin/env python
"""
Example demonstrating gradient and energy differential analysis.

This script shows how to:
1. Calculate wave energy density and power
2. Compute spatial gradients between buoys
3. Identify significant energy differentials
4. Use the analyze_gradients API
"""

import sys
from buoy_data import (
    calculate_wave_energy_density,
    calculate_wave_power,
    calculate_spatial_gradient,
    identify_significant_gradients
)


def example_wave_energy_calculations():
    """Demonstrate wave energy and power calculations."""
    print("=" * 70)
    print("WAVE ENERGY CALCULATIONS")
    print("=" * 70)
    
    # Example 1: Small wave
    wave_height = 1.5  # meters
    wave_period = 7.0  # seconds
    
    energy = calculate_wave_energy_density(wave_height, wave_period)
    power = calculate_wave_power(wave_height, wave_period)
    
    print(f"\nSmall Wave (H={wave_height}m, T={wave_period}s):")
    print(f"  Energy Density: {energy:.2f} J/m²")
    print(f"  Wave Power: {power:.2f} W/m ({power/1000:.2f} kW/m)")
    
    # Example 2: Large wave
    wave_height = 4.0  # meters
    wave_period = 12.0  # seconds
    
    energy = calculate_wave_energy_density(wave_height, wave_period)
    power = calculate_wave_power(wave_height, wave_period)
    
    print(f"\nLarge Wave (H={wave_height}m, T={wave_period}s):")
    print(f"  Energy Density: {energy:.2f} J/m²")
    print(f"  Wave Power: {power:.2f} W/m ({power/1000:.2f} kW/m)")
    
    print("\nNote: Larger waves carry significantly more energy!")


def example_spatial_gradient():
    """Demonstrate spatial gradient calculation."""
    print("\n" + "=" * 70)
    print("SPATIAL GRADIENT CALCULATION")
    print("=" * 70)
    
    # Two buoys with different wave heights
    buoy1_height = 2.5  # meters
    buoy1_lat, buoy1_lon = 40.7, -72.0
    
    buoy2_height = 1.8  # meters
    buoy2_lat, buoy2_lon = 40.5, -69.2
    
    gradient = calculate_spatial_gradient(
        buoy1_height, buoy2_height,
        buoy1_lat, buoy1_lon,
        buoy2_lat, buoy2_lon
    )
    
    print(f"\nBuoy 1: {buoy1_height}m at ({buoy1_lat}, {buoy1_lon})")
    print(f"Buoy 2: {buoy2_height}m at ({buoy2_lat}, {buoy2_lon})")
    print(f"\nGradient Results:")
    print(f"  Distance: {gradient['distance_km']:.1f} km")
    print(f"  Wave height change: {gradient['value_diff']:.2f} m")
    print(f"  Gradient: {gradient['gradient']:.4f} m/km")
    print(f"  Direction: {gradient['direction']}")
    
    if gradient['direction'] == 'decreasing':
        print(f"\n  → Wave energy is flowing from Buoy 1 toward Buoy 2")


def example_significant_gradients():
    """Demonstrate identifying significant gradients."""
    print("\n" + "=" * 70)
    print("IDENTIFYING SIGNIFICANT GRADIENTS")
    print("=" * 70)
    
    # Simulated buoy data
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
        },
        {
            'station_id': '44013',
            'latitude': 42.35,
            'longitude': -70.65,
            'wave_height_m': 2.1,
            'wave_period': 7.5
        },
        {
            'station_id': '44025',
            'latitude': 40.25,
            'longitude': -73.17,
            'wave_height_m': 3.2,
            'wave_period': 9.0
        }
    ]
    
    print(f"\nAnalyzing {len(stations)} buoy stations...")
    
    gradients = identify_significant_gradients(stations, threshold_percentile=50)
    
    print(f"\nFound {len(gradients)} significant gradients:\n")
    
    for i, grad in enumerate(gradients, 1):
        print(f"{i}. {grad['station1']} ↔ {grad['station2']}")
        print(f"   Wave heights: {grad['wave_height_1']:.2f}m → {grad['wave_height_2']:.2f}m")
        print(f"   Gradient: {grad['gradient']:.4f} m/km over {grad['distance_km']:.1f} km")
        
        if 'energy_differential' in grad:
            print(f"   Energy differential: {grad['energy_differential']:.2f} W/m")
            print(f"   Energy flows from: {grad['energy_flow_direction']}")
        print()
    
    print("Interpretation:")
    print("  - Higher gradients indicate more active wave dynamics")
    print("  - Energy flows from high to low potential (Carnot's principle)")
    print("  - These areas may be good for wave energy extraction")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  WAVE GRADIENT AND ENERGY DIFFERENTIAL ANALYSIS EXAMPLES".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        example_wave_energy_calculations()
        example_spatial_gradient()
        example_significant_gradients()
        
        print("\n" + "=" * 70)
        print("SUCCESS: All examples completed!")
        print("=" * 70)
        print("\nThese features enable:")
        print("  ✓ Wave energy resource assessment")
        print("  ✓ Maritime safety (detecting rapid wave changes)")
        print("  ✓ Understanding wave dynamics and energy transfer")
        print("  ✓ Identifying optimal locations for energy extraction")
        print("\nFor real-time analysis, use:")
        print("  python predict.py --buoys 44017 44008 44013 44025 --mode gradients")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
