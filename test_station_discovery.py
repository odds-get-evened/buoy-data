#!/usr/bin/env python
"""Test script to verify station discovery functionality."""

from buoy_data import get_available_stations, filter_stations_by_region

print("="*70)
print("BUOY STATION DISCOVERY TEST")
print("="*70)

# Test 1: Get all available stations
print("\nTest 1: Fetching all available stations...")
try:
    all_stations = get_available_stations()
    print(f"✓ Found {len(all_stations)} active stations")
    print(f"First 10: {all_stations[:10]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Filter by Northeast region
print("\nTest 2: Filtering Northeast stations...")
try:
    northeast = filter_stations_by_region(all_stations, 'northeast')
    print(f"✓ Found {len(northeast)} Northeast stations")
    print(f"Northeast stations: {northeast[:10]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Filter by Pacific region
print("\nTest 3: Filtering Pacific stations...")
try:
    pacific = filter_stations_by_region(all_stations, 'pacific')
    print(f"✓ Found {len(pacific)} Pacific stations")
    print(f"Pacific stations: {pacific[:10]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Filter by Caribbean region
print("\nTest 4: Filtering Caribbean stations...")
try:
    caribbean = filter_stations_by_region(all_stations, 'caribbean')
    print(f"✓ Found {len(caribbean)} Caribbean stations")
    print(f"Caribbean stations: {caribbean[:10]}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*70)
print("All tests completed!")
print("="*70)
