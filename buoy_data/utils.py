"""Utility functions for buoy data operations."""

import re
import logging
import requests
import math
import numpy as np
from typing import List, Optional, Union, Any, Dict, Tuple

logger = logging.getLogger(__name__)


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Converted integer or default value

    Example:
        >>> safe_int("123")
        123
        >>> safe_int("invalid", default=-1)
        -1
        >>> safe_int(None, default=0)
        0
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug(f"Failed to convert {value!r} to int, using default {default}")
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Converted float or default value

    Example:
        >>> safe_float("123.45")
        123.45
        >>> safe_float("invalid", default=-1.0)
        -1.0
        >>> safe_float(None, default=0.0)
        0.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug(f"Failed to convert {value!r} to float, using default {default}")
        return default


def safe_str(value: Any, default: str = "") -> str:
    """
    Safely convert a value to string.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Converted string or default value
    """
    if value is None:
        return default
    try:
        return str(value)
    except (TypeError, ValueError):
        logger.debug(f"Failed to convert {value!r} to str, using default {default!r}")
        return default


def get_available_stations(timeout: int = 10) -> List[str]:
    """
    Fetch list of all available buoy stations from NOAA's realtime2 directory.

    This function scrapes the NOAA realtime2 directory to find all active buoy
    stations that currently have data available.

    Args:
        timeout: Request timeout in seconds (default: 10)

    Returns:
        List of station IDs (e.g., ['44017', '44008', '44013'])

    Raises:
        requests.RequestException: If the request fails

    Example:
        >>> stations = get_available_stations()
        >>> print(f"Found {len(stations)} active stations")
        >>> print(stations[:5])  # Show first 5
    """
    url = 'https://www.ndbc.noaa.gov/data/realtime2/'

    logger.info(f"Fetching available stations from {url}")

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Parse HTML to extract .txt file links
        # Pattern matches links like: <a href="44017.txt">
        pattern = r'<a href="([^"]+\.txt)">'
        matches = re.findall(pattern, response.text)

        # Extract station IDs (filename without .txt extension)
        stations = []
        for match in matches:
            # Remove .txt extension
            station_id = match.replace('.txt', '')

            # Filter out non-buoy files (like .drift.txt, .spec.txt, etc.)
            # Valid station IDs are typically 5 characters (numeric or alphanumeric)
            if '.' not in station_id and len(station_id) <= 6:
                stations.append(station_id)

        # Sort for consistency
        stations.sort()

        logger.info(f"Found {len(stations)} available stations")

        return stations

    except requests.RequestException as e:
        logger.error(f"Failed to fetch available stations: {e}")
        raise


def filter_stations_by_region(
    stations: List[str],
    region: Optional[str] = None
) -> List[str]:
    """
    Filter stations by geographic region based on station ID prefix.

    Station ID prefixes indicate geographic regions:
    - 41xxx: Southwest North Atlantic (Caribbean)
    - 42xxx: Southeast North Atlantic
    - 44xxx: Northeast North Atlantic
    - 45xxx: Great Lakes
    - 46xxx: Northeast Pacific
    - 51xxx: Southwest Pacific (Hawaii)

    Args:
        stations: List of station IDs
        region: Region code ('northeast', 'southeast', 'caribbean', 'pacific',
                'greatlakes', 'hawaii') or None for all

    Returns:
        Filtered list of station IDs

    Example:
        >>> all_stations = get_available_stations()
        >>> ne_stations = filter_stations_by_region(all_stations, 'northeast')
        >>> print(f"Found {len(ne_stations)} Northeast Atlantic stations")
    """
    if not region:
        return stations

    region_prefixes = {
        'caribbean': ['41'],
        'southeast': ['42'],
        'northeast': ['44'],
        'greatlakes': ['45'],
        'pacific': ['46'],
        'hawaii': ['51']
    }

    region_lower = region.lower()

    if region_lower not in region_prefixes:
        logger.warning(f"Unknown region '{region}'. Valid regions: {list(region_prefixes.keys())}")
        return stations

    prefixes = region_prefixes[region_lower]
    filtered = [s for s in stations if any(s.startswith(p) for p in prefixes)]

    logger.info(f"Filtered to {len(filtered)} stations in {region} region")

    return filtered


def validate_station_ids(station_ids: List[str]) -> List[str]:
    """
    Validate and clean a list of station IDs.

    Args:
        station_ids: List of station IDs to validate

    Returns:
        List of valid station IDs
    """
    valid_stations = []

    for station_id in station_ids:
        # Clean up whitespace
        station_id = station_id.strip()

        # Basic validation
        if station_id and len(station_id) <= 6:
            valid_stations.append(station_id)
        else:
            logger.warning(f"Skipping invalid station ID: {station_id}")

    return valid_stations


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in meters

    Example:
        >>> # Distance from New York to Boston (approx)
        >>> dist = haversine_distance(40.7128, -74.0060, 42.3601, -71.0589)
        >>> print(f"{dist/1000:.1f} km")  # Should be ~300 km
    """
    # Earth's radius in meters
    R = 6371000.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance


def find_stations_by_location(
    center_lat: float,
    center_lon: float,
    radius_meters: float,
    station_ids: Optional[List[str]] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Find buoy stations within a specified radius from a center point.

    Args:
        center_lat: Latitude of center point in degrees
        center_lon: Longitude of center point in degrees
        radius_meters: Search radius in meters
        station_ids: Optional list of station IDs to search within.
                    If None, searches all stations in STATIONS database.

    Returns:
        List of dictionaries containing station_id and distance, sorted by distance.
        Example: [{'station_id': '44017', 'distance': 15000.0}, ...]

    Raises:
        ValueError: If center coordinates are invalid

    Example:
        >>> # Find all buoys within 100km of New York City
        >>> nearby = find_stations_by_location(40.7128, -74.0060, 100000)
        >>> for station in nearby:
        ...     print(f"{station['station_id']}: {station['distance']/1000:.1f}km")
    """
    from .stations_data import STATIONS

    # Validate coordinates
    if not (-90 <= center_lat <= 90):
        raise ValueError(f"Invalid latitude: {center_lat}. Must be between -90 and 90.")
    if not (-180 <= center_lon <= 180):
        raise ValueError(f"Invalid longitude: {center_lon}. Must be between -180 and 180.")
    if radius_meters <= 0:
        raise ValueError(f"Invalid radius: {radius_meters}. Must be positive.")

    # Determine which stations to search
    if station_ids is None:
        search_stations = list(STATIONS.keys())
    else:
        search_stations = station_ids

    nearby_stations = []

    for station_id in search_stations:
        if station_id not in STATIONS:
            logger.debug(f"Station {station_id} not found in STATIONS database")
            continue

        station_data = STATIONS[station_id]

        # Skip stations without valid coordinates
        station_lat = station_data.get('lat')
        station_lon = station_data.get('long')
        if station_lat is None or station_lon is None:
            continue

        # Calculate distance
        distance = haversine_distance(center_lat, center_lon, station_lat, station_lon)

        # Check if within radius
        if distance <= radius_meters:
            nearby_stations.append({
                'station_id': station_id,
                'distance': distance,
                'latitude': station_lat,
                'longitude': station_lon,
                'location': station_data.get('location', '')
            })

    # Sort by distance (closest first)
    nearby_stations.sort(key=lambda x: x['distance'])

    logger.info(
        f"Found {len(nearby_stations)} stations within {radius_meters/1000:.1f}km "
        f"of ({center_lat}, {center_lon})"
    )

    return nearby_stations


def calculate_wave_energy_density(
    wave_height: float,
    wave_period: float,
    water_density: float = 1025.0,
    gravity: float = 9.81
) -> float:
    """
    Calculate wave energy density (energy per unit area) for ocean waves.
    
    Based on linear wave theory, the energy density of a wave is:
    E = (1/8) * ρ * g * H²
    
    Where:
    - ρ (rho) is water density (kg/m³)
    - g is gravitational acceleration (m/s²)
    - H is wave height (m)
    
    Args:
        wave_height: Significant wave height in meters
        wave_period: Wave period in seconds (used for validation)
        water_density: Seawater density in kg/m³ (default: 1025 kg/m³)
        gravity: Gravitational acceleration in m/s² (default: 9.81 m/s²)
    
    Returns:
        Wave energy density in Joules per square meter (J/m²)
        Returns 0.0 if wave_height is invalid or negative
    
    Example:
        >>> # 2m wave with 8s period
        >>> energy = calculate_wave_energy_density(2.0, 8.0)
        >>> print(f"{energy:.2f} J/m²")
    """
    if wave_height is None or wave_height <= 0 or wave_period is None or wave_period <= 0:
        return 0.0
    
    # Wave energy density formula: E = (1/8) * ρ * g * H²
    energy_density = (1.0 / 8.0) * water_density * gravity * (wave_height ** 2)
    
    return energy_density


def calculate_wave_power(
    wave_height: float,
    wave_period: float,
    water_density: float = 1025.0,
    gravity: float = 9.81
) -> float:
    """
    Calculate wave power (energy flux) per unit width of wave crest.
    
    Wave power represents the rate of energy transport and is given by:
    P = E * Cg
    
    Where:
    - E is wave energy density
    - Cg is group velocity (for deep water: Cg ≈ gT/4π)
    
    This represents the power available for extraction or the energy differential
    that drives wave motion (relevant to Carnot's principle of energy flow).
    
    Args:
        wave_height: Significant wave height in meters
        wave_period: Wave period in seconds
        water_density: Seawater density in kg/m³ (default: 1025 kg/m³)
        gravity: Gravitational acceleration in m/s² (default: 9.81 m/s²)
    
    Returns:
        Wave power in Watts per meter of wave crest (W/m)
        Returns 0.0 if inputs are invalid
    
    Example:
        >>> # 2m wave with 8s period
        >>> power = calculate_wave_power(2.0, 8.0)
        >>> print(f"{power:.2f} W/m")
    """
    if wave_height is None or wave_height <= 0 or wave_period is None or wave_period <= 0:
        return 0.0
    
    # Calculate energy density
    energy_density = calculate_wave_energy_density(wave_height, wave_period, water_density, gravity)
    
    # Calculate group velocity for deep water waves: Cg = gT/(4π)
    group_velocity = (gravity * wave_period) / (4 * math.pi)
    
    # Wave power: P = E * Cg
    wave_power = energy_density * group_velocity
    
    return wave_power


def calculate_spatial_gradient(
    value1: float,
    value2: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> Dict[str, float]:
    """
    Calculate the spatial gradient between two points.
    
    The gradient represents the rate of change of a value (e.g., wave height,
    energy density) per unit distance. This is useful for identifying energy
    differentials that drive wave dynamics (Carnot's principle: energy flows
    from high to low potential).
    
    Args:
        value1: Value at first location (e.g., wave height, energy)
        value2: Value at second location
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
    
    Returns:
        Dictionary containing:
        - 'gradient': Value change per kilometer (value_units/km)
        - 'distance_km': Distance between points in kilometers
        - 'value_diff': Absolute difference in values
        - 'direction': Direction of gradient ('increasing' or 'decreasing')
    
    Example:
        >>> # Wave height gradient between two buoys
        >>> gradient = calculate_spatial_gradient(
        ...     2.5, 1.8, 40.0, -74.0, 41.0, -73.0
        ... )
        >>> print(f"Gradient: {gradient['gradient']:.4f} m/km")
    """
    if any(v is None for v in [value1, value2, lat1, lon1, lat2, lon2]):
        return {
            'gradient': 0.0,
            'distance_km': 0.0,
            'value_diff': 0.0,
            'direction': 'none'
        }
    
    # Calculate distance
    distance_m = haversine_distance(lat1, lon1, lat2, lon2)
    distance_km = distance_m / 1000.0
    
    if distance_km == 0:
        return {
            'gradient': 0.0,
            'distance_km': 0.0,
            'value_diff': 0.0,
            'direction': 'none'
        }
    
    # Calculate value difference and gradient
    value_diff = value2 - value1
    gradient = value_diff / distance_km
    
    # Determine direction
    direction = 'increasing' if value_diff > 0 else ('decreasing' if value_diff < 0 else 'constant')
    
    return {
        'gradient': gradient,
        'distance_km': distance_km,
        'value_diff': abs(value_diff),
        'direction': direction
    }


def identify_significant_gradients(
    stations_data: List[Dict[str, Any]],
    threshold_percentile: float = 75.0
) -> List[Dict[str, Any]]:
    """
    Identify significant energy or wave height gradients between stations.
    
    This function analyzes gradients between all pairs of stations and identifies
    those with significant energy differentials, which indicate areas where wave
    dynamics are most active (following Carnot's principle of energy flow).
    
    Args:
        stations_data: List of station dictionaries, each containing:
            - 'station_id': Station identifier
            - 'latitude': Station latitude
            - 'longitude': Station longitude
            - 'wave_height_m': Wave height in meters
            - 'wave_period': Wave period in seconds (optional, for energy calc)
        threshold_percentile: Percentile threshold for significance (default: 75)
                            Gradients above this percentile are considered significant
    
    Returns:
        List of dictionaries describing significant gradients, sorted by magnitude.
        Each dictionary contains:
        - 'station1': First station ID
        - 'station2': Second station ID
        - 'gradient': Gradient value
        - 'distance_km': Distance between stations
        - 'value_diff': Absolute difference in values
        - 'direction': Direction of energy flow
        - 'energy_differential': If wave_period available, energy flux difference (W/m)
    
    Example:
        >>> stations = [
        ...     {'station_id': '44017', 'latitude': 40.7, 'longitude': -72.0,
        ...      'wave_height_m': 2.5, 'wave_period': 8.0},
        ...     {'station_id': '44008', 'latitude': 40.5, 'longitude': -69.2,
        ...      'wave_height_m': 1.8, 'wave_period': 7.0}
        ... ]
        >>> gradients = identify_significant_gradients(stations)
    """
    if not stations_data or len(stations_data) < 2:
        logger.warning("Need at least 2 stations to calculate gradients")
        return []
    
    gradients = []
    
    # Calculate gradients for all pairs
    for i in range(len(stations_data)):
        for j in range(i + 1, len(stations_data)):
            station1 = stations_data[i]
            station2 = stations_data[j]
            
            # Skip if required fields are missing
            required_fields = ['station_id', 'latitude', 'longitude', 'wave_height_m']
            if not all(field in station1 and field in station2 for field in required_fields):
                continue
            
            # Convert wave heights to float and skip if invalid
            try:
                height1 = float(station1['wave_height_m']) if station1['wave_height_m'] not in (None, '') else None
                height2 = float(station2['wave_height_m']) if station2['wave_height_m'] not in (None, '') else None
            except (ValueError, TypeError):
                continue
            
            # Skip if wave heights are missing or invalid
            if height1 is None or height2 is None or height1 <= 0 or height2 <= 0:
                continue
            
            # Calculate wave height gradient
            gradient_info = calculate_spatial_gradient(
                height1,
                height2,
                station1['latitude'],
                station1['longitude'],
                station2['latitude'],
                station2['longitude']
            )
            
            gradient_entry = {
                'station1': station1['station_id'],
                'station2': station2['station_id'],
                'gradient': gradient_info['gradient'],
                'gradient_abs': abs(gradient_info['gradient']),
                'distance_km': gradient_info['distance_km'],
                'value_diff': gradient_info['value_diff'],
                'direction': gradient_info['direction'],
                'wave_height_1': height1,
                'wave_height_2': height2
            }
            
            # Calculate energy differential if wave periods are available
            if ('wave_period' in station1 and 'wave_period' in station2 and
                station1['wave_period'] is not None and station2['wave_period'] is not None):
                
                # Convert wave periods to float, handling string values
                try:
                    period1 = float(station1['wave_period']) if station1['wave_period'] not in (None, '') else None
                    period2 = float(station2['wave_period']) if station2['wave_period'] not in (None, '') else None
                except (ValueError, TypeError):
                    period1 = None
                    period2 = None
                
                if period1 is not None and period1 > 0 and period2 is not None and period2 > 0:
                    power1 = calculate_wave_power(height1, period1)
                    power2 = calculate_wave_power(height2, period2)
                    
                    gradient_entry['wave_power_1'] = power1
                    gradient_entry['wave_power_2'] = power2
                    gradient_entry['energy_differential'] = abs(power2 - power1)
                    gradient_entry['energy_flow_direction'] = station1['station_id'] if power1 > power2 else station2['station_id']
            
            gradients.append(gradient_entry)
    
    if not gradients:
        logger.warning("No valid gradients calculated")
        return []
    
    # Calculate threshold based on percentile
    gradient_magnitudes = [g['gradient_abs'] for g in gradients]
    threshold = np.percentile(gradient_magnitudes, threshold_percentile)
    
    # Filter significant gradients
    significant = [g for g in gradients if g['gradient_abs'] >= threshold]
    
    # Sort by magnitude (descending)
    significant.sort(key=lambda x: x['gradient_abs'], reverse=True)
    
    logger.info(
        f"Identified {len(significant)} significant gradients "
        f"(threshold: {threshold:.4f}, {threshold_percentile}th percentile)"
    )
    
    return significant
