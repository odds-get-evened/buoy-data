"""Utility functions for buoy data operations."""

import re
import logging
import requests
import math
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

        # Skip stations without coordinates
        if not station_data.get('lat') or not station_data.get('long'):
            continue

        station_lat = station_data['lat']
        station_lon = station_data['long']

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
