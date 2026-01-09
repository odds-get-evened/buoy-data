"""Utility functions for buoy data operations."""

import re
import logging
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)


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
