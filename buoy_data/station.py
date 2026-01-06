"""Station information classes for buoy data."""

from typing import Optional, Tuple
from .stations_data import STATIONS


class Station:
    """
    Manages buoy station information.

    Provides access to station metadata including location, depth,
    and geographical information.
    """

    def __init__(self, station_id: str):
        """
        Initialize Station with a station ID.

        Args:
            station_id: The buoy station ID number

        Raises:
            ValueError: If station ID does not exist
        """
        self._station_id = station_id
        self._station_list = STATIONS
        self._station_info = {}
        self._set_station()

    def _set_station(self):
        """
        Set station information from the station list.

        Raises:
            ValueError: If station ID is not found
        """
        if self._station_id not in self._station_list:
            raise ValueError(f"Station {self._station_id} does not exist.")

        self._station_info = self._station_list[self._station_id]

    def get_latitude(self) -> float:
        """Get station latitude."""
        return self._station_info['lat']

    def get_longitude(self) -> float:
        """Get station longitude."""
        return self._station_info['long']

    def get_description(self) -> str:
        """Get station location description."""
        return self._station_info['location']

    def get_shore_distance(self):
        """Get distance from shore in nautical miles."""
        return self._station_info['distance']

    def get_shore_direction(self):
        """Get compass direction from shore."""
        return self._station_info['dir_from']

    def get_depth(self):
        """Get water depth in meters."""
        return self._station_info['depth']


class BuoyDataStation:
    """
    Represents a buoy station with additional metadata.

    This class is designed for database storage and extended station information.
    """

    def __init__(self):
        """Initialize an empty BuoyDataStation."""
        self.station_id: Optional[str] = None
        self.owner_id: Optional[str] = None
        self.ttype: Optional[str] = None
        self.hull: Optional[str] = None
        self.name: Optional[str] = None
        self.payload: Optional[str] = None
        self.location: Optional[Tuple[float, float]] = None
        self.note: Optional[str] = None

    def set_station_id(self, station_id: str):
        """Set the station ID."""
        self.station_id = station_id

    def get_station_id(self) -> Optional[str]:
        """Get the station ID."""
        return self.station_id

    def set_owner_id(self, owner_id: str):
        """Set the owner ID."""
        self.owner_id = owner_id

    def get_owner_id(self) -> Optional[str]:
        """Get the owner ID."""
        return self.owner_id

    def set_ttype(self, ttype: str):
        """Set the station type."""
        self.ttype = ttype

    def get_ttype(self) -> Optional[str]:
        """Get the station type."""
        return self.ttype

    def set_hull(self, hull: str):
        """Set the hull type."""
        self.hull = hull

    def get_hull(self) -> Optional[str]:
        """Get the hull type."""
        return self.hull

    def set_name(self, name: str):
        """Set the station name."""
        self.name = name

    def get_name(self) -> Optional[str]:
        """Get the station name."""
        return self.name

    def set_payload(self, payload: str):
        """Set the payload information."""
        self.payload = payload

    def get_payload(self) -> Optional[str]:
        """Get the payload information."""
        return self.payload

    def set_location(self, coords: Tuple[float, float]):
        """
        Set the location coordinates.

        Args:
            coords: Tuple of (latitude, longitude)
        """
        self.location = coords

    def get_location(self) -> Optional[Tuple[float, float]]:
        """Get the location coordinates as (latitude, longitude)."""
        return self.location

    def set_note(self, note: str):
        """Set a note about the station."""
        self.note = note

    def get_note(self) -> Optional[str]:
        """Get the station note."""
        return self.note
