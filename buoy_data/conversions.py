"""Conversion utilities for buoy data measurements."""


class Conversions:
    """Utility class for converting and processing buoy data measurements."""

    @staticmethod
    def get_compass_direction(degrees):
        """
        Convert degree direction to compass direction.

        Args:
            degrees: Wind direction in degrees (0-360)

        Returns:
            str: Compass direction (e.g., 'North', 'Northeast')
        """
        if degrees == 0 or degrees == 360:
            return 'North'
        elif 0 < degrees < 90:
            return 'Northeast'
        elif degrees == 90:
            return 'East'
        elif 90 < degrees < 180:
            return 'Southeast'
        elif degrees == 180:
            return 'South'
        elif 180 < degrees < 270:
            return 'Southwest'
        elif degrees == 270:
            return 'West'
        elif 270 < degrees < 360:
            return 'Northwest'
        return ''

    @staticmethod
    def get_wave_height(height):
        """
        Get the height of the waves.

        Args:
            height: Wave height value or 'MM' if not measured

        Returns:
            dict: Dictionary with 'metric' (meters) and 'stnd' (feet) keys
        """
        result = {"metric": "", "stnd": ""}

        if height != "MM":
            try:
                height_float = float(height)
                result['metric'] = height_float
                result['stnd'] = round(3.28 * height_float, 1)
            except (ValueError, TypeError):
                pass

        return result

    @staticmethod
    def get_wind_speed(speed):
        """
        Get the wind speed.

        Args:
            speed: Wind speed value or 'MM' if not measured

        Returns:
            float or str: Wind speed in nautical miles/hour, or empty string if not measured
        """
        if speed != "MM":
            try:
                return float(speed)
            except (ValueError, TypeError):
                pass
        return ''

    @staticmethod
    def get_wind_direction(wind_dir):
        """
        Get the actual wind direction.

        Args:
            wind_dir: Wind direction in degrees or 'MM' if not measured

        Returns:
            dict: Dictionary with 'degree' and 'compass' keys
        """
        result = {'degree': '', 'compass': ''}

        if wind_dir != "MM":
            try:
                wind_dir_int = int(wind_dir)
                result['degree'] = wind_dir_int
                result['compass'] = Conversions.get_compass_direction(wind_dir_int)
            except (ValueError, TypeError):
                pass

        return result

    @staticmethod
    def get_wave_period(period):
        """
        Get the length of the frequency.

        Args:
            period: Wave period value or 'MM' if not measured

        Returns:
            float or str: Wave period, or empty string if not measured
        """
        if period != "MM":
            try:
                return float(period)
            except (ValueError, TypeError):
                pass
        return ''

    @staticmethod
    def get_mean_wave_direction(direction):
        """
        Get the average reported wave direction.

        Args:
            direction: Wave direction in degrees or 'MM' if not measured

        Returns:
            dict: Dictionary with 'degree' and 'compass' keys
        """
        result = {"degree": "", "compass": ""}

        if direction != "MM":
            try:
                dir_int = int(direction)
                result['degree'] = dir_int
                result['compass'] = Conversions.get_compass_direction(dir_int)
            except (ValueError, TypeError):
                pass

        return result

    @staticmethod
    def get_barometer(reading):
        """
        Get air pressure reading.

        Args:
            reading: Barometric pressure in millibars or 'MM' if not measured

        Returns:
            float or str: Pressure reading, or empty string if not measured
        """
        if reading != "MM":
            try:
                return float(reading)
            except (ValueError, TypeError):
                pass
        return ''

    @staticmethod
    def get_temp(reading):
        """
        Get temperature reading.

        Args:
            reading: Temperature in Celsius or 'MM' if not measured

        Returns:
            dict: Dictionary with 'celsius' and 'fahr' keys
        """
        result = {"celsius": "", "fahr": ""}

        if reading != "MM":
            try:
                celsius = float(reading)
                result['celsius'] = celsius
                result['fahr'] = (celsius * 1.8) + 32
            except (ValueError, TypeError):
                pass

        return result

    @staticmethod
    def get_visibility(miles):
        """
        Get the visibility.

        Args:
            miles: Visibility in miles or 'MM' if not measured

        Returns:
            float or str: Visibility in miles, or empty string if not measured
        """
        if miles != "MM":
            try:
                return float(miles)
            except (ValueError, TypeError):
                pass
        return ''

    @staticmethod
    def get_tide(val):
        """
        Get tide height.

        Args:
            val: Reported tide height or 'MM' if not measured

        Returns:
            float or str: Tide height, or empty string if not measured
        """
        if val != "MM":
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return ''

    @staticmethod
    def get_pressure_trend(val):
        """
        Get the pressure trend.

        Args:
            val: Pressure trend value or 'MM' if not measured

        Returns:
            float or str: Trend value, or empty string if not measured
        """
        if val != "MM":
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return ''
