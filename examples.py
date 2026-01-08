"""Example usage of the buoy-data package."""

from buoy_data import BuoyRealTime, BuoyHourly, Station
from pprint import pprint


def example_real_time():
    """Example: Fetch real-time data for a buoy."""
    print("=" * 60)
    print("REAL-TIME DATA EXAMPLE")
    print("=" * 60)

    try:
        # Get real-time data for buoy 44017 (Montauk Point)
        buoy = BuoyRealTime('44017')
        data = buoy.get_data()

        print(f"\nReal-time data for Buoy {buoy.get_buoy_id()}:")
        print("-" * 60)
        pprint(data)

        # Access station information
        station = buoy.get_station()
        print(f"\nStation Information:")
        print(f"  Location: {station.get_description()}")
        print(f"  Latitude: {station.get_latitude()}")
        print(f"  Longitude: {station.get_longitude()}")
        print(f"  Depth: {station.get_depth()} meters")

    except Exception as e:
        print(f"Error fetching real-time data: {e}")


def example_hourly():
    """Example: Fetch hourly data for a buoy."""
    print("\n" + "=" * 60)
    print("HOURLY DATA EXAMPLE")
    print("=" * 60)

    try:
        # Get hourly data for buoy 44017 from hour 9 to 12
        buoy = BuoyHourly('44017', 9, 12)
        data = buoy.get_data()

        print(f"\nHourly data for Buoy {buoy.get_buoy_id()} (hours 9-12):")
        print("-" * 60)

        for i, reading in enumerate(data):
            print(f"\nReading {i + 1}:")
            print(f"  Time: {reading['year']}/{reading['month']}/{reading['day']} "
                  f"{reading['hour']}:{reading['min']}")
            print(f"  Wind Speed: {reading['wind_speed']} knots")
            print(f"  Wind Direction: {reading['wind_direction']['compass']} "
                  f"({reading['wind_direction']['degree']}°)")
            print(f"  Wave Height: {reading['wave_height']['stnd']} ft "
                  f"({reading['wave_height']['metric']} m)")
            print(f"  Water Temp: {reading['water_temp']['fahr']:.1f}°F "
                  f"({reading['water_temp']['celsius']:.1f}°C)")

    except Exception as e:
        print(f"Error fetching hourly data: {e}")


def example_station_info():
    """Example: Access station information."""
    print("\n" + "=" * 60)
    print("STATION INFORMATION EXAMPLE")
    print("=" * 60)

    try:
        # Create station object
        station = Station('44017')

        print(f"\nStation 44017 Information:")
        print("-" * 60)
        print(f"  Description: {station.get_description()}")
        print(f"  Latitude: {station.get_latitude()}")
        print(f"  Longitude: {station.get_longitude()}")
        print(f"  Shore Distance: {station.get_shore_distance()} nm")
        print(f"  Shore Direction: {station.get_shore_direction()}°")
        print(f"  Water Depth: {station.get_depth()} meters")

    except Exception as e:
        print(f"Error accessing station info: {e}")


def example_database():
    """Example: Store buoy data in a database."""
    print("\n" + "=" * 60)
    print("DATABASE STORAGE EXAMPLE")
    print("=" * 60)

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from buoy_data.database import BuoyDataDB, BuoyReading

        # Create in-memory SQLite database
        engine = create_engine('sqlite:///:memory:', echo=False)

        # Create tables
        BuoyDataDB.create_tables(engine)

        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create database handler
        db = BuoyDataDB(session)

        # Fetch real-time data
        buoy = BuoyRealTime('44017')
        data = buoy.get_data()

        # Store data in database
        wind_dir = data['wind_direction']['degree'] or 0
        wind_speed = data['wind_speed'] or 0.0
        wave_height = data['wave_height']['metric'] or 0.0
        water_temp = data['water_temp']['celsius'] or 0.0
        timestamp = int(data['timestamp'])

        db.log_data(
            buoy_id='44017',
            wind_dir=wind_dir,
            wind_spd=wind_speed,
            wave_height=wave_height,
            water_temp=water_temp,
            reading_time=timestamp
        )

        print("\nData stored successfully!")

        # Retrieve latest reading
        latest = db.get_latest_reading('44017')
        if latest:
            print(f"\nLatest reading from database:")
            print(f"  Buoy ID: {latest.buoy_id}")
            print(f"  Wind Speed: {latest.wind_spd} knots")
            print(f"  Wave Height: {latest.wave_height} meters")
            print(f"  Water Temp: {latest.water_temp}°C")

        session.close()

    except Exception as e:
        print(f"Error with database operations: {e}")


if __name__ == '__main__':
    # Run all examples
    example_real_time()
    example_hourly()
    example_station_info()
    example_database()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
