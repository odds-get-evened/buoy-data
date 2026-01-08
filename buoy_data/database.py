"""Database storage for buoy data using SQLAlchemy."""

import time
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    BigInteger,
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


class BuoyReading(Base):
    """
    SQLAlchemy model for buoy readings.

    Database schema for storing buoy data readings.
    """

    __tablename__ = 'buoys'

    id = Column(Integer, primary_key=True, autoincrement=True)
    buoy_id = Column(String(50), nullable=False, default='')
    wind_dir = Column(Integer, nullable=False, default=0)
    wind_spd = Column(Float, nullable=False, default=0.0)
    wave_height = Column(Float, default=0.0)
    water_temp = Column(Float, nullable=False, default=0.0)
    reading_time = Column(BigInteger, nullable=False, default=0)
    insert_stamp = Column(BigInteger, nullable=False, default=0)
    update_stamp = Column(BigInteger, nullable=False, default=0)

    def __repr__(self):
        return (
            f"<BuoyReading(buoy_id={self.buoy_id}, "
            f"reading_time={self.reading_time})>"
        )


class BuoyDataDB:
    """
    Stores data from a BuoyData object into a database.

    This class provides methods for inserting and querying buoy data
    using SQLAlchemy.
    """

    def __init__(self, db_session: Session):
        """
        Initialize database handler.

        Args:
            db_session: SQLAlchemy session object
        """
        self._db = db_session
        self._table = 'buoys'

    @staticmethod
    def create_tables(engine):
        """
        Create database tables.

        Args:
            engine: SQLAlchemy engine object
        """
        Base.metadata.create_all(engine)

    def insert_buoy_data(
        self,
        buoy_id: str,
        wind_dir: int,
        wind_spd: float,
        wave_height: float,
        water_temp: float,
        reading_time: int
    ) -> bool:
        """
        Insert buoy data into the database.

        Args:
            buoy_id: Station ID number
            wind_dir: Wind direction reading
            wind_spd: Wind speed reading
            wave_height: Wave height reading
            water_temp: Water temperature reading
            reading_time: Time of reading (Unix timestamp)

        Returns:
            True if insert was successful

        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            reading = BuoyReading(
                buoy_id=buoy_id,
                wind_dir=wind_dir,
                wind_spd=wind_spd,
                wave_height=wave_height,
                water_temp=water_temp,
                reading_time=reading_time,
                insert_stamp=int(time.time())
            )

            self._db.add(reading)
            self._db.commit()
            return True

        except SQLAlchemyError as e:
            self._db.rollback()
            raise SQLAlchemyError(f"Failed to insert buoy data: {e}")

    def is_logged(self, buoy_id: str, reading_time: int) -> bool:
        """
        Check if a reading has already been logged.

        Args:
            buoy_id: Station ID number
            reading_time: Time of reading (Unix timestamp)

        Returns:
            True if reading exists in database

        Raises:
            SQLAlchemyError: If database query fails
        """
        try:
            result = (
                self._db.query(BuoyReading.id)
                .filter(BuoyReading.buoy_id == buoy_id)
                .filter(BuoyReading.reading_time == reading_time)
                .first()
            )

            return result is not None

        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"Failed to check logged status: {e}")

    def log_data(
        self,
        buoy_id: str,
        wind_dir: int,
        wind_spd: float,
        wave_height: float,
        water_temp: float,
        reading_time: int
    ):
        """
        Log data if it hasn't been logged already.

        Checks if the reading exists before inserting to avoid duplicates.

        Args:
            buoy_id: Station ID number
            wind_dir: Wind direction reading
            wind_spd: Wind speed reading
            wave_height: Wave height reading
            water_temp: Water temperature reading
            reading_time: Time of reading (Unix timestamp)
        """
        if not self.is_logged(buoy_id, reading_time):
            self.insert_buoy_data(
                buoy_id,
                wind_dir,
                wind_spd,
                wave_height,
                water_temp,
                reading_time
            )

    def get_latest_reading(self, buoy_id: str) -> Optional[BuoyReading]:
        """
        Get the most recent reading for a buoy.

        Args:
            buoy_id: Station ID number

        Returns:
            BuoyReading object or None if no readings found

        Raises:
            SQLAlchemyError: If database query fails
        """
        try:
            result = (
                self._db.query(BuoyReading)
                .filter(BuoyReading.buoy_id == buoy_id)
                .order_by(BuoyReading.reading_time.desc())
                .first()
            )

            return result

        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"Failed to get latest reading: {e}")
