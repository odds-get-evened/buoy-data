"""Tests for database module."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from buoy_data.database import BuoyDataDB, BuoyReading, Base


class TestBuoyDataDB:
    """Test suite for BuoyDataDB class."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing."""
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    @pytest.fixture
    def buoy_db(self, db_session):
        """Create BuoyDataDB instance."""
        return BuoyDataDB(db_session)

    def test_insert_buoy_data(self, buoy_db):
        """Test inserting buoy data."""
        result = buoy_db.insert_buoy_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )
        assert result is True

    def test_is_logged_true(self, buoy_db):
        """Test checking if data is logged (exists)."""
        buoy_db.insert_buoy_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )

        assert buoy_db.is_logged('44017', 1000000000) is True

    def test_is_logged_false(self, buoy_db):
        """Test checking if data is logged (doesn't exist)."""
        assert buoy_db.is_logged('44017', 1000000000) is False

    def test_log_data_new(self, buoy_db):
        """Test logging new data."""
        buoy_db.log_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )

        assert buoy_db.is_logged('44017', 1000000000) is True

    def test_log_data_duplicate(self, buoy_db, db_session):
        """Test that duplicate data is not logged twice."""
        buoy_db.log_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )

        # Try to log the same data again
        buoy_db.log_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )

        # Should only have one entry
        count = db_session.query(BuoyReading).count()
        assert count == 1

    def test_get_latest_reading(self, buoy_db):
        """Test getting latest reading."""
        # Insert multiple readings
        buoy_db.insert_buoy_data(
            buoy_id='44017',
            wind_dir=180,
            wind_spd=15.5,
            wave_height=2.5,
            water_temp=20.0,
            reading_time=1000000000
        )

        buoy_db.insert_buoy_data(
            buoy_id='44017',
            wind_dir=190,
            wind_spd=16.5,
            wave_height=3.5,
            water_temp=21.0,
            reading_time=1000000100
        )

        latest = buoy_db.get_latest_reading('44017')
        assert latest is not None
        assert latest.reading_time == 1000000100
        assert latest.wind_dir == 190

    def test_get_latest_reading_none(self, buoy_db):
        """Test getting latest reading when none exist."""
        latest = buoy_db.get_latest_reading('99999')
        assert latest is None
