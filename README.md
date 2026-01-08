# buoy-data

A Python package that aggregates and retrieves data from NOAA's National Data Buoy Center (NDBC).

## Overview

This package provides a simple interface to fetch real-time and historical buoy data from NOAA's National Data Buoy Center. It supports:

- Real-time buoy readings
- Hourly historical data
- Station metadata and information
- Database storage with SQLAlchemy
- Automatic unit conversions (metric/imperial)

## Installation

### From source

```bash
git clone https://github.com/odds-get-evened/buoy-data.git
cd buoy-data
pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- requests >= 2.31.0
- sqlalchemy >= 2.0.0
- pandas >= 2.0.0 (for ML features)
- numpy >= 1.24.0 (for ML features)
- scikit-learn >= 1.3.0 (for ML features)

## Quick Start

### Fetch Real-Time Data

```python
from buoy_data import BuoyRealTime

# Get real-time data for buoy 44017 (Montauk Point)
buoy = BuoyRealTime('44017')
data = buoy.get_data()

print(f"Wind Speed: {data['wind_speed']} knots")
print(f"Wave Height: {data['wave_height']['stnd']} feet")
print(f"Water Temperature: {data['water_temp']['fahr']:.1f}°F")
```

### Fetch Hourly Data

```python
from buoy_data import BuoyHourly

# Get hourly data from 9 AM to 12 PM
buoy = BuoyHourly('44017', 9, 12)
readings = buoy.get_data()

for reading in readings:
    print(f"Time: {reading['hour']}:{reading['min']}")
    print(f"Wind Speed: {reading['wind_speed']} knots")
```

### Access Station Information

```python
from buoy_data import Station

station = Station('44017')
print(f"Location: {station.get_description()}")
print(f"Coordinates: {station.get_latitude()}, {station.get_longitude()}")
print(f"Depth: {station.get_depth()} meters")
```

### Store Data in Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from buoy_data import BuoyRealTime
from buoy_data.database import BuoyDataDB

# Setup database
engine = create_engine('sqlite:///buoy_data.db')
BuoyDataDB.create_tables(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Fetch and store data
buoy = BuoyRealTime('44017')
data = buoy.get_data()

db = BuoyDataDB(session)
db.log_data(
    buoy_id='44017',
    wind_dir=data['wind_direction']['degree'] or 0,
    wind_spd=data['wind_speed'] or 0.0,
    wave_height=data['wave_height']['metric'] or 0.0,
    water_temp=data['water_temp']['celsius'] or 0.0,
    reading_time=int(data['timestamp'])
)

session.close()
```

## API Reference

### BuoyRealTime

Retrieves real-time data from NOAA's data feeds.

**Constructor:**
- `BuoyRealTime(buoy_id: str)` - Initialize with a buoy station ID

**Methods:**
- `get_data()` - Returns dictionary with current readings

### BuoyHourly

Retrieves historical hourly data.

**Constructor:**
- `BuoyHourly(buoy_id: str, hr_from: int = 0, hr_to: int = 23)` - Initialize with station ID and hour range (0-23)

**Methods:**
- `get_data()` - Returns list of dictionaries with hourly readings

### Station

Provides station metadata and location information.

**Constructor:**
- `Station(station_id: str)` - Initialize with a station ID

**Methods:**
- `get_latitude()` - Returns latitude
- `get_longitude()` - Returns longitude
- `get_description()` - Returns location description
- `get_shore_distance()` - Returns distance from shore (nautical miles)
- `get_shore_direction()` - Returns direction from shore (degrees)
- `get_depth()` - Returns water depth (meters)

### BuoyDataDB

Database interface for storing buoy readings.

**Constructor:**
- `BuoyDataDB(db_session: Session)` - Initialize with SQLAlchemy session

**Methods:**
- `insert_buoy_data(...)` - Insert a new reading
- `log_data(...)` - Insert reading if not already logged
- `is_logged(buoy_id, timestamp)` - Check if reading exists
- `get_latest_reading(buoy_id)` - Get most recent reading

## Data Format

Reading dictionaries contain:

```python
{
    'year': str,
    'month': str,
    'day': str,
    'hour': str,
    'min': str,
    'timestamp': int,  # Unix timestamp
    'wind_direction': {'degree': int, 'compass': str},
    'wind_speed': float,  # knots
    'gusts': float,  # knots
    'wave_height': {'metric': float, 'stnd': float},  # meters/feet
    'dominant_wave_period': float,  # seconds
    'avg_wave_period': float,  # seconds
    'mean_wave_direction': {'degree': int, 'compass': str},
    'barometer': float,  # millibars
    'air_temp': {'celsius': float, 'fahr': float},
    'water_temp': {'celsius': float, 'fahr': float},
    'dewpoint': {'celsius': float, 'fahr': float},
    'visibility': float,  # miles
    'pressure_tendency': float,
    'tide': float
}
```

## Examples

See `examples.py` for complete working examples including:
- Real-time data retrieval
- Hourly data retrieval
- Station information access
- Database storage and retrieval

Run examples:
```bash
python examples.py
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Structure

```
buoy_data/
├── __init__.py           # Package initialization
├── buoy_data.py          # Base class
├── buoy_real_time.py     # Real-time data retrieval
├── buoy_hourly.py        # Hourly data retrieval
├── station.py            # Station information
├── stations_data.py      # Station database
├── conversions.py        # Unit conversion utilities
├── database.py           # Database models and operations
└── ml/                   # Machine learning module
    ├── __init__.py       # ML module initialization
    ├── data_collector.py # Historical data collection
    ├── feature_engineering.py  # Feature creation
    ├── wave_predictor.py # ML models
    └── forecaster.py     # High-level forecasting interface
```

## License

GNU General Public License (GPL)

## Credits

Originally authored by C.J. Walsh (cj@odewebdesigns.com)
Rewritten in Python for modern usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Machine Learning Features

### Wave Height Forecasting

The package includes ML models for forecasting wave heights between buoy stations using ensemble methods (Random Forest, Gradient Boosting).

#### Quick Start - ML

**Train a model:**
```bash
python train_model.py --buoys 44017 44008 44013 44025 --days 7
```

**Make predictions:**
```bash
python predict.py --buoys 44017 44008 --model models/wave_predictor.pkl --mode forecast
```

#### Python API

**Train a forecasting model:**
```python
from buoy_data.ml import BuoyForecaster

# Initialize and train
forecaster = BuoyForecaster()
metrics = forecaster.train(
    buoy_ids=['44017', '44008', '44013', '44025'],
    days_back=7,
    model_type='random_forest',
    save_path='models/wave_predictor.pkl'
)

print(f"Model RMSE: {metrics['test_rmse']:.3f}m")
print(f"Model R²: {metrics['test_r2']:.3f}")
```

**Query current readings:**
```python
from buoy_data.ml import BuoyForecaster

forecaster = BuoyForecaster()
readings = forecaster.get_current_readings(['44017', '44008'])
print(readings)
```

**Forecast wave heights:**
```python
from buoy_data.ml import BuoyForecaster

forecaster = BuoyForecaster(model_path='models/wave_predictor.pkl')

# Forecast for multiple buoys
forecast = forecaster.forecast_between_buoys(['44017', '44008', '44013'])
print(forecast[['buoy_id', 'predicted_wave_height_m', 'predicted_wave_height_ft']])

# Get regional summary
summary = forecaster.get_regional_summary(['44017', '44008', '44013'])
print(f"Mean wave height: {summary['predicted']['mean_wave_height_m']:.2f}m")
print(f"Max wave height: {summary['predicted']['max_wave_height_m']:.2f}m")
```

**Single buoy prediction:**
```python
result = forecaster.predict_wave_height('44017')
print(f"Predicted: {result['predicted_wave_height_m']:.2f}m")
print(f"95% CI: [{result['confidence']['lower_95']:.2f}, {result['confidence']['upper_95']:.2f}]m")
```

### ML Features

The forecasting model uses:
- **Spatial features**: Buoy locations, depth, inter-buoy distances
- **Temporal features**: Hour of day, day of year (cyclical encoding)
- **Meteorological features**: Wind speed/direction, barometric pressure, temperature
- **Lag features**: Historical values (1h, 3h, 6h, 12h, 24h)
- **Rolling statistics**: Moving averages and standard deviations
- **Inter-buoy relationships**: Weighted averages from nearby buoys

### ML API Reference

#### BuoyForecaster

High-level interface for wave height forecasting.

**Methods:**
- `train(buoy_ids, days_back, model_type, save_path)` - Train a new model
- `get_current_readings(buoy_ids)` - Get real-time buoy data
- `predict_wave_height(buoy_id, current_conditions)` - Predict single buoy
- `forecast_between_buoys(buoy_ids)` - Forecast multiple buoys
- `get_regional_summary(buoy_ids)` - Get regional statistics

#### DataCollector

Collects historical and real-time data for training.

**Methods:**
- `collect_current_data(buoy_ids)` - Fetch current readings
- `collect_hourly_data(buoy_ids, hr_from, hr_to)` - Fetch hourly data
- `collect_training_dataset(buoy_ids, days_back)` - Build training dataset
- `load_from_database(buoy_ids, start_timestamp, end_timestamp)` - Load stored data

#### WaveHeightPredictor

ML model for wave height prediction.

**Methods:**
- `train(df, target_col, test_size, cv_folds)` - Train the model
- `predict(df)` - Make predictions
- `predict_with_confidence(df)` - Predictions with confidence intervals
- `save(filepath)` - Save trained model
- `load(filepath)` - Load trained model

### ML Examples

See `examples_ml.py` for complete examples:
```bash
python examples_ml.py
```

## Links

- [NOAA National Data Buoy Center](https://www.ndbc.noaa.gov/)
- [NDBC Data Documentation](https://www.ndbc.noaa.gov/docs/)
