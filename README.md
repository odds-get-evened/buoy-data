# buoy-data

A Python package that aggregates and retrieves data from NOAA's National Data Buoy Center (NDBC).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Data Format](#data-format)
- [Examples](#examples)
- [Development](#development)
- [Machine Learning Features](#machine-learning-features)
  - [Wave Height Forecasting](#wave-height-forecasting)
  - [Complete ML Workflow](#complete-ml-workflow)
  - [ML API Reference](#ml-api-reference)
  - [CLI Tools Reference](#cli-tools-reference)
- [Station Discovery](#station-discovery-and-region-filtering)
- [Contributing](#contributing)
- [License](#license)

## Overview

This package provides a simple interface to fetch real-time and historical buoy data from NOAA's National Data Buoy Center. It supports:

- Real-time buoy readings
- Hourly historical data
- Station metadata and information
- Database storage with SQLAlchemy
- Automatic unit conversions (metric/imperial)
- **Optional Machine Learning features** for wave height forecasting
- Location-based station discovery (find buoys by lat/lon/radius)
- Region-based filtering (Northeast, Southeast, Caribbean, Pacific, Great Lakes, Hawaii)

## Installation

### From source

```bash
git clone https://github.com/odds-get-evened/buoy-data.git
cd buoy-data
pip install -e .
```

For ML features, install with the `ml` extras:

```bash
pip install -e ".[ml]"
```

### Using pip (with requirements files)

Basic installation (core features only):
```bash
pip install -r requirements-core.txt
```

Full installation (including ML features):
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

For core functionality tests:
```bash
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_analyze_model.py
```

For all tests (including ML):
```bash
pip install -e ".[dev,ml]"
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

**Basic Training:**
```bash
python train_model.py --buoys 44017 44008 44013 44025 --days 7
```

**Advanced Training (Recommended for better confidence):**
```bash
# With improved default parameters
python train_model_advanced.py --buoys 44017 44008 44013 44025 44065 44066 --days 14

# With hyperparameter tuning (slower but best results)
python train_model_advanced.py --tune --days 21 --buoys 44017 44008 44013 44025 44065 44066
```

**Analyze Model Performance:**
```bash
python analyze_model.py --model models/wave_predictor.pkl
```

**Make Predictions:**
```bash
# Current readings
python predict.py --buoys 44017 44008 --model models/wave_predictor.pkl --mode current

# Wave height forecast
python predict.py --buoys 44017 44008 44013 --model models/wave_predictor.pkl --mode forecast

# Regional summary
python predict.py --buoys 44017 44008 44013 --model models/wave_predictor.pkl --mode summary
```

### Complete ML Workflow

#### 1. Train Your Model

**Option A: Basic Training (Quick)**
```bash
# Train with default parameters (7 days, 4 buoys)
python train_model.py --buoys 44017 44008 44013 44025 --days 7
```

**Option B: Advanced Training (Recommended)**
```bash
# Train with improved parameters (14 days, 6 buoys)
python train_model_advanced.py \
    --buoys 44017 44008 44013 44025 44065 44066 \
    --days 14 \
    --output models/wave_predictor_optimized.pkl
```

**Option C: Hyperparameter Tuning (Best Quality)**
```bash
# Optimize model with RandomizedSearchCV (takes longer)
python train_model_advanced.py \
    --tune \
    --days 21 \
    --buoys 44017 44008 44013 44025 44065 44066 \
    --output models/wave_predictor_tuned.pkl
```

#### 2. Analyze Model Performance

After training, analyze your model to understand its performance:

```bash
python analyze_model.py --model models/wave_predictor.pkl
```

This will show:
- **Model confidence** (R² score converted to percentage)
- **Error metrics** (RMSE, MAE, R²)
- **Feature importance** (top 15 features)
- **Actionable recommendations** based on confidence level

**Example output:**
```
======================================================================
MODEL DIAGNOSTICS
======================================================================

Model Type: random_forest
Number of features: 72

Performance Metrics:
  Train RMSE: 0.215m
  Test RMSE:  0.342m
  Test MAE:   0.267m
  Test R²:    0.878
  CV RMSE:    0.351 ± 0.045m

✓ Model Confidence: 87.8%
  (Based on R² = 0.878)

Top 15 Most Important Features:
  wave_height_m_lag_1h                     0.1234
  neighbor_wave_avg                        0.0987
  wind_speed                               0.0876
  ...

======================================================================
RECOMMENDATIONS
======================================================================

✓✓ HIGH CONFIDENCE MODEL - Great job!
  Model is performing well with 87.8% confidence

Error Analysis:
  Average prediction error: ±1.12 feet
  95% of predictions within: ±2.24 feet
  ✓ Excellent accuracy!
======================================================================
```

**Understanding Confidence Levels:**
- **< 70%**: Low confidence - Needs improvement (more data/tuning)
- **70-85%**: Moderate confidence - Good for most purposes
- **> 85%**: High confidence - Production ready

#### 3. Make Predictions

**Single Buoy Current Reading:**
```bash
python predict.py --buoys 44017 --model models/wave_predictor.pkl --mode current
```

**Multi-Buoy Forecast:**
```bash
python predict.py \
    --buoys 44017 44008 44013 44025 \
    --model models/wave_predictor.pkl \
    --mode forecast
```

**Regional Summary:**
```bash
python predict.py \
    --buoys 44017 44008 44013 44025 44065 44066 \
    --model models/wave_predictor.pkl \
    --mode summary
```

### Improving Model Confidence

If your model has low confidence (<70%), follow these steps:

**Step 1: Collect More Historical Data**
```bash
# Increase from 7 to 21 days
python train_model_advanced.py --days 21 --buoys 44017 44008 44013 44025
```

**Step 2: Add More Buoy Stations**
```bash
# More buoys = better spatial coverage
python train_model_advanced.py --days 14 --buoys 44017 44008 44013 44025 44065 44066 44007
```

**Step 3: Use Hyperparameter Tuning**
```bash
# Let the model find optimal parameters
python train_model_advanced.py --tune --days 21 --buoys 44017 44008 44013 44025 44065 44066
```

**Step 4: Re-analyze**
```bash
python analyze_model.py --model models/wave_predictor_optimized.pkl
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
- **Gradient features**: Spatial gradients of wave height between neighboring buoys (max, min, average, magnitude)
- **Energy features**: Wave energy density and wave power calculations based on wave dynamics

These features capture both temporal evolution and spatial patterns, including energy differentials that drive wave dynamics according to Carnot's principle.

### ML API Reference

#### BuoyForecaster

High-level interface for wave height forecasting.

**Methods:**
- `train(buoy_ids, days_back, model_type, save_path)` - Train a new model
- `get_current_readings(buoy_ids)` - Get real-time buoy data
- `predict_wave_height(buoy_id, current_conditions)` - Predict single buoy
- `forecast_between_buoys(buoy_ids)` - Forecast multiple buoys
- `get_regional_summary(buoy_ids)` - Get regional statistics
- `analyze_gradients(buoy_ids, threshold_percentile, include_energy)` - Analyze spatial gradients and energy differentials

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

### CLI Tools Reference

#### train_model.py

Basic training script with default parameters.

**Usage:**
```bash
python train_model.py [OPTIONS]
```

**Options:**
- `--buoys BUOY_ID [BUOY_ID ...]` - List of buoy station IDs (default: 44017 44008 44013 44025)
- `--all-stations` - Use all available stations from NOAA realtime2 directory
- `--region {northeast,southeast,caribbean,pacific,greatlakes,hawaii}` - Filter by region (use with --all-stations)
- `--lat LAT` - Latitude of center point for location-based search (requires --lon and --radius)
- `--lon LON` - Longitude of center point for location-based search (requires --lat and --radius)
- `--radius RADIUS` - Search radius in meters from center point (requires --lat and --lon)
- `--days N` - Number of days of historical data (default: 7)
- `--model-type {random_forest,gradient_boosting}` - Model type (default: random_forest)
- `--output PATH` - Path to save trained model (default: models/wave_predictor.pkl)
- `--db CONNECTION_STRING` - Database connection string (default: sqlite:///buoy_ml_data.db)

**Examples:**
```bash
# Train with specific buoys
python train_model.py \
    --buoys 44017 44008 44013 44025 \
    --days 7 \
    --model-type random_forest \
    --output models/my_model.pkl

# Train with all available stations
python train_model.py --all-stations --days 7

# Train with all Northeast Atlantic stations
python train_model.py --all-stations --region northeast --days 7

# Train with buoys near New York City (within 200km)
python train_model.py \
    --lat 40.7128 \
    --lon -74.0060 \
    --radius 200000 \
    --days 7
```

#### train_model_advanced.py

Advanced training with hyperparameter tuning and improved defaults.

**Usage:**
```bash
python train_model_advanced.py [OPTIONS]
```

**Options:**
- `--buoys BUOY_ID [BUOY_ID ...]` - List of buoy station IDs (default: 44017 44008 44013 44025 44065 44066)
- `--all-stations` - Use all available stations from NOAA realtime2 directory
- `--region {northeast,southeast,caribbean,pacific,greatlakes,hawaii}` - Filter by region (use with --all-stations)
- `--lat LAT` - Latitude of center point for location-based search (requires --lon and --radius)
- `--lon LON` - Longitude of center point for location-based search (requires --lat and --radius)
- `--radius RADIUS` - Search radius in meters from center point (requires --lat and --lon)
- `--days N` - Number of days of historical data (default: 14, recommended: 21+)
- `--tune` - Enable hyperparameter tuning with RandomizedSearchCV (slower but better)
- `--model-type {random_forest,gradient_boosting}` - Model type (default: random_forest)
- `--output PATH` - Path to save trained model (default: models/wave_predictor_optimized.pkl)
- `--db CONNECTION_STRING` - Database connection string (default: sqlite:///buoy_ml_data.db)

**Examples:**
```bash
# Quick training with improved parameters
python train_model_advanced.py --days 14

# Best quality with hyperparameter tuning
python train_model_advanced.py --tune --days 21

# Custom buoys and output path
python train_model_advanced.py \
    --buoys 44017 44008 44013 \
    --days 14 \
    --output models/northeast_predictor.pkl

# Train with ALL available stations (comprehensive model)
python train_model_advanced.py --all-stations --days 14

# Train with all Pacific region stations
python train_model_advanced.py --all-stations --region pacific --days 14 --tune

# Train with buoys near San Francisco (within 300km)
python train_model_advanced.py \
    --lat 37.7749 \
    --lon -122.4194 \
    --radius 300000 \
    --days 14 \
    --tune
```

**Performance Tips:**
- More days = better model (recommended: 14-21 days)
- More buoys = better spatial coverage (recommended: 6+ buoys)
- Use `--tune` for production models (takes 5-10x longer)

#### analyze_model.py

Model diagnostics and performance analysis.

**Usage:**
```bash
python analyze_model.py [OPTIONS]
```

**Options:**
- `--model PATH` - Path to trained model (default: models/wave_predictor.pkl)
- `--db CONNECTION_STRING` - Database connection string (default: sqlite:///buoy_ml_data.db)

**Example:**
```bash
python analyze_model.py --model models/wave_predictor_optimized.pkl
```

**Output includes:**
- Model type and feature count
- Performance metrics (RMSE, MAE, R², CV scores)
- Model confidence percentage
- Top 15 most important features
- Actionable recommendations
- Error analysis with 95% confidence intervals

#### predict.py

Make predictions with trained models.

**Usage:**
```bash
python predict.py [--buoys BUOY_ID [BUOY_ID ...] | --lat LAT --lon LON --radius RADIUS] --model PATH --mode {current,forecast,summary,gradients}
```

**Options:**
- `--buoys BUOY_ID [BUOY_ID ...]` - List of buoy station IDs
- `--lat LAT` - Latitude of center point for location-based search (requires --lon and --radius)
- `--lon LON` - Longitude of center point for location-based search (requires --lat and --radius)
- `--radius RADIUS` - Search radius in meters from center point (requires --lat and --lon)
- `--model PATH` - Path to trained model (default: models/wave_predictor.pkl)
- `--mode {current,forecast,summary,gradients}` - Prediction mode (default: forecast)
- `--gradient-threshold PERCENT` - Percentile threshold for significant gradients (default: 75, only used with gradients mode)
- `--db CONNECTION_STRING` - Database connection string (default: sqlite:///buoy_ml_data.db)

**Modes:**
- `current` - Show current readings for specified buoys
- `forecast` - Predict wave heights with confidence intervals
- `summary` - Regional summary with statistics and high wave alerts
- `gradients` - Analyze spatial gradients and energy differentials between buoys

**Examples:**
```bash
# Get current readings for specific buoys
python predict.py --buoys 44017 44008 --mode current

# Forecast wave heights
python predict.py --buoys 44017 44008 44013 --mode forecast

# Regional summary
python predict.py --buoys 44017 44008 44013 44025 --mode summary

# Analyze energy gradients and differentials
python predict.py --buoys 44017 44008 44013 44025 --mode gradients

# Get forecast for all buoys near Boston (within 100km)
python predict.py \
    --lat 42.3601 \
    --lon -71.0589 \
    --radius 100000 \
    --mode forecast

# Analyze gradients for buoys near New York (within 200km)
python predict.py \
    --lat 40.7128 \
    --lon -74.0060 \
    --radius 200000 \
    --mode gradients \
    --gradient-threshold 80

# Get regional summary for buoys near Miami (within 150km)
python predict.py \
    --lat 25.7617 \
    --lon -80.1918 \
    --radius 150000 \
    --mode summary
```

### Wave Gradient and Energy Differential Analysis

The package now includes advanced features for analyzing spatial gradients and energy differentials between buoy stations. This functionality is based on wave dynamics principles and Carnot's law, which states that energy flows from high to low potential, driving wave motion and energy transfer.

#### Understanding Energy Differentials

Wave energy differentials represent the spatial variation in wave power across a region, which can indicate:
- **Active wave dynamics**: Areas where wave energy is being transferred or transformed
- **Energy hotspots**: Locations with high energy concentration or rapid energy changes
- **Potential for energy extraction**: Areas suitable for wave energy harvesting
- **Wave propagation patterns**: How waves travel and evolve across a region

#### Key Concepts

**Wave Energy Density**: The energy per unit area stored in ocean waves:
```
E = (1/8) × ρ × g × H²
```
Where:
- ρ = water density (1025 kg/m³)
- g = gravitational acceleration (9.81 m/s²)
- H = significant wave height (m)

**Wave Power**: The rate of energy transport per meter of wave crest:
```
P = E × Cg
```
Where Cg is the group velocity (for deep water: gT/4π)

**Spatial Gradient**: The rate of change of wave height or energy per unit distance:
```
∇H = (H₂ - H₁) / distance
```

#### Using Gradient Analysis

**Python API:**

```python
from buoy_data.ml import BuoyForecaster

# Initialize forecaster
with BuoyForecaster(model_path='models/wave_predictor.pkl') as forecaster:
    # Analyze gradients for a region
    analysis = forecaster.analyze_gradients(
        buoy_ids=['44017', '44008', '44013', '44025'],
        threshold_percentile=75.0,  # Report top 25% of gradients
        include_energy=True
    )
    
    # Access significant gradients
    for grad in analysis['significant_gradients']:
        print(f"{grad['station1']} → {grad['station2']}")
        print(f"  Wave height gradient: {grad['gradient']:.4f} m/km")
        print(f"  Distance: {grad['distance_km']:.1f} km")
        print(f"  Energy differential: {grad['energy_differential']:.2f} W/m")
        print(f"  Energy flow: from {grad['energy_flow_direction']}")
    
    # View summary statistics
    summary = analysis['summary']
    print(f"Significant gradient pairs: {summary['significant_pairs']}")
    print(f"Max gradient: {summary['max_gradient']:.4f} m/km")
    
    # Identify energy hotspots
    if 'energy_hotspots' in summary:
        for hotspot in summary['energy_hotspots']:
            print(f"Station {hotspot['station_id']}: {hotspot['gradient_count']} connections")
```

**CLI Tool:**

```bash
# Analyze gradients for specific buoys
python predict.py \
    --buoys 44017 44008 44013 44025 \
    --mode gradients \
    --gradient-threshold 75

# Find and analyze gradients near a location
python predict.py \
    --lat 40.7128 \
    --lon -74.0060 \
    --radius 200000 \
    --mode gradients \
    --gradient-threshold 80
```

#### Interpreting Results

**High Gradient Magnitude**: Indicates rapid changes in wave conditions over short distances, suggesting:
- Active wave dynamics and energy transfer
- Potential wave-wave interactions
- Bathymetric effects (shallow water, continental shelf)
- Wind field variations

**Energy Flow Direction**: Shows the dominant direction of energy transfer:
- From station with higher wave power → station with lower wave power
- Follows Carnot's principle of energy flowing from high to low potential
- Useful for predicting wave propagation

**Energy Hotspots**: Buoys involved in many significant gradients:
- Central locations in active wave fields
- Good candidates for detailed monitoring
- Potential sites for wave energy extraction

#### Applications

1. **Wave Energy Resource Assessment**: Identify optimal locations for wave energy converters
2. **Maritime Safety**: Detect areas with rapidly changing wave conditions
3. **Coastal Management**: Understand wave energy distribution near shores
4. **Research**: Study wave dynamics and energy transfer mechanisms
5. **Model Validation**: Compare predicted vs. actual energy gradients

#### Enhanced ML Features

The gradient analysis automatically enhances ML model predictions with:
- Spatial gradient features (max, min, average gradients to neighbors)
- Wave energy density and power calculations
- Gradient magnitude features for better predictions

These features are automatically included when training models and making predictions.

### Station Discovery and Region Filtering

The package includes utilities to automatically discover all available buoy stations from NOAA's real-time data feeds, including location-based search.

#### Location-based Search

Find buoy stations within a specified radius from a geographic coordinate point. This is useful when specific station IDs are unavailable or when you want to discover all buoys in a region dynamically.

**Using Python API:**

```python
from buoy_data import find_stations_by_location

# Find all buoys within 100km of New York City
nearby = find_stations_by_location(
    center_lat=40.7128,
    center_lon=-74.0060,
    radius_meters=100000
)

for station in nearby:
    print(f"{station['station_id']}: {station['distance']/1000:.1f}km - {station['location']}")
```

**Using CLI Tools:**

All training and prediction tools support location-based search:

```bash
# Train model with buoys near Boston
python train_model.py \
    --lat 42.3601 \
    --lon -71.0589 \
    --radius 200000 \
    --days 7

# Predict with buoys near San Francisco
python predict.py \
    --lat 37.7749 \
    --lon -122.4194 \
    --radius 150000 \
    --mode forecast
```

**Benefits:**
- No need to know specific station IDs in advance
- Automatically adapts to station availability (handles 404 errors gracefully)
- Finds closest operational buoys to your area of interest
- Works with any geographic location worldwide

#### Region-based Discovery

```python
from buoy_data import get_available_stations, filter_stations_by_region

# Get all available stations
all_stations = get_available_stations()
print(f"Found {len(all_stations)} active stations")

# Filter by region
northeast = filter_stations_by_region(all_stations, 'northeast')
pacific = filter_stations_by_region(all_stations, 'pacific')
```

#### Region Mapping

Station IDs use prefixes to indicate geographic regions:

- **northeast** (44xxx) - Northeast North Atlantic (New England, Mid-Atlantic)
- **southeast** (42xxx) - Southeast North Atlantic (South Atlantic Coast)
- **caribbean** (41xxx) - Southwest North Atlantic (Caribbean, Gulf of Mexico)
- **pacific** (46xxx) - Northeast Pacific (West Coast, Alaska)
- **greatlakes** (45xxx) - Great Lakes
- **hawaii** (51xxx) - Southwest Pacific (Hawaiian Islands)

#### Training with All Stations

Using `--all-stations` automatically trains on all buoys with available data:

```bash
# Train on ALL available stations (200+ buoys)
python train_model_advanced.py --all-stations --days 14

# Train on all Northeast stations (40+ buoys)
python train_model_advanced.py --all-stations --region northeast --days 14
```

**Benefits:**
- Maximum spatial coverage for better predictions
- No need to manually track active stations
- Automatically adapts to station availability
- Better model generalization across regions

**Considerations:**
- More stations = longer training time
- Some stations may have incomplete data (handled gracefully)
- Regional models often perform better than global models

### Common Buoy Stations (US East Coast)

- **44017** - Montauk Point, NY (65 NM S)
- **44008** - Nantucket, MA (54 NM SE)
- **44013** - Boston, MA (16 NM E)
- **44025** - Long Island, NY (33 NM S)
- **44065** - New York Harbor Entrance (15 NM SE)
- **44066** - Texas Tower #4, NY (39 NM SE)
- **44007** - Portland, ME (12 NM SE)

Find more stations at: https://www.ndbc.noaa.gov/

## Troubleshooting

### Common Issues

**ImportError: No module named 'pandas' (or 'numpy', 'scikit-learn')**
- The ML features require optional dependencies. Install with: `pip install -e ".[ml]"`

**404 Errors when accessing buoy data**
- Some buoy stations go offline or stop reporting. Use location-based search (`--lat`, `--lon`, `--radius`) to automatically find active nearby stations.

**Model confidence is low (<70%)**
- Collect more historical data: increase `--days` parameter (recommended: 14-21+ days)
- Add more buoy stations for better spatial coverage
- Use `--tune` flag for hyperparameter optimization
- See [Improving Model Confidence](#improving-model-confidence) section for details

**Database connection errors**
- Check that the database path is writable
- Default is `sqlite:///buoy_ml_data.db` in the current directory
- For custom paths, use `--db` parameter with full connection string

## Package Structure

```
buoy-data/
├── buoy_data/              # Main package directory
│   ├── __init__.py         # Package initialization and exports
│   ├── buoy_data.py        # Base buoy data class
│   ├── buoy_real_time.py   # Real-time data retrieval
│   ├── buoy_hourly.py      # Historical hourly data
│   ├── station.py          # Station metadata and information
│   ├── stations_data.py    # Station database
│   ├── conversions.py      # Unit conversion utilities
│   ├── database.py         # Database models and ORM
│   ├── utils.py            # Utility functions (geospatial, validation)
│   └── ml/                 # Machine learning module (optional)
│       ├── __init__.py
│       ├── data_collector.py    # Historical data collection
│       ├── feature_engineering.py # Feature creation for ML
│       ├── wave_predictor.py     # ML prediction models
│       └── forecaster.py         # High-level forecasting interface
├── train_model.py          # Basic model training CLI
├── train_model_advanced.py # Advanced training with tuning
├── predict.py              # Prediction CLI tool
├── analyze_model.py        # Model diagnostics tool
├── download_data.py        # Data download utility
├── examples.py             # Core feature examples
├── examples_ml.py          # ML feature examples
└── tests/                  # Test suite

```

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| **Install (core only)** | `pip install -e .` |
| **Install (with ML)** | `pip install -e ".[ml]"` |
| **Run core examples** | `python examples.py` |
| **Run ML examples** | `python examples_ml.py` |
| **Train basic model** | `python train_model.py --buoys 44017 44008 --days 7` |
| **Train optimized model** | `python train_model_advanced.py --tune --days 21` |
| **Analyze model** | `python analyze_model.py --model models/wave_predictor.pkl` |
| **Make predictions** | `python predict.py --buoys 44017 44008 --mode forecast` |
| **Find nearby buoys** | `python predict.py --lat 40.7 --lon -74.0 --radius 100000 --mode current` |
| **Run tests (core)** | `pytest tests/ --ignore=tests/test_analyze_model.py` |
| **Run all tests** | `pytest tests/` (requires ML dependencies) |

### Key Python Classes

| Class | Purpose | Module |
|-------|---------|--------|
| `BuoyRealTime` | Fetch real-time buoy data | `from buoy_data import BuoyRealTime` |
| `BuoyHourly` | Fetch hourly historical data | `from buoy_data import BuoyHourly` |
| `Station` | Station metadata and location | `from buoy_data import Station` |
| `BuoyDataDB` | Database storage interface | `from buoy_data import BuoyDataDB` |
| `BuoyForecaster` | ML wave height forecasting | `from buoy_data.ml import BuoyForecaster` |
| `DataCollector` | ML data collection | `from buoy_data.ml import DataCollector` |

### Useful Functions

| Function | Purpose | Import |
|----------|---------|--------|
| `get_available_stations()` | List all active NOAA stations | `from buoy_data import get_available_stations` |
| `filter_stations_by_region()` | Filter stations by geographic region | `from buoy_data import filter_stations_by_region` |
| `find_stations_by_location()` | Find stations near lat/lon | `from buoy_data import find_stations_by_location` |
| `haversine_distance()` | Calculate distance between coordinates | `from buoy_data import haversine_distance` |

## Links

- [NOAA National Data Buoy Center](https://www.ndbc.noaa.gov/)
- [NDBC Data Documentation](https://www.ndbc.noaa.gov/docs/)
- [Station Map and IDs](https://www.ndbc.noaa.gov/maps/)
