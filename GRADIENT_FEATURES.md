# Wave Gradient and Energy Differential Analysis

## Overview

This feature enables the analysis of spatial gradients and energy differentials between buoy stations, based on wave dynamics principles and Carnot's law. Energy flows from high to low potential, driving wave motion and energy transfer across regions.

## Quick Start

### CLI Usage

```bash
# Analyze gradients for specific buoys
python predict.py --buoys 44017 44008 44013 44025 --mode gradients

# Find buoys near a location and analyze gradients
python predict.py \
    --lat 40.7128 \
    --lon -74.0060 \
    --radius 200000 \
    --mode gradients \
    --gradient-threshold 75
```

### Python API Usage

```python
from buoy_data.ml import BuoyForecaster

# Initialize forecaster with trained model
with BuoyForecaster(model_path='models/wave_predictor.pkl') as forecaster:
    # Analyze gradients
    analysis = forecaster.analyze_gradients(
        buoy_ids=['44017', '44008', '44013', '44025'],
        threshold_percentile=75.0,
        include_energy=True
    )
    
    # View significant gradients
    for grad in analysis['significant_gradients']:
        print(f"{grad['station1']} → {grad['station2']}")
        print(f"  Gradient: {grad['gradient']:.4f} m/km")
        print(f"  Energy differential: {grad['energy_differential']:.2f} W/m")
```

## Key Concepts

### Wave Energy Density
The energy per unit area stored in ocean waves:
```
E = (1/8) × ρ × g × H²
```
Where:
- ρ = water density (1025 kg/m³)
- g = gravitational acceleration (9.81 m/s²)
- H = significant wave height (m)

### Wave Power
The rate of energy transport per meter of wave crest:
```
P = E × Cg
```
Where Cg is the group velocity (for deep water: gT/4π)

### Spatial Gradient
The rate of change of wave height or energy per unit distance:
```
∇H = (H₂ - H₁) / distance
```

## Applications

1. **Wave Energy Resource Assessment**
   - Identify optimal locations for wave energy converters
   - Estimate available wave power in a region
   - Map energy distribution patterns

2. **Maritime Safety**
   - Detect areas with rapidly changing wave conditions
   - Identify potentially hazardous wave gradients
   - Monitor wave field evolution

3. **Coastal Management**
   - Understand wave energy distribution near shores
   - Predict coastal erosion patterns
   - Plan coastal protection measures

4. **Scientific Research**
   - Study wave dynamics and energy transfer mechanisms
   - Validate wave propagation models
   - Analyze wave-wave interactions

## Understanding Results

### Gradient Magnitude
- **High gradient**: Rapid wave changes over short distances
  - Active wave dynamics
  - Potential bathymetric effects
  - Wind field variations

- **Low gradient**: Gradual wave changes
  - Stable wave conditions
  - Uniform wave field

### Energy Flow Direction
Indicates the dominant direction of energy transfer:
- Energy flows from station with higher wave power → lower wave power
- Follows Carnot's principle
- Useful for predicting wave propagation

### Energy Hotspots
Buoys involved in many significant gradients:
- Central locations in active wave fields
- Good candidates for detailed monitoring
- Potential sites for energy extraction

## ML Feature Integration

Gradient analysis automatically enhances ML predictions with:
- Spatial gradient features (max, min, average)
- Wave energy density calculations
- Wave power estimates
- Gradient magnitude features

These features are included when training models and making predictions.

## Examples

See `example_gradients.py` for complete working examples:
```bash
python example_gradients.py
```

## API Reference

### Core Functions

```python
from buoy_data import (
    calculate_wave_energy_density,
    calculate_wave_power,
    calculate_spatial_gradient,
    identify_significant_gradients
)
```

#### calculate_wave_energy_density(wave_height, wave_period)
Calculate energy per unit area (J/m²).

#### calculate_wave_power(wave_height, wave_period)
Calculate energy flux per meter of wave crest (W/m).

#### calculate_spatial_gradient(value1, value2, lat1, lon1, lat2, lon2)
Calculate spatial gradient between two points.

#### identify_significant_gradients(stations_data, threshold_percentile=75.0)
Identify significant energy differentials across multiple stations.

### BuoyForecaster Methods

```python
forecaster.analyze_gradients(
    buoy_ids,
    threshold_percentile=75.0,
    include_energy=True
)
```

Returns dictionary with:
- `significant_gradients`: List of gradient pairs
- `summary`: Summary statistics
- `energy_hotspots`: Buoy IDs with highest energy differentials (if include_energy=True)

## Physical Background

### Carnot's Principle
Energy naturally flows from regions of high potential to low potential. In wave systems:
- High wave power → Low wave power
- Energy transfer drives wave propagation
- Gradients indicate active energy transfer

### Wave Dynamics
- Wave energy is proportional to the square of wave height
- Wave power increases with both height and period
- Energy transfer occurs through group velocity
- Spatial gradients reveal energy flux patterns

## Testing

Run comprehensive tests:
```bash
python -m unittest tests.test_gradients -v
```

All functionality is thoroughly tested with 20+ unit tests covering:
- Wave energy calculations
- Spatial gradient computations
- Significant gradient identification
- Integration with location-based search
