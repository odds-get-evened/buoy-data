# Training Script Updates

## Summary of Changes

This PR merges the separate `train_model.py` and `train_model_advanced.py` scripts into a single unified training script with the following improvements:

### 1. Unified Training Script

Instead of two separate scripts, there is now a single `train_model.py` that supports both basic and advanced training modes.

**Before:**
- `train_model.py` - Basic training (7 days, 4 buoys, default params)
- `train_model_advanced.py` - Advanced training (14 days, 6 buoys, tuning, better params)

**After:**
- `train_model.py` - Unified script with `--advanced` and `--tune` flags

### 2. New Flags

#### `--advanced`
Enables advanced training mode with improved defaults:
- 14 days of data (vs. 7 days in basic mode)
- 6 default buoys (vs. 4 in basic mode)
- Better model parameters (200 estimators, max_depth=30)
- 10-fold cross-validation (vs. 5-fold)

#### `--tune`
Enables hyperparameter tuning using RandomizedSearchCV:
- Searches hyperparameter space (n_estimators, max_depth, etc.)
- 20 iterations of random search
- 5-fold cross-validation during tuning
- Automatically uses best parameters found

#### `--retrain`
Retrains an existing model instead of creating a new one:
- Loads model from `--output` path
- Preserves model structure and parameters
- Updates with new training data
- Requires `--output` to specify existing model file

### 3. Automatic Timestamped Filenames

Models are now automatically saved with unique timestamped filenames to prevent overwriting:

**Format:** `<basename>_YYYYMMDD-<randomhex>.pkl`

**Example:** 
- Input: `--output models/wave_predictor.pkl`
- Saved as: `models/wave_predictor_20260118-a1b2c3d4.pkl`

**Note:** When using `--retrain`, the specified output filename is used as-is (no timestamp added).

The random hex string is generated using `secrets.token_hex(4)` for cryptographic security.

### 4. Updated BuoyForecaster API

The `BuoyForecaster.train()` method now supports a `retrain` parameter:

```python
from buoy_data.ml import BuoyForecaster

# Train new model
with BuoyForecaster(db_path="sqlite:///buoy_ml_data.db") as forecaster:
    metrics = forecaster.train(
        buoy_ids=['44017', '44008'],
        days_back=7,
        save_path='models/my_model.pkl',
        retrain=False  # Create new model
    )

# Retrain existing model
with BuoyForecaster(db_path="sqlite:///buoy_ml_data.db") as forecaster:
    metrics = forecaster.train(
        buoy_ids=['44017', '44008'],
        days_back=7,
        save_path='models/my_model_20260118-abc12345.pkl',
        retrain=True  # Load and retrain existing model
    )
```

## Migration Guide

### For Users of `train_model.py`

No changes needed! The basic training workflow remains the same:

```bash
python train_model.py --buoys 44017 44008 --days 7
```

### For Users of `train_model_advanced.py`

Replace `train_model_advanced.py` with `train_model.py --advanced`:

**Before:**
```bash
python train_model_advanced.py --buoys 44017 44008 --days 14
python train_model_advanced.py --tune --days 21
```

**After:**
```bash
python train_model.py --advanced --buoys 44017 44008 --days 14
python train_model.py --advanced --tune --days 21
```

## Examples

### Basic Training
```bash
# Default: 7 days, 4 buoys
python train_model.py

# Custom buoys and days
python train_model.py --buoys 44017 44008 44013 --days 7

# Outputs: models/wave_predictor_20260118-abc12345.pkl
```

### Advanced Training
```bash
# Advanced mode with improved defaults
python train_model.py --advanced

# Advanced with hyperparameter tuning
python train_model.py --advanced --tune --days 21

# Outputs: models/wave_predictor_20260118-def67890.pkl
```

### Retraining
```bash
# Retrain existing model
python train_model.py --retrain \
    --output models/wave_predictor_20260118-abc12345.pkl \
    --days 7

# Outputs: models/wave_predictor_20260118-abc12345.pkl (overwrites)
```

### Location-Based Training
```bash
# Train with buoys near New York City
python train_model.py --advanced \
    --lat 40.7128 \
    --lon -74.0060 \
    --radius 200000 \
    --days 14
```

## Benefits

1. **Simpler CLI**: One script instead of two
2. **No Overwrites**: Unique timestamps prevent accidental model overwrites
3. **Easy Retraining**: Update existing models with new data
4. **Backward Compatible**: Basic usage unchanged
5. **Cleaner Codebase**: Removed code duplication

## Breaking Changes

None! The basic training workflow remains exactly the same. Users of `train_model_advanced.py` just need to add `--advanced` flag.
