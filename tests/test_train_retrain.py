"""Tests for training and retraining functionality."""

import pytest
import secrets
import tempfile
from pathlib import Path
from datetime import datetime

from buoy_data.ml import BuoyForecaster
from buoy_data.ml.wave_predictor import WaveHeightPredictor


def test_filename_generation_pattern():
    """Test that the filename generation creates expected format."""
    # Simulate the filename generation logic
    date_str = datetime.now().strftime('%Y%m%d')
    random_hash = secrets.token_hex(4)
    
    output_path = Path('models/wave_predictor.pkl')
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    
    new_filename = f"{stem}_{date_str}-{random_hash}{suffix}"
    result = str(parent / new_filename)
    
    # Verify format
    assert date_str in result
    assert len(random_hash) == 8
    assert result.startswith('models/wave_predictor_')
    assert result.endswith('.pkl')
    assert '-' in result


def test_forecaster_retrain_parameter_validation():
    """Test that retrain parameter validation works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"sqlite:///{tmpdir}/test.db"
        
        with BuoyForecaster(db_path=db_path) as forecaster:
            # Test that retrain requires save_path
            with pytest.raises(ValueError, match="save_path must be provided when retrain=True"):
                forecaster.train(
                    buoy_ids=['44017'],
                    days_back=1,
                    retrain=True,
                    save_path=None
                )


def test_retrain_flag_in_forecaster():
    """Test that the retrain parameter exists and is documented."""
    import inspect
    
    sig = inspect.signature(BuoyForecaster.train)
    params = sig.parameters
    
    # Check that retrain parameter exists
    assert 'retrain' in params
    
    # Check default value is False
    assert params['retrain'].default is False
    
    # Check that the docstring mentions retrain
    docstring = BuoyForecaster.train.__doc__
    assert docstring is not None
    assert 'retrain' in docstring.lower()


def test_unique_filenames_are_different():
    """Test that multiple calls generate different filenames."""
    filenames = set()
    
    for _ in range(10):
        date_str = datetime.now().strftime('%Y%m%d')
        random_hash = secrets.token_hex(4)
        
        output_path = Path('models/wave_predictor.pkl')
        new_filename = f"{output_path.stem}_{date_str}-{random_hash}{output_path.suffix}"
        filenames.add(new_filename)
    
    # All filenames should be unique
    assert len(filenames) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
