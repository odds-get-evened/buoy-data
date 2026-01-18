# Small pytest suite to validate analyze_model() handles full and missing metric dicts.
# These tests monkeypatch WaveHeightPredictor.load so no real model file is required.

import logging
import pandas as pd
import pytest

import analyze_model as am


class DummyPredictor:
    """Minimal predictor stub used by tests."""

    def __init__(self, metrics):
        self.model_type = "random_forest"
        self.feature_columns = ["f1", "f2", "f3"]
        self.metrics = metrics


def _full_metrics_predictor():
    """Create a DummyPredictor that mimics a full metrics dict the analyzer expects."""
    feature_imp = pd.DataFrame(
        [
            {"feature": "wave_height_m_lag_1h", "importance": 0.1234},
            {"feature": "neighbor_wave_avg", "importance": 0.0987},
        ]
    )
    metrics = {
        "train_rmse": 0.215,
        "test_rmse": 0.342,
        "test_mae": 0.267,
        "test_r2": 0.878,
        "cv_rmse": 0.351,
        "cv_rmse_std": 0.045,
        "feature_importance": feature_imp,
    }
    return DummyPredictor(metrics)


def test_analyze_runs_with_full_metrics(monkeypatch, caplog):
    """analyze_model should run without errors and log expected entries when metrics are present."""
    # Set logging level for the analyze_model module logger
    caplog.set_level(logging.INFO, logger="analyze_model")
    
    monkeypatch.setattr(
        "buoy_data.ml.WaveHeightPredictor.load",
        staticmethod(lambda path: _full_metrics_predictor()),
    )

    # Run analyzer (no actual model file required due to monkeypatch)
    am.analyze_model("models/fake.pkl", db_path="sqlite:///:memory:")

    # Verify that high-level logs were emitted
    assert "Model Type: random_forest" in caplog.text
    assert "Performance Metrics" in caplog.text
    assert "Model Confidence" in caplog.text
    assert "Top 15 Most Important Features" in caplog.text


def test_analyze_handles_missing_metrics(monkeypatch, caplog):
    """analyze_model should not crash when metrics dict is empty or missing keys."""
    # Set logging level for the analyze_model module logger
    caplog.set_level(logging.WARNING, logger="analyze_model")
    
    monkeypatch.setattr(
        "buoy_data.ml.WaveHeightPredictor.load",
        staticmethod(lambda path: DummyPredictor(metrics={})),
    )

    am.analyze_model("models/fake.pkl", db_path="sqlite:///:memory:")

    # Expect a warning about missing test_r2 (confidence cannot be computed)
    assert "test_r2 missing" in caplog.text or "Cannot compute model confidence" in caplog.text
    # Also expect it to mention that test_rmse is missing (error analysis cannot be computed)
    assert "test_rmse missing" in caplog.text or "Error Analysis: test_rmse not available" in caplog.text
