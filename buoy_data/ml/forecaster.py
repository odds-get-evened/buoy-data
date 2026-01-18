"""High-level forecasting interface for buoy wave heights."""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .data_collector import DataCollector
from .feature_engineering import FeatureEngineer
from .wave_predictor import WaveHeightPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuoyForecaster:
    """
    High-level interface for querying current readings and forecasting
    wave heights between buoy stations.

    This class provides a simple API for:
    - Getting current buoy readings
    - Predicting wave heights at specific buoys
    - Forecasting wave heights between multiple buoy stations
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: str = "sqlite:///buoy_ml_data.db"
    ):
        """
        Initialize BuoyForecaster.

        Args:
            model_path: Path to pre-trained model (None = train new model)
            db_path: Database connection string
        """
        self.data_collector = DataCollector(db_path)
        self.feature_engineer = FeatureEngineer()
        self.predictor = None

        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.predictor = WaveHeightPredictor.load(model_path)
        else:
            logger.info("No model loaded. Call train() to create a new model.")

    def get_current_readings(
        self,
        buoy_ids: List[str]
    ) -> pd.DataFrame:
        """
        Get current real-time readings from buoy stations.

        Args:
            buoy_ids: List of buoy station IDs

        Returns:
            DataFrame with current readings
            
        Raises:
            ValueError: If buoy_ids is empty or invalid
        """
        if not buoy_ids or not isinstance(buoy_ids, list):
            raise ValueError("buoy_ids must be a non-empty list")
        if not all(isinstance(bid, str) for bid in buoy_ids):
            raise ValueError("All buoy_ids must be strings")
            
        logger.info(f"Fetching current readings for {len(buoy_ids)} buoys...")
        return self.data_collector.collect_current_data(buoy_ids)

    def train(
        self,
        buoy_ids: List[str],
        days_back: int = 7,
        model_type: str = "random_forest",
        save_path: Optional[str] = None,
        use_cache: bool = False
    ) -> Dict[str, float]:
        """
        Train a new forecasting model.

        Args:
            buoy_ids: List of buoy station IDs to include in training
            days_back: Number of days of historical data to collect
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            save_path: Optional path to save trained model
            use_cache: If True, use cached data from database (default: False)

        Returns:
            Dictionary of training metrics
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not buoy_ids or not isinstance(buoy_ids, list):
            raise ValueError("buoy_ids must be a non-empty list")
        if not all(isinstance(bid, str) for bid in buoy_ids):
            raise ValueError("All buoy_ids must be strings")
        if days_back <= 0 or days_back > 365:
            raise ValueError("days_back must be between 1 and 365")
        if model_type not in ["random_forest", "gradient_boosting"]:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
            
        logger.info(f"Training model on {len(buoy_ids)} buoy stations...")

        # Collect training data
        logger.info("Collecting training data...")
        raw_data = self.data_collector.collect_training_dataset(
            buoy_ids,
            days_back=days_back,
            use_cache=use_cache
        )

        if raw_data.empty:
            raise ValueError("No data collected for training")

        logger.info(f"Collected {len(raw_data)} data points")

        # Prepare features
        logger.info("Engineering features...")
        prepared_data = self.feature_engineer.prepare_features(
            raw_data,
            add_lags=True,
            add_rolling=True,
            add_inter_buoy=True
        )

        # Train model
        self.predictor = WaveHeightPredictor(model_type=model_type)
        metrics = self.predictor.train(prepared_data)

        # Save model if requested
        if save_path:
            self.predictor.save(save_path)

        return metrics

    def predict_wave_height(
        self,
        buoy_id: str,
        current_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict wave height at a specific buoy.

        Args:
            buoy_id: Buoy station ID
            current_conditions: Optional dict with current conditions
                               (if None, will fetch real-time data)

        Returns:
            Dictionary with prediction and metadata
        """
        if not self.predictor or not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get current data if not provided
        if current_conditions is None:
            current_data = self.get_current_readings([buoy_id])
            if current_data.empty:
                raise ValueError(f"No current data available for buoy {buoy_id}")
        else:
            current_data = pd.DataFrame([current_conditions])

        # Prepare features (without lag/rolling since we don't have historical data)
        prepared_data = self.feature_engineer.prepare_features(
            current_data,
            add_lags=False,  # Can't compute lags for single point
            add_rolling=False,
            add_inter_buoy=False
        )

        # Ensure all required features are present, fill missing with 0
        for col in self.predictor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0.0

        # Make prediction
        prediction = self.predictor.predict(prepared_data)[0]

        # Get confidence interval if possible
        confidence = None
        if self.predictor.model_type == "random_forest":
            try:
                _, std = self.predictor.predict_with_confidence(prepared_data)
                confidence = {
                    'std': float(std[0]),
                    'lower_95': float(prediction - 1.96 * std[0]),
                    'upper_95': float(prediction + 1.96 * std[0])
                }
            except (AttributeError, ValueError, IndexError) as e:
                logger.warning(f"Could not compute confidence intervals for {buoy_id}: {e}")

        result = {
            'buoy_id': buoy_id,
            'predicted_wave_height_m': float(prediction),
            'predicted_wave_height_ft': float(prediction * 3.28),
            'confidence': confidence
        }

        # Add actual reading if available
        if 'wave_height_m' in current_data.columns:
            actual = current_data['wave_height_m'].iloc[0]
            if not pd.isna(actual):
                result['actual_wave_height_m'] = float(actual)
                result['prediction_error_m'] = float(prediction - actual)

        return result

    def forecast_between_buoys(
        self,
        buoy_ids: List[str],
        include_current: bool = True
    ) -> pd.DataFrame:
        """
        Forecast wave heights for multiple buoy stations.

        This is useful for understanding wave patterns across a region
        and identifying potential areas between monitored stations.

        Args:
            buoy_ids: List of buoy station IDs
            include_current: Whether to include current readings

        Returns:
            DataFrame with predictions for all buoys
        """
        if not self.predictor or not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Forecasting for {len(buoy_ids)} buoy stations...")

        # Get current data
        current_data = self.get_current_readings(buoy_ids)

        if current_data.empty:
            raise ValueError("No current data available for specified buoys")

        # Prepare features (without lag/rolling since we don't have historical data)
        prepared_data = self.feature_engineer.prepare_features(
            current_data,
            add_lags=False,
            add_rolling=False,
            add_inter_buoy=True  # Important for spatial relationships
        )

        # Ensure all required features are present, fill missing with 0
        for col in self.predictor.feature_columns:
            if col not in prepared_data.columns:
                prepared_data[col] = 0.0

        # Make predictions
        predictions = self.predictor.predict(prepared_data)

        # Create results DataFrame
        results = pd.DataFrame({
            'buoy_id': current_data['buoy_id'],
            'predicted_wave_height_m': predictions,
            'predicted_wave_height_ft': predictions * 3.28,
        })

        # Add confidence intervals if available
        if self.predictor.model_type == "random_forest":
            try:
                _, std = self.predictor.predict_with_confidence(prepared_data)
                results['confidence_std'] = std
                results['lower_95_m'] = predictions - 1.96 * std
                results['upper_95_m'] = predictions + 1.96 * std
            except (AttributeError, ValueError, IndexError) as e:
                logger.warning(f"Could not compute confidence intervals: {e}")

        # Include current readings if requested
        if include_current:
            current_heights = current_data[['buoy_id', 'wave_height_m']].copy()
            current_heights.columns = ['buoy_id', 'actual_wave_height_m']

            # Convert to numeric to handle any string values
            current_heights['actual_wave_height_m'] = pd.to_numeric(
                current_heights['actual_wave_height_m'],
                errors='coerce'
            )

            results = results.merge(current_heights, on='buoy_id', how='left')

            # Calculate errors where actual data exists
            mask = ~results['actual_wave_height_m'].isna()
            results.loc[mask, 'prediction_error_m'] = (
                results.loc[mask, 'predicted_wave_height_m'] -
                results.loc[mask, 'actual_wave_height_m']
            )

        # Add spatial info (from prepared_data which has lat/lon added by feature engineering)
        if 'latitude' in prepared_data.columns and 'longitude' in prepared_data.columns:
            spatial_info = prepared_data[['buoy_id', 'latitude', 'longitude']].copy()
            # Remove duplicates in case there are multiple rows per buoy
            spatial_info = spatial_info.drop_duplicates(subset=['buoy_id'])
            results = results.merge(spatial_info, on='buoy_id', how='left')

        logger.info(f"Generated forecasts for {len(results)} buoys")

        return results

    def get_regional_summary(
        self,
        buoy_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get a summary of wave conditions across a region.

        Args:
            buoy_ids: List of buoy station IDs

        Returns:
            Dictionary with regional statistics
        """
        forecast = self.forecast_between_buoys(buoy_ids, include_current=True)

        summary = {
            'num_buoys': len(forecast),
            'predicted': {
                'mean_wave_height_m': float(forecast['predicted_wave_height_m'].mean()),
                'max_wave_height_m': float(forecast['predicted_wave_height_m'].max()),
                'min_wave_height_m': float(forecast['predicted_wave_height_m'].min()),
                'std_wave_height_m': float(forecast['predicted_wave_height_m'].std()),
            },
            'buoys': forecast.to_dict('records')
        }

        # Add actual statistics if available
        if 'actual_wave_height_m' in forecast.columns:
            actual_data = forecast['actual_wave_height_m'].dropna()
            if not actual_data.empty:
                summary['actual'] = {
                    'mean_wave_height_m': float(actual_data.mean()),
                    'max_wave_height_m': float(actual_data.max()),
                    'min_wave_height_m': float(actual_data.min()),
                }

        # Identify areas of concern (high waves)
        high_wave_threshold = 3.0  # meters
        high_wave_buoys = forecast[forecast['predicted_wave_height_m'] > high_wave_threshold]
        if not high_wave_buoys.empty:
            summary['high_wave_alerts'] = high_wave_buoys['buoy_id'].tolist()

        return summary

    def close(self):
        """Close data collector connection."""
        self.data_collector.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
