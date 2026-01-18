"""Machine learning model for wave height prediction."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveHeightPredictor:
    """
    ML model for predicting wave heights at buoy stations.

    Uses ensemble methods (Random Forest or Gradient Boosting) to
    predict wave heights based on current conditions and spatial patterns.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42
    ):
        """
        Initialize WaveHeightPredictor.

        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.is_trained = False
        self.metrics = {}

        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'wave_height_m',
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train the model on buoy data.

        Args:
            df: DataFrame with prepared features
            target_col: Name of target column
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Prepare features
        X, y = self.feature_engineer.split_features_target(df, target_col)

        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Features: {X.columns.tolist()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Train model
        logger.info("Fitting model...")
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # Calculate metrics
        self.metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }

        # Cross-validation
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            X_train_scaled,
            y_train,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        self.metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
        self.metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())

        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            self.metrics['feature_importance'] = feature_importance

        self.is_trained = True

        logger.info("\nModel Performance:")
        logger.info(f"  Train RMSE: {self.metrics['train_rmse']:.3f}m")
        logger.info(f"  Test RMSE:  {self.metrics['test_rmse']:.3f}m")
        logger.info(f"  Test MAE:   {self.metrics['test_mae']:.3f}m")
        logger.info(f"  Test R²:    {self.metrics['test_r2']:.3f}")
        logger.info(f"  CV RMSE:    {self.metrics['cv_rmse']:.3f} ± {self.metrics['cv_rmse_std']:.3f}m")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: DataFrame with features (must match training features)

        Returns:
            Array of predictions

        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure all required features are present
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_with_confidence(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for Random Forest).

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if self.model_type != "random_forest":
            raise ValueError("Confidence intervals only available for Random Forest")

        # Prepare data
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)

        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.model.estimators_
        ])

        # Calculate mean and std
        predictions = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0)

        return predictions, std

    def save(self, filepath: str):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, filepath)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'WaveHeightPredictor':
        """
        Load trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded WaveHeightPredictor instance
        """
        model_data = joblib.load(filepath)

        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_engineer = model_data['feature_engineer']
        predictor.feature_columns = model_data['feature_columns']
        predictor.metrics = model_data['metrics']
        predictor.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Test RMSE: {predictor.metrics.get('test_rmse', 'N/A'):.3f}m")

        return predictor

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")

        return self.metrics.get('feature_importance', pd.DataFrame())
