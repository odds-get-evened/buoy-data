"""Feature engineering for buoy data ML models."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime
from ..station import Station


class FeatureEngineer:
    """
    Creates features for ML models from buoy data.

    Handles feature extraction, spatial relationships between buoys,
    temporal features, and data preparation for training.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_columns = []
        self.target_column = 'wave_height_m'

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two lat/lon points using Haversine formula.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        r = 6371

        return c * r

    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add spatial features based on buoy station locations.

        Args:
            df: DataFrame with buoy data

        Returns:
            DataFrame with added spatial features
        """
        df = df.copy()

        # Add station coordinates
        def add_coordinates(row):
            try:
                station = Station(row['buoy_id'])
                row['latitude'] = station.get_latitude()
                row['longitude'] = station.get_longitude()
                row['depth'] = station.get_depth()
            except:
                row['latitude'] = None
                row['longitude'] = None
                row['depth'] = None
            return row

        df = df.apply(add_coordinates, axis=1)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features from timestamp.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        if 'timestamp' in df.columns:
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # Extract temporal features
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['day_of_year'] = df['datetime'].dt.dayofyear

            # Cyclical encoding for hour (24-hour cycle)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Cyclical encoding for day of year (365-day cycle)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        return df

    def add_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wind-related features.

        Args:
            df: DataFrame with wind data

        Returns:
            DataFrame with added wind features
        """
        df = df.copy()

        if 'wind_direction_deg' in df.columns:
            # Convert to numeric, replacing non-numeric with NaN
            df['wind_direction_deg'] = pd.to_numeric(df['wind_direction_deg'], errors='coerce')
            # Cyclical encoding for wind direction
            df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_deg'] / 360)
            df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_deg'] / 360)

        if 'wind_speed' in df.columns and 'gusts' in df.columns:
            # Convert to numeric
            df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
            df['gusts'] = pd.to_numeric(df['gusts'], errors='coerce')
            # Gust factor (ratio of gusts to wind speed)
            df['gust_factor'] = df['gusts'] / (df['wind_speed'] + 0.1)

        if 'mean_wave_direction_deg' in df.columns:
            # Convert to numeric
            df['mean_wave_direction_deg'] = pd.to_numeric(df['mean_wave_direction_deg'], errors='coerce')
            # Cyclical encoding for wave direction
            df['wave_dir_sin'] = np.sin(2 * np.pi * df['mean_wave_direction_deg'] / 360)
            df['wave_dir_cos'] = np.cos(2 * np.pi * df['mean_wave_direction_deg'] / 360)

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        lag_hours: List[int] = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add lagged features for time series forecasting.

        Args:
            df: DataFrame sorted by timestamp
            lag_hours: List of lag periods in hours

        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        df = df.sort_values(['buoy_id', 'timestamp'])

        # Define columns to lag
        lag_columns = ['wave_height_m', 'wind_speed', 'barometer', 'water_temp_c']

        for col in lag_columns:
            if col in df.columns:
                for lag in lag_hours:
                    # Create lag feature per buoy
                    df[f'{col}_lag_{lag}h'] = df.groupby('buoy_id')[col].shift(lag)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Args:
            df: DataFrame sorted by timestamp
            windows: List of window sizes in hours

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        df = df.sort_values(['buoy_id', 'timestamp'])

        # Define columns for rolling stats
        roll_columns = ['wave_height_m', 'wind_speed', 'barometer']

        for col in roll_columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{col}_mean_{window}h'] = (
                        df.groupby('buoy_id')[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )

                    # Rolling std
                    df[f'{col}_std_{window}h'] = (
                        df.groupby('buoy_id')[col]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(0, drop=True)
                    )

        return df

    def create_inter_buoy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on relationships between nearby buoys.

        For forecasting wave heights between buoys, this creates features
        that capture spatial patterns.

        Args:
            df: DataFrame with multiple buoys

        Returns:
            DataFrame with inter-buoy features
        """
        df = df.copy()

        # Group by timestamp to get concurrent readings
        grouped = df.groupby('timestamp')

        inter_buoy_data = []

        for timestamp, group in grouped:
            if len(group) < 2:
                continue

            buoys = group['buoy_id'].values
            wave_heights = group['wave_height_m'].values
            lats = group['latitude'].values
            lons = group['longitude'].values

            # Create features for each buoy based on neighbors
            for i, buoy_id in enumerate(buoys):
                if pd.isna(wave_heights[i]):
                    continue

                # Calculate weighted average of nearby buoys
                distances = []
                neighbor_heights = []

                for j in range(len(buoys)):
                    if i != j and not pd.isna(wave_heights[j]):
                        dist = self.calculate_distance(
                            lats[i], lons[i], lats[j], lons[j]
                        )
                        distances.append(dist)
                        neighbor_heights.append(wave_heights[j])

                if distances:
                    # Inverse distance weighting
                    weights = [1 / (d + 1) for d in distances]
                    total_weight = sum(weights)
                    weighted_avg = sum(h * w for h, w in zip(neighbor_heights, weights)) / total_weight

                    inter_buoy_data.append({
                        'buoy_id': buoy_id,
                        'timestamp': timestamp,
                        'neighbor_wave_avg': weighted_avg,
                        'neighbor_wave_std': np.std(neighbor_heights) if len(neighbor_heights) > 1 else 0,
                        'min_neighbor_distance': min(distances),
                        'num_neighbors': len(distances)
                    })

        if inter_buoy_data:
            inter_df = pd.DataFrame(inter_buoy_data)
            df = df.merge(inter_df, on=['buoy_id', 'timestamp'], how='left')

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        add_lags: bool = True,
        add_rolling: bool = True,
        add_inter_buoy: bool = True
    ) -> pd.DataFrame:
        """
        Prepare all features for ML model.

        Args:
            df: Raw buoy data DataFrame
            add_lags: Whether to add lag features
            add_rolling: Whether to add rolling window features
            add_inter_buoy: Whether to add inter-buoy features

        Returns:
            DataFrame with all features prepared
        """
        df = df.copy()

        # Convert all numeric columns to proper numeric types
        numeric_columns = [
            'wave_height_m', 'wave_height_ft', 'wind_speed', 'gusts',
            'dominant_wave_period', 'avg_wave_period', 'barometer',
            'air_temp_c', 'water_temp_c', 'dewpoint_c', 'timestamp'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add spatial features
        df = self.add_spatial_features(df)

        # Add temporal features
        df = self.add_temporal_features(df)

        # Add wind features
        df = self.add_wind_features(df)

        # Add lag features
        if add_lags:
            df = self.add_lag_features(df)

        # Add rolling features
        if add_rolling:
            df = self.add_rolling_features(df)

        # Add inter-buoy features
        if add_inter_buoy:
            df = self.create_inter_buoy_features(df)

        return df

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'wave_height_m'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.

        Args:
            df: Prepared DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Drop non-feature columns
        drop_cols = [
            'buoy_id', 'timestamp', 'datetime',
            'year', 'month', 'day', 'hour', 'minute',
            'wave_height_ft',  # Duplicate of wave_height_m
        ]

        # Keep target column for now
        feature_cols = [col for col in df.columns if col not in drop_cols]

        # Extract target
        if target_col in df.columns:
            y = df[target_col].copy()
            feature_cols.remove(target_col)
        else:
            y = None

        X = df[feature_cols].copy()

        # Convert all columns to numeric (handle any remaining strings)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill NaN values
        X = X.fillna(X.median())

        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()

        return X, y
