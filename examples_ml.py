"""Machine learning examples for buoy wave height forecasting."""

import logging
from buoy_data.ml import BuoyForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_train_model():
    """Example: Train a wave height forecasting model."""
    print("=" * 70)
    print("EXAMPLE 1: Training a Wave Height Forecasting Model")
    print("=" * 70)

    # Select buoy stations (East Coast USA)
    buoy_ids = ['44017', '44008', '44013', '44025']

    print(f"\nTraining on buoys: {buoy_ids}")
    print("This will collect recent data and train a model...")

    try:
        with BuoyForecaster() as forecaster:
            # Train model (this will take a few minutes)
            metrics = forecaster.train(
                buoy_ids=buoy_ids,
                days_back=3,  # Use 3 days for quick demo
                model_type='random_forest',
                save_path='models/demo_model.pkl'
            )

            print("\n" + "="*70)
            print("MODEL TRAINED SUCCESSFULLY!")
            print("="*70)
            print(f"Test RMSE: {metrics['test_rmse']:.3f} meters")
            print(f"Test R²:   {metrics['test_r2']:.3f}")
            print("\nModel saved to: models/demo_model.pkl")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Note: Training requires internet connection to fetch NOAA data")


def example_current_readings():
    """Example: Get current buoy readings."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Query Current Buoy Readings")
    print("=" * 70)

    buoy_ids = ['44017', '44008']

    try:
        with BuoyForecaster() as forecaster:
            readings = forecaster.get_current_readings(buoy_ids)

            print(f"\nCurrent readings for {len(buoy_ids)} buoys:")
            print("-" * 70)

            for _, row in readings.iterrows():
                print(f"\nBuoy {row['buoy_id']}:")
                print(f"  Wave Height: {row['wave_height_ft']:.1f} ft ({row['wave_height_m']:.1f} m)")
                print(f"  Wind Speed:  {row['wind_speed']:.1f} knots")
                print(f"  Wind Dir:    {row['wind_direction_deg']}° ({row.get('compass', 'N/A')})")
                print(f"  Water Temp:  {row['water_temp_c']:.1f}°C")

    except Exception as e:
        print(f"Error fetching readings: {e}")


def example_forecast():
    """Example: Forecast wave heights using trained model."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Forecast Wave Heights Between Buoys")
    print("=" * 70)

    buoy_ids = ['44017', '44008', '44013', '44025']
    model_path = 'models/demo_model.pkl'

    try:
        # Load pre-trained model
        with BuoyForecaster(model_path=model_path) as forecaster:
            forecast = forecaster.forecast_between_buoys(buoy_ids)

            print(f"\nWave height forecast for {len(forecast)} buoys:")
            print("-" * 70)

            for _, row in forecast.iterrows():
                print(f"\nBuoy {row['buoy_id']}:")
                print(f"  Predicted:   {row['predicted_wave_height_ft']:.2f} ft "
                      f"({row['predicted_wave_height_m']:.2f} m)")

                if 'actual_wave_height_m' in row and not pd.isna(row['actual_wave_height_m']):
                    print(f"  Actual:      {row['actual_wave_height_m']*3.28:.2f} ft "
                          f"({row['actual_wave_height_m']:.2f} m)")
                    print(f"  Error:       {abs(row['prediction_error_m']):.2f} m")

                if 'confidence_std' in row:
                    print(f"  95% CI:      [{row['lower_95_m']:.2f}, {row['upper_95_m']:.2f}] m")

    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Run example_train_model() first to train a model")
    except Exception as e:
        print(f"Error during forecast: {e}")


def example_regional_summary():
    """Example: Get regional wave height summary."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Regional Wave Height Summary")
    print("=" * 70)

    buoy_ids = ['44017', '44008', '44013', '44025']
    model_path = 'models/demo_model.pkl'

    try:
        with BuoyForecaster(model_path=model_path) as forecaster:
            summary = forecaster.get_regional_summary(buoy_ids)

            print(f"\nRegional Summary ({summary['num_buoys']} buoys):")
            print("-" * 70)

            pred = summary['predicted']
            print(f"\nPredicted Wave Heights:")
            print(f"  Mean:  {pred['mean_wave_height_m']:.2f} m")
            print(f"  Max:   {pred['max_wave_height_m']:.2f} m")
            print(f"  Min:   {pred['min_wave_height_m']:.2f} m")
            print(f"  Std:   {pred['std_wave_height_m']:.2f} m")

            if 'actual' in summary:
                act = summary['actual']
                print(f"\nActual Wave Heights:")
                print(f"  Mean:  {act['mean_wave_height_m']:.2f} m")
                print(f"  Max:   {act['max_wave_height_m']:.2f} m")
                print(f"  Min:   {act['min_wave_height_m']:.2f} m")

            if 'high_wave_alerts' in summary:
                print(f"\n⚠ HIGH WAVE ALERTS:")
                for buoy in summary['high_wave_alerts']:
                    print(f"  - Buoy {buoy}")

    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Run example_train_model() first to train a model")
    except Exception as e:
        print(f"Error: {e}")


def example_single_buoy_prediction():
    """Example: Predict wave height for a single buoy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Single Buoy Wave Height Prediction")
    print("=" * 70)

    buoy_id = '44017'
    model_path = 'models/demo_model.pkl'

    try:
        with BuoyForecaster(model_path=model_path) as forecaster:
            # Predict using current conditions
            result = forecaster.predict_wave_height(buoy_id)

            print(f"\nPrediction for Buoy {result['buoy_id']}:")
            print("-" * 70)
            print(f"Predicted wave height: {result['predicted_wave_height_ft']:.2f} ft "
                  f"({result['predicted_wave_height_m']:.2f} m)")

            if result['confidence']:
                conf = result['confidence']
                print(f"\n95% Confidence Interval:")
                print(f"  [{conf['lower_95']:.2f}, {conf['upper_95']:.2f}] m")
                print(f"  Std Dev: {conf['std']:.2f} m")

            if 'actual_wave_height_m' in result:
                print(f"\nActual wave height: {result['actual_wave_height_m']:.2f} m")
                print(f"Prediction error:   {abs(result['prediction_error_m']):.2f} m")

    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Run example_train_model() first to train a model")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    import pandas as pd

    print("\n" + "=" * 70)
    print("BUOY WAVE HEIGHT FORECASTING - ML EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate:")
    print("1. Training a forecasting model")
    print("2. Querying current buoy readings")
    print("3. Forecasting wave heights between multiple buoys")
    print("4. Getting regional summaries")
    print("5. Single buoy predictions with confidence intervals")
    print("\nNote: Examples require internet connection to fetch NOAA data")
    print("=" * 70)

    # Run examples
    try:
        # Comment out training to save time after first run
        # example_train_model()

        example_current_readings()

        # Uncomment after training:
        # example_forecast()
        # example_regional_summary()
        # example_single_buoy_prediction()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
