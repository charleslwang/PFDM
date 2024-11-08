import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def check_dependencies():
    """
    Check and install required dependencies.
    Returns True if all dependencies are satisfied, False otherwise.
    """
    required_packages = {
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib'
    }

    missing_packages = []

    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")

    if missing_packages:
        print("\nMissing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def create_remainder_columns_list():
    """
    Create a compatibility layer for older scikit-learn models.
    """
    import sklearn
    from sklearn.compose._column_transformer import _get_column_indices

    class _RemainderColsList(list):
        def __init__(self, columns):
            self.columns = columns
            super().__init__([columns])

        def __reduce__(self):
            return _RemainderColsList, (self.columns,)

    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList


def load_model_with_compatibility(model_path):
    """
    Load the model with backwards compatibility support.
    """
    import joblib
    try:
        return joblib.load(model_path)
    except AttributeError as e:
        if '_RemainderColsList' in str(e):
            print("Detected older scikit-learn model format. Applying compatibility fix...")
            create_remainder_columns_list()
            return joblib.load(model_path)
        raise e


def identify_fraud_in_csv(csv_file):
    """
    Process a CSV file for fraud detection with dependency checking.
    """
    try:
        # Check dependencies first
        if not check_dependencies():
            raise ImportError("Missing required packages. Please install them as indicated above.")

        # Now that dependencies are confirmed, import the required packages
        import joblib
        import sklearn
        import xgboost

        # Load the trained model
        model_path = os.path.join('models', 'fraud_detection_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file not found. Please ensure you have trained the model first by running fraud_detection_model.py"
            )

        print(f"Current scikit-learn version: {sklearn.__version__}")
        print(f"Current XGBoost version: {xgboost.__version__}")
        print("Loading model with compatibility mode...")

        try:
            model_pipeline = load_model_with_compatibility(model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

        # Read and process the data
        print(f"Reading CSV file: {csv_file}")
        new_data = pd.read_csv(csv_file)

        def preprocess_new_data(data):
            data_copy = data.copy()
            required_columns = [
                'Transaction Amount',
                'Merchant Category',
                'Payment Method',
                'Card Type',
                'CVV Verification',
                'Device Information',
                'Location',
                'Transaction Type',
                'Discount/Promo Code'
            ]

            # Check for missing required columns
            missing_columns = [col for col in required_columns if col not in data_copy.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            # Handle missing values
            for col in required_columns:
                if data_copy[col].dtype == object:
                    data_copy[col] = data_copy[col].fillna('Unknown')
                else:
                    data_copy[col] = data_copy[col].fillna(data_copy[col].mean())

            return data_copy[required_columns]

        print("Preprocessing data...")
        new_data_processed = preprocess_new_data(new_data)

        print("Making predictions...")
        try:
            predictions = model_pipeline.predict(new_data_processed)
        except Exception as e:
            print(f"Warning: Initial prediction failed, attempting type conversion...")
            predictions = model_pipeline.predict(new_data_processed.astype(float))

        # Add predictions to the original data
        new_data['Fraud Prediction'] = predictions

        # Add fraud reasons
        conditions = [
            (new_data['Fraud Prediction'] == 1) & (new_data['Transaction Amount'] > 300),
            (new_data['Fraud Prediction'] == 1) & (new_data['Merchant Category'].isin(['Electronics', 'Travel'])),
            (new_data['Fraud Prediction'] == 1)
        ]
        choices = ['High Amount', 'Suspicious Merchant', 'Unusual Behavior']
        new_data['Fraud Reason'] = np.select(conditions, choices, default='')

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'fraud_detection_results_{timestamp}_{os.path.basename(csv_file)}')

        # Save results
        new_data.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        return output_file

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)

        # Create error log
        error_log = os.path.join('data', 'error_logs.txt')
        with open(error_log, 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")

        raise Exception(error_msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process CSV file for fraud detection')
    parser.add_argument('csv_file', help='Path to the CSV file to process')

    args = parser.parse_args()

    try:
        output_file = identify_fraud_in_csv(args.csv_file)
        print(f"Processing completed successfully. Results saved to: {output_file}")
    except Exception as e:
        print(f"Processing failed: {str(e)}")