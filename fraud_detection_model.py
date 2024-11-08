# fraud_detection_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import os


def train_fraud_detection_model():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)

    # Check for training data
    data_file = os.path.join("data", "transactions.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Training data not found at {data_file}. Please run generate_data.py first."
        )

    print(f"Loading training data from {data_file}")
    data = pd.read_csv(data_file)

    # Verify required columns are present
    required_columns = [
        'Transaction Amount',
        'Merchant Category',
        'Payment Method',
        'Card Type',
        'CVV Verification',
        'Device Information',
        'Location',
        'Transaction Type',
        'Discount/Promo Code',
        'Fraud Flag'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Training data missing required columns: {', '.join(missing_columns)}")

    print("Preparing features...")
    # Define features and target variable
    feature_columns = required_columns[:-1]  # All except Fraud Flag
    X = data[feature_columns]
    y = data['Fraud Flag']

    # Define numerical and categorical features
    numerical_features = ['Transaction Amount']
    categorical_features = [
        'Merchant Category',
        'Payment Method',
        'Card Type',
        'CVV Verification',
        'Device Information',
        'Location',
        'Transaction Type',
        'Discount/Promo Code'
    ]

    # Create preprocessing steps
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.2%}")
    print(f"Test accuracy: {test_score:.2%}")

    # Save the model
    model_file = os.path.join("models", "fraud_detection_model.pkl")
    joblib.dump(model_pipeline, model_file)
    print(f"Model saved to {model_file}")

    return model_pipeline


if __name__ == "__main__":
    train_fraud_detection_model()