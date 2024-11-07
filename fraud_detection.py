import pandas as pd
import joblib
import numpy as np

# Load the trained model
model_pipeline = joblib.load('models/fraud_detection_model.pkl')

def identify_fraud_in_csv(csv_file):
    new_data = pd.read_csv(csv_file)
    print(f"Processing file: {csv_file}")
    print(new_data.head())

    # Preprocessing function for new data
    def preprocess_new_data(data):
        data_processed = data.drop(columns=['Transaction ID', 'User ID', 'Transaction Status', 'Cardholder Name'], errors='ignore')
        data_processed = data_processed.reindex(columns=[
            'Transaction Amount',
            'Merchant Category',
            'Payment Method',
            'Card Type',
            'CVV Verification',
            'Device Information',
            'Location',
            'Transaction Type',
            'Discount/Promo Code'
        ], fill_value='')
        return data_processed

    new_data_processed = preprocess_new_data(new_data)
    predictions = model_pipeline.predict(new_data_processed)
    new_data['Fraud Prediction'] = predictions

    new_data['Fraud Reason'] = np.where(
        (new_data['Fraud Prediction'] == 1) & (new_data['Transaction Amount'] > 300), 'High Amount',
        np.where(
            (new_data['Fraud Prediction'] == 1) & (new_data['Merchant Category'].isin(['Electronics', 'Travel'])), 'Suspicious Merchant',
            'Unusual Behavior'
        )
    )

    output_file = f'fraud_detection_results_{os.path.basename(csv_file)}'
    new_data.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")
    print(new_data[['Transaction ID', 'Fraud Prediction', 'Fraud Reason']].head())
