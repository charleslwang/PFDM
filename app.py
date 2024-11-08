import streamlit as st
import os
import pandas as pd
from fraud_detection import identify_fraud_in_csv

# Create necessary directories if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

st.title("Credit Card Fraud Detection App")
st.write("Upload CSV files containing transaction data for fraud detection analysis.")

# Add file uploader with specific file types
uploaded_files = st.file_uploader(
    "Choose CSV files",
    accept_multiple_files=True,
    type=['csv']
)

# Add sample data format information
st.info("""
Expected CSV columns:
- Transaction Amount
- Merchant Category
- Payment Method
- Card Type
- CVV Verification
- Device Information
- Location
- Transaction Type
- Discount/Promo Code
""")

if st.button("Process Files"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Create a unique filename for the uploaded file
            file_path = os.path.join("data", uploaded_file.name)

            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f'Processing {uploaded_file.name}...'):
                try:
                    # Process the file and get results
                    result_file = identify_fraud_in_csv(file_path)

                    # Read and display results
                    results_df = pd.read_csv(result_file)
                    st.success(f"Processed {uploaded_file.name}")

                    # Display fraud detection results
                    fraud_count = len(results_df[results_df['Fraud Prediction'] == 1])
                    st.metric("Detected Fraud Transactions", fraud_count)

                    # Show the results table
                    st.dataframe(results_df[['Transaction ID', 'Transaction Amount',
                                              'Fraud Prediction', 'Fraud Reason']])

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue

            # Cleanup temporary files
            os.remove(file_path)

    else:
        st.warning("Please upload at least one CSV file.")
