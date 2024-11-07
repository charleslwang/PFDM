import streamlit as st
import os
from fraud_detection import identify_fraud_in_csv

st.title("Credit Card Fraud Detection App")

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True)

if st.button("Process Files"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary directory
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Identify fraud in the uploaded CSV
            identify_fraud_in_csv(file_path)
            st.success(f"Processed {uploaded_file.name}")
    else:
        st.warning("Please upload at least one CSV file.")
