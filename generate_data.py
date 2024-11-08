# generate_data.py
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Initialize Faker for generating fake data
fake = Faker()


def generate_transactions(num_transactions=10000, fraud_probability=0.01):
    merchant_categories = ['Groceries', 'Electronics', 'Clothing', 'Health', 'Travel', 'Entertainment']
    transactions = []

    for _ in range(num_transactions):
        # Generate transaction details
        transaction_id = fake.uuid4()
        user_id = fake.uuid4()
        transaction_amount = round(random.uniform(1, 500), 2)
        transaction_time = fake.date_time_this_year().strftime("%Y-%m-%d %H:%M:%S")
        merchant_id = fake.uuid4()
        merchant_category = random.choice(merchant_categories)
        payment_method = random.choice(['Credit Card', 'Debit Card', 'Digital Wallet'])
        card_type = random.choice(['Visa', 'MasterCard', 'American Express'])
        cardholder_name = fake.name()
        transaction_status = random.choice(['Completed', 'Pending', 'Refunded', 'Failed'])
        billing_address = fake.address().replace('\n', ', ')
        shipping_address = fake.address().replace('\n', ', ')
        cvv_verification = random.choice(['Yes', 'No'])
        ip_address = fake.ipv4()
        device_information = random.choice(['Mobile', 'Desktop'])
        location = f"{fake.city()}, {fake.country()}"
        transaction_type = random.choice(['Purchase', 'Refund'])
        discount_code = random.choice(['SPRING20', 'SUMMER2024', 'FALL2024', 'WINTER2024', ''])
        refund_amount = round(random.uniform(0, 100), 2) if transaction_status == 'Refunded' else 0.00
        fraud_flag = 1 if random.random() < fraud_probability else 0

        transactions.append({
            'Transaction ID': transaction_id,
            'User ID': user_id,
            'Transaction Amount': transaction_amount,
            'Transaction Date/Time': transaction_time,
            'Merchant ID': merchant_id,
            'Merchant Category': merchant_category,
            'Payment Method': payment_method,
            'Card Type': card_type,
            'Cardholder Name': cardholder_name,
            'Transaction Status': transaction_status,
            'Billing Address': billing_address,
            'Shipping Address': shipping_address,
            'CVV Verification': cvv_verification,
            'IP Address': ip_address,
            'Device Information': device_information,
            'Location': location,
            'Transaction Type': transaction_type,
            'Discount/Promo Code': discount_code,
            'Refund Amount': refund_amount,
            'Fraud Flag': fraud_flag
        })

    return pd.DataFrame(transactions)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Generate main training dataset
    print("Generating training data...")
    df = generate_transactions(num_transactions=10000, fraud_probability=0.01)
    training_file = os.path.join("data", "transactions.csv")
    df.to_csv(training_file, index=False)
    print(f"Generated {len(df)} transactions and saved to {training_file}")

    # Generate a small test dataset
    print("\nGenerating test data...")
    test_df = generate_transactions(num_transactions=100, fraud_probability=0.05)
    test_file = os.path.join("data", "test_transactions.csv")
    test_df.to_csv(test_file, index=False)
    print(f"Generated {len(test_df)} test transactions and saved to {test_file}")