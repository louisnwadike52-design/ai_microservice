#!/usr/bin/env python3
"""
Test script for the add-transaction endpoint
"""

import requests
import json
from datetime import datetime, timedelta

# Base URL for your API
BASE_URL = "http://localhost:8000/api"  # Adjust port if needed

def test_add_transaction():
    """Test the add-transaction endpoint with sample transaction data."""
    
    # Sample transaction data
    sample_transaction = {
        "transaction_id": "txn_12345",
        "amount": -25.50,
        "currency": "USD",
        "description": "Coffee Shop Purchase",
        "merchant": "Starbucks Downtown",
        "category": "Food & Dining",
        "account_id": "acc_67890",
        "timestamp": datetime.now().isoformat()
    }
    
    # Request payload
    payload = {
        "user_id": "test_user_123",
        "transaction": sample_transaction
    }
    
    print(f"🔍 Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make the request
        response = requests.post(
            f"{BASE_URL}/add-transaction",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        # Check the response
        if response.status_code == 200:
            print("✅ Transaction added successfully!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Error adding transaction: {response.status_code}")
            print(f"Response text: {response.text}")
            try:
                error_json = response.json()
                print(f"Response JSON: {json.dumps(error_json, indent=2)}")
            except:
                print("Could not parse response as JSON")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_add_multiple_transactions():
    """Test adding multiple transactions for the same user."""
    
    user_id = "test_user_123"
    
    transactions = [
        {
            "transaction_id": "txn_001",
            "amount": -15.99,
            "currency": "USD",
            "description": "Lunch",
            "merchant": "McDonald's",
            "category": "Food & Dining"
        },
        {
            "transaction_id": "txn_002",
            "amount": -89.99,
            "currency": "USD",
            "description": "Grocery Shopping",
            "merchant": "Whole Foods",
            "category": "Groceries"
        },
        {
            "transaction_id": "txn_003",
            "amount": 2500.00,
            "currency": "USD",
            "description": "Salary Deposit",
            "merchant": "ABC Company",
            "category": "Income"
        }
    ]
    
    for i, transaction in enumerate(transactions, 1):
        payload = {
            "user_id": user_id,
            "transaction": transaction
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/add-transaction",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Transaction {i}/3 added successfully!")
            else:
                print(f"❌ Error adding transaction {i}/3: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request {i}/3 failed: {e}")

if __name__ == "__main__":
    print("Testing add-transaction endpoint...")
    print("\n1. Testing single transaction:")
    test_add_transaction()
    
    print("\n2. Testing multiple transactions:")
    test_add_multiple_transactions()
    
    print("\nDone!") 