"""
Project: Supply Chain Analytics Portfolio
Week 3: Exploratory Data Analysis & Advanced Feature Engineering
Author: Vinay (Data Analyst Candidate)
Description: This script handles data cleaning, handling missing values, 
             temporal feature extraction, lead time calculations, and 
             demand correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")

def load_and_preprocess_data(file_path):
    """Loads dataset and performs initial data cleaning and profiling."""
    print("[INFO] Loading dataset...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[WARNING] {file_path} not found. Generating sample data for demonstration.")
        df = generate_sample_data()
        
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        df['day_of_week'] = df['date'].dt.day_name()
    
    print(f"[INFO] Dataset shape: {df.shape}")
    return df

def generate_sample_data():
    """Generates a synthetic supply chain dataset if local CSV is missing."""
    np.random.seed(42)
    n = 5000
    dates = pd.date_range(start='2024-01-01', periods=n, freq='H')
    df = pd.DataFrame({
        'date': dates,
        'product_id': np.random.choice([f'P{str(i).zfill(3)}' for i in range(1, 20)], n),
        'warehouse_id': np.random.choice(['WH_North', 'WH_South', 'WH_East'], n),
        'units_sold': np.random.poisson(lam=45, size=n),
        'inventory_level': np.random.randint(10, 500, n),
        'unit_price': np.random.uniform(20.0, 150.0, n),
        'lead_time_days': np.random.choice([2, 4, 7, 10], n, p=[0.4, 0.3, 0.2, 0.1]),
        'promotion_active': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    return df

def engineer_features(df):
    """Performs advanced feature engineering for supply chain optimization."""
    print("[INFO] Engineering features...")
    
    # 1. Lead Time Risk Flag (Lead time > 5 days considered high risk)
    df['high_lead_time_risk'] = np.where(df['lead_time_days'] > 5, 1, 0)
    
    # 2. Stockout Risk Flag (Inventory level below 20 units)
    df['stockout_risk'] = np.where(df['inventory_level'] < 20, 1, 0)
    
    # 3. Revenue calculation
    if 'unit_price' in df.columns and 'units_sold' in df.columns:
        df['total_revenue'] = df['units_sold'] * df['unit_price']
        
    # 4. Rolling average demand (7-day window if sorted by date and product)
    df = df.sort_values(['product_id', 'date'])
    df['rolling_demand_7d'] = df.groupby('product_id')['units_sold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    print("[INFO] Feature engineering completed successfully.")
    return df

def save_outputs(df):
    """Saves cleaned dataset and summary statistics."""
    df.to_csv('cleaned_supply_chain_week3.csv', index=False)
    print("[INFO] Saved cleaned dataset to 'cleaned_supply_chain_week3.csv'")
    
    # Summary metrics
    summary = df.groupby('warehouse_id').agg(
        Total_Sales=('units_sold', 'sum'),
        Avg_Inventory=('inventory_level', 'mean'),
        Stockout_Incidents=('stockout_risk', 'sum')
    ).reset_index()
    print("\n--- Warehouse Performance Summary ---")
    print(summary)

if __name__ == "__main__":
    file_name = 'supply_chain_data.csv'
    df_raw = load_and_preprocess_data(file_name)
    df_featured = engineer_features(df_raw)
    save_outputs(df_featured)