"""
Project: Supply Chain Analytics Portfolio
Week 4: Predictive Modeling & Inventory Stockout Classification
Author: Vinay (Data Analyst Candidate)
Description: This script builds a machine learning pipeline (Random Forest) 
             to predict stockout risks and evaluate operational performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def load_data():
    """Loads the engineered dataset from Week 3."""
    try:
        df = pd.read_csv('cleaned_supply_chain_week3.csv')
        print("[INFO] Loaded 'cleaned_supply_chain_week3.csv' successfully.")
    except FileNotFoundError:
        print("[INFO] Cleaned dataset not found. Running quick simulation for Week 4.")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'units_sold': np.random.poisson(45, n),
            'inventory_level': np.random.randint(5, 300, n),
            'unit_price': np.random.uniform(20, 150, n),
            'lead_time_days': np.random.choice([2, 4, 7, 10], n),
            'promotion_active': np.random.choice([0, 1], n),
            'high_lead_time_risk': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'rolling_demand_7d': np.random.uniform(30, 60, n),
            'stockout_risk': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
    return df

def train_stockout_model(df):
    """Trains a Random Forest Classifier to predict stockout risks."""
    print("[INFO] Preparing features for machine learning...")
    
    features = ['units_sold', 'inventory_level', 'unit_price', 'lead_time_days', 
                'promotion_active', 'high_lead_time_risk', 'rolling_demand_7d']
    
    X = df[features]
    y = df['stockout_risk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[INFO] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature Importance Plot
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='crest')
    plt.title('Feature Importance for Stockout Risk Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stockout_feature_importance.png', dpi=300)
    print("[INFO] Saved feature importance plot to 'stockout_feature_importance.png'")
    
    return model

if __name__ == "__main__":
    df_data = load_data()
    trained_model = train_stockout_model(df_data)