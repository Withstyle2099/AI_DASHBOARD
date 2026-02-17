"""
Predictive AI Model - Customer Purchase Prediction
This model predicts whether a customer will make a purchase based on their behavior.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PREDICTIVE AI MODEL - CUSTOMER PURCHASE PREDICTION")
print("=" * 60)

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
print("\n[1] Generating synthetic customer data...")

n_samples = 500
data = {
    'age': np.random.randint(18, 80, n_samples),
    'monthly_income': np.random.randint(1000, 10000, n_samples),
    'purchase_frequency': np.random.randint(0, 50, n_samples),
    'avg_purchase_amount': np.random.uniform(10, 500, n_samples),
    'website_visits': np.random.randint(0, 100, n_samples),
    'customer_tenure_months': np.random.randint(1, 60, n_samples),
    'email_clicks': np.random.randint(0, 30, n_samples),
    'cart_abandonment_rate': np.random.uniform(0, 1, n_samples)
}

df = pd.DataFrame(data)

# Create target variable (whether customer will purchase)
# Logic: based on features, create a realistic target
df['will_purchase'] = (
    (df['purchase_frequency'] > 10) & 
    (df['website_visits'] > 20) |
    (df['avg_purchase_amount'] > 200) &
    (df['customer_tenure_months'] > 12)
).astype(int)

print(f"[OK] Generated {len(df)} customer records")
print(f"[OK] Features: {list(data.keys())}")
print(f"[OK] Target classes: {df['will_purchase'].unique()}")
print(f"\nDataset preview:")
print(df.head())
print(f"\nTarget distribution:\n{df['will_purchase'].value_counts()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("[2] Exploratory Data Analysis")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nBasic statistics:")
print(df.describe())

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n" + "=" * 60)
print("[3] Preparing data for training...")
print("=" * 60)

# Separate features and target
X = df.drop('will_purchase', axis=1)
y = df['will_purchase']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Training set: {X_train.shape[0]} samples")
print(f"[OK] Testing set: {X_test.shape[0]} samples")

# Scale the features (important for model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[OK] Features scaled using StandardScaler")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 60)
print("[4] Training Predictive Models...")
print("=" * 60)

# Train Random Forest model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("[OK] Model training completed")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("[5] Model Evaluation")
print("=" * 60)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 60)
print("[6] Feature Importance Analysis")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features for Purchase Prediction:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:.<30} {row['importance']:.4f}")

# ============================================================================
# 7. MAKING PREDICTIONS
# ============================================================================
print("\n" + "=" * 60)
print("[7] Making Predictions on New Customers")
print("=" * 60)

# Example: Create new customer data
new_customers = pd.DataFrame({
    'age': [35, 55, 25],
    'monthly_income': [5000, 3000, 2000],
    'purchase_frequency': [15, 5, 30],
    'avg_purchase_amount': [250, 50, 400],
    'website_visits': [45, 10, 80],
    'customer_tenure_months': [24, 3, 36],
    'email_clicks': [15, 2, 20],
    'cart_abandonment_rate': [0.2, 0.8, 0.1]
})

print("\nNew customers to predict:")
print(new_customers)

# Scale and predict
new_customers_scaled = scaler.transform(new_customers)
predictions = model.predict(new_customers_scaled)
probabilities = model.predict_proba(new_customers_scaled)

print("\nPredictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nCustomer {i+1}:")
    print(f"  Will Purchase: {'Yes' if pred == 1 else 'No'}")
    print(f"  Confidence: {max(prob)*100:.2f}%")
    print(f"  Probability breakdown: No={prob[0]:.2%}, Yes={prob[1]:.2%}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("[8] Model Summary")
print("=" * 60)
print(f"""
Model: Random Forest Classifier
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Samples: {len(df)}
Features Used: {len(X.columns)}
Test Accuracy: {accuracy*100:.2f}%
Most Important Feature: {feature_importance.iloc[0]['feature']}

This model can predict whether customers will make a purchase
based on their behavioral patterns.
""")

print("\n[OK] Predictive AI Model Complete!")
print("=" * 60)
