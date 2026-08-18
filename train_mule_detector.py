import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. ENTERPRISE-GRADE DATA SIMULATION (5,000 Accounts) ---
# Profile A: Legitimate Accounts (3,500 samples)
legit = pd.DataFrame({
    'Inward_Tx_Count_24h': np.random.poisson(3, 3500),
    'In_Out_Fan_Ratio': np.random.normal(1.2, 0.3, 3500),
    'Avg_Drain_Time_Mins': np.random.exponential(500, 3500) + 60,
    'Zero_Balance_Reset_Freq': np.random.uniform(0, 0.2, 3500),
    'Account_Age_Months': np.random.randint(6, 120, 3500),
    'Night_Tx_Percentage': np.random.uniform(0, 0.15, 3500),
    'IP_Change_Count_24h': np.random.poisson(0.5, 3500),
    'Auth_Method': np.random.choice(['Biometric', 'PIN', 'OTP'], 3500, p=[0.7, 0.2, 0.1]),
    'Target': 0
})

# Profile B: Classic Burner Mule Accounts (1,000 samples)
fast_mule = pd.DataFrame({
    'Inward_Tx_Count_24h': np.random.poisson(45, 1000),
    'In_Out_Fan_Ratio': np.random.normal(6.5, 1.5, 1000),
    'Avg_Drain_Time_Mins': np.random.exponential(15, 1000) + 2,
    'Zero_Balance_Reset_Freq': np.random.uniform(0.7, 1.0, 1000),
    'Account_Age_Months': np.random.randint(0, 3, 1000), 
    'Night_Tx_Percentage': np.random.uniform(0.6, 1.0, 1000), 
    'IP_Change_Count_24h': np.random.poisson(5, 1000), 
    'Auth_Method': np.random.choice(['Biometric', 'PIN', 'OTP'], 1000, p=[0.05, 0.45, 0.50]),
    'Target': 1
})

# Profile C: Compromised "Sleeper" Accounts (500 samples)
sleeper_mule = pd.DataFrame({
    'Inward_Tx_Count_24h': np.random.poisson(35, 500),
    'In_Out_Fan_Ratio': np.random.normal(5.0, 1.2, 500),
    'Avg_Drain_Time_Mins': np.random.exponential(25, 500) + 5,
    'Zero_Balance_Reset_Freq': np.random.uniform(0.6, 0.9, 500),
    'Account_Age_Months': np.random.randint(36, 120, 500), 
    'Night_Tx_Percentage': np.random.uniform(0.5, 0.9, 500),
    'IP_Change_Count_24h': np.random.poisson(3, 500),
    'Auth_Method': np.random.choice(['Biometric', 'PIN', 'OTP'], 500, p=[0.0, 0.3, 0.7]), 
    'Target': 1
})

# Add explicit Subtype column for visualization
legit['Mule_Type'] = 'Legitimate'
fast_mule['Mule_Type'] = 'Fast Burner'
sleeper_mule['Mule_Type'] = 'Sleeper Mule'

# Combine Dataset and Clean
df = pd.concat([legit, fast_mule, sleeper_mule]).sample(frac=1, random_state=42).reset_index(drop=True)
df['In_Out_Fan_Ratio'] = df['In_Out_Fan_Ratio'].clip(lower=0.1)

# Generate synthetic Account ID
df['Account_ID'] = [f"ACC-{100000 + i}" for i in range(len(df))]

# Save the dataset to a CSV file so you can view/submit it
df.to_csv('fintech_mule_dataset.csv', index=False)
print("[1/3] Dataset saved as 'fintech_mule_dataset.csv'")

# --- 2. ML PIPELINE & PREPROCESSING ---
features_df = df.drop(['Target', 'Mule_Type', 'Account_ID'], axis=1)
X = features_df
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['Inward_Tx_Count_24h', 'In_Out_Fan_Ratio', 'Avg_Drain_Time_Mins', 
                    'Zero_Balance_Reset_Freq', 'Account_Age_Months', 'Night_Tx_Percentage', 'IP_Change_Count_24h']
categorical_features = ['Auth_Method']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, l2_regularization=1.5, random_state=42))
])

# --- 3. TRAIN & EVALUATE ---
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
y_probs = model_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)
cm = confusion_matrix(y_test, y_pred)

print("\n--- ADVANCED MULE DETECTION METRICS ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

# --- 4. EXPORT VISUALS & MODEL ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', 
            xticklabels=['Legitimate', 'Mule (Fast & Sleeper)'], 
            yticklabels=['Legitimate', 'Mule (Fast & Sleeper)'])
plt.title('Advanced Mule Detector - Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('advanced_mule_cm.png', dpi=300)
print("[2/3] Confusion matrix saved as 'advanced_mule_cm.png'")
plt.close()

joblib.dump(model_pipeline, 'advanced_mule_pipeline.pkl')
print("[3/3] Model saved as 'advanced_mule_pipeline.pkl'")

# Export summary JSON for Web Application UI integration
summary = {
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    },
    "counts": {
        "total_accounts": len(df),
        "legitimate_count": int((df['Target'] == 0).sum()),
        "fast_mule_count": int((df['Mule_Type'] == 'Fast Burner').sum()),
        "sleeper_mule_count": int((df['Mule_Type'] == 'Sleeper Mule').sum())
    }
}
with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n[SUCCESS] ALL FILES ARE READY FOR DOWNLOAD!")
