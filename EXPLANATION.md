# Technical Explanation: How the FinTech Money Mule Detector Works

This document provides a complete, step-by-step breakdown of how the **FinTech Money Mule Account Detection System** works, from synthetic data simulation to machine learning model training, evaluation, and real-time frontend inference.

---

## 🧭 System Overview

Money mules are bank account holders who transfer illegally acquired funds on behalf of fraudsters. This program simulates retail banking account telemetry, builds a machine learning pipeline using **scikit-learn**, evaluates model accuracy, saves the trained pipeline, and provides an interactive web dashboard for real-time risk simulation.

```
+-----------------------------------------------------------------------+
|                         PROGRAM LIFECYCLE                             |
+-----------------------------------------------------------------------+
|  1. Data Simulation     ---> Generates 5,000 synthetic account profiles  |
|  2. Feature Engineering  ---> Preprocesses numerical & categorical data|
|  3. Model Training      ---> Fits HistGradientBoostingClassifier      |
|  4. Evaluation & Export ---> Computes metrics, exports PNG and PKL    |
|  5. Web Frontend UI     ---> Interactive risk calculator & UI dashboard|
+-----------------------------------------------------------------------+
```

---

## 1. Enterprise Data Simulation Phase

The program uses `numpy` statistical distributions to simulate **5,000 synthetic accounts** across three realistic behavioral profiles:

### A. Profile 1: Legitimate Retail Accounts (3,500 Samples / 70%)
- **Inward Transaction Count (24 Hours)**: Modeled with a **Poisson distribution** ($\lambda = 3$), representing normal daily transaction activity.
- **Inward-to-Outward Fan Ratio**: Modeled with a **Normal distribution** ($\mu = 1.2, \sigma = 0.3$). Indicates balanced inward and outward transfers.
- **Average Funds Drain Time**: Modeled with an **Exponential distribution** ($\lambda = 500$) plus a 60-minute baseline. Legitimate users retain money in accounts for days before spending.
- **Zero-Balance Reset Frequency**: Modeled with a **Uniform distribution** ($0.0 \text{ to } 0.2$). Legitimate accounts rarely drop to zero balance repeatedly.
- **Account Age**: Uniformly distributed between 6 and 120 months (0.5 to 10 years).
- **Nighttime Transactions Percentage**: Uniformly distributed between 0% and 15%.
- **Authentication Method**: 70% Biometric, 20% PIN, 10% OTP.
- **Target Label**: `0` (Legitimate).

### B. Profile 2: Classic Burner Mule Accounts (1,000 Samples / 20%)
- **Inward Transaction Count (24 Hours)**: High transaction volume ($\text{Poisson}(\lambda = 45)$).
- **Inward-to-Outward Fan Ratio**: High fan-out ($\text{Normal}(\mu = 6.5, \sigma = 1.5)$), meaning money is split and sent to many external accounts.
- **Average Funds Drain Time**: Extremely short ($\text{Exponential}(\lambda = 15) + 2$ minutes). Money is transferred out almost immediately upon arrival.
- **Zero-Balance Reset Frequency**: High ($0.7 \text{ to } 1.0$). Accounts are continuously drained to zero.
- **Account Age**: Freshly created accounts ($0 \text{ to } 3$ months old).
- **Nighttime Transactions Percentage**: High night activity ($60\% \text{ to } 100\%$).
- **Authentication Method**: 50% OTP, 45% PIN, 5% Biometric.
- **Target Label**: `1` (Money Mule).

### C. Profile 3: Compromised Sleeper Mule Accounts (500 Samples / 10%)
- **Behavioral Shift**: Mimics older accounts ($36 \text{ to } 120$ months old) that were recently bought, rented, or compromised via remote access tools.
- **Inward Transaction Count**: Spikes to $\text{Poisson}(\lambda = 35)$ per 24 hours.
- **Average Funds Drain Time**: Rapid drain ($\text{Exponential}(\lambda = 25) + 5$ minutes).
- **Authentication Method**: 70% OTP (indicating remote takeover/SMS intercept), 30% PIN.
- **Target Label**: `1` (Money Mule).

After generating the three datasets, they are combined using `pd.concat`, randomly shuffled using `.sample(frac=1)`, and exported to `fintech_mule_dataset.csv`.

---

## 2. Machine Learning Preprocessing Pipeline

Machine learning algorithms require clean, properly scaled numerical inputs and encoded categorical variables. The program constructs a scikit-learn `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
```

1. **`StandardScaler`**: Standardizes numerical features by removing the mean and scaling to unit variance ($z = \frac{x - \mu}{\sigma}$). This prevents features with large numeric scales (like drain time in minutes) from overwhelming features with small numeric scales (like zero balance frequency).
2. **`OneHotEncoder`**: Converts the text categorical attribute `Auth_Method` into three binary columns (`Auth_Method_Biometric`, `Auth_Method_PIN`, `Auth_Method_OTP`).

---

## 3. Classifier Model: `HistGradientBoostingClassifier`

The dataset is passed to a **Histogram-Based Gradient Boosting Classifier**:

```python
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        l2_regularization=1.5,
        random_state=42
    ))
])
```

### Why Histogram-Based Gradient Boosting?
- **Speed & Efficiency**: Discretizes continuous numeric features into integer bins (256 bins), significantly accelerating tree building.
- **Non-Linear Pattern Recognition**: Capable of detecting complex non-linear combinations of risk indicators (e.g., *Short Drain Time* **AND** *High Fan Ratio* **AND** *Fresh Account Age*).
- **L2 Regularization**: Prevents model overfitting by penalizing overly complex decision trees (`l2_regularization=1.5`).

---

## 4. Model Training & Evaluation Phase

1. **Stratified Train-Test Split**: The dataset is split into 80% training (4,000 accounts) and 20% test (1,000 accounts), preserving class proportions (`stratify=y`).
2. **Model Training**: `model_pipeline.fit(X_train, y_train)` fits both the feature scaling transformers and the gradient boosting classifier.
3. **Performance Metrics Calculation**:
   - **Accuracy**: Proportion of correct predictions overall ($\frac{TP + TN}{TP + TN + FP + FN}$).
   - **Precision**: Proportion of flagged accounts that are actual mules ($\frac{TP}{TP + FP}$).
   - **Recall**: Proportion of actual mules correctly caught by the model ($\frac{TP}{TP + FN}$).
   - **F1 Score**: Harmonic mean of Precision and Recall ($\frac{2 \cdot P \cdot R}{P + R}$).
   - **ROC-AUC**: Probability that the model ranks a random mule account higher than a random legitimate account.
4. **Visual & Artifact Export**:
   - Renders a confusion matrix heatmap using `seaborn` and saves it as `advanced_mule_cm.png`.
   - Serializes the entire trained pipeline object into `advanced_mule_pipeline.pkl` using `joblib.dump()`.

---

## 5. Real-Time Frontend Inference Engine (`index.html`)

The interactive web dashboard allows users to test account scenarios in real time:

- **Interactive Sliders**: Users adjust account attributes dynamically.
- **Inference Engine in JavaScript**: Computes a continuous risk percentage score ($0\% \text{ to } 100\%$) based on weighted feature risk signals matching the ML model's decision boundaries.
- **Status Classification**: Accounts with risk scores $> 40\%$ are flagged as **MONEY MULE ACCOUNT DETECTED** (red status badge), while scores $\le 40\%$ display **LEGITIMATE ACCOUNT** (green status badge).

---

## 📋 Summary of Output Files

| File Name | Description |
| :--- | :--- |
| `train_mule_detector.py` | Python script that executes dataset generation, pipeline preprocessing, model fitting, and evaluation. |
| `fintech_mule_dataset.csv` | Full dataset containing 5,000 simulated account records. |
| `advanced_mule_cm.png` | Seaborn heatmap plot visualizing True Positives, True Negatives, False Positives, and False Negatives. |
| `advanced_mule_pipeline.pkl` | Serialized scikit-learn model pipeline object ready for deployment. |
| `index.html` | Clean single-page interactive risk calculator web application. |
| `README.md` | GitHub repository documentation. |
| `EXPLANATION.md` | Complete technical breakdown of the system architecture and code logic. |
