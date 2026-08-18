# Technical Explanation: End-to-End ML Architecture & Working

This document provides a complete technical breakdown of how the **FinTech Money Mule Account Detection System** operates as a real end-to-end Machine Learning web application.

---

## 🧭 End-to-End System Architecture

```text
+-------------------------------------------------------------------------------+
|                            END-TO-END REQUEST FLOW                            |
+-------------------------------------------------------------------------------+
| 1. User Input         ---> User adjusts sliders / selects sample on UI        |
| 2. Frontend Request   ---> index.html sends async POST /predict to FastAPI    |
| 3. Input Validation   ---> Pydantic validates datatypes and ranges            |
| 4. DataFrame Formatter---> Constructs DataFrame matching exact model columns   |
| 5. Pipeline Transform ---> StandardScaler & OneHotEncoder fit by Pipeline      |
| 6. Model Inference    ---> HistGradientBoostingClassifier computes probability|
| 7. JSON Response      ---> Returns prediction, probability, score, reasons    |
| 8. Dashboard Update   ---> UI renders risk gauge, level, and ML explanations  |
+-------------------------------------------------------------------------------+
```

---

## 1. Python FastAPI Backend (`backend/main.py`)

The backend is built using **FastAPI** and **Uvicorn**.

### A. Input Data Contract (`PredictRequest`)
The API defines a strict Pydantic model matching the exact feature names and data types expected by `advanced_mule_pipeline.pkl`:

- `Inward_Tx_Count_24h`: Incoming transactions count in 24 hours (Float)
- `In_Out_Fan_Ratio`: Inward to outward transfer destination ratio (Float)
- `Avg_Drain_Time_Mins`: Average funds drain time in minutes (Float)
- `Zero_Balance_Reset_Freq`: Zero balance reset frequency ($0.0 \text{ to } 1.0$)
- `Account_Age_Months`: Account age in months (Float)
- `Night_Tx_Percentage`: Nighttime transaction percentage ($0.0 \text{ to } 1.0$)
- `IP_Change_Count_24h`: Unique IP address changes in 24 hours (Float)
- `Auth_Method`: Authentication method string (`Biometric`, `PIN`, `OTP`)

### B. Prediction Execution (`POST /predict`)
When a request arrives:
1. Constructs a pandas DataFrame with exact column order:
   `['Inward_Tx_Count_24h', 'In_Out_Fan_Ratio', 'Avg_Drain_Time_Mins', 'Zero_Balance_Reset_Freq', 'Account_Age_Months', 'Night_Tx_Percentage', 'IP_Change_Count_24h', 'Auth_Method']`
2. Executes `model_pipeline.predict(df)` and `model_pipeline.predict_proba(df)`.
3. Extracts probability for Class 1 (Money Mule).
4. Computes `risk_score` ($0 \text{ to } 100$) and `risk_level` (`LOW`, `MEDIUM`, `HIGH`).
5. Evaluates input parameters to construct dynamic feature risk explanations.
6. Returns JSON:
```json
{
  "prediction": "HIGH RISK",
  "is_mule": true,
  "probability": 0.9989,
  "risk_score": 100,
  "risk_level": "HIGH",
  "reasons": [
    "Rapid funds drain time (15 minutes)",
    "High zero-balance reset frequency (85%)",
    "High inward transaction count (45 in 24 hours)"
  ],
  "disclaimer": "Evaluation based on synthetic dataset distributions. Real-world banking validation required."
}
```

---

## 2. Frontend Dashboard Integration (`index.html`)

- **Asynchronous Fetching**: Makes non-blocking `fetch()` calls to `${API_BASE_URL}/predict` whenever inputs change.
- **Dynamic API Base URL**: Supports configurable `window.VITE_API_URL` or defaults to `http://localhost:8000`.
- **Health Monitoring & Graceful Error Handling**: Periodically checks `/health`. If the server is offline or unreachable, displays a clean connection warning banner without crashing JavaScript.

---

## 3. Synthetic Data Limitation Disclosure

> 📌 **Synthetic Data Disclosure:** The model was evaluated on a synthetic dataset and achieved high classification performance under the generated data distributions. Real-world financial deployment requires validation on representative banking transaction telemetry.
