import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Initialize FastAPI App
app = FastAPI(
    title="FinTech Money Mule Detector API",
    description="Machine Learning REST API for detecting money mule accounts using trained HistGradientBoosting model.",
    version="1.0.0"
)

# Configure CORS Middleware
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://127.0.0.1:8000,http://localhost:8000,*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Model Pipeline
MODEL_PATH_LOCATIONS = [
    os.path.join(os.path.dirname(__file__), "advanced_mule_pipeline.pkl"),
    "advanced_mule_pipeline.pkl",
    "../advanced_mule_pipeline.pkl"
]

model_pipeline = None

for path in MODEL_PATH_LOCATIONS:
    if os.path.exists(path):
        try:
            model_pipeline = joblib.load(path)
            print(f"[SUCCESS] Loaded model pipeline from: {path}")
            break
        except Exception as e:
            print(f"[WARNING] Failed to load model from {path}: {e}")

if model_pipeline is None:
    print("[ERROR] Could not load advanced_mule_pipeline.pkl from any location.")


# Pydantic Request Model
class PredictRequest(BaseModel):
    Inward_Tx_Count_24h: float = Field(..., ge=0, description="Incoming transactions count in 24 hours")
    In_Out_Fan_Ratio: float = Field(..., ge=0.0, description="Inward to outward transfer destination ratio")
    Avg_Drain_Time_Mins: float = Field(..., ge=0.0, description="Average funds drain time in minutes")
    Zero_Balance_Reset_Freq: float = Field(..., ge=0.0, le=1.0, description="Frequency of balance resetting to zero (0.0 to 1.0)")
    Account_Age_Months: float = Field(..., ge=0, description="Account age in months")
    Night_Tx_Percentage: float = Field(..., ge=0.0, le=1.0, description="Percentage of transactions executed at night (0.0 to 1.0)")
    IP_Change_Count_24h: float = Field(..., ge=0, description="Unique IP changes in 24 hours")
    Auth_Method: str = Field(..., description="Authentication method: Biometric, PIN, or OTP")


# Pydantic Response Model
class PredictResponse(BaseModel):
    prediction: str
    is_mule: bool
    probability: float
    risk_score: int
    risk_level: str
    reasons: List[str]
    disclaimer: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FinTech Money Mule Account Detection API",
        "model_loaded": model_pipeline is not None,
        "disclaimer": "Evaluated on a synthetic dataset under generated distributions. Real-world validation required."
    }


@app.get("/health")
def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="ML Model pipeline is not loaded.")
    return {"status": "healthy", "model": "HistGradientBoostingClassifier"}


@app.post("/predict", response_model=PredictResponse)
def predict_mule_risk(payload: PredictRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction model is currently unavailable on the server.")

    # Validate Auth_Method
    valid_auth = ["Biometric", "PIN", "OTP"]
    auth_method = payload.Auth_Method.strip()
    if auth_method not in valid_auth:
        # Fallback to closest match or default
        if "bio" in auth_method.lower():
            auth_method = "Biometric"
        elif "pin" in auth_method.lower():
            auth_method = "PIN"
        else:
            auth_method = "OTP"

    # Construct DataFrame with exact feature names and order expected by trained ColumnTransformer
    input_data = pd.DataFrame([{
        'Inward_Tx_Count_24h': float(payload.Inward_Tx_Count_24h),
        'In_Out_Fan_Ratio': float(payload.In_Out_Fan_Ratio),
        'Avg_Drain_Time_Mins': float(payload.Avg_Drain_Time_Mins),
        'Zero_Balance_Reset_Freq': float(payload.Zero_Balance_Reset_Freq),
        'Account_Age_Months': float(payload.Account_Age_Months),
        'Night_Tx_Percentage': float(payload.Night_Tx_Percentage),
        'IP_Change_Count_24h': float(payload.IP_Change_Count_24h),
        'Auth_Method': auth_method
    }])

    try:
        # Run Prediction using loaded trained pipeline
        pred_class = int(model_pipeline.predict(input_data)[0])
        probabilities = model_pipeline.predict_proba(input_data)[0]
        
        # Probabilities: index 0 is Legit (0), index 1 is Mule (1)
        mule_prob = float(probabilities[1])
        risk_score = int(round(mule_prob * 100))

        # Risk Classification & Level
        if mule_prob >= 0.70:
            prediction_label = "HIGH RISK"
            risk_level = "HIGH"
            is_mule = True
        elif mule_prob >= 0.35:
            prediction_label = "SUSPICIOUS"
            risk_level = "MEDIUM"
            is_mule = True
        else:
            prediction_label = "LEGITIMATE"
            risk_level = "LOW"
            is_mule = False

        # Generate Feature Explanations / Reasons based on inputs & model features
        reasons = []
        if payload.Avg_Drain_Time_Mins < 30:
            reasons.append(f"Rapid funds drain time ({payload.Avg_Drain_Time_Mins:.0f} minutes)")
        elif payload.Avg_Drain_Time_Mins < 90:
            reasons.append(f"Accelerated drain time ({payload.Avg_Drain_Time_Mins:.0f} minutes)")

        if payload.Zero_Balance_Reset_Freq > 0.5:
            reasons.append(f"High zero-balance reset frequency ({payload.Zero_Balance_Reset_Freq * 100:.0f}%)")

        if payload.Inward_Tx_Count_24h > 20:
            reasons.append(f"High inward transaction count ({payload.Inward_Tx_Count_24h} in 24 hours)")

        if payload.In_Out_Fan_Ratio > 3.5:
            reasons.append(f"High outward fan-out ratio ({payload.In_Out_Fan_Ratio:.1f}x)")

        if payload.Night_Tx_Percentage > 0.4:
            reasons.append(f"High nighttime transaction volume ({payload.Night_Tx_Percentage * 100:.0f}%)")

        if payload.IP_Change_Count_24h >= 3:
            reasons.append(f"Frequent IP address velocity ({payload.IP_Change_Count_24h} changes in 24h)")

        if payload.Account_Age_Months <= 3 and is_mule:
            reasons.append(f"Freshly created bank account ({payload.Account_Age_Months} months old)")
        elif payload.Account_Age_Months >= 36 and is_mule:
            reasons.append(f"Aged account exhibiting sudden behavioral anomaly ({payload.Account_Age_Months} months old)")

        if not is_mule and not reasons:
            reasons = [
                "Normal transaction volume and frequency",
                "Standard funds retention time",
                "Low zero-balance reset activity"
            ]

        return PredictResponse(
            prediction=prediction_label,
            is_mule=is_mule,
            probability=round(mule_prob, 4),
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            disclaimer="Evaluation based on synthetic dataset distributions. Real-world banking validation required."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
