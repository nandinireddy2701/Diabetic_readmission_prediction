
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Diabetic Readmission Prediction API",
    description="Predicts 30-day readmission risk for diabetic patients",
    version="1.0.0"
)

# Load models and artifacts
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model         = joblib.load(f"{BASE}/models/model_xgb_multi.pkl")
feature_names = joblib.load(f"{BASE}/models/feature_names.pkl")
specialty_means = joblib.load(f"{BASE}/models/specialty_means.pkl")
thresholds    = joblib.load(f"{BASE}/models/optimal_thresholds.pkl")

# Input schema
class PatientData(BaseModel):
    age: float
    time_in_hospital: float
    num_lab_procedures: float
    num_procedures: float
    num_medications: float
    number_diagnoses: float
    admission_type_id: float
    discharge_disposition_id: float
    admission_source_id: float
    race: float
    gender: float
    A1c_normal: float
    A1c_elevated: float
    medical_specialty: Optional[str] = "Unknown"
    prior_utilization: Optional[float] = 0.0
    emergency_admission: Optional[float] = 0.0
    long_stay: Optional[float] = 0.0
    high_diagnosis_burden: Optional[float] = 0.0
    total_diabetes_meds: Optional[float] = 0.0
    med_complexity_score: Optional[float] = 0.0
    high_med_burden: Optional[float] = 0.0
    on_insulin: Optional[float] = 0.0
    multiple_med_changes: Optional[float] = 0.0
    meds_x_stay: Optional[float] = 0.0
    age_x_diagnoses: Optional[float] = 0.0
    composite_risk: Optional[float] = 0.0

def get_risk_tier(score):
    if score >= 0.30:
        return "High"
    elif score >= 0.15:
        return "Medium"
    else:
        return "Low"

def get_intervention(tier):
    if tier == "High":
        return [
            "Schedule follow-up call within 24 hours of discharge",
            "Assign care coordinator before discharge",
            "Review and simplify medication regimen",
            "Arrange home health visit",
            "Ensure medication counseling completed"
        ]
    elif tier == "Medium":
        return [
            "Schedule follow-up appointment within 7 days",
            "Provide medication counseling",
            "Share discharge instructions clearly",
            "Call patient within 48 hours"
        ]
    else:
        return [
            "Standard discharge process",
            "Provide patient education materials",
            "Schedule routine follow-up appointment"
        ]

def get_risk_color(tier):
    if tier == "High":
        return "red"
    elif tier == "Medium":
        return "yellow"
    else:
        return "green"

@app.get("/")
def home():
    return {
        "message": "Diabetic Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST — predict readmission risk for one patient",
            "/health":  "GET  — check API health",
            "/docs":    "GET  — interactive API documentation"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost multi-source",
        "features": len(feature_names)
    }

@app.post("/predict")
def predict(patient: PatientData):
    # Build feature dict
    input_data = {f: 0.0 for f in feature_names}

    # Fill in provided values
    patient_dict = patient.dict()

    # Apply Target Encoding for medical_specialty
    overall_mean = 0.0892
    specialty = patient_dict.pop("medical_specialty", "Unknown")
    input_data["specialty_risk"] = float(
        specialty_means.get(specialty, overall_mean)
    )

    # Fill other features
    for key, value in patient_dict.items():
        if key in input_data and value is not None:
            input_data[key] = float(value)

    # Create DataFrame
    X = pd.DataFrame([input_data])[feature_names]

    # Predict
    risk_score = float(model.predict_proba(X)[0][1])
    risk_tier  = get_risk_tier(risk_score)
    color      = get_risk_color(risk_tier)
    interventions = get_intervention(risk_tier)

    return {
        "risk_score":      round(risk_score, 4),
        "risk_percentage": f"{risk_score * 100:.1f}%",
        "risk_tier":       risk_tier,
        "color":           color,
        "interventions":   interventions,
        "model":           "XGBoost multi-source",
        "threshold":       0.30
    }