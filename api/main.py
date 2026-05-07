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
model           = joblib.load(f"{BASE}/models/model_xgb_multi.pkl")
feature_names   = joblib.load(f"{BASE}/models/feature_names.pkl")
specialty_means = joblib.load(f"{BASE}/models/specialty_means.pkl")

# Load real test data
X_test = np.load(f"{BASE}/data/processed/X_test.npy")
X_test_df = pd.DataFrame(X_test, columns=feature_names)
feature_means = X_test_df.mean().to_dict()
probs_test = model.predict_proba(X_test_df)[:, 1]

# Get exact real patient rows for samples
low_idx    = np.where(probs_test < 0.15)[0][0]
medium_idx = np.where((probs_test >= 0.15) & (probs_test < 0.30))[0][0]
high_idx   = np.where(probs_test >= 0.30)[0][0]

low_patient    = X_test_df.iloc[low_idx].to_dict()
medium_patient = X_test_df.iloc[medium_idx].to_dict()
high_patient   = X_test_df.iloc[high_idx].to_dict()

print(f"Model loaded! Features: {len(feature_names)}")
print(f"Sample patients loaded!")
print(f"   Low:    {probs_test[low_idx]:.4f}")
print(f"   Medium: {probs_test[medium_idx]:.4f}")
print(f"   High:   {probs_test[high_idx]:.4f}")

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
    medical_specialty: Optional[str]        = "Unknown"
    prior_utilization: Optional[float]      = None
    emergency_admission: Optional[float]    = None
    long_stay: Optional[float]              = None
    high_diagnosis_burden: Optional[float]  = None
    total_diabetes_meds: Optional[float]    = None
    on_insulin: Optional[float]             = None
    multiple_med_changes: Optional[float]   = None
    specialty_risk: Optional[float]         = None
    change: Optional[float]                 = None
    diabetesMed: Optional[float]            = None
    insulin_encoded: Optional[float]        = None
    metformin: Optional[float]              = None
    repaglinide: Optional[float]            = None
    glimepiride: Optional[float]            = None
    glipizide: Optional[float]              = None
    glyburide: Optional[float]              = None
    pioglitazone: Optional[float]           = None
    rosiglitazone: Optional[float]          = None
    diag_1_Diabetes: Optional[float]        = None
    diag_1_Digestive: Optional[float]       = None
    diag_1_Genitourinary: Optional[float]   = None
    diag_1_Injury: Optional[float]          = None
    diag_1_Musculoskeletal: Optional[float] = None
    diag_1_Neoplasms: Optional[float]       = None
    diag_1_Other: Optional[float]           = None
    diag_1_Respiratory: Optional[float]     = None
    diag_2_Diabetes: Optional[float]        = None
    diag_2_Digestive: Optional[float]       = None
    diag_2_Genitourinary: Optional[float]   = None
    diag_2_Injury: Optional[float]          = None
    diag_2_Musculoskeletal: Optional[float] = None
    diag_2_Neoplasms: Optional[float]       = None
    diag_2_Other: Optional[float]           = None
    diag_2_Respiratory: Optional[float]     = None
    diag_3_Diabetes: Optional[float]        = None
    diag_3_Digestive: Optional[float]       = None
    diag_3_Genitourinary: Optional[float]   = None
    diag_3_Injury: Optional[float]          = None
    diag_3_Musculoskeletal: Optional[float] = None
    diag_3_Neoplasms: Optional[float]       = None
    diag_3_Other: Optional[float]           = None
    diag_3_Respiratory: Optional[float]     = None
    diag_3_Unknown: Optional[float]         = None
    BPHIGH: Optional[float]                 = None
    CHECKUP: Optional[float]                = None
    CHOLSCREEN: Optional[float]             = None
    DEPRESSION: Optional[float]             = None
    DIABETES: Optional[float]               = None
    OBESITY: Optional[float]                = None

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

def format_patient(patient_dict, risk_score, risk_tier):
    color_map = {"High": "red", "Medium": "yellow", "Low": "green"}
    return {
        "age":                    int(patient_dict.get("age", 0)),
        "time_in_hospital":       int(patient_dict.get("time_in_hospital", 0)),
        "num_medications":        int(patient_dict.get("num_medications", 0)),
        "number_diagnoses":       int(patient_dict.get("number_diagnoses", 0)),
        "prior_utilization":      round(float(patient_dict.get("prior_utilization", 0)), 2),
        "risk_score":             round(risk_score, 4),
        "risk_percentage":        f"{risk_score*100:.1f}%",
        "risk_tier":              risk_tier,
        "color":                  color_map[risk_tier],
        "interventions":          get_intervention(risk_tier)
    }

@app.get("/")
def home():
    return {
        "message": "Diabetic Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict":       "POST — predict readmission risk",
            "/health":        "GET  — check API health",
            "/sample/low":    "GET  — low risk sample patient",
            "/sample/medium": "GET  — medium risk sample patient",
            "/sample/high":   "GET  — high risk sample patient",
            "/docs":          "GET  — interactive documentation"
        }
    }

@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "model":    "XGBoost multi-source",
        "features": len(feature_names)
    }

@app.get("/sample/low")
def sample_low():
    risk_score = float(probs_test[low_idx])
    risk_tier  = get_risk_tier(risk_score)
    result = format_patient(low_patient, risk_score, risk_tier)
    result['_full_features'] = {k: float(v) for k, v in low_patient.items()}
    return result

@app.get("/sample/medium")
def sample_medium():
    risk_score = float(probs_test[medium_idx])
    risk_tier  = get_risk_tier(risk_score)
    result = format_patient(medium_patient, risk_score, risk_tier)
    result['_full_features'] = {k: float(v) for k, v in medium_patient.items()}
    return result

@app.get("/sample/high")
def sample_high():
    risk_score = float(probs_test[high_idx])
    risk_tier  = get_risk_tier(risk_score)
    result = format_patient(high_patient, risk_score, risk_tier)
    result['_full_features'] = {k: float(v) for k, v in high_patient.items()}
    return result

@app.post("/predict")
def predict(patient: PatientData):

    # Start with real test data means
    input_data = feature_means.copy()

    patient_dict = patient.dict()

    # Apply Target Encoding for medical_specialty
    overall_mean = float(np.mean(list(specialty_means)))
    specialty = patient_dict.pop("medical_specialty", "Unknown")
    input_data["specialty_risk"] = float(
        specialty_means.get(specialty, overall_mean)
    )

    # Override specialty_risk if directly provided
    if patient_dict.get('specialty_risk') is not None:
        input_data['specialty_risk'] = float(patient_dict['specialty_risk'])

    # Handle insulin_encoded separately
    if patient_dict.get('insulin_encoded') is not None:
        input_data['insulin'] = float(patient_dict['insulin_encoded'])

    # Override all provided fields
    all_feature_fields = [
        'age', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_diagnoses',
        'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'race', 'gender',
        'A1c_normal', 'A1c_elevated',
        'prior_utilization', 'emergency_admission',
        'long_stay', 'high_diagnosis_burden',
        'total_diabetes_meds', 'on_insulin',
        'multiple_med_changes', 'change', 'diabetesMed',
        'metformin', 'repaglinide', 'glimepiride',
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
        'diag_1_Diabetes', 'diag_1_Digestive', 'diag_1_Genitourinary',
        'diag_1_Injury', 'diag_1_Musculoskeletal', 'diag_1_Neoplasms',
        'diag_1_Other', 'diag_1_Respiratory',
        'diag_2_Diabetes', 'diag_2_Digestive', 'diag_2_Genitourinary',
        'diag_2_Injury', 'diag_2_Musculoskeletal', 'diag_2_Neoplasms',
        'diag_2_Other', 'diag_2_Respiratory',
        'diag_3_Diabetes', 'diag_3_Digestive', 'diag_3_Genitourinary',
        'diag_3_Injury', 'diag_3_Musculoskeletal', 'diag_3_Neoplasms',
        'diag_3_Other', 'diag_3_Respiratory', 'diag_3_Unknown',
        'BPHIGH', 'CHECKUP', 'CHOLSCREEN',
        'DEPRESSION', 'DIABETES', 'OBESITY'
    ]

    for field in all_feature_fields:
        value = patient_dict.get(field)
        if value is not None and field in input_data:
            input_data[field] = float(value)

    # Recalculate engineered features
    input_data['med_complexity_score'] = (
        input_data['num_medications']  * 0.4441 +
        input_data['num_procedures']   * 0.0018 +
        input_data['number_diagnoses'] * 0.5541
    )
    input_data['high_med_burden'] = (
        1.0 if input_data['num_medications'] >= 18 else 0.0
    )
    input_data['meds_x_stay'] = (
        input_data['num_medications'] *
        input_data['time_in_hospital']
    )
    input_data['age_x_diagnoses'] = (
        input_data['age'] *
        input_data['number_diagnoses']
    )
    input_data['obesity_x_diabetes'] = (
        input_data['OBESITY'] *
        input_data['DIABETES']
    )
    input_data['composite_risk'] = (
        input_data['high_med_burden'] +
        input_data['high_diagnosis_burden'] +
        input_data['long_stay'] +
        input_data['emergency_admission']
    )

    # Create DataFrame with correct feature order
    X = pd.DataFrame([input_data])[feature_names]

    # Predict
    risk_score    = float(model.predict_proba(X)[0][1])
    risk_tier     = get_risk_tier(risk_score)
    color         = get_risk_color(risk_tier)
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

@app.post("/predict_raw")
def predict_raw(features: dict):
    # Start with real test data means
    input_data = feature_means.copy()
    
    # Only use known feature names — ignore extra fields
    for key, value in features.items():
        if key in feature_names:
            try:
                input_data[key] = float(value)
            except (TypeError, ValueError):
                pass
    
    X = pd.DataFrame([input_data])[feature_names]
    risk_score    = float(model.predict_proba(X)[0][1])
    risk_tier     = get_risk_tier(risk_score)
    color         = get_risk_color(risk_tier)
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