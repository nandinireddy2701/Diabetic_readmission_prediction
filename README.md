# Diabetic Readmission Prediction & Decision-Support System

A machine learning platform to predict 30-day hospital readmissions for diabetic patients
using clinical EHR data, Social Determinants of Health (SDOH), and medication signals.

Yeshiva University | DAV Capstone Project | Group 3 - Predictive Healthcare / Applied Machine Learning

## Project Overview
Diabetes is one of the most prevalent chronic conditions in the United States, affecting 
millions of patients and placing enormous strain on the healthcare system.

Hospital readmissions represent a major financial burden hospitals face CMS penalties
under the Hospital Readmission Reduction Program (HRRP) when rates exceed benchmarks.
This project builds an end-to-end ML platform that:
- Predicts 30-day readmission risk for diabetic patients
- Fuses 3 data sources: EHR + SDOH + Medication signals
- Deploys an interactive risk stratification dashboard
- Uses SHAP explainability for clinical decision support

The key innovation of this project is the multi-modal data fusion approach.
Most existing readmission models rely solely on clinical data. By incorporating
SDOH and medication signals, this model captures a more complete picture of
patient risk - including socioeconomic barriers that clinical data alone cannot reflect.

The final deliverable is a cloud-deployed risk stratification dashboard that helps
hospital administrators and care coordination teams identify high-risk patients
before discharge and prioritize intervention resources using data-driven decision making.

**Research Question:**
Can integrating clinical EHR data, Social Determinants of Health, and
medication/polypharmacy signals into a unified machine learning model improve
30-day readmission prediction for diabetic patients compared to models relying
on clinical features alone?

## Weekly Progress
### Week 1 & 2 — Project Setup 
- Project folder created
- All 9 folders created (api, dashboard, data, etl, explainability, features, models, notebooks, tests)
- Conda environment `Diabetic_readmission_prediction` created
- All libraries installed (pandas, sklearn, xgboost, shap + more)
- Jupyter kernel registered
- requirements.txt generated
- Git initialized & configured
- .gitignore created
- .env created
- All 3 datasets downloaded into `data/raw/`
- Setup test notebook passed

### Week 3 — ETL Pipeline & Data Cleaning 
- Loaded UCI dataset — 101,766 rows, 50 columns
- Replaced all `?` with NaN
- Dropped high-missing columns (weight, max_glu_serum, A1Cresult, payer_code)
- Filled remaining missing values
- Created target variable `readmitted_30` (11.2% positive rate)
- Converted age ranges to numbers
- Simplified 700+ diagnosis codes into 9 categories
- Encoded all categorical columns
- Loaded CDC PLACES SDOH data
- Pivoted SDOH to wide format
- Created state level SDOH averages
- Joined SDOH data to main dataset
- Saved all processed datasets to `data/processed/`

### Week 4 — EDA & Feature Engineering 
- `02_eda.ipynb` — 7 EDA plots created and saved
- `03_feature_engineering.ipynb` — Feature engineering complete
- 5 medication features engineered
- 5 clinical risk features engineered
- 32 total features across 4 groups (clinical + SDOH + medication + risk)
- SMOTE applied — balanced to 90,409 each class
- Features scaled with StandardScaler
- Feature store saved (scaled + unscaled)
- Scaler saved for deployment

### Week 5 — Model Development (upcoming)
- Train Logistic Regression (clinical only — baseline)
- Train Random Forest (clinical only)
- Train XGBoost (clinical only)
- Train XGBoost (multi-source — all 32 features)
- Model comparison table (ROC-AUC, Recall, Precision, F1)

### Week 6 — Explainability & Tuning (upcoming)
- SHAP summary plot + waterfall charts
- GridSearchCV hyperparameter tuning
- Best model saved with joblib

### Week 7 — Azure Deployment & Power BI (upcoming)
- Deploy model to Azure ML
- Build REST API with FastAPI
- Power BI dashboard — 3 pages

### Week 8 — Paper & Presentation (upcoming)
- Write academic paper (6–8 pages APA)
- Build PowerPoint presentation (7 slides)
- Final demo rehearsal & submission
