# Diabetic Hospital Readmission Prediction & Decision-Support System

Predicting 30-day hospital readmissions for diabetic patients using Machine Learning and Social Determinants of Health.

Yeshiva University | DAV Capstone Project | Predictive Healthcare / Applied Machine Learning
Student: Nandini Reddy Basupally

## Project Overview

Hospital readmissions within 30 days of discharge are one of the biggest quality and cost challenges in healthcare. Hospitals face financial penalties under the CMS Hospital Readmission Reduction Program (HRRP) when readmission rates exceed national benchmarks.

This project builds an end-to-end machine learning platform that predicts whether a diabetic patient will be readmitted within 30 days before they are discharged. This gives care teams time to intervene with extra follow-up calls, medication counseling, and home visits.

What makes this project unique is that most existing systems only use clinical data. We fuse three data sources together — clinical EHR records, Social Determinants of Health (SDOH), and medication signals — to build a more complete picture of patient risk.

## Research Question

Can integrating clinical EHR data, Social Determinants of Health (SDOH), and medication/polypharmacy signals into a unified machine learning model improve 30-day readmission prediction for diabetic patients compared to models relying on clinical features alone?

Thesis: A multi-source machine learning model that fuses structured EHR records with socioeconomic and medication data outperforms single-source clinical models in predicting 30-day diabetic readmissions, while remaining interpretable and deployable for real-world clinical decision support.

## Datasets Used

Dataset 1 — UCI Diabetes 130-US Hospitals (1999–2008)
Source: https://www.kaggle.com/datasets/brandao/diabetes
Size: 101,766 patient records from 130 US hospitals
Purpose: Primary dataset — core EHR clinical records

Dataset 2 — CDC PLACES County Data
Source: https://places.cdc.gov/
Size: 229,298 rows of county-level health indicators
Purpose: Social Determinants of Health — obesity, depression, blood pressure, diabetes prevalence by county

Dataset 3 — CMS HRRP Hospital Data
Source: https://data.cms.gov/provider-data/dataset/9n3s-kdb3
Purpose: Hospital readmission rates and CMS penalty benchmarks

Note: Large files over 50MB are stored locally and excluded from GitHub via .gitignore

## Project Structure

The project is organized into the following folders:
- notebooks/ — all 7 Jupyter notebooks one per project phase
- data/raw/ — original downloaded datasets
- data/processed/ — cleaned and engineered datasets
- models/ — saved model pkl files and scalers
- dashboard/ — all charts and Tableau export CSV files
- explainability/ — SHAP plots (bar, beeswarm, waterfall)
- api/ — FastAPI inference endpoint
- tests/ — unit tests

## How to Run

Step 1 — Clone the repository
git clone https://github.com/nandinireddy2701/Diabetic_readmission_prediction.git
cd Diabetic_readmission_prediction

Step 2 — Create Conda environment
conda create -n Diabetic_readmission_prediction python=3.11 -y
conda activate Diabetic_readmission_prediction

Step 3 — Install dependencies
pip install -r requirements.txt

Step 4 — Download datasets and place in data/raw/
- diabetic_data.csv from https://www.kaggle.com/datasets/brandao/diabetes
- SDOH_data.csv from https://places.cdc.gov/
- Unplanned_Hospital_Visits.csv from https://data.cms.gov/provider-data/dataset/9n3s-kdb3

Step 5 — Run notebooks in order
00_setup_test.ipynb → 01_etl_pipeline.ipynb → 02_eda.ipynb →
03_feature_engineering.ipynb → 04_model_training.ipynb →
05_explainability.ipynb → 06_model_improvement.ipynb → 09_optuna_tuning.ipynb

Step 6 — Start the FastAPI
```bash
python -m uvicorn api.main:app --reload --port 8000
```
Access at: http://127.0.0.1:8000
Interactive docs at: http://127.0.0.1:8000/docs

Step 7 — Start the Streamlit Dashboard (open a new terminal)
```bash
python -m streamlit run dashboard/streamlit_app.py
```
Access at: http://localhost:8501

## Project Phases

### Phase 1 — Project Setup 

- Python 3.11 Conda environment created and configured on Mac
- All required libraries installed including pandas, scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn, FastAPI, Streamlit
- Project folder structure created with organized folders
- All 3 datasets downloaded into data/raw/
- Git initialized and GitHub repository connected
- Setup verification notebook confirmed all libraries working correctly

### Phase 2 — Data Collection and ETL Pipeline

- Loaded 101,766 patient records from 130 US hospitals
- Replaced all question marks with NaN since pandas cannot detect them as missing values
- Dropped 4 columns with 40 to 97 percent missing data — weight, max_glu_serum, A1Cresult, payer_code
- Created binary target variable readmitted_30 where 1 means readmitted within 30 days and 0 means not readmitted
- Patient deduplication — kept first encounter only, reducing 101,766 to 69,973 unique patients — prevents data leakage
- Removed deceased and hospice patients with discharge codes 11, 13, 14, 18, 19, 20, 21
- Preserved A1C result as 3 binary flags — A1c_measured, A1c_normal, A1c_elevated
- Converted age text ranges like [50-60) to numeric midpoints like 55
- Simplified 700+ ICD-9 diagnosis codes into 9 meaningful disease categories — Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other
- Label encoded all categorical columns for machine learning compatibility
- Loaded CDC PLACES SDOH data, computed state-level averages, joined 6 SDOH features — BPHIGH, CHECKUP, CHOLSCREEN, DEPRESSION, DIABETES, OBESITY
- Final cleaned dataset saved with 69,973 rows and 69 columns

### Phase 3 — Exploratory Data Analysis (EDA) and Feature Engineering 

- Created 9 visualizations saved to dashboard/ folder — target distribution, age analysis, clinical features, SDOH features, correlation heatmap, diagnosis analysis, inpatient analysis, gender analysis, race analysis
- Statistical t-tests confirmed key predictors — number_inpatient t=−26.71 p<0.001 strongest predictor, number_emergency t=−19.41 p<0.001, number_diagnoses t=−15.82 p<0.001
- OBESITY SDOH p=0.94 individually but contributes significantly inside the model — confirmed by SHAP
- Created 13 new clinical features:
  - prior_utilization — LACE-inspired weighted prior hospital usage (inpatient×0.5 + emergency×0.3 + outpatient×0.2)
  - med_complexity_score — Pearson-weighted medication complexity combining medications, procedures, and diagnoses
  - high_med_burden — polypharmacy flag for 18 or more medications
  - emergency_admission — flag for ER admissions (admission_source_id = 7)
  - long_stay — flag for 6 or more day hospital stays
  - composite_risk — sum of 4 binary risk flags (0 to 4 scale)
  - meds_x_stay — num_medications multiplied by time_in_hospital
  - age_x_diagnoses — age multiplied by number_diagnoses
  - obesity_x_diabetes — OBESITY multiplied by DIABETES (SDOH interaction)
  - on_insulin, total_diabetes_meds, multiple_med_changes, high_diagnosis_burden
- Removed 16 near-zero variance features
- Final engineered dataset saved with 69,973 rows and 69 features

### Phase 4 — Model Training 

- 60/30/10 train/validation/test split with random seed 123 — test set sealed before any model development
- SMOTE applied only to training data after splitting at sampling_strategy=0.30 — prevents data leakage — applying before splitting is a documented error in published papers
- PCA applied to Logistic Regression only — 68 to 50 components preserving 95.8 percent variance — excluded from tree models which perform internal feature selection
- Target encoding for medical_specialty (73 unique values) using training means only
- 5-fold stratified cross-validation on all advanced and Optuna-tuned models
- Threshold optimized from ROC curve — not default 0.5 which is wrong for imbalanced data
- Trained and compared 9 models — Logistic Regression, Random Forest, XGBoost clinical only (thesis baseline), XGBoost multi-source (thesis model), LR with CV, XGBoost with CV, LightGBM with CV, XGBoost Optuna, LightGBM Optuna
- Thesis confirmed — XGBoost multi-source outperforms clinical-only baseline proving multi-source data fusion improves prediction

### Phase 5 — SHAP Explainability and Optuna Tuning

- Applied SHAP TreeExplainer to XGBoost multi-source model on sealed test patients
- Generated 3 SHAP visualizations — shap_summary_bar.png, shap_beeswarm.png, shap_waterfall.png saved to explainability/ folder
- Top predictor: change (medication changes) — SHAP value 0.4703
- Second: discharge_disposition_id — SHAP value 0.2782
- Third: prior_utilization — SHAP value 0.2540 — our LACE-inspired engineered feature
- Fourth: med_complexity_score — SHAP value 0.2393 — our Pearson-weighted engineered feature
- OBESITY SDOH appeared in top 15 SHAP features — validates multi-source thesis despite individual non-significance
- 7 of top 15 features are engineered features created in this project
- Optuna Bayesian optimization with Tree-structured Parzen Estimator — 100 trials each for XGBoost and LightGBM
- XGBoost Optuna best params: n_estimators=499, max_depth=10, learning_rate=0.083, scale_pos_weight=7
- Best recall: 60.8 percent (XGBoost Optuna) — catches 60 of every 100 high-risk patients at point of discharge
- Risk tier distribution on 6,998 sealed test patients: 304 High risk, 1,171 Medium risk, 5,523 Low risk

### Phase 6 — Deployment, Documentation and Submission

- Built FastAPI REST API with 8 endpoints serving real-time predictions at port 8000
- Built Streamlit 4-page clinical dashboard connecting to the FastAPI backend
- Wrote 6 to 8 page APA academic paper covering all methodology, results, and limitations
- Built PowerPoint presentation covering all 7 professor-required sections with live demo
- Final GitHub cleanup, README documentation, and project submission

## Key Results Summary

- Best model for ROC-AUC: XGBoost Multi-Source — 0.6103
- Best model for clinical utility: XGBoost Optuna
- Best recall: 60.8 percent — catches 60 of every 100 high-risk patients at point of discharge
- Thesis proven: Multi-source model outperforms clinical-only baseline by +0.0259 ROC-AUC
- Top predictor: change (medication changes) with SHAP score of 0.4703
- Prior utilization ranked third in SHAP — our LACE-inspired engineered feature with score 0.2540
- SDOH validated: OBESITY appears in top 15 SHAP features despite being individually non-significant — confirms multi-source approach
- Engineered features: 7 of top 15 SHAP features are features we created in this project
- High risk patients identified: 304 of 6,998 sealed test patients (4.3 percent)
- Consistent with literature: Published papers report 0.62 to 0.68 with proper methodology. Our results are honest and consistent. Papers reporting higher scores typically apply SMOTE before splitting which constitutes data leakage.

## Tech Stack

- Language: Python 3.11
- Environment: Conda on Mac
- IDE: VS Code with Jupyter kernel
- ML Models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Class Balancing: SMOTE from imbalanced-learn
- Hyperparameter Tuning: Optuna with TPE Bayesian optimization
- Explainability: SHAP TreeExplainer
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn, Plotly
- API: FastAPI with Uvicorn
- Dashboard: Streamlit
- Version Control: Git and GitHub

## References

- Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission rates. BioMed Research International.
- Donze, J., et al. (2013). Potentially avoidable 30-day hospital readmissions. JAMA Internal Medicine, 173(8), 632–638.
- Parekh, A. K., and Barton, M. B. (2010). The challenge of multiple comorbidity. JAMA, 303(13), 1303–1304.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.
- Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD 2019.




