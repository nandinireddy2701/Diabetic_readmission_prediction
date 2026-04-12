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

Diabetic_readmission_prediction/
├── notebooks/
│   ├── 00_setup_test.ipynb
│   ├── 01_etl_pipeline.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_model_improvement.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── dashboard/
├── explainability/
├── api/
├── tests/
├── .gitignore
└── requirements.txt

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
05_explainability.ipynb → 06_model_improvement.ipynb

## Weekly Progress

### Week 1 and 2 — Project Setup 

- Python 3.11 Conda environment created and configured on Mac
- All required libraries installed including pandas, scikit-learn, XGBoost, SHAP, imbalanced-learn, matplotlib, seaborn
- Project folder structure created with 9 organized folders
- All 3 datasets downloaded into data/raw/
- Git initialized and GitHub repository connected
- Setup verification notebook confirmed all libraries working correctly

### Week 3 — ETL Pipeline and Data Cleaning 

- Loaded 101,766 patient records from 130 US hospitals
- Replaced all question marks with NaN since pandas cannot detect them as missing values
- Dropped 4 columns with 40 to 97 percent missing data — weight, max_glu_serum, A1Cresult, payer_code
- Created binary target variable readmitted_30 where 1 means readmitted within 30 days and 0 means not readmitted
- Discovered class imbalance — only 11.2 percent of patients were readmitted
- Converted age text ranges like [50-60) to numeric midpoints like 55
- Simplified 700+ ICD-9 diagnosis codes into 9 meaningful disease categories — Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other
- Label encoded all categorical columns for machine learning compatibility
- Loaded CDC PLACES SDOH data, pivoted to wide format, created state-level averages
- Joined 6 SDOH features to patient records — BPHIGH, CHECKUP, CHOLSCREEN, DEPRESSION, DIABETES, OBESITY
- Final merged dataset saved with 101,766 rows and 48 columns

### Week 4 — Exploratory Data Analysis and Feature Engineering 

Created 9 visualizations saved to dashboard folder:
- Target distribution showing 88.8 percent vs 11.2 percent class imbalance
- Age analysis showing older patients aged 65 to 85 have highest readmission rates
- Clinical features analysis — readmitted patients have more medications and longer hospital stays
- SDOH features analysis showing social health indicator distributions by readmission status
- Correlation heatmap — number_inpatient has the strongest correlation with readmission
- Diagnosis analysis by disease category
- Inpatient visit analysis — more prior visits means much higher readmission risk
- Gender analysis
- Race analysis

#### Statistical significance testing confirmed using t-tests:
- number_inpatient: t = -53.42, p = 0.0000 — strongest predictor
- number_emergency: t = -19.41, p = 0.0000 — highly significant
- number_diagnoses: t = -15.82, p = 0.0000 — highly significant
- time_in_hospital: t = -14.11, p = 0.0000 — significant
- OBESITY (SDOH): p = 0.94 — not individually significant but contributes in model combination

#### Feature engineering created 10 new features:

##### Medication features (5 new):
- total_diabetes_meds — count of active diabetes medications
- med_complexity_score — weighted medication complexity score
- high_med_burden — flag for top 25 percent medication count (27.1 percent of patients)
- on_insulin — insulin usage flag (88.0 percent of patients)
- multiple_med_changes — drug change flag (53.8 percent of patients)

##### Clinical risk features (5 new):
- prior_utilization — weighted prior healthcare usage score (became number 1 SHAP predictor)
- high_prior_inpatient — flag for 2+ prior inpatient stays (14.4 percent of patients)
- emergency_admission — emergency admission flag (56.5 percent of patients)
- long_stay — flag for top 25 percent hospital stay length (28.2 percent of patients)
- high_diagnosis_burden — flag for 7+ diagnoses (69.4 percent of patients)

Removed 13 near-zero variance features that added no predictive value.
Final engineered dataset saved with 101,766 rows and 44 features.

### Week 5 — Model Training with Proper Methodology 

Used academically correct approach to prevent data leakage:
- Split data first into 70 percent training, 15 percent validation, 15 percent test (sealed)
- Applied SMOTE only to training set — validation and test kept real unbalanced distribution
- Fitted StandardScaler on training data only

#### Trained and compared 4 machine learning models on validation set:

##### Logistic Regression (clinical only — baseline)
- ROC-AUC: 0.5996 — just above random guessing
- This is our floor — every other model must beat this

##### Random Forest (clinical only)
- ROC-AUC: 0.6041 — improvement over linear baseline
- Confirms non-linear models handle this data better

##### XGBoost clinical only
- ROC-AUC: 0.6043 — strong model but missing SDOH and medication data

##### XGBoost multi-source (all 44 features) — BEST MODEL
- ROC-AUC: 0.6332 — best of all four models
- Recall: 0.0423, Precision: 0.3172, F1: 0.0746

Thesis confirmed: XGBoost multi-source outperforms clinical-only baseline by +0.0289 ROC-AUC
Test set remains sealed for final evaluation.

Note on scores: Published academic literature on this exact dataset with proper methodology reports ROC-AUC of 0.62 to 0.68. Our 0.6332 is honest and consistent with published research. Papers reporting higher scores typically apply SMOTE before splitting which constitutes data leakage.

### Week 6 — SHAP Explainability and Model Improvement

Applied SHAP TreeExplainer to XGBoost multi-source model on 1000 test patients.

Generated 3 SHAP visualizations:
- shap_summary_bar.png — top 15 features by overall importance
- shap_beeswarm.png — direction of impact for each feature across all patients
- shap_waterfall.png — single high-risk patient breakdown showing exactly why they were flagged

##### Top 5 most important features from SHAP:
1. prior_utilization — importance score 1.087 (strongest by far)
2. change — importance score 0.633
3. insulin — importance score 0.354
4. number_outpatient — importance score 0.317
5. number_inpatient — importance score 0.289
14. OBESITY (SDOH feature) — importance score 0.134 — validates multi-source approach

Key finding: OBESITY appears in top 15 SHAP features despite p-value of 0.94 in isolation. This proves SDOH features contribute to predictions when combined with clinical data inside the model.

Hyperparameter tuning using GridSearchCV tested 12 parameter combinations:
- Best parameters found: max_depth=8, learning_rate=0.1, n_estimators=200
- ROC-AUC improved from 0.6332 to 0.6362 (+0.0030)

#### Model improvement strategies applied:

##### Strategy 1 — scale_pos_weight=8 (class weighting):
- Trained on original unbalanced data with XGBoost class weighting
- ROC-AUC improved to 0.6559 (+0.0227)
- Recall improved dramatically from 4.2 percent to 39.8 percent — nearly 10x improvement

##### Strategy 2 — Interaction features (7 new features added):
- inpatient_x_meds: number_inpatient multiplied by num_medications
- utilization_x_diagnoses: number_inpatient multiplied by number_diagnoses
- age_x_diagnoses: age multiplied by number_diagnoses
- emergency_x_inpatient: number_emergency multiplied by number_inpatient
- meds_x_stay: num_medications multiplied by time_in_hospital
- obesity_x_diabetes: OBESITY multiplied by DIABETES (SDOH interaction)
- composite_risk: sum of 5 binary risk flags (0 to 5 scale)

##### Final best model results:
- ROC-AUC: 0.6616 — total improvement of +0.0284 over original model
- Recall: 0.3979
- Precision: 0.1990
- F1-Score: 0.2653
- Features: 51 total (44 original + 7 interaction features)

##### Complete improvement journey:
- Original XGBoost: 0.6332
- After hyperparameter tuning: 0.6362 (+0.0030)
- After scale_pos_weight: 0.6559 (+0.0227)
- After interaction features: 0.6616 (+0.0284 total)

### Week 7 - Tableau Dashboard 

Exported 4 CSV files for Tableau dashboard:
- powerbi_patient_risks.csv — 15,265 patients with risk scores and tiers
- powerbi_model_comparison.csv — performance metrics for all 4 models
- powerbi_shap_importance.csv — top 15 SHAP feature importance scores
- powerbi_improvement.csv — model improvement journey data

Risk tier distribution (threshold: High >= 0.30, Medium >= 0.15, Low < 0.15):
- Low risk: 9,576 patients
- Medium risk: 4,252 patients
- High risk: 1,437 patients

Building 6-sheet Tableau dashboard covering patient risk distribution, risk score distribution, model comparison, SHAP feature importance, improvement journey, and clinical feature analysis.

### Week 8 — Academic Paper and Presentation (upcoming)

- Write 6 to 8 page APA academic paper
- Build 7-slide PowerPoint presentation
- Final demo with live Tableau dashboard

## Key Results Summary

- Best model: XGBoost with interaction features
- Final ROC-AUC: 0.6616
- Thesis proven: Multi-source model outperforms clinical-only baseline by +0.0289
- Top predictor: prior_utilization with SHAP score of 1.087
- SDOH validated: OBESITY appears in top 15 SHAP features despite being individually non-significant
- Recall improvement: 4.2 percent to 39.8 percent using scale_pos_weight class weighting
- Consistent with literature: Published papers report 0.62 to 0.68 with proper methodology

## Tech Stack

Language: Python 3.11
Environment: Conda on Mac
IDE: VS Code with Jupyter kernel
ML Models: Logistic Regression, Random Forest, XGBoost
Class Balancing: SMOTE from imbalanced-learn
Explainability: SHAP
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn, Tableau
Statistical Tests: scipy.stats
Version Control: Git and GitHub
Deployment: Azure ML (Week 7 — in progress)

## References

- Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission rates. BioMed Research International.
- Donze, J., et al. (2013). Potentially avoidable 30-day hospital readmissions. JAMA Internal Medicine, 173(8), 632–638.
- Parekh, A. K., and Barton, M. B. (2010). The challenge of multiple comorbidity. JAMA, 303(13), 1303–1304.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.
- Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.





