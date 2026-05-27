# Capstone Project Review — Diabetic Readmission Prediction
## General
This is a good project overall, but the dataset is very imbalanced which poses a lot of challenges for us. 

## Feedback & Suggestions

---

### 1. Multiple Encounters Per Patient — Data Leakage Risk

**Where:** `notebooks/01_etl_pipeline.ipynb`, cell 9 — you drop `encounter_id` and `patient_nbr`

**What's happening:**  
Imagine a patient named John who visited the hospital 3 times. Your dataset has 3 rows for John. When you randomly split the data into training and test sets, John's 2nd visit might land in training and his 3rd visit in the test set. Your model has already "met" John during training, so it does better on the test set than it would on a brand new patient it has never seen. This makes your model look more accurate than it really is.

**Suggestion:**
```python
# Keep only the first encounter per patient before splitting OR alternatively, you could just keep all encounters for a single patient, group them, and just keep the patients separated in your test, train, and val sets. 
df = df.sort_values('encounter_id')
df = df.groupby('patient_nbr').first().reset_index()
df = df.drop(columns=['encounter_id', 'patient_nbr'])
```
After deduplication you will have  unique patients rather than encounters where patients are duplicated. Technically, your AUC score may drop slightly, but it will be more honest. 

---

### 2. Patients Who Died or Went to Hospice Are Included as "Not Readmitted"

**Where:** `notebooks/01_etl_pipeline.ipynb`, cell 8

**What's happening:**  
When you create `readmitted_30 = 0` for everyone who was not readmitted within 30 days, you are including patients who died in the hospital or were transferred to hospice care. These patients are not readmitted because they cannot be — they are deceased or in end-of-life care. They are very different from a patient who went home healthy and simply was not readmitted. Mixing these groups together confuses your model.

**Suggestion:**
```python
# These discharge codes indicate the patient cannot be readmitted
ineligible_dispositions = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(ineligible_dispositions)]
```
This is a standard preprocessing step in clinical readmission research. Removing these rows will make your negative class cleaner and your model more valid.

---

### 3. Categorical Variables Are Encoded as If They Have a Natural Order 

**Where:** `notebooks/01_etl_pipeline.ipynb`, cell 14 — Label Encoding section

**What's happening:**  
You used Label Encoding to convert text categories like diagnosis type into numbers: Circulatory=0, Diabetes=1, Digestive=2, etc. The problem is that your model now thinks "Digestive is bigger than Diabetes, which is bigger than Circulatory." That ordering is completely made up — there is no reason digestive should be mathematically "greater than" diabetes. This can send your model in the wrong direction.

A better approach is One-Hot Encoding (venctor encoding) for the diagnosis categories (since there are only 9 of them), which gives each category its own yes/no column. For `medical_specialty`, which has 73 categories, you should use wwhat's called target encoding— replace the specialty name with the average readmission rate for patients with that specialty, calculated only from training data. See here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html


**Suggestion:**
```python
# One-hot encode the 9-category diagnosis columns
df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], drop_first=True)

# Target encode medical_specialty on training data only
specialty_means = X_train.join(y_train).groupby('medical_specialty')['readmitted_30'].mean()
X_train['specialty_risk'] = X_train['medical_specialty'].map(specialty_means)
X_val['specialty_risk']   = X_val['medical_specialty'].map(specialty_means)
```

---

### 4. A1Cresult Was Dropped — But "Not Measured" Is Itself Clinically Meaningful

**Where:** `notebooks/01_etl_pipeline.ipynb`, cell 7 — columns dropped for missing data

**What's happening:**  
A1C is a blood test that measures how well a diabetic patient's blood sugar has been controlled over the past 3 months. You dropped this column because 83% of the values are missing. However, the fact that A1C was not measured is itself a signal — it often means the hospital did not assess the patient's long-term diabetes control, which is associated with higher risk. You are throwing away meaningful information.

**Suggestion:**  
Instead of dropping the column entirely, convert it into binary flags:
```python
df['A1c_measured'] = (df['A1Cresult'] != '?').astype(int)
df['A1c_normal']   = (df['A1Cresult'] == 'Norm').astype(int)
df['A1c_elevated'] = (df['A1Cresult'].isin(['>7', '>8'])).astype(int)
# Then drop the original text column
df = df.drop(columns=['A1Cresult'])
```
This preserves the clinical signal with zero missing value problems.

---

### 5. SMOTE Ratio Is Too Aggressive (50/50 Balance Is Unrealistic)

**Where:** `notebooks/04_model_training.ipynb`, cell 8 — SMOTE application

**What's happening:**  
You used SMOTE to fix the class imbalance, which is good methodology. However, you balanced it all the way to 50% readmitted and 50% not readmitted. In the real world, only 11.2% of patients are readmitted. By training on a world where 50% are readmitted, your model learns the wrong sense of "normal." This is why your default threshold of 0.50 produces near-zero recall (0.04) — the model's internal probability scale is calibrated to the wrong distribution.

The results in Notebook 06 actually confirm this: using `scale_pos_weight=8` without SMOTE gave better AUC (0.6559 vs 0.6332) and dramatically better recall (39.8% vs 4.2%). That is your signal to prefer weighting over heavy SMOTE.

**Suggestion:**  
Either use `scale_pos_weight` on XGBoost/LightGBM directly (which already outperformed), or if you keep SMOTE, reduce the sampling ratio:
```python
# Instead of 50/50, oversample minority to only 30% of majority
smote = SMOTE(random_state=42, sampling_strategy=0.30)
```

---

### 6. `prior_utilization` Is Redundant — It Is a Linear Combination of Features Already in the Model

**Where:** `notebooks/03_feature_engineering.ipynb`, cell 6

**What's happening in plain English:**  
You created `prior_utilization` as:
```
prior_utilization = number_inpatient × 0.5 + number_emergency × 0.3 + number_outpatient × 0.2
```
But `number_inpatient`, `number_emergency`, and `number_outpatient` are all still in the dataset too. This is like adding a column that says "total score = A + B + C" and then also keeping columns A, B, and C. The composite adds no new information — it is just a weighted average of what the model already has. The very high SHAP importance for `prior_utilization` is partially an artifact of this: the model "borrows" predictive power from the other three features into this combined feature, making it look more important than it really is.

**Suggestion:**  
Either drop `prior_utilization` and let the model learn from the components directly, or keep the composite and drop its three components. Do not keep both.

---

### 7. SDOH Join Is a Methodology Limitation That Should Be Acknowledged

**Where:** `notebooks/01_etl_pipeline.ipynb`, cells 22–24

**What's happening:**  
Because the original dataset does not include the hospital's state or location, you used `np.random.seed(42)` followed by a random state assignment to attach SDOH features to patients. This is a reproducible and defensible workaround given the data constraints — the seed guarantees the same patient always gets the same state. However, the state assignment is still not based on where the patient actually lives or where the hospital is located. A patient from Alabama may have been assigned Oregon's SDOH statistics.

This means the SDOH features carry limited real-world signal.

**Recommendation:**  
The thesis claim that "multi-source fusion outperforms clinical-only models" should be qualified with this limitation clearly stated. I would maybe consider hard coding the mapping between the state and the patient ensuring this is a variable that can be used in your analysis.

---

### 8. Only apply PCA to your Linear Regression model

**Where:** `notebooks/07_professor_suggestions.ipynb`, cells 10 and 13

**What's happening:**  
PCA is a technique that combines many features into fewer "summary" features. It works well for models like Logistic Regression that prefer clean, uncorrelated inputs. However, tree-based models like XGBoost and LightGBM work by asking questions one feature at a time — "Is number_inpatient > 2? Yes/No." The tree model can no longer make clean splits, which is why your LightGBM + PCA dropped. 

**Suggestion:** I wouldn't use it for the XGboost or LightGBM at all just exporation and linear regression and I would put linear regression in a separate model path (i.e., do not use the same treatment for these models as you would for XGBoost or lightGBM). So remove PCA from the LightGBM pipeline entirely. 

---

### 9. Hyperparameter Search Coverage Is Very Limited

**Where:** `notebooks/06_model_improvement.ipynb` — GridSearchCV with 12 combinations

**What's happening:**  
You tested 12 combinations of hyperparameters using GridSearchCV. For a model as complex as XGBoost, 12 combinations barely scratches the surface. It is like trying 12 different oven temperatures for a recipe when there are hundreds of possible settings. You may be leaving significant performance on the table.

**Suggestion:** Use optuna, a Bayesian hyperparameter optimization library. Instead of checking combinations in a grid, optuna learns which parameter regions look promising and focuses its search there — much more efficient. https://github.com/optuna/optuna

```python
!pip install optuna
import optuna
```
100 trials with Optuna is more effective than 12 trials with GridSearch, and takes roughly the same time.

---

### 10. No Cross-Validation — Results Are From a Single Split

**Where:** All model training notebooks

**What's happening:**  
All of your model comparisons are based on a single train/validation split. This is like giving a student one exam and using that one result to judge them forever. What if that particular split happened to put easy or hard patients in the validation set? Your reported AUC of ~.66 could be slightly optimistic or pessimistic depending purely on which random rows ended up where.

**Suggestion:** Use k or 5-fold stratified cross-validation on the training set to get a more reliable estimate of true model performance. The test set stays sealed.

https://scikit-learn.org/stable/modules/cross_validation.html

The `mean ± std` gives reviewers and readers confidence that your reported number is stable, not just lucky.

---

