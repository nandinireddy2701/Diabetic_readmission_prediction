import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Diabetic Readmission Risk Dashboard",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# Title
st.title("Diabetic Hospital Readmission Prediction")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Patient Risk Predictor", "Model Performance",
     "SHAP Feature Importance", "Patient Risk Overview"]
)

# ============================================================
# PAGE 1 — Patient Risk Predictor
# ============================================================
if page == "Patient Risk Predictor":
    st.header("Patient Risk Predictor")
    st.markdown("Load a sample patient or enter details manually")

    # Sample patient buttons
    st.subheader("Quick Load Sample Patient")
    s1, s2, s3 = st.columns(3)

    with s1:
        if st.button("🟢 Load Low Risk Sample"):
            resp = requests.get(f"{API_URL}/sample/low")
            data = resp.json()
            st.session_state['sample']         = data['_full_features']
            st.session_state['sample_display'] = data
            st.session_state['sample_tier']    = 'Low'
            st.success(f"Low risk patient loaded! Expected score: {data['risk_percentage']}")

    with s2:
        if st.button("🟡 Load Medium Risk Sample"):
            resp = requests.get(f"{API_URL}/sample/medium")
            data = resp.json()
            st.session_state['sample']         = data['_full_features']
            st.session_state['sample_display'] = data
            st.session_state['sample_tier']    = 'Medium'
            st.success(f"Medium risk patient loaded! Expected score: {data['risk_percentage']}")

    with s3:
        if st.button("🔴 Load High Risk Sample"):
            resp = requests.get(f"{API_URL}/sample/high")
            data = resp.json()
            st.session_state['sample']         = data['_full_features']
            st.session_state['sample_display'] = data
            st.session_state['sample_tier']    = 'High'
            st.success(f"High risk patient loaded! Expected score: {data['risk_percentage']}")

    if 'sample' in st.session_state:
        display = st.session_state.get('sample_display', {})
        st.info(f"Patient loaded — Age: {display.get('age')} | "
                f"Medications: {display.get('num_medications')} | "
                f"Expected: {display.get('risk_percentage')}")

        if st.button("🔮 Predict Sample Patient", type="primary"):
            sample = st.session_state['sample'].copy()
            response = requests.post(f"{API_URL}/predict", json=sample)
            result   = response.json()

            st.markdown("---")
            st.subheader("Prediction Results")

            risk_score = result['risk_score']
            risk_tier  = result['risk_tier']
            color_map  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{result['risk_percentage']}")
            with col2:
                st.metric("Risk Tier", f"{color_map[risk_tier]} {risk_tier}")
            with col3:
                st.metric("Prior Utilization",
                          f"{sample.get('prior_utilization', 0):.2f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={'text': "Readmission Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': result['color']},
                    'steps': [
                        {'range': [0, 15],   'color': '#ccffcc'},
                        {'range': [15, 30],  'color': '#fff3cc'},
                        {'range': [30, 100], 'color': '#ffcccc'}
                    ],
                    'threshold': {
                        'line':      {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value':     risk_score * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"{color_map[risk_tier]} Recommended Interventions")
            st.markdown(f"**Risk Level: {risk_tier}**")
            for i, intervention in enumerate(result['interventions'], 1):
                st.markdown(f"{i}. {intervention}")
                
    st.markdown("---")
    st.subheader("Manual Patient Entry")
    st.markdown("Or enter patient details manually below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age    = st.slider("Age", 0, 100, 65)
        gender = st.selectbox("Gender", [0, 1],
                              format_func=lambda x: "Female" if x == 0 else "Male")
        race   = st.selectbox("Race", [0, 1, 2, 3, 4],
                              format_func=lambda x: ["AfricanAmerican", "Asian",
                                                      "Caucasian", "Hispanic",
                                                      "Other"][x])

    with col2:
        st.subheader("Clinical Information")
        time_in_hospital   = st.slider("Days in Hospital", 1, 14, 4)
        num_medications    = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses   = st.slider("Number of Diagnoses", 1, 16, 7)
        num_lab_procedures = st.slider("Lab Procedures", 1, 132, 43)
        num_procedures     = st.slider("Procedures", 0, 6, 1)

    with col3:
        st.subheader("Admission Details")
        admission_type_id = st.selectbox(
            "Admission Type", [1, 2, 3],
            format_func=lambda x: {1: "Emergency", 2: "Urgent",
                                    3: "Elective"}[x])
        discharge_disposition_id = st.selectbox(
            "Discharge Disposition", [1, 2, 3, 4],
            format_func=lambda x: {1: "Home", 2: "Care Facility",
                                    3: "Skilled Nursing", 4: "Other"}[x])
        admission_source_id = st.selectbox(
            "Admission Source", [1, 7],
            format_func=lambda x: {1: "Physician Referral",
                                    7: "Emergency Room"}[x])
        medical_specialty = st.selectbox(
            "Medical Specialty",
            ["Unknown", "Cardiology", "InternalMedicine",
             "Surgery-General", "Orthopedics", "Radiology"])

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Lab Results")
        A1c_normal   = st.selectbox("A1C Normal", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
        A1c_elevated = st.selectbox("A1C Elevated", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
        on_insulin   = st.selectbox("On Insulin", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")

    with col5:
        st.subheader("Prior Utilization")
        number_inpatient  = st.slider("Prior Inpatient Visits",  0, 10, 0)
        number_emergency  = st.slider("Prior Emergency Visits",  0, 10, 0)
        number_outpatient = st.slider("Prior Outpatient Visits", 0, 10, 0)

    # Calculate engineered features
    prior_utilization     = (number_inpatient * 0.5 +
                             number_emergency  * 0.3 +
                             number_outpatient * 0.2)
    emergency_admission   = 1 if admission_source_id == 7 else 0
    long_stay             = 1 if time_in_hospital >= 6 else 0
    high_diagnosis_burden = 1 if number_diagnoses >= 7 else 0
    total_diabetes_meds   = on_insulin
    multiple_med_changes  = 0

    if st.button("🔮 Predict Manual Patient", type="secondary"):
        payload = {
            "age":                      age,
            "time_in_hospital":         time_in_hospital,
            "num_lab_procedures":       num_lab_procedures,
            "num_procedures":           num_procedures,
            "num_medications":          num_medications,
            "number_diagnoses":         number_diagnoses,
            "admission_type_id":        admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id":      admission_source_id,
            "race":                     race,
            "gender":                   gender,
            "A1c_normal":               A1c_normal,
            "A1c_elevated":             A1c_elevated,
            "medical_specialty":        medical_specialty,
            "prior_utilization":        prior_utilization,
            "emergency_admission":      emergency_admission,
            "long_stay":                long_stay,
            "high_diagnosis_burden":    high_diagnosis_burden,
            "total_diabetes_meds":      total_diabetes_meds,
            "on_insulin":               on_insulin,
            "multiple_med_changes":     multiple_med_changes
        }

        response = requests.post(f"{API_URL}/predict_raw", json=payload)
        result   = response.json()

        st.markdown("---")
        st.subheader("Manual Prediction Results")

        risk_score = result['risk_score']
        risk_tier  = result['risk_tier']
        color_map  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("Risk Score", f"{result['risk_percentage']}")
        with col7:
            st.metric("Risk Tier", f"{color_map[risk_tier]} {risk_tier}")
        with col8:
            st.metric("Prior Utilization Score", f"{prior_utilization:.2f}")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "Readmission Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': result['color']},
                'steps': [
                    {'range': [0, 15],   'color': '#ccffcc'},
                    {'range': [15, 30],  'color': '#fff3cc'},
                    {'range': [30, 100], 'color': '#ffcccc'}
                ],
                'threshold': {
                    'line':      {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value':     risk_score * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Interventions
        st.subheader(f"{color_map[risk_tier]} Recommended Interventions")
        st.markdown(f"**Risk Level: {risk_tier}**")
        for i, intervention in enumerate(result['interventions'], 1):
            st.markdown(f"{i}. {intervention}")

# ============================================================
# PAGE 2 — Model Performance
# ============================================================
elif page == "Model Performance":
    st.header("📊 Model Performance Comparison")

    df = pd.read_csv("dashboard/powerbi_model_comparison.csv")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df, x='Model', y='ROC_AUC',
            title='ROC-AUC by Model',
            color='Type',
            color_discrete_map={'Basic': '#95a5a6', 'Advanced': '#3498db'},
            text='ROC_AUC'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0.50, 0.65])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df, x='Model', y='Recall',
            title='Recall by Model',
            color='Type',
            color_discrete_map={'Basic': '#95a5a6', 'Advanced': '#e74c3c'},
            text='Recall'
        )
        fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Complete Model Results")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Results")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Best ROC-AUC", "0.6103", "XGBoost multi-source")
    with col4:
        st.metric("Best Recall", "0.5114", "LR + PCA + CV")
    with col5:
        st.metric("Thesis Confirmed", "+0.0259",
                  "Multi-source vs Clinical-only")

# ============================================================
# PAGE 3 — SHAP Feature Importance
# ============================================================
elif page == "SHAP Feature Importance":
    st.header("🔍 SHAP Feature Importance")
    st.markdown("Features ranked by their average impact on readmission predictions")

    df_shap = pd.read_csv("dashboard/powerbi_shap_importance.csv")

    fig = px.bar(
        df_shap.sort_values('Importance'),
        x='Importance', y='Feature',
        orientation='h',
        title='Top 15 Features by SHAP Importance',
        color='Importance',
        color_continuous_scale='RdYlGn_r',
        text='Importance'
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Predictors")
        for i, row in df_shap.head(5).iterrows():
            st.markdown(f"**{i+1}. {row['Feature']}** — {row['Importance']:.4f}")

    with col2:
        st.subheader("Key Insights")
        st.markdown("✅ **change** — Medication changes are the strongest predictor")
        st.markdown("✅ **prior_utilization** — Prior hospital visits signal high risk")
        st.markdown("✅ **med_complexity_score** — Complex medication regime increases risk")
        st.markdown("✅ **OBESITY** — SDOH feature validates multi-source approach")
        st.markdown("✅ **Engineered features** dominate top 15 rankings")

    st.markdown("---")
    st.subheader("SHAP Visualizations")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.image("explainability/shap_summary_bar.png",
                 caption="SHAP Summary Bar")
    with col4:
        st.image("explainability/shap_beeswarm.png",
                 caption="SHAP Beeswarm")
    with col5:
        st.image("explainability/shap_waterfall.png",
                 caption="SHAP Waterfall")

# ============================================================
# PAGE 4 — Patient Risk Overview
# ============================================================
elif page == "Patient Risk Overview":
    st.header("Patient Risk Overview")

    df_patients = pd.read_csv("dashboard/powerbi_patient_risks.csv")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{len(df_patients):,}")
    with col2:
        high = (df_patients['risk_tier'] == 'High').sum()
        st.metric("🔴 High Risk", f"{high:,}")
    with col3:
        medium = (df_patients['risk_tier'] == 'Medium').sum()
        st.metric("🟡 Medium Risk", f"{medium:,}")
    with col4:
        low = (df_patients['risk_tier'] == 'Low').sum()
        st.metric("🟢 Low Risk", f"{low:,}")

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        tier_counts = df_patients['risk_tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk Tier', 'Count']
        fig = px.pie(
            tier_counts, values='Count', names='Risk Tier',
            title='Patient Risk Distribution',
            color='Risk Tier',
            color_discrete_map={
                'High':   '#e74c3c',
                'Medium': '#f39c12',
                'Low':    '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig2 = px.histogram(
            df_patients, x='risk_score',
            title='Risk Score Distribution',
            nbins=50,
            color_discrete_sequence=['#3498db']
        )
        fig2.add_vline(x=0.15, line_dash="dash", line_color="orange",
                       annotation_text="Medium threshold")
        fig2.add_vline(x=0.30, line_dash="dash", line_color="red",
                       annotation_text="High threshold")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Age vs Risk Score")
    fig3 = px.scatter(
        df_patients.sample(500),
        x='age', y='risk_score',
        color='risk_tier',
        color_discrete_map={
            'High':   '#e74c3c',
            'Medium': '#f39c12',
            'Low':    '#2ecc71'
        },
        title='Age vs Risk Score by Risk Tier',
        opacity=0.6
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("High Risk Patients Sample")
    high_risk = df_patients[df_patients['risk_tier'] == 'High'].head(20)
    st.dataframe(
        high_risk[['patient_id', 'risk_score', 'risk_tier',
                   'age', 'time_in_hospital', 'num_medications',
                   'prior_utilization', 'actual']],
        use_container_width=True
    )