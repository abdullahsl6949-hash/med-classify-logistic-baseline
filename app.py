import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = joblib.load("cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

cancer = load_breast_cancer()
features = cancer.feature_names

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Clinical Cancer Risk Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🏥 Breast Cancer Risk Assessment Dashboard")
st.write("""
This is a **clinical-style prototype** that predicts whether a tumor is  
**Malignant (Cancerous)** or **Benign (Non-Cancerous)** using Logistic Regression.
""")

st.warning("⚠️ Educational prototype only — not medical advice.")

st.divider()

# -----------------------------
# Sidebar Inputs (Grouped)
# -----------------------------
st.sidebar.header("🧾 Tumor Measurements Input")

user_input = []

# ---- Group 1: Mean Features ----
with st.sidebar.expander("📌 Mean Tumor Measurements", expanded=True):
    mean_features = [f for f in features if "mean" in f]
    for f in mean_features:
        val = st.number_input(f, min_value=0.0, step=0.1)
        user_input.append(val)

# ---- Group 2: Standard Error Features ----
with st.sidebar.expander("📌 Measurement Error (SE)", expanded=False):
    se_features = [f for f in features if "error" in f]
    for f in se_features:
        val = st.number_input(f, min_value=0.0, step=0.1)
        user_input.append(val)

# ---- Group 3: Worst Features ----
with st.sidebar.expander("📌 Worst-Case Tumor Measurements", expanded=False):
    worst_features = [f for f in features if "worst" in f]
    for f in worst_features:
        val = st.number_input(f, min_value=0.0, step=0.1)
        user_input.append(val)

# -----------------------------
# Threshold Slider
# -----------------------------
threshold = st.sidebar.slider(
    "Cancer Detection Threshold",
    0.1, 0.9, 0.5,
    help="Lower threshold catches more cancer cases (reduces false negatives)."
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("🔍 Run Risk Assessment"):

    # Prepare input
    x = np.array(user_input).reshape(1, -1)
    x_scaled = scaler.transform(x)

    # Probabilities
    prob_benign = model.predict_proba(x_scaled)[0][1]
    prob_malignant = 1 - prob_benign

    # Decision
    pred = 0 if prob_benign < threshold else 1

    # -----------------------------
    # Clinical Dashboard Output
    # -----------------------------
    st.subheader("📊 Clinical Risk Output")

    col1, col2, col3 = st.columns(3)
    col1.metric("Malignant Risk", f"{prob_malignant*100:.2f}%")
    col2.metric("Benign Confidence", f"{prob_benign*100:.2f}%")
    col3.metric("Threshold Used", threshold)

    st.divider()

    # -----------------------------
    # Risk Category + Recommendation
    # -----------------------------
    if prob_malignant > 0.80:
        st.error("🚨 VERY HIGH RISK: Malignant Tumor Likely")
        st.write("**Recommendation:** Immediate biopsy + oncology referral.")
    elif prob_malignant > 0.55:
        st.warning("⚠️ MODERATE RISK: Further Screening Required")
        st.write("**Recommendation:** MRI / follow-up clinical evaluation.")
    else:
        st.success("✅ LOW RISK: Benign Tumor Likely")
        st.write("**Recommendation:** Routine monitoring.")

    st.divider()

    # -----------------------------
    # Explainability: Top Features
    # -----------------------------
    st.subheader("🔎 Top Influential Features (Interpretability)")

    coef = model.coef_[0]
    importance = pd.DataFrame({
        "Feature": features,
        "Weight": coef
    })

    top10 = importance.reindex(
        importance.Weight.abs().sort_values(ascending=False).index
    ).head(10)

    st.dataframe(top10)

    st.divider()

    # -----------------------------
    # Patient Input Summary
    # -----------------------------
    st.subheader("🧾 Patient Measurement Summary")
    input_df = pd.DataFrame([user_input], columns=features)
    st.dataframe(input_df)

    st.caption("Prototype dashboard for ML portfolio + clinical simulation.")
    