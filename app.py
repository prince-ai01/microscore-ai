import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MicroScore AI - Explainable Credit Engine",
    layout="wide"
)

# 1. Load trained model
@st.cache_resource
def load_model():
    with open("credit_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 2. Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

st.title("MicroScore AI: Explainable Credit Risk Underwriter")
st.markdown("Physics-guided, XAI-driven risk scoring for informal micro-merchants.")
st.divider()

# Sidebar inputs
st.sidebar.header("Applicant Financials")
daily_income = st.sidebar.number_input("Average Daily Income (₹)", min_value=500, max_value=50000, value=3500, step=100)
daily_expense = st.sidebar.number_input("Daily Operating Expense (₹)", min_value=100, max_value=45000, value=2200, step=100)
daily_upi_count = st.sidebar.slider("Daily UPI Scan Count", min_value=1, max_value=250, value=45)

monthly_savings = (daily_income - daily_expense) * 30
st.sidebar.markdown(f"**Monthly Savings Buffer:** ₹{monthly_savings:,.2f}")

input_df = pd.DataFrame([{
    "daily_income": daily_income,
    "daily_expense": daily_expense,
    "monthly_savings": monthly_savings,
    "daily_upi_count": daily_upi_count
}])

# Inference
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]
shap_values = explainer.shap_values(input_df)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Financial Profile Summary")
    st.metric(label="Daily Net Margin", value=f"₹{daily_income - daily_expense:,}")
    st.metric(label="Expense-to-Income Ratio", value=f"{(daily_expense / daily_income) * 100:.1f}%")
    st.metric(label="Calculated Savings Buffer", value=f"₹{monthly_savings:,}")

with col2:
    st.subheader("Underwriting Verdict")
    if prediction == 1:
        st.success("### Loan Approved")
        st.write(f"**Approval Confidence:** {prediction_proba[1] * 100:.1f}%")
        safe_limit = max(monthly_savings * 1.5, 10000)
        st.info(f"**Recommended Credit Limit:** Up to ₹{safe_limit:,.0f}")
    else:
        st.error("### High Default Risk (Rejected)")
        st.write(f"**Default Probability:** {prediction_proba[0] * 100:.1f}%")

st.divider()

# Explainable AI (SHAP) Visual Section
st.subheader("Explainable AI (TreeSHAP) Feature Attribution")
st.markdown("Quantifies how each financial feature pushed the model toward approval (+ values) or rejection (- values).")

if isinstance(shap_values, list):
    applicant_shap = shap_values[1][0]
else:
    applicant_shap = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

features = ["Daily Income", "Daily Expense", "Monthly Savings", "UPI Count"]
shap_df = pd.DataFrame({
    "Feature": features,
    "SHAP Contribution Value": applicant_shap
}).sort_values(by="SHAP Contribution Value", ascending=True)

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in shap_df["SHAP Contribution Value"]]
ax.barh(shap_df["Feature"], shap_df["SHAP Contribution Value"], color=colors)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Impact on Loan Approval (SHAP Value)")
ax.set_title("Local Feature Contribution Breakdown")
plt.tight_layout()

st.pyplot(fig)

st.divider()
st.subheader("Historical Batch Records")
if st.checkbox("Show Sample Dataset"):
    df_raw = pd.read_csv("vendor_data.csv")
    st.dataframe(df_raw.head(15), use_container_width=True)