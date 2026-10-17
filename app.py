import joblib
import pandas as pd
import streamlit as st

from features import add_features
from train import train_model

st.set_page_config(page_title="Loan Amount Predictor", page_icon="💰")


@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception:
        # Fallback if the pickle doesn't match the installed scikit-learn version
        return train_model()


model = load_model()

st.title("💰 Loan Amount Predictor")
st.caption("Linear Regression model trained on the Kaggle Loan Approval dataset.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 80, 28)
    gender = st.selectbox("Gender", ["female", "male"])
    education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Yearly income ($)", 5000, 500000, 70000, step=1000)
    emp_exp = st.number_input("Work experience (years)", 0, 60, 4)
    home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
with col2:
    intent = st.selectbox("Loan purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    rate = st.slider("Interest rate (%)", 5.0, 20.0, 11.0, 0.1)
    hist = st.number_input("Credit history length (years)", 2, 30, 5)
    score = st.slider("Credit score", 390, 850, 640)
    default = st.selectbox("Previous loan default?", ["No", "Yes"])

if st.button("Predict loan amount", type="primary"):
    row = pd.DataFrame([{
        "person_age": age, "person_gender": gender, "person_education": education,
        "person_income": income, "person_emp_exp": emp_exp, "person_home_ownership": home,
        "loan_intent": intent, "loan_int_rate": rate, "cb_person_cred_hist_length": hist,
        "credit_score": score, "previous_loan_defaults_on_file": default,
    }])
    pred = float(model.predict(add_features(row))[0])
    pred = max(pred, 500.0)
    st.success(f"Estimated loan amount: **${pred:,.0f}**")
    st.caption("Typical error is about ±$4,300 (MAE), so treat this as a rough estimate.")

with st.expander("About this model"):
    st.write(
        "Test R² is about 0.24 and MAE is about $4,300. The dataset has a weak "
        "link between applicant features and loan amount, so predictions are approximate. "
        "`loan_percent_income` was left out on purpose because it is derived from the target."
    )
