"""Bank Customer Churn — minimal Streamlit UI over the FastAPI backend."""
import os
import requests
import streamlit as st

API = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Bank Customer Churn Predictor")
st.caption("Predicts whether a bank customer is likely to leave — and why.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Years with bank", 0, 10, 3)
    credit_score = st.number_input("Credit score", 350, 850, 650)
with col2:
    balance = st.number_input("Account balance", 0.0, 300000.0, 120000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated salary", 1000.0, 250000.0, 95000.0, step=1000.0)
    num_of_products = st.selectbox("Products held", [1, 2, 3, 4])
    is_active_member = st.radio("Active member?", ["Yes", "No"], horizontal=True)
    has_cr_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)

payload = {
    "credit_score": int(credit_score), "geography": geography, "gender": gender,
    "age": int(age), "tenure": int(tenure), "balance": float(balance),
    "num_of_products": int(num_of_products),
    "has_cr_card": 1 if has_cr_card == "Yes" else 0,
    "is_active_member": 1 if is_active_member == "Yes" else 0,
    "estimated_salary": float(estimated_salary),
}

if st.button("Predict churn risk", type="primary", use_container_width=True):
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json()
            proba, level = d["churn_probability"], d["risk_level"]
            color = {"high": "#D1495B", "medium": "#E9C46A", "low": "#4C9F70"}[level]
            st.markdown(f"### <span style='color:{color}'>{proba:.1%} — {level.upper()} risk</span>",
                        unsafe_allow_html=True)
            st.progress(min(proba, 1.0))
            st.caption(f"Model: {d['model_name']}")
            st.markdown("**Contributing factors**")
            for f in d["top_factors"]:
                st.markdown(f"- {f}")
        else:
            st.error(f"API {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}. Is the API running on {API}?")
