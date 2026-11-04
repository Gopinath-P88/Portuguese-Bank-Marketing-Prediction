
# Portuguese Bank Marketing Prediction - Streamlit App

# 1. Import Required Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 2. Load Trained Model and Preprocessing Files

model = joblib.load("random_forest_bank_model.pkl")
encoder = joblib.load("bank_encoder.pkl")
scaler = joblib.load("bank_scaler.pkl")

st.success("Machine Learning Model Loaded Successfully!")

# 3. Configure Streamlit Page
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# 4. Application Heading
st.title("🏦 Portuguese Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a bank term deposit.")

st.divider()

# 5. Customer Information
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=17, max_value=100, value=35)

    job = st.selectbox("Job", [
        "admin.", "blue-collar.", "entrepreneur.", "housemaid.",
        "management.", "retired.", "self-employed.", "services.",
        "student.", "technician.", "unemployed.", "unknown"
    ])

    marital = st.selectbox("Marital Status", [
        "married", "single", "divorced", "unknown"
    ])

    education = st.selectbox("Education", [
        "university.degree", "high.school", "basic.9y",
        "professional.course", "basic.4y", "basic.6y",
        "illiterate", "unknown"
    ])

with col2:
    default = st.selectbox("Credit in Default", [
        "no", "yes", "unknown"
    ])

    housing = st.selectbox("Housing Loan", [
        "yes", "no", "unknown"
    ])

    loan = st.selectbox("Personal Loan", [
        "no", "yes", "unknown"
    ])

    contact = st.selectbox("Contact Type", [
        "cellular", "telephone"
    ])

with col3:
    month = st.selectbox("Last Contact Month", [
        "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ])

    day_of_week = st.selectbox("Last Contact Day", [
        "mon", "tue", "wed", "thu", "fri"
    ])

    campaign = st.number_input(
        "Number of Contacts During Campaign",
        min_value=1, value=2
    )

    pdays = st.number_input(
        "Days Since Previous Contact (999 = Never Contacted)",
        min_value=0, value=999
    )

# 6. Previous Campaign Information
st.subheader("📞 Previous Campaign Information")

col4, col5 = st.columns(2)

with col4:
    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0, value=0
    )

    poutcome = st.selectbox("Previous Campaign Outcome", [
        "nonexistent", "failure", "success"
    ])

with col5:
    emp_var_rate = st.number_input(
        "Employment Variation Rate", value=1.1
    )

    cons_price_idx = st.number_input(
        "Consumer Price Index", value=93.994
    )

    cons_conf_idx = st.number_input(
        "Consumer Confidence Index", value=-36.4
    )

    euribor3m = st.number_input(
        "Euribor 3 Month Rate", value=4.857
    )

    nr_employed = st.number_input(
        "Number of Employees", value=5191.0
    )

st.divider()

# 7. Prediction Button
if st.button("🔍 Predict Subscription", use_container_width=True):

    # Create Customer DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    # Identify Numerical and Categorical Features
    numerical_features = [
        "age", "campaign", "pdays", "previous",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed"
    ]

    categorical_features = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month",
        "day_of_week", "poutcome"
    ]

    # Apply Saved Preprocessing
    scaled_data = scaler.transform(
        input_data[numerical_features]
    )

    encoded_data = encoder.transform(
        input_data[categorical_features]
    )

    # Combine Processed Features
    final_data = np.hstack([scaled_data, encoded_data])

    # Make Prediction
    prediction = model.predict(final_data)[0]
    probability = model.predict_proba(final_data)[0][1]

    # Display Result
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to subscribe! 🎉")
    else:
        st.warning("The customer is unlikely to subscribe.")

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))
