
# Portuguese Bank Marketing Prediction - Streamlit App

# 1. Import Required Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 2. Configure Professional Page
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 3. Professional Banking Dashboard Styling
st.markdown("""
<style>
    /* Main Application Background */
    .stApp {
        background-color: #f4f7fb;
    }

    /* Main Content Width */
section.main > div.block-container {
    max-width: 1200px !important;
    width: 95% !important;
    padding: 2rem 2rem !important;
    margin: auto;
}

/* Responsive Layout */
@media (max-width: 768px) {
    section.main > div.block-container {
        width: 100% !important;
        padding: 1rem !important;
    }
}

    /* Headings */
    h1, h2, h3 {
        color: #142d4e;
        font-family: Arial, sans-serif;
        font-weight: 700;
    }

    /* Normal Text */
    p, label, span {
        font-family: Arial, sans-serif;
    }

    /* Buttons */
    .stButton > button {
        background-color: #153e68;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-weight: 600;
        min-height: 45px;
    }

    .stButton > button:hover {
        background-color: #24649e;
        color: white;
        border: none;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e1e8f0;
    }

   /* Improve Input Label Readability */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #142d4e !important;
}

/* Improve Selected Input Values */
[data-baseweb="select"] div,
[data-testid="stNumberInput"] input {
    font-size: 16px !important;
    color: #142d4e !important;
}

/* Improve General Text */
.stMarkdown p {
    font-size: 16px !important;
    line-height: 1.5;
}
    /* Dividers */
    hr {
        border-color: #dce4ee;
    }
</style>
""", unsafe_allow_html=True)

# 4. Load Trained Model and Preprocessing Files
model = joblib.load("random_forest_bank_model.pkl")
encoder = joblib.load("bank_encoder.pkl")
scaler = joblib.load("bank_scaler.pkl")

# 5. Professional Dashboard Header
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("🏦 Portuguese Bank Marketing")
    st.markdown(
        "### Customer Subscription Prediction"
    )
    st.write(
        "Analyze customer information and predict "
        "the likelihood of subscribing to a term deposit."
    )

with header_col2:
    st.markdown(
        """
        <div style="
            background-color: #153e68;
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 15px;
        ">
            <h2 style="color: white;">BANK AI</h2>
            <p style="color: white;">PREDICTION SYSTEM</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# 6. Project KPI Cards
st.divider()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Records", "41,176")
kpi2.metric("Subscription Rate", "11.27%")
kpi3.metric("Models Evaluated", "6")
kpi4.metric("Final Model", "Random Forest")

st.divider()

# 7. Customer Information
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
    default = st.selectbox("Credit in Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
    loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])

with col3:
    month = st.selectbox("Last Contact Month", [
        "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ])

    day_of_week = st.selectbox("Last Contact Day", [
        "mon", "tue", "wed", "thu", "fri"
    ])

    campaign = st.number_input(
        "Number of Contacts During Campaign", min_value=1, value=2
    )

    pdays = st.number_input(
        "Days Since Previous Contact (999 = Never Contacted)",
        min_value=0, value=999
    )

st.divider()
# 8. Previous Campaign Information
st.subheader("📞 Previous Campaign Information")

col4, col5 = st.columns(2)

with col4:
    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["nonexistent", "failure", "success"]
    )

with col5:
    emp_var_rate = st.number_input(
        "Employment Variation Rate",
        value=1.1
    )

    cons_price_idx = st.number_input(
        "Consumer Price Index",
        value=93.994
    )

    cons_conf_idx = st.number_input(
        "Consumer Confidence Index",
        value=-36.4
    )

    euribor3m = st.number_input(
        "Euribor 3 Month Rate",
        value=4.857
    )

    nr_employed = st.number_input(
        "Number of Employees",
        value=5191.0
    )

st.divider()

# 9. Prediction Button
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

    # Display Professional Prediction Result
    st.divider()
    st.subheader("📊 Subscription Prediction")

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        if prediction == 1:
            st.success("✅ Customer is likely to subscribe!")
        else:
            st.warning("⚠️ Customer is unlikely to subscribe.")

        st.write("**Subscription Probability**")
        st.progress(float(probability))

    with result_col2:
        st.metric(
            label="Subscription Probability",
            value=f"{probability * 100:.2f}%"
        )

    st.caption(
        "This prediction is generated using the trained "
        "Random Forest Machine Learning model."
    )
