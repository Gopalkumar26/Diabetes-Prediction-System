import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("model_joblib.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

.title{
    font-size:42px;
    color:#1565C0;
    text-align:center;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result{
    padding:15px;
    border-radius:10px;
    font-size:22px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<p class="title">🩺 Diabetes Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Diabetes Prediction using Logistic Regression</p>',
    unsafe_allow_html=True
)

st.write("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("About")

st.sidebar.info("""
This project predicts whether a patient is diabetic or not using Logistic Regression.

Dataset:
PIMA Indians Diabetes Dataset

Developed using

✔ Python

✔ Scikit-Learn

✔ Streamlit

✔ Logistic Regression
""")

st.sidebar.write("---")

st.sidebar.success("Enter patient details.")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=250,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.5
    )

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.35
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.write("---")

# ---------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------

errors = []

if glucose <= 0:
    errors.append("Glucose must be greater than 0")

if blood_pressure <= 0:
    errors.append("Blood Pressure must be greater than 0")

if bmi <= 0:
    errors.append("BMI must be greater than 0")

if age <= 0:
    errors.append("Age must be greater than 0")

if pedigree < 0:
    errors.append("Pedigree cannot be negative")

if insulin < 0:
    errors.append("Insulin cannot be negative")

if skin_thickness < 0:
    errors.append("Skin Thickness cannot be negative")

if len(errors) > 0:

    for error in errors:
        st.error(error)

    st.stop()

# ---------------------------------------------------
# CREATE INPUT ARRAY
# ---------------------------------------------------

patient_data = np.array([
    [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]
])

scaled_data = scaler.transform(patient_data)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

predict = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict:

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    st.write("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🔴 High Risk: Patient is likely Diabetic")

    else:

        st.success("🟢 Low Risk: Patient is likely Not Diabetic")

    st.write("---")

    # -----------------------------
    # Probability
    # -----------------------------

    st.subheader("Prediction Probability")

    st.progress(float(probability))

    st.metric(
        label="Probability of Diabetes",
        value=f"{probability*100:.2f}%"
    )

    # -----------------------------
    # Risk Level
    # -----------------------------

    st.subheader("Risk Level")

    if probability < 0.30:

        st.success("🟢 Low Risk")

    elif probability < 0.60:

        st.warning("🟡 Moderate Risk")

    else:

        st.error("🔴 High Risk")

    st.write("---")

    # -----------------------------
    # Patient Details
    # -----------------------------

    st.subheader("Patient Details")

    patient_df = pd.DataFrame({

        "Feature":[
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],

        "Value":[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]

    })

    st.dataframe(
        patient_df,
        use_container_width=True
    )

    st.write("---")

    # -----------------------------
    # Health Recommendation
    # -----------------------------

    st.subheader("Health Recommendation")

    if prediction == 1:

        st.warning("""
### Recommendations

✔ Consult a doctor.

✔ Monitor blood sugar regularly.

✔ Reduce sugar intake.

✔ Exercise at least 30 minutes daily.

✔ Maintain a healthy BMI.

✔ Drink sufficient water.

✔ Get regular health check-ups.
""")

    else:

        st.success("""
### Recommendations

✔ Continue healthy eating.

✔ Exercise regularly.

✔ Maintain a healthy weight.

✔ Avoid excessive sugar intake.

✔ Have routine health check-ups.

✔ Stay physically active.
""")

    st.write("---")

    # -----------------------------
    # Download Report
    # -----------------------------

    report = f"""
DIABETES PREDICTION REPORT

----------------------------------------

Pregnancies : {pregnancies}

Glucose : {glucose}

Blood Pressure : {blood_pressure}

Skin Thickness : {skin_thickness}

Insulin : {insulin}

BMI : {bmi}

Diabetes Pedigree Function : {pedigree}

Age : {age}

----------------------------------------

Prediction :

{"Diabetic" if prediction==1 else "Non-Diabetic"}

Probability :

{probability*100:.2f} %

----------------------------------------

Generated using

Diabetes Prediction System

Logistic Regression
"""

    st.download_button(

        label="📄 Download Report",

        data=report,

        file_name="Diabetes_Report.txt",

        mime="text/plain"

    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("---")

st.markdown(
"""
<div class="footer">

Developed using ❤️ Streamlit | Scikit-Learn | Logistic Regression

</div>
""",
unsafe_allow_html=True
)