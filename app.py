#now we make app for heart disease prediction
import streamlit as st
import pandas as pd 
import joblib

model=joblib.load('best_heart_disease_model.pkl')
scaler=joblib.load('scaler.pkl')
model_columns=joblib.load('model_columns.pkl')

st.title("Heart Disease Prediction App🫀")
st.write("This app predicts whether a person has heart disease or not based on the input features.")
st.markdown("please provide the following details👇")
age=st.slider("Age", 18, 100, 40)
sex=st.selectbox("Sex", ["Male", "Female"])
chest_pain_type=st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp=st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol=st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg=st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr=st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
oldpeak=st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope=st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])
exercise_angina=st.selectbox("Exercise Induced Angina", ["Yes", "No"])

if st.button("Predict"):

    # Convert user inputs to training format
    sex = "M" if sex == "Male" else "F"
    exercise_angina = "Y" if exercise_angina == "Yes" else "N"
    fasting_bs = 1 if fasting_bs == "Yes" else 0

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain_type: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing dummy columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale ONLY numeric columns
    numeric_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("☠️ The model predicts that the person may have heart disease. Please consult a doctor.")
    else:
        st.success("✅ The model predicts that the person is unlikely to have heart disease.")