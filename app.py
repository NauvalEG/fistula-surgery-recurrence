import streamlit as st
import pandas as pd
import joblib 
import numpy as np

# --- Load your trained model ---
# Replace this with your actual model path
MODEL_PATH = "D:/package/fistula-recurrence-prediction/training/catboost001.pkl"
model = joblib.load(MODEL_PATH)
# print(model.feature_names_)

# --- Streamlit UI ---
st.title("Fistula Surgery Recurrence Prediction App")


# Inputs
with st.form(key='columns_in_form'):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=100, step=1)
        weight = st.number_input("Body Weight (kg)", min_value=10.0, max_value=200.0, step=0.1)
        cause_options = {
            "Gynecology": 1,
            "Obstetric": 2,
            "Cancer": 3,
            "Others": 4
            }
        cause_label = st.selectbox("Cause of Initial Surgery", list(cause_options.keys()))
        cause = cause_options[cause_label]
    with c2:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=0.1)
        fistula_size = st.number_input("Fistula Size (cm)", min_value=0.01, max_value=10.0, step=0.01)
        route_options = {
            "TAH": 1,
            "TVH": 2,
            "TLH": 3,
            "RH": 4,
            "SVH": 5,
            "Others": 6
        }
        route_label = st.selectbox("Surgery Route", list(route_options.keys()))
        surgery_route = route_options[route_label]

    if st.form_submit_button("Predict"):

    # --- Preprocessing ---
        def categorize_age(age):
            if age < 18:
                return 1
            elif age <= 35:
                return 2
            elif age <= 59:
                return 3
            else:
                return 4
            
        
        def calculate_bmi(weight, height_cm):
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
            return round(bmi, 2)

        def categorize_bmi(bmi):
            if bmi < 18.5:
                bmi_label = 'Underweight'
                return bmi_label, 1  # Underweight
            elif bmi < 25:
                bmi_label = 'Normal'
                return bmi_label, 2  # Normal
            elif bmi < 30:
                bmi_label = 'Overweight'
                return bmi_label, 3  # Overweight
            else:
                bmi_label = 'Obese'
                return bmi_label, 4  # Obese

        def categorize_fistula(size):
            if size < 0.15:
                return 1
            elif size <= 1:
                return 2
            elif size <= 2:
                return 3
            else:
                return 4

    # --- Prediction ---

        # Calculate BMI and categorize
        bmi_value = calculate_bmi(weight, height_cm)
        age_cat = categorize_age(age)
        bmi_cat = categorize_bmi(bmi_value)
        fistula_cat = categorize_fistula(fistula_size)

        # Final input for model
        input_data = pd.DataFrame([{
            'cause': cause,
            'route': surgery_route,
            'BMI': bmi_cat[1],
            'fistula_size': fistula_cat,
            'age_category': age_cat,
            'raw_bmi': bmi_value  # just for display
        }])

        st.subheader("Processed Input")
        st.write({
            "Cause": f"{cause_label} ({cause})",
            "Surgery Route": f"{route_label} ({surgery_route})",
            "BMI Category": bmi_cat[0],
            "Fistula Size (Category)": f"{fistula_size}cm ({fistula_cat})",
            "Age (Category)": f"{age} ({age_cat})",
            "Raw BMI": bmi_value
        })

        


        # Predict
        model_input = input_data.drop(columns='raw_bmi')
        prediction_proba = model.predict_proba(model_input)[0]
        prediction = model.predict(model_input)[0]
        confidence = np.max(prediction_proba)

        st.subheader("Prediction Result")
        st.markdown(f"**Prediction:** {'Recurring' if prediction > 0.75 else 'Not Recurring'}")
        st.markdown(f"**Confidence:** {confidence:.2%}")