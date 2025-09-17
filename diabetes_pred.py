# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 00:41:33 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

# load the saved model (fix path issue)
loaded_model = pickle.load(open(
    r"C:\Users\HP\OneDrive\Desktop\Deploy\trained_model.sav", "rb"
))

# function for Prediction
def diabetes_prediction(input_data):
    # convert input to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    # title
    st.title("Diabetes Prediction Web App")

    # input fields
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person")

    # prediction result
    diagnosis = ""

    # button for prediction
    if st.button("Diabetes Test Result"):
        try:
            # convert all inputs to float before prediction
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]

            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "❌ Please enter valid numeric values!"

    st.success(diagnosis)


if __name__ == "__main__":
    main()
