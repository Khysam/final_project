import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model dan preprocessing
model = joblib.load("laptop_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

def run_ml_app():
    st.title("💻 Prediksi Harga Laptop")

    # Input user
    company = st.text_input("Company")
    product = st.text_input("Product")
    typename = st.text_input("TypeName")
    inches = st.number_input("Inches", min_value=10.0, max_value=20.0, step=0.1)
    screenresolution = st.text_input("ScreenResolution")
    cpu = st.text_input("CPU")
    ram = st.text_input("RAM")
    memory = st.text_input("Memory")
    gpu = st.text_input("GPU")
    opsys = st.text_input("Operating System")
    weight = st.text_input("Weight")

    if st.button("Prediksi Harga"):
        input_data = pd.DataFrame({
            'Company':[company],
            'Product':[product],
            'TypeName':[typename],
            'Inches':[inches],
            'ScreenResolution':[screenresolution],
            'Cpu':[cpu],
            'Ram':[ram],
            'Memory':[memory],
            'GPU':[gpu],
            'OpSys':[opsys],
            'Weight':[weight]
        })

        # Encode sesuai training
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(input_scaled)[0]

        st.success(f"💶 Prediksi harga laptop: {prediction:.2f} Euro")

        # Tambahan visualisasi
        st.metric("Predicted Price", f"{prediction:.2f} €")
        st.progress(min(int(prediction/5000*100), 100))  # progress bar relatif
