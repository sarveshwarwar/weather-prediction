import streamlit as st
import pickle
import numpy as np

# Load the model
with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌦️ Weather Prediction App")
st.write("Enter weather parameters below to predict if it will rain.")

# User inputs
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=150, value=10)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)

# Prediction
if st.button("Predict"):
    features = np.array([[temperature, humidity, wind_speed, pressure]])
    prediction = model.predict(features)
    result = "🌧️ Rain" if prediction[0] == 1 else "☀️ No Rain"
    st.success(f"Prediction: {result}")
