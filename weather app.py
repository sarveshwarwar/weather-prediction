import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_data
def load_model():
    try:
        with open("weather_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Train and save the model as 'weather_model.pkl'.")
        return None

model = load_model()

st.title("🌦️ Weather Prediction App")
st.write("Enter weather parameters below to predict if it will rain.")

temperature = st.number_input("Temperature (°C)", value=25)
humidity = st.number_input("Humidity (%)", value=70)
wind_speed = st.number_input("Wind Speed (km/h)", value=10)
pressure = st.number_input("Pressure (hPa)", value=1010)

if st.button("Predict"):
    if model:
        features = np.array([[temperature, humidity, wind_speed, pressure]])
        prediction = model.predict(features)[0]
        result = "🌧️ Rain predicted!" if prediction == 1 else "☀️ No rain predicted."
        st.success(result)
