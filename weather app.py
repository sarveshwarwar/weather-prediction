
import streamlit as st
import pickle
import numpy as np


with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌦️ Weather Prediction App")
st.write("Enter weather parameters below to predict if it will rain.")


temperature = st.number_input("Temperature (°C)", value=25)
humidity = st.number_input("Humidity (%)", value=70)
wind_speed = st.number_input("Wind Speed (km/h)", value=10)
pressure = st.number_input("Pressure (hPa)", value=1010)


if st.button("Predict"):
    features = np.array([[temperature, humidity, wind_speed, pressure]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🌧️ It will rain.")
    else:
        st.info("☀️ No rain expected.")
