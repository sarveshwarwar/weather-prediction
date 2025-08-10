import streamlit as st
import pickle

# Load model
with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌦️ Weather Prediction App")
st.write("Enter weather parameters below to predict if it will rain.")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-50, max_value=60, value=25)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=200, value=10)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)

if st.button("Predict"):
    prediction = model.predict([[temp, humidity, wind_speed, pressure]])[0]
    if prediction == 1:
        st.success("🌧️ It will rain!")
    else:
        st.success("☀️ No rain expected.")
