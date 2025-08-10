
import streamlit as st
import pickle

# Load trained model
with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌦️ Weather Prediction App")
st.write("Enter weather parameters below to predict if it will rain.")

# Input fields
temperature = st.number_input("Temperature (°C)", min_value=-20, max_value=50, value=25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=150, value=10)
pressure = st.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1010)

if st.button("Predict"):
    prediction = model.predict([[temperature, humidity, wind_speed, pressure]])[0]
    if prediction == 1:
        st.success("🌧️ It will rain!")
    else:
        st.info("☀️ No rain expected.")

