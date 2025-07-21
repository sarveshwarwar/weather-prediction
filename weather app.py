import streamlit as st
import pandas as pd
import pickle

# Load the trained model
def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

st.set_page_config(page_title="Weather Predictor")
st.title("🌤️ Weather Prediction App")

model = load_model()

# User inputs
temperature = st.slider("🌡️ Temperature (°C)", -10, 50, 25)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)

# Prepare input for model
features = pd.DataFrame([[temperature, humidity]], columns=["temperature", "humidity"])

# Prediction
if st.button("🔍 Predict Weather"):
    prediction = model.predict(features)
    st.success(f"🌈 Predicted Weather: **{prediction[0]}**")
