# weather_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

st.title("🌦️ Weather Prediction App")

# Step 1: Training data
st.subheader("📊 Step 1: Train the Model")
train_button = st.button("Train Model")

if train_button:
    # Sample data (Temp, Humidity, Pressure, Wind Speed)
    data = pd.DataFrame({
        'temperature': [30, 22, 25, 28, 32, 26],
        'humidity': [80, 60, 65, 70, 90, 75],
        'pressure': [1012, 1010, 1011, 1013, 1009, 1012],
        'wind': [10, 5, 7, 8, 12, 6],
        'label': [1, 0, 0, 1, 1, 0]  # 1 = Rainy, 0 = Not Rainy
    })

    X = data[['temperature', 'humidity', 'pressure', 'wind']]
    y = data['label']

    model = LogisticRegression()
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("✅ Model trained and saved successfully as model.pkl")

# Step 2: Load model and make prediction
st.subheader("🌤️ Step 2: Predict Weather")

temperature = st.slider("Temperature (°C)", 10, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 70)
pressure = st.slider("Pressure (hPa)", 900, 1100, 1010)
wind = st.slider("Wind Speed (km/h)", 0, 100, 10)

predict_button = st.button("Predict")

if predict_button:
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        features = np.array([[temperature, humidity, pressure, wind]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("🌧️ Prediction: Rainy Weather")
        else:
            st.info("☀️ Prediction: No Rain")

    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please train the model first.")

