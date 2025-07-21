import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample training data
data = {
    'Temperature': [30, 25, 27, 22, 35],
    'Humidity': [80, 60, 75, 50, 40],
    'WindSpeed': [10, 5, 15, 7, 12],
    'Rain': [1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Train the model
X = df[['Temperature', 'Humidity', 'WindSpeed']]
y = df['Rain']
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("🌦️ Weather Prediction App")
st.markdown("Enter weather parameters to predict whether it will rain.")

temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50, max_value=60, value=25)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
wind_speed = st.number_input("🍃 Wind Speed (km/h)", min_value=0, max_value=150, value=10)

if st.button("Predict"):
    input_data = pd.DataFrame([[temperature, humidity, wind_speed]], 
                              columns=['Temperature', 'Humidity', 'WindSpeed'])
    prediction = model.predict(input_data)[0]
    result = "🌧️ It will rain!" if prediction == 1 else "☀️ No rain expected."
    st.success(result)
