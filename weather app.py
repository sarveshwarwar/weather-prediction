
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌤 Weather Prediction App (Self-Contained)")

# Sample weather dataset
data = {
    "temperature": [30, 22, 25, 28, 35, 20, 18, 40, 32, 26],
    "humidity": [80, 60, 65, 70, 50, 90, 95, 40, 55, 75],
    "wind_speed": [12, 7, 15, 9, 20, 5, 4, 25, 13, 8],
    "weather": ["Rain", "Cloudy", "Cloudy", "Sunny", "Sunny", "Rain", "Rain", "Sunny", "Sunny", "Cloudy"]
}
df = pd.DataFrame(data)

# Features and labels
X = df[["temperature", "humidity", "wind_speed"]]
y = df["weather"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (no saving/loading from file)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# User input section
st.subheader("Enter Weather Conditions:")
temp = st.number_input("🌡 Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, value=60)
wind = st.number_input("💨 Wind Speed (km/h)", min_value=0, max_value=100, value=10)

# Predict button
if st.button("Predict Weather"):
    prediction = model.predict([[temp, humidity, wind]])
    st.success(f"Predicted Weather: **{prediction[0]}**")

# Show dataset
if st.checkbox("Show training dataset"):
    st.dataframe(df)
