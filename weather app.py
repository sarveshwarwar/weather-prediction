import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Title
st.title("🌤️ Weather Prediction App")

# 2. Collect user input
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)

# 3. Dummy data for training
data = pd.DataFrame({
    'temperature': [30, 22, 35, 18, 40],
    'humidity': [40, 60, 20, 90, 15],
    'pressure': [1012, 1015, 1010, 1008, 1020],
    'wind_speed': [10, 5, 20, 12, 7],
    'rain': [0, 1, 0, 1, 0]  # 0 = No rain, 1 = Rain
})

# 4. Train model inline
X = data[['temperature', 'humidity', 'pressure', 'wind_speed']]
y = data['rain']
model = LogisticRegression()
model.fit(X, y)

# 5. Predict with user input
input_df = pd.DataFrame([[temp, humidity, pressure, wind_speed]],
                        columns=['temperature', 'humidity', 'pressure', 'wind_speed'])
prediction = model.predict(input_df)[0]

# 6. Show result
if st.button("Predict"):
    st.success("☔ Rain Expected!" if prediction == 1 else "🌞 No Rain Expected.")

