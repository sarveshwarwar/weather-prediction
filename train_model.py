# train_weather_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("weather_data.csv")

# Features & Target
X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = df['rain']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Weather model trained and saved as weather_model.pkl")

