import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example dataset (you can replace with your own weather dataset)
data = {
    "temperature": [25, 30, 18, 22, 28, 35, 15, 20],
    "humidity": [70, 80, 60, 65, 85, 90, 50, 55],
    "wind_speed": [10, 5, 20, 15, 8, 12, 25, 18],
    "pressure": [1010, 1005, 1015, 1012, 1008, 1003, 1018, 1014],
    "rain": [1, 1, 0, 0, 1, 1, 0, 0]  # 1 = Rain, 0 = No Rain
}

df = pd.DataFrame(data)

# Features & Target
X = df[["temperature", "humidity", "wind_speed", "pressure"]]
y = df["rain"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to file
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as weather_model.pkl")
