import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example dataset
data = pd.DataFrame({
    "temperature": [25, 30, 18, 22, 28, 15],
    "humidity": [70, 65, 80, 75, 60, 85],
    "wind_speed": [10, 12, 8, 15, 7, 5],
    "pressure": [1010, 1012, 1008, 1011, 1013, 1007],
    "rain": [1, 0, 1, 0, 0, 1]  # 1 = rain, 0 = no rain
})

X = data[["temperature", "humidity", "wind_speed", "pressure"]]
y = data["rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as weather_model.pkl")

