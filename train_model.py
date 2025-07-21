import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset
data = {
    'temperature': [30, 22, 25, 28, 35, 18, 15, 27, 24, 29],
    'humidity': [80, 65, 70, 75, 90, 60, 55, 72, 68, 85],
    'wind_speed': [12, 10, 9, 11, 13, 7, 6, 8, 9, 10],
    'pressure': [1010, 1005, 1008, 1012, 1002, 1007, 1006, 1011, 1009, 1003],
    'rain': [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and label
X = df.drop('rain', axis=1)
y = df['rain']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'weather_model.pkl'")
