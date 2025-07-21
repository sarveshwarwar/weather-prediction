import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dummy data
df = pd.DataFrame({
    'temperature': [30, 22, 35, 18, 40],
    'humidity': [40, 60, 20, 90, 15],
    'pressure': [1012, 1015, 1010, 1008, 1020],
    'wind_speed': [10, 5, 20, 12, 7],
    'rain': [0, 1, 0, 1, 0]  # Target variable
})

X = df[['temperature', 'humidity', 'pressure', 'wind_speed']]
y = df['rain']

model = LogisticRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
