import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample weather dataset
data = {
    'temperature': [30, 22, 25, 28, 35, 18, 15],
    'humidity': [80, 65, 70, 75, 90, 60, 55],
    'wind_speed': [12, 10, 9, 11, 13, 7, 6],
    'pressure': [1010, 1005, 1008, 1012, 1002, 1007, 1006],
    'rain': [1, 0, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df.drop('rain', axis=1)
y = df['rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)
