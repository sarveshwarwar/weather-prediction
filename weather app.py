import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Example training data
data = pd.DataFrame({
    'temperature': [30, 22, 25, 28],
    'humidity': [80, 60, 65, 70],
    'pressure': [1012, 1010, 1011, 1013],
    'wind': [10, 5, 7, 8],
    'label': [1, 0, 0, 1]  # 1 = Rainy, 0 = Not Rainy
})

X = data[['temperature', 'humidity', 'pressure', 'wind']]
y = data['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
