import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample weather dataset
data = {
    "temperature": [30, 35, 40, 20, 25, 10, 15, 45],
    "humidity": [70, 80, 90, 40, 50, 20, 30, 95],
    "label": ["Rainy", "Rainy", "Rainy", "Clear", "Clear", "Clear", "Clear", "Rainy"]
}

df = pd.DataFrame(data)

# Train with only 2 features
X = df[["temperature", "humidity"]]
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
