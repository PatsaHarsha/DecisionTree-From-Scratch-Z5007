import numpy as np
import pandas as pd
from src.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv("data/dsm_dataset.csv")

# ---------- PREPROCESSING (CRITICAL) ----------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
    .str.replace("/", "_")
)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df = df.drop_duplicates()
# --------------------------------------------

X = df.drop(columns=["DSM"]).values
y = df["DSM"].values

# Train-test split
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
tree.fit(X_train, y_train)

# Predict
predictions = tree.predict(X_test)

# Sanity checks
print("Predictions shape:", predictions.shape)
print("Any NaNs:", np.isnan(predictions).any())
print("First 5 predictions:", predictions[:5])
print("Min/Max:", predictions.min(), predictions.max())


