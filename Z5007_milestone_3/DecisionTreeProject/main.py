import numpy as np
import pandas as pd
from src.tree import DecisionTreeRegressor
from src.utils import rmse, r2
import time

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

# tree stats
print("Tree depth:", tree.max_depth)
print("Total nodes:", tree.count_nodes())
print("Leaf nodes:", tree.count_leaves())

# ---------------- TIMING: CUSTOM TREE ----------------
start_time = time.time()
tree.fit(X_train, y_train)
train_time = time.time() - start_time

start_time = time.time()
preds = tree.predict(X_test)
predict_time = time.time() - start_time

print("Custom Tree Training Time (s):", round(train_time, 4))
print("Custom Tree Prediction Time (s):", round(predict_time, 4))

# ---------------- BASELINE: MEAN PREDICTOR ----------------
mean_value = y_train.mean()
mean_preds = np.full_like(y_test, mean_value)

print("Mean Baseline RMSE:", rmse(y_test, mean_preds))
print("Mean Baseline R2:", r2(y_test, mean_preds))

# ---------------- BENCHMARK: SKLEARN TREE ----------------
from sklearn.tree import DecisionTreeRegressor as SkTree

sk_tree = SkTree(max_depth=10, random_state=42)

start_time = time.time()
sk_tree.fit(X_train, y_train)
sk_train_time = time.time() - start_time

start_time = time.time()
sk_preds = sk_tree.predict(X_test)
sk_predict_time = time.time() - start_time

print("Sklearn Tree Training Time (s):", round(sk_train_time, 4))
print("Sklearn Tree Prediction Time (s):", round(sk_predict_time, 4))
print("Sklearn RMSE:", rmse(y_test, sk_preds))
print("Sklearn R2:", r2(y_test, sk_preds))
