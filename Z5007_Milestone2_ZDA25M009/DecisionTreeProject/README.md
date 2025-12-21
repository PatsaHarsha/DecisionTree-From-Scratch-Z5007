

# Decision Tree Regression from Scratch

**Z5007: Programming and Data Structures – IIT Madras Zanzibar**

## Project Overview

This project implements a **Decision Tree Regression algorithm from scratch** using only
fundamental Python data structures. No machine learning libraries (e.g., scikit-learn) are used
for model implementation. The objective is to understand how **data structures, recursion,
and impurity measures** drive decision tree learning.

The model predicts a continuous target variable using **Mean Squared Error (MSE)** as the
splitting criterion and supports configurable stopping conditions.

---

## Project Structure

```
DecisionTreeProject/
│
├── src/
│   ├── node.py        # Custom tree node data structure
│   ├── tree.py        # Decision Tree Regressor implementation
│   └── utils.py       # MSE, splitting logic, stopping conditions
│
├── data/
│   └── dsm_dataset.csv
│
├── main.py            # Training & evaluation script
└── README.md
```

---

## Requirements

* Python **3.8+**
* Allowed libraries:

  * `numpy`
  * `pandas`
  * `matplotlib` (for evaluation/plots)

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## Dataset

**Dataset Name:** DSM Strength Prediction Dataset  
**Samples:** 1664  
**Features:** 25  
**Target:** Ultimate Strength (DSM)

The dataset is preprocessed to:

* Standardize column names
* Convert features to numeric
* Remove redundant columns
* Split into training and testing sets (70:30)

---

## How to Run

1. Place the dataset in the `data/` folder.
2. Run the training script:

```bash
python main.py
```

### Example Usage (inside `main.py`)

```python
from src.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
```

---

## Model Details

* **Algorithm:** Decision Tree Regression
* **Split Criterion:** Mean Squared Error (MSE)
* **Threshold Selection:** Midpoints between sorted unique feature values
* **Stopping Conditions:**

  * Maximum depth
  * Minimum samples per split
  * Pure node (MSE = 0)

---

## Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Performance is compared against:

* A naive mean predictor
* scikit-learn’s `DecisionTreeRegressor` (reference only)

---

## Time & Space Complexity

- **Training:** O(d · n · log n)  
- **Prediction:** O(tree depth) per sample  
- **Space:** O(n · d)

Where:
- n = number of samples  
- d = number of features


---

## Notes

* All core data structures are implemented manually.
* No forbidden libraries are used.
* The code is modular, readable, and easy to extend.

---

## Author

**Patsa Harsha Sai**
M.Tech Data Science & Artificial Intelligence
IIT Madras Zanzibar
Roll No: **ZDA25M009**

Instructor: **Innocent Nyalala**

---

