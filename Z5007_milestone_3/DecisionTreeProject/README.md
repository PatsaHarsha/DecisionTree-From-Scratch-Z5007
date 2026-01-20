Decision Tree Regression from Scratch

Z5007: Programming and Data Structures – IIT Madras Zanzibar

Project Overview

This project implements a Decision Tree Regression algorithm from scratch using only
fundamental Python data structures and numerical libraries.
No machine learning libraries (such as scikit-learn) are used for model construction.

The primary objective of this project is to demonstrate how data structures, recursion,
impurity measures, and algorithmic design collectively enable a working decision tree
learning system.

The model predicts a continuous target variable using Mean Squared Error (MSE) as the
splitting criterion and supports configurable stopping conditions to control model complexity.

Project Structure
DecisionTreeProject/
│
├── src/
│   ├── __init__.py
│   ├── node.py        # Custom tree node data structure
│   ├── tree.py        # Decision Tree Regressor (from scratch)
│   └── utils.py       # MSE, splitting logic, stopping conditions
│
├── data/
│   └── dsm_dataset.csv
│
├── tests/
│   └── test_tree.py   # Minimal unit tests
│
├── screenshots/
│   ├── predictions_output.png
│   ├── tree_structure_stats.png
│   ├── benchmarking_output.png
│   └── tests_passed.png
│
├── main.py            # Training, evaluation & benchmarking script
├── requirements.txt
└── README.md

Requirements

Python 3.8+

Allowed libraries:

numpy

pandas

matplotlib (for evaluation plots only)

scikit-learn (used strictly for benchmarking, not model implementation)

Install dependencies:

pip install -r requirements.txt

Dataset

Dataset Name: DSM Strength Prediction Dataset
Number of Samples: 1664
Number of Features: 25
Target Variable: Ultimate Strength (DSM)

Preprocessing Steps

Removal of missing values

Conversion to numeric format

Feature–target separation

Train–test split (70% training / 30% testing)

The dataset file must be placed in the data/ directory.

How to Run

From the project root directory:

python main.py


This command will:

Train the custom Decision Tree Regressor

Generate predictions on the test set

Print evaluation metrics

Display tree structure statistics

Run benchmarking comparisons

Sample Output

When running main.py, the following outputs are produced:

Predictions shape: (498,)
Any NaNs: False

Tree depth: 10
Total nodes: 603
Leaf nodes: 302

Custom Tree Training Time (s): 2.2184
Custom Tree Prediction Time (s): 0.0004

Mean Baseline RMSE: 67.25
Mean Baseline R2: -0.03

Sklearn RMSE: 25.66
Sklearn R2: 0.85


These outputs confirm correct end-to-end execution, numerical stability, and valid
performance evaluation.

Corresponding screenshots are provided in the screenshots/ directory.

Model Details

Algorithm: Decision Tree Regression

Split Criterion: Mean Squared Error (MSE)

Threshold Selection: Midpoints between sorted unique feature values

Stopping Conditions:

Maximum tree depth

Minimum samples per split

Pure node (MSE = 0)

The model is implemented using recursive tree construction and a custom Node
data structure.

Evaluation Metrics

The model is evaluated using standard regression metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Benchmarking Comparisons

Performance is compared against:

A naive mean predictor (baseline)

scikit-learn’s DecisionTreeRegressor (reference implementation only)

Benchmarking includes:

Accuracy comparison (RMSE, R²)

Training time

Prediction time

Testing

A minimal unit test suite is included in the tests/ directory to verify:

Successful model training

Correct prediction dimensionality

Absence of NaN values in predictions

Valid tree structure (node and leaf counts)

Run tests using:

python tests/test_tree.py


A successful test run prints:

All tests passed successfully!

Time & Space Complexity

Training Time Complexity:
O(d · n · log n)

Prediction Time Complexity:
O(tree depth) per sample

Space Complexity:
O(n · d)

Where:

n = number of samples

d = number of features

Notes

All core algorithms and data structures are implemented manually.

No forbidden libraries are used for learning logic.

The project is fully reproducible using relative paths.

Visual and terminal-based evidence is included for transparency.

Author

Patsa Harsha Sai
M.Tech Data Science & Artificial Intelligence
IIT Madras Zanzibar
Roll No: ZDA25M009

Instructor: Innocent Nyalala