import numpy as np

def mse(y):
    """
    Calculate Mean Squared Error of a target array.
    Used as impurity measure for Decision Tree Regression.
    """
    if len(y) == 0:
        return 0
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)


def split_dataset(X, y, feature_index, threshold):
    """
    Split dataset into left and right subsets based on threshold.

    LEFT  = samples where X[:, feature_index] <= threshold
    RIGHT = samples where X[:, feature_index] > threshold
    """
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]

    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]


def best_split(X, y):
    """
    Find the best feature and threshold that minimizes MSE.
    Returns: feature_index, threshold, mse_value
    """
    best_feature = None
    best_threshold = None
    best_mse = float("inf")

    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        # Consider all unique values of this feature as potential thresholds
        values = np.unique(X[:, feature_index])
        thresholds = (values[:-1] + values[1:]) / 2


        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

            # Skip invalid splits
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Weighted MSE of children
            mse_left = mse(y_left)
            mse_right = mse(y_right)

            weighted_mse = (len(y_left) * mse_left + len(y_right) * mse_right) / n_samples

            # Update best split if better found
            if weighted_mse < best_mse:
                best_mse = weighted_mse
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_mse


def should_stop(depth, max_depth, n_samples, min_samples_split):
    """
    Determine whether to stop splitting the node.
    """
    if depth >= max_depth:
        return True
    if n_samples < min_samples_split:
        return True

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    """Calculate R^2 Score."""
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_tot == 0:
        return 0
        
    return 1 - (ss_res / ss_tot)
