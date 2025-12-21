import numpy as np
from src.node import Node
from src.utils import mse, split_dataset, best_split, should_stop

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Custom Decision Tree Regressor implemented from scratch.
        Parameters:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum samples required to split a node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Train the regression tree."""
        self.root = self._build_tree(X, y, depth=0, parent_value=np.mean(y))

    def _build_tree(self, X, y, depth, parent_value=None):
        """
        Recursively build the decision tree.
        """
        # Handle empty node safely
        if len(y) == 0:
            return Node(value=parent_value)

        n_samples, n_features = X.shape

        # Pure node check 
        if mse(y) == 0:
            return Node(value=np.mean(y) if len(y) > 0 else parent_value)
        
        # Stopping conditions
        if should_stop(depth, self.max_depth, n_samples, self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Find best split
        feature_index, threshold, best_mse = best_split(X, y)

        # If no split is possible, create leaf
        if feature_index is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        # Split dataset
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value=np.mean(y) if len(y) > 0 else parent_value)

        # Recursively build subtrees
        current_mean = np.mean(y)

        left_child = self._build_tree(
            X_left, y_left, depth + 1, parent_value=current_mean
        )

        right_child = self._build_tree(
            X_right, y_right, depth + 1, parent_value=current_mean
        )


        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
            value=current_mean
        )

    def predict_one(self, x):
        """Predict a single sample by traversing the tree."""
        node = self.root
        while node.left is not None and node.right is not None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """Predict multiple samples."""
        return np.array([self.predict_one(sample) for sample in X])
