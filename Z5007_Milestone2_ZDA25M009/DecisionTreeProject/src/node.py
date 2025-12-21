class Node:
    """
    A single node in the Decision Tree Regression model.
    Stores split information for internal nodes
    and prediction value for leaf nodes.
    """

    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        """
        Initialize a decision tree node.

        Parameters
        ----------
        feature_index : int
            Index of the feature used for the split.
        threshold : float
            Threshold value for the split.
        left : Node
            Left child node.
        right : Node
            Right child node.
        value : float
            Mean value of target when node is a leaf.
        """

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only used for leaf nodes

    def is_leaf(self):
        """Return True if the node is a leaf node."""
        return self.value is not None
