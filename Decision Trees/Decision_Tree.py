# Decision Trees

import numpy as np
from collections import Counter

class Node:
    def __init__(self,
                 feature=None,
                 threshold=None,
                 left=None,
                 right=None,
                 value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeIG:

    def __init__(self,
                 max_depth=5):

        self.max_depth = max_depth
        self.root = None


    def entropy(self, y):

        counts = np.bincount(y)

        probs = counts / len(y)

        entropy = 0

        for p in probs:

            if p > 0:

                entropy -= p * np.log2(p)

        return entropy


    def information_gain(self,
                         parent,
                         left,
                         right):

        if len(left) == 0 or len(right) == 0:

            return 0

        parent_entropy = self.entropy(parent)

        n = len(parent)

        child_entropy = (
            (len(left)/n) * self.entropy(left)
            +
            (len(right)/n) * self.entropy(right)
        )

        return parent_entropy - child_entropy


    def split(self,
              column,
              threshold):

        left = np.where(
            column <= threshold
        )[0]

        right = np.where(
            column > threshold
        )[0]

        return left, right


    def best_split(self,
                   X,
                   y):

        best_gain = -1

        best_feature = None

        best_threshold = None

        for feature in range(X.shape[1]):

            column = X[:, feature]

            thresholds = np.unique(column)

            for threshold in thresholds:

                left, right = self.split(
                    column,
                    threshold
                )

                gain = self.information_gain(
                    y,
                    y[left],
                    y[right]
                )

                if gain > best_gain:

                    best_gain = gain

                    best_feature = feature

                    best_threshold = threshold

        return best_feature, best_threshold


    def leaf(self, y):

        return Counter(y).most_common(1)[0][0]


    def build(self,
              X,
              y,
              depth=0):

        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
        ):

            return Node(
                value=self.leaf(y)
            )

        feature, threshold = self.best_split(
            X,
            y
        )

        left_idx, right_idx = self.split(
            X[:, feature],
            threshold
        )

        left = self.build(
            X[left_idx],
            y[left_idx],
            depth+1
        )

        right = self.build(
            X[right_idx],
            y[right_idx],
            depth+1
        )

        return Node(
            feature,
            threshold,
            left,
            right
        )


    def fit(self,
            X,
            y):

        self.root = self.build(
            X,
            y
        )


    def traverse(self,
                 x,
                 node):

        if node.value is not None:

            return node.value

        if x[node.feature] <= node.threshold:

            return self.traverse(
                x,
                node.left
            )

        return self.traverse(
            x,
            node.right
        )


    def predict(self,
                X):

        return np.array([
            self.traverse(
                x,
                self.root
            )
            for x in X
        ])

X = np.array([[1],[2],[3],[4],[5]])

y = np.array([0,0,1,1,1])

tree = DecisionTreeIG()

tree.fit(X,y)

print(tree.predict(X))
