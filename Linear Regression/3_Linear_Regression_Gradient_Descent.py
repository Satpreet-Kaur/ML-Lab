# Linear Regression using Gradient Descent

import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr                  # learning rate
        self.iterations = iterations  # number of steps
        self.weights = None
        self.bias = None

    # Training function
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Predicted values
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # Prediction function
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X = np.array([[1],
              [2],
              [3],
              [4],
              [5]])

y = np.array([2, 4, 6, 8, 10])

model = LinearRegression(lr=0.01, iterations=1000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("Predictions:", predictions)
print("Weights:", model.weights)
print("Bias:", model.bias)
