# Linear Regression Using Library
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1],
              [2],
              [3],
              [4],
              [5]])

y = np.array([2, 4, 6, 8, 10])

model = LinearRegression() # Create model

model.fit(X, y) # Train model

# Predictions
predictions = model.predict(X)

print("Predictions:", predictions)

# Model parameters
print("Weight (slope):", model.coef_)
print("Bias (intercept):", model.intercept_)
