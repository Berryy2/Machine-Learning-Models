# Imports NumPy for vectorized math.
import numpy as np

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Logistic Regression (Binary Classifier) — Implemented from Scratch using Gradient Descent.

    This class models the probability that a given input belongs to the positive class (y=1)
    using the logistic (sigmoid) function. Parameters are optimized via batch gradient descent.
    Equation:
        ŷ = σ(X·θ + b)
    where:
        ŷ → predicted probability of class 1
        X → input features matrix (m x n)
        θ → parameters/weights (n x 1)
        b → bias term (scalar)
        σ(z) → sigmoid activation = 1 / (1 + exp(-z))
    Author: Mohamed ElBerry
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Initialize the Logistic Regression model.
        Parameters:
        learning_rate : float, optional
            Step size for the gradient descent update.
        n_iterations : int, optional
            Number of training iterations.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Defining Model as class.
class LogesticRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.parameters = None
        self.bias = None


# ---------------------------- Training Phase ---------------------------- #
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Fit the Logistic Regression model using batch gradient descent.
        Theoretical  we should be aiming to maximize the likelihood function,
        but sklearn library's most implementations like to use the decent approach
        to minimize cost function rather than maximizing the likelihood and 
        both are the same only the sign is the difference (so both approaches are valid) 
        Parameters
        X : ndarray, shape (m, n)
        Training data, where m = number of samples, n = number of features.
        y : ndarray, shape (m,)
        Binary labels {0, 1} corresponding to each sample.
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters and bias
        self.parameters = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Loop.
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.parameters) + self.bias

            # Apply sigmoid activation to get predicted probabilities
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dp = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Apply sigmoid activation to get predicted probabilities
            self.parameters -= self.learning_rate * dp
            self.bias -= self.learning_rate * db

 # ---------------------------- Prediction Phase ---------------------------- #


        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        Fit the Logistic Regression model using batch gradient descent.
            Parameters
            X : ndarray, shape (m, n)
                Training data, where m = number of samples, n = number of features.
            y : ndarray, shape (m,)
                Binary labels {0, 1} corresponding to each sample.
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def hypothesis(self, X):
        linear_model = np.dot(X, self.parameters) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

 # ---------------------------- Helper Functions ---------------------------- #
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    '''
    Newton's equation method:
    θ:= θ − H^(−1) ∇J(θ)
    Where:
    H -> is the Hessian matrix (second derivatives of the cost function)
    ∇𝐽(𝜃) -> is the gradient

     n_samples, n_features = X.shape
        self.parameters = np.zeros((n_features, 1))
        self.bias = 0
        X = np.hstack((np.ones((n_samples, 1)), X))  # Add bias term
        theta = np.zeros((n_features + 1, 1))

        for i in range(self.n_iterations):
            z = X.dot(theta)
            y_pred = self._sigmoid(z)

            # Gradient
            g = X.T.dot(y_pred - y)

            # Hessian
            R = np.diag((y_pred * (1 - y_pred)).flatten())
            H = X.T.dot(R).dot(X)

            # Parameter update
            delta = np.linalg.inv(H).dot(g)
            theta -= delta

            # Stop if change is small
            if np.linalg.norm(delta) < self.tolerance:
                break

        self.bias = theta[0]
        self.parameters = theta[1:]
    '''