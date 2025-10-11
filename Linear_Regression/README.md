# ============================================================
# 📈 Linear Regression — From Scratch (Gradient Descent)
# ============================================================

description: |
  A complete implementation of Linear Regression from scratch using only NumPy.
  No `sklearn.LinearRegression` — all computations and optimization are manually implemented
  using the Gradient Descent algorithm to minimize the Mean Squared Error.

objective: |
  Linear Regression aims to find the best-fitting line that predicts a continuous target variable `y`
  from an input feature `x` by minimizing the difference between predicted and actual values.

# ------------------------------------------------------------
# 🧠 Theoretical Overview
# ------------------------------------------------------------

theory:
  hypothesis_function:
    single_feature: "ŷ = w * x + b"
    multiple_features: "ŷ = X · θ + b"
    explanation: |
      - X → Input feature matrix of shape (m, n)
      - θ → Weight vector (parameters) of shape (n, 1)
      - b → Bias term (scalar)
      - m → Number of training examples
      - n → Number of features

  cost_function:
    formula: "J(θ, b) = (1 / (2m)) * Σ(ŷᵢ - yᵢ)²"
    goal: "Minimize the cost function J(θ, b) — the mean squared error between predictions and true values."

  gradient_descent:
    update_rules: |
      θ := θ - α * (1/m) * Xᵀ(ŷ - y)
      b := b - α * (1/m) * Σ(ŷᵢ - yᵢ)
    definitions:
      α: "Learning rate (controls the step size)"
      ŷ: "Model predictions = Xθ + b"

# ------------------------------------------------------------
# ⚙️ Implementation Details
# ------------------------------------------------------------

implementation:
  Linear_Regression.py:
    purpose: "Defines the LinearRegression class."
    attributes:
      - learning_rate: "Step size for gradient updates"
      - n_iterations: "Number of gradient descent iterations"
      - parameters: "Model weights"
      - bias: "Intercept term"
    methods:
      - fit(X, y): "Trains the model using gradient descent."
      - predict(X): "Computes predictions for given input data."

  Linear_Regression_test.py:
    purpose: "Demonstrates training and testing of the custom Linear Regression model."
    steps:
      - "Generates a dataset using sklearn.datasets.make_regression."
      - "Splits data into training and testing sets."
      - "Trains the model and computes Mean Squared Error (MSE)."
      - "Plots actual vs predicted values using Matplotlib."

# ------------------------------------------------------------
# 📊 Visualization
# ------------------------------------------------------------

visualization:
  description: |
    After training, the model plots:
      - Blue points → Actual data samples
      - Black line → Predicted regression line
  run_command: "python Linear_Regression_test.py"

# ------------------------------------------------------------
# 🔍 Example Parameters
# ------------------------------------------------------------

parameters:
  - learning_rate: "0.01 — controls convergence speed"
  - n_iterations: "1000 — number of gradient descent steps"
  - θ: "[w₁, w₂, …, wₙ] — feature weights"
  - b: "scalar — bias/intercept term"

# ------------------------------------------------------------
# 🧾 References
# ------------------------------------------------------------

references:
  - "Andrew Ng — Machine Learning (Coursera)"
  - "Scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
  - "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron"
  - "Wikipedia — Linear Regression: https://en.wikipedia.org/wiki/Linear_regression"

# ------------------------------------------------------------
# 🌟 Author
# ------------------------------------------------------------

author:
  name: "Mohamed Elberry"
  location: "Cairo, Egypt"
  links:
    github: "https://github.com/mohamedelberry"
  passion: "Exploring AI, ML, and Embedded Systems from first principles."
