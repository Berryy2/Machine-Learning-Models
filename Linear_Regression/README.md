# 📈 Linear Regression — From Scratch (Gradient Descent Implementation)

This folder contains a complete implementation of **Linear Regression from scratch** using only **NumPy**.  
No `sklearn.LinearRegression` is used — the algorithm is implemented manually using **Gradient Descent**.

---

## 🎯 Objective

Linear Regression aims to find the best-fitting line that predicts a continuous target variable `y` from an input feature `x`  
by minimizing the difference between predicted and actual values.

---

## 🧠 Theoretical Overview

### 🔹 Hypothesis Function

For a single feature:
\[
\hat{y} = w x + b
\]

For multiple features:
\[
\hat{y} = X \cdot \theta + b
\]

where:
- \( X \) → input features matrix of shape \((m, n)\)
- \( \theta \) → parameter (weight) vector of shape \((n, 1)\)
- \( b \) → bias term (scalar)
- \( m \) → number of training examples
- \( n \) → number of features

---

### 🔹 Cost Function — Mean Squared Error (MSE)

\[
J(\theta, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
\]

The goal is to minimize \( J(\theta, b) \).

---

### 🔹 Gradient Descent Update Rules

To minimize the cost, we iteratively update parameters:

\[
\theta := \theta - \alpha \frac{1}{m} X^T (\hat{y} - y)
\]
\[
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
\]

where:
- \( \alpha \) → learning rate (controls step size)
- \( \hat{y} = X\theta + b \)

---

## ⚙️ Implementation Details

### 🔸 `Linear_Regression.py`

Defines the `LinearRegression` class:
- **Attributes**
  - `Learning_rate`: step size for gradient updates
  - `n_Iterations`: number of training iterations
  - `parameters`: model weights
  - `bias`: intercept term
- **Methods**
  - `fit(X, y)`: trains model via gradient descent
  - `predict(X)`: computes predictions

### 🔸 `Linear_Regression_test.py`

1. Generates a dataset using `sklearn.datasets.make_regression`.
2. Splits it into training and testing sets.
3. Trains the model and computes MSE.
4. Visualizes results using Matplotlib.

---

## 📊 Visualization

After training, the code plots:
- Blue/yellow dots → actual data points
- Black line → predicted regression line

Example:

```bash
python Linear_Regression_test.py
