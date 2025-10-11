# 📈 Linear Regression — From Scratch (Gradient Descent Implementation)

This folder contains a complete implementation of **Linear Regression from scratch** using only **NumPy**.  
No `sklearn.LinearRegression` is used — the algorithm is implemented manually using **Gradient Descent**.

---

## 🎯 Objective

Linear Regression aims to find the **best-fitting line** that predicts a continuous target variable `y` from an input feature `x`,  
by minimizing the difference between **predicted** and **actual** values.

---

## 🧠 Theoretical Overview

## 🔹 Hypothesis Function

For a single feature:

> **ŷ = w · x + b**

For multiple features:

> **ŷ = X · θ + b**


Where

| Symbol | Meaning |
|:-------|:---------|
| `X` | Input features matrix of shape (m, n) |
| `θ` | Weight vector (parameters) of shape (n, 1) |
| `b` | Bias term (scalar) |
| `m` | Number of training examples |
| `n` | Number of features |

---

### 🔹 Cost Function — Mean Squared Error (MSE)

The cost function measures how well the model fits the data:

> **J(θ, b) = (1 / 2m) Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²**


Our goal is to minimize **J(θ, b)**.

---

### 🔹 Gradient Descent Update Rules

To minimize the cost function, parameters are updated iteratively as:

> **θ := θ − α × (1/m) × Xᵀ (ŷ − y)**  
> **b := b − α × (1/m) × Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)**


Where:

| Symbol | Meaning |
|:-------|:---------|
| `α` | Learning rate (controls the step size of each update) |
| `ŷ` | Model predictions = Xθ + b |

---

## ⚙️ Implementation Details

### 🔸 `Linear_Regression.py`

Defines the `LinearRegression` class:

**Attributes**
- `learning_rate` → step size for gradient updates  
- `n_iterations` → number of training iterations  
- `parameters` → model weights  
- `bias` → intercept term  

**Methods**
- `fit(X, y)` → trains the model using gradient descent  
- `predict(X)` → computes predictions for input data  

---

### 🔸 `Linear_Regression_test.py`

Demonstrates how to use the model:
1. Generates a random regression dataset using `sklearn.datasets.make_regression`
2. Splits data into training and testing sets
3. Trains the model and evaluates it using Mean Squared Error
4. Visualizes predictions using Matplotlib

---

## 📊 Visualization

After training, the code plots:
- 🟦 Actual data points  
- ⚫ Predicted regression line  

Run the test file to visualize:
```bash
python Linear_Regression_test.py
Example output:
A line fitting the data points showing the learned relationship.
```

🔍 Example Parameters
Parameter	Example Value	Description
learning_rate	0.01	Controls convergence speed
n_iterations	1000	Number of gradient descent steps
θ	[w₁, w₂, …, wₙ]	Feature weights
b	scalar	Bias/intercept term

🧾 References

Andrew Ng — Machine Learning Course (Coursera)

Scikit-learn documentation: LinearRegression

“Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron

Additional explanations adapted from Wikipedia - Linear Regression

Author: Mohamed Elberry

💡 Passionate about understanding ML from first principles



