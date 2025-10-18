# 🧩 Logistic Regression — From Scratch (Using NumPy)

A complete implementation of **Logistic Regression** built entirely from scratch using **NumPy**, without relying on `sklearn.LogisticRegression`.  
This model is trained using **Gradient Descent** to minimize the **Binary Cross-Entropy Loss**.

---

## 🎯 Objective

Logistic Regression is a **supervised classification algorithm** used to predict a binary outcome (0 or 1).  
It models the **probability** that a given input `x` belongs to class `1` using the **sigmoid function**.

---

## 🧮 Theoretical Foundation

### 🔹 Hypothesis Function

For a dataset with `n` features:

\[
\hat{y} = \sigma(w^T x + b)
\]

where:

- \( w \) → weight vector of shape \((n, 1)\)  
- \( b \) → bias (scalar)  
- \( \sigma(z) = \frac{1}{1 + e^{-z}} \) → **sigmoid activation**  
- Output \( \hat{y} \in (0, 1) \) represents the predicted probability of class `1`

---

### 🔹 Decision Rule

\[
\hat{y}_{class} =
\begin{cases}
1, & \text{if } \hat{y} \geq 0.5 \\
0, & \text{if } \hat{y} < 0.5
\end{cases}
\]

---

### 🔹 Cost Function — Binary Cross-Entropy Loss

To measure prediction error:

\[
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
\]

where:
- \( m \) → number of training examples  
- \( y^{(i)} \in \{0, 1\} \) → true label  
- \( \hat{y}^{(i)} \) → predicted probability

---

### 🔹 Gradient Descent Optimization

To minimize the cost \( J(w, b) \), parameters are updated as follows:

\[
\begin{aligned}
w &:= w - \alpha \frac{1}{m} X^T (\hat{y} - y) \\
b &:= b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
\end{aligned}
\]

where \( \alpha \) is the learning rate.

---

## ⚙️ Implementation Overview

### `Logistic_Regression_Model.py`

Defines the class **`LogisticRegression`** with:
- `__init__`: initializes hyperparameters (learning rate, iterations)
- `fit(X, y)`: trains using gradient descent
- `predict(X)`: returns class predictions (0 or 1)
- `_sigmoid(x)`: computes the sigmoid activation

---

### `Logistic_Regression_Test.py`

1. Loads the **Breast Cancer dataset** from `sklearn.datasets`.  
2. Splits the dataset into training and testing sets.  
3. Trains the model and evaluates accuracy.  
4. Prints model performance.

Example Output:
Logistic Regression classification accuracy: 0.93



---

## 🧑‍💻 Example Usage


python Logistic_Regression_Test.py
📈 Visualization (Optional)
You can visualize the sigmoid function for intuition:

math
\sigma(z) = \frac{1}{1 + e^{-z}
 ```bash
python
Copy code
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 200)
sigma = 1 / (1 + np.exp(-z))

plt.plot(z, sigma, color="black")
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.grid(True)
plt.show()
🧠 Key Takeaways
✅ Implements Logistic Regression from first principles
✅ Uses Gradient Descent optimization
✅ Builds intuition about sigmoid and cost function
✅ Prepares you for extensions to Softmax and Multiclass Classification

📚 References
Andrew Ng, Machine Learning (Stanford CS229 Notes)

ISLR: An Introduction to Statistical Learning

Géron, Aurélien — Hands-On Machine Learning with Scikit-Learn and TensorFlow

👨‍💻 Author
Mohamed Elberry
📍 Cairo, Egypt
💼 Passionate about AI, Machine Learning & Embedded Systems
