\# 🧮 Logistic Regression from Scratch (NumPy Implementation)



A clean and well-documented implementation of \*\*Logistic Regression\*\* using only \*\*NumPy\*\*, built entirely from scratch — without using `scikit-learn`’s built-in model.



This project demonstrates the core principles behind gradient-based optimization and binary classification using the \*\*sigmoid hypothesis function\*\*.



---



\## 📘 Project Overview



This repository contains a minimal and educational implementation of \*\*Logistic Regression\*\*, one of the most fundamental algorithms in Machine Learning.



The goal is to:

\- Understand logistic regression mathematically and programmatically.

\- Implement gradient descent manually.

\- Evaluate model accuracy on a real dataset (Breast Cancer dataset from `sklearn.datasets`).



---



\## 🧠 Algorithm Summary



Logistic Regression is a \*\*linear classifier\*\* that models the probability of class membership using the \*\*sigmoid function\*\*:



\\\[

h\_\\theta(x) = \\frac{1}{1 + e^{-(w^T x + b)}}

\\]



We optimize the parameters \\( w \\) and \\( b \\) by minimizing the \*\*logistic loss\*\* (binary cross-entropy) using \*\*gradient descent\*\*:



\\\[

J(w, b) = -\\frac{1}{m} \\sum\_{i=1}^{m} \[y^{(i)} \\log(h\_\\theta(x^{(i)})) + (1 - y^{(i)}) \\log(1 - h\_\\theta(x^{(i)}))]

\\]



---



\## ⚙️ Implementation Details



\### 🔹 `LogesticRegression` class

Defined in `Logestic\_Regression\_Model.py`:



\- `\_\_init\_\_(learning\_rate, n\_iterations)`: Initializes hyperparameters.  

\- `fit(X, y)`: Trains the model using gradient descent.  

\- `hypothesis(X)`: Predicts labels for new samples.  

\- `\_sigmoid(x)`: Applies the sigmoid activation function.



\### 🚀 Optimization Logic

Gradients are computed as:



\\\[

\\frac{\\partial J}{\\partial w} = \\frac{1}{m} X^T (h - y), \\quad

\\frac{\\partial J}{\\partial b} = \\frac{1}{m} \\sum (h - y)

\\]



Parameters are updated iteratively using:



\\\[

w := w - \\alpha \\cdot \\frac{\\partial J}{\\partial w}, \\quad

b := b - \\alpha \\cdot \\frac{\\partial J}{\\partial b}

\\]



---



\## 🧪 Testing \& Evaluation



Testing code is provided in `Logestic\_Regression\_Test.py`.  

The \*\*Breast Cancer dataset\*\* from scikit-learn is used to validate model accuracy.



Example:

```python

regressor = LogesticRegression(learning\_rate=0.001, n\_iterations=1000)

regressor.fit(x\_train, y\_train)

predictions = regressor.hypothesis(x\_test)

print("Logistic Regression accuracy:", accuracy(y\_test, predictions))

✅ Expected Accuracy: ~0.90–0.95 depending on learning rate and iterations.



🗂️ Repository Structure

bash

Copy code

machine-learning-models/

│

├── Logestic Regression/

│   ├── Logestic\_Regression\_Model.py    # Core model implementation

│   ├── Logestic\_Regression\_Test.py     # Dataset loading, training, and evaluation

│   └── README.md                       # You are here

│

└── ...

🚀 How to Run

Clone this repository:



bash

Copy code

git clone https://github.com/Berryy2/machine-learning-models.git

cd machine-learning-models/Logestic\\ Regression

Install dependencies:



bash

Copy code

pip install numpy scikit-learn matplotlib

Run the model:



bash

Copy code

python Logestic\_Regression\_Test.py

📊 Example Output

yaml

Copy code

Training Logistic Regression...

Model converged after 1000 iterations.

Logistic regression classification accuracy: 0.931578947368421

📈 Future Improvements

Add Newton’s Method optimization for faster convergence.



Support multiclass classification via One-vs-Rest strategy.



Add visualization of decision boundaries for 2D datasets.



Compare performance with sklearn.linear\_model.LogisticRegression.



💡 Credits

Developed by Mohamed Maged Elberry

🎓 Master’s Student — Machine Learning \& AI

