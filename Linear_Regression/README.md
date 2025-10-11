title: "🔹 Linear Regression — From Scratch"
description: >
  Linear Regression is a supervised learning algorithm used to predict a continuous target variable `y` based on one or more input features `x`.
  It assumes a linear relationship between input and output.

overview:
  hypothesis_function: |
    ẏ = wX + b
    - w: weight (slope)
    - b: bias (intercept)
    - X: input features
    - ẏ: predicted output

parameters:
  - name: w
    meaning: Determines slope of regression line
    update_rule: "w = w - α ∂J/∂w"
  - name: b
    meaning: Shifts line up or down
    update_rule: "b = b - α ∂J/∂b"

cost_function:
  formula: "J(w,b) = (1 / 2m) * Σ(ẏᵢ - yᵢ)²"
  explanation: >
    Measures the average squared difference between predicted and actual values.
    Minimizing this function gives the best-fit line.

gradient_descent:
  partial_derivatives: |
    ∂J/∂w = (1/m) Σ(ẏᵢ - yᵢ) Xᵢ
    ∂J/∂b = (1/m) Σ(ẏᵢ - yᵢ)
  update_equations: |
    w := w - α * ∂J/∂w
    b := b - α * ∂J/∂b
  learning_rate: α (controls step size)

implementation:
  language: "Python 3.x"
  libraries: ["NumPy", "Matplotlib"]
  code_example: |
    import numpy as np
    import matplotlib.pyplot as plt

    class LinearRegression:
        def __init__(self, lr=0.01, epochs=1000):
            self.lr = lr
            self.epochs = epochs
            self.w = 0
            self.b = 0

        def fit(self, X, y):
            n = len(y)
            for _ in range(self.epochs):
                y_pred = self.w * X + self.b
                dw = -(2/n) * np.sum(X * (y - y_pred))
                db = -(2/n) * np.sum(y - y_pred)
                self.w -= self.lr * dw
                self.b -= self.lr * db

        def predict(self, X):
            return self.w * X + self.b

    if __name__ == "__main__":
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=80, n_features=1, noise=10, random_state=42)
        X, y = X.flatten(), y.flatten()

        model = LinearRegression(lr=0.01, epochs=1000)
        model.fit(X, y)
        y_pred = model.predict(X)

        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, y_pred, color='red', label='Regression Line')
        plt.legend()
        plt.show()

evaluation:
  metric: "Mean Squared Error (MSE)"
  formula: "MSE = (1/m) Σ(ẏᵢ - yᵢ)²"
  meaning: >
    Represents the average of the squares of the errors between predicted and actual values.

key_learnings:
  - Learned how gradient descent optimizes weights.
  - Understood the difference between fit and predict.
  - Implemented MSE manually as a loss function.

references:
  - "Andrew Ng - Machine Learning (Coursera)"
  - "Aurelien Geron - Hands-On Machine Learning with Scikit-Learn & TensorFlow"
  - "Wikipedia: https://en.wikipedia.org/wiki/Linear_regression"

author:
  name: "Mohamed Elberry"
  location: "Cairo, Egypt"
  interests: ["AI", "ML", "Embedded Systems"]
  links:
    linkedin: "https://linkedin.com/in/mohamed-elberry"
    github: "https://github.com/<your-username>"

license:
  type: "MIT License"
  year: 2025
  holder: "Mohamed Elberry"
