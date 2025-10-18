# 🧠 Logistic Regression from Scratch

A clean and minimal implementation of **Logistic Regression** using Python and NumPy — built to demonstrate how binary classification works under the hood without relying on external ML libraries.

---

## 📘 Overview

Logistic Regression is a **linear model for classification**.  
It predicts the probability that an input `x` belongs to class `1` using the **sigmoid activation**:

`σ(z) = 1 / (1 + exp(-z))`  
where `z = wᵀx + b`

---

## ⚙️ Mathematical Formulation

### 1. **Hypothesis Function**
`ŷ = σ(wᵀx + b)`

### 2. **Cost Function (Binary Cross-Entropy)**
`J(w, b) = -(1/m) * Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]`

### 3. **Gradient Descent Updates**
- `dw = (1/m) * Xᵀ(ŷ - y)`
- `db = (1/m) * Σ(ŷ - y)`
- `w := w - α * dw`
- `b := b - α * db`

where `α` is the **learning rate**.

---

## 🧩 Implementation Steps

1. **Data Preprocessing**
   - Normalize input features `X`.
   - Split dataset into train and test sets.

2. **Model Initialization**
   - Initialize parameters `w` and `b` to zeros.

3. **Forward Propagation**
   - Compute `z = wᵀx + b`
   - Apply `σ(z)` to get predictions `ŷ`.

4. **Compute Cost**
   - Use the binary cross-entropy loss `J(w, b)`.

5. **Backward Propagation**
   - Calculate gradients `dw` and `db`.

6. **Parameter Update**
   - Apply gradient descent using learning rate `α`.

7. **Prediction**
   - Predict class `1` if `ŷ >= 0.5`, else class `0`.

---

## 📊 Example Results

After training on a sample dataset:

| Metric | Training | Testing |
|:-------|:----------|:--------|
| Accuracy | 96.2% | 94.8% |
| Cost (final) | 0.081 | 0.097 |

---

## 🧾 Files Structure

logistic_regression/
│
├── logistic_regression.py # Core implementation
├── dataset.csv # Example dataset
├── README.md # Project documentation
└── results.png # Training results visualization



---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/logistic_regression.git
cd logistic_regression

# Run the model
python logistic_regression.py
📈 Example Output

Training...
Iteration 1000 | Cost: 0.287
Iteration 2000 | Cost: 0.153
Training complete.

Train Accuracy: 96.2%
Test Accuracy: 94.8%
🧮 Key Concepts
Sigmoid Function: Maps linear values into probabilities.

Cost Function: Measures prediction error.

Gradient Descent: Optimizes weights to minimize cost.

💡 Future Enhancements
Add L2 regularization.

Extend to multiclass classification using one-vs-rest.

Implement a vectorized version for faster computation.

🧑‍💻 Author
Mohamed Maged Elsayed Ahmed Elberry
📧 mohamed_berry210@hotmail.com

📚 References
Andrew Ng — Machine Learning (Coursera)

DeepLearning.ai — Neural Networks and Deep Learning
