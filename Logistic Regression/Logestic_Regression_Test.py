# ============================================================
# This script tests the custom Logistic Regression model
# implemented from scratch using NumPy. The model is trained
# and evaluated on the Breast Cancer dataset from scikit-learn.
# ============================================================

# -------------------------------
# Import Required Libraries
# -------------------------------
import numpy as np
# Split dataset into train/test sets
from sklearn.model_selection import train_test_split
# Load pre-built datasets
from sklearn import datasets
# For future visualization (optional)
#from matplotlib import pyplot as plt


# Import the Logistic Regression model from our implementation
from Logestic_Regression_Model  import LogesticRegression

# Loading the dataset
# Breast Cancer dataset:
# A binary classification dataset with 30 features (e.g., cell size, texture, etc.)
bc = datasets.load_breast_cancer()
x,y = bc.data,bc.target

# Spliting dataset to 80% training, 20% testing.
# random_state ensures reproducibility of results.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


# Calculates classification accuracy.
# Accuracy = (Number of correct predictions / Total samples).
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
# Initializing and train the model:
# learning_rate controls the step size in gradient descent.
# n_iterations controls how many times the parameters are updated.
regressor = LogesticRegression(learning_rate=0.0001, n_iterations=1000)

# Train the model on the training data.
regressor.fit(x_train, y_train)

# make the prediction -> labels (0 or 1) "Binary classification" on the test data.
predictions = regressor.hypothesis(x_test)

# Compute accuracy on test set. (93%)
print("Logestic regression classification accuraccy",accuracy(y_test, predictions))


