
# NumPy (np) → helps with mathematical operations and arrays (e.g., vectorized math).
import numpy as np
# train_test_split → splits your dataset into training and testing parts.
from sklearn.model_selection import train_test_split
# datasets → part of scikit-learn; lets you generate or load datasets easily.
from sklearn import datasets
# matplotlib.pyplot (plt) → used for plotting (to visualize our data).
import matplotlib.pyplot as plt
# n_samples -> number of data points.
# n_features -> number of input features (1 = simple linear regression) -> (x).
# noise -> adds randomness/noise to make it realistic.
# random_state ->  ensures results are the same each run.
x,y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=5)
# test_size = 0.2 -> 20% test data, 80% training data.
# random_state=1234 -> again for reproducibility.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# This makes a figure window of size 8×6 inches, so our plot looks larger and clearer.
# fig = plt.figure(figsize=(8, 6))
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
x → feature (horizontal axis)
y → target (vertical axis)
Blue dots (color="b")
Round markers (marker="o")
Size 30 (s=30)
Since we have one feature, x is a 2D array of shape (100, 1).
We use x[:, 0] to extract the first column (the actual feature values).
This gives us a plot of points that roughly form a straight line with some scatter — a visual of linear data.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# plt.scatter(x[:, 0], y, color = "b", marker="o", s = 30)
# This tells matplotlib to display the plot window.
# plt.show()
from Linear_Regression import LinearRegression

regressor = LinearRegression(Learning_rate= 0.01)

# Now gradient descent runs → updates θ & b until convergence.
regressor.fit(x_train, y_train)
regressor.fit(x_train, y_train)

# This uses your trained model to predict on unseen (test) data.
predictions = regressor.predict(x_test)

# Here we compute Mean Squared Error between actual and predicted test values.
def mse(y_true, y_predictions):
    return np.mean((y_true - y_predictions)**2)

# And computes Mean Squared Error between actual and predicted test values.
mse_value = mse(y_test, predictions)
print(mse_value)

# Visualize the Results.
y_pred_line = regressor.predict(x)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(x, y_pred_line, color='black', linewidth=2, label = "Prediction")
plt.title("Linear Regression")
plt.show()
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Here we show the the training samples is 80% of the data and the test sample is 20% of the data.
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
