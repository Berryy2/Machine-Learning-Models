import numpy as np
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
We define a class to encapsulate everything related to the model:
1) Its parameters (weights, bias).
2) Its training process (fit).
3) Its prediction function (predict).
This structure lets us reuse the model like any other ML library model.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class LinearRegression:
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    This runs __init__, storing:
    1) Learning_rate: how big each gradient descent step is.
    2) n_Iterations: how many times we update our parameters.
    -> Initially, we don’t know our model’s parameters (weights & bias), so we set them to None.
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __init__(self, Learning_rate = 0.001, n_Iterations = 1000):
        self.Learning_rate = Learning_rate
        self.n_Iterations = n_Iterations
        # Function's weights -> the Line's inclination.
        self.parameters = None
        # Function's bias -> the starting point of the Line drawn.
        self.bias = None
    # Implementing Gradient descent training function.
    # Where the goal is to minimize the cost function which is "mean square error"
    def fit(self, X, y):
        # Init parameters -> Having 100 samples, each with 1 feature X.shape (100,1)
        n_samples, n_features = X.shape
        # initialize weight by zero or can be small random values.
        self.parameters = np.zeros(n_features)
        # initialize bias by zero.
        self.bias = 0
        # y^(i) = X^(i)⋅θ + b -> for all training samples. (i) is i th number of training example.
        for _ in range(self.n_Iterations):
            y_predicted = np.dot(X, self.parameters) + self.bias
            # These are the partial derivatives of the cost function (Mean Squared Error) w.r.t. θ and b.
            # X.T --> X transpose.
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # Derivative of the bias.
            db = (1/n_samples) * np.sum(y_predicted - y)
            # Gradient descent function : parameter = old parameter - Learn.ratee * Partial derv. of parameters in the cost function
            # We subtract because we want to move opposite to the gradient direction (to minimize the cost).
            self.parameters -= self.Learning_rate * dw
            self.bias -= self.Learning_rate * db
    # Output function
    # Once training is done, we can compute predictions on new data using the final θ and b values.
    def predict(self, X):
        y_predicted = np.dot(X, self.parameters) + self.bias
        return y_predicted

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    In the Linear: Regression implementation file.
    Function	Purpose
   __init__()	Set hyperparameters and placeholders.
     fit()	    Train using gradient descent updating to get the best parameters and bias.
   predict()	Output predictions using trained weights.
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''