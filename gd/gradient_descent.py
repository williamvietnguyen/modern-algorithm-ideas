# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class GradientDescent:
    """
    Gradient descent is a first-order iterative optimization algorithm for
    finding local minimum of a differentiable function. The idea is to take
    repeated steps in the opposite direction of the gradient at the current
    point, as this is the direction of steepest descent. Here we use mean
    squared error for objective to minimize, and L2 norm for regularization.
    - train(X, y, step_size, tolerance, max_itr): trains our model with X and y
    - predict(X): predicts output for X
    """

    def __init__(self, reg_lambda = 1e-6):
        """
        Constructor for gradient descent.
        - self.reg_lambda: The regularization parameter
        - self.w = The weights
        - self.b = The bias/intercept/offset
        """
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = None

    def mse(self, X, y):
        """
        Returns the objective mean squared error of X, y, self.w, and self.b.
        :param X: input data
        :param y: output data
        :return: Returns the objective evaluated with self.w and self.b.
        """
        n = X.shape[0]
        return (1/n) * np.sum((X.dot(self.w) + self.b - y)**2) + self.reg_lambda * np.sum(self.w**2)

    def train(self, X, y, step_size = 1e-4, tolerance=1e-6, max_itr = 10000):
        """
        Trains the model by performing gradient descent.
        :param X: input data
        :param y: output data
        :param step_size: learning rate
        :param tolerance: the margin in which we considered convergence
        """
        n, d = X.shape
        self.w = np.random.normal(0, 1, size=(d, 1))
        self.b = 0 # bias term
        prev_mse = float('inf')
        cur_mse = self.mse(X, y)
        # iterate until convergence
        while np.abs(cur_mse - prev_mse) > tolerance:
            gradient_w = (2/n) * X.T.dot(X.dot(self.w) + self.b - y) + 2 * self.reg_lambda * np.sum(self.w)
            # note we do not regularize the bias term, never do it!
            gradient_b = (2/n) * (X.dot(self.w) - self.b - y)
            self.w = self.w - step_size * gradient_w
            self.b = self.b - step_size * gradient_b
            prev_mse = cur_mse
            cur_mse = self.mse(X, y)

    def predict(self, X):
        """
        Predicts the output for X. Note we
        must run self.train before predicting.
        :param X: input data
        :return: prediction output for X, or None if not trained yet
        """
        if self.w is None or self.b is None:
            return None
        return X.dot(self.w) + self.b
