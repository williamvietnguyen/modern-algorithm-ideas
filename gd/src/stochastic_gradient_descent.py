# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class StochasticGradientDescent:
    """
    Stochastic gradient descent (SGD) is a first-order iterative optimization algorithm for
    finding local minimum of a differentiable or suitably smooth function. 
    The idea is the same as gradient descent, except we compute gradients on random 
    subsets of the data each iteration. This tends to be faster in high-dimensional
    optimization problems where the computational burden is high. Here we use L2
    norm for regularlization and mean squared error for the objective function.
    - train(X, y, step_size, tolerance, max_itr): trains our model with X and y
    - predict(X): predicts output for X
    """

    def __init__(self, reg_lambda = 1e-6):
        """
        Constructor for gradient descent.
        - self.reg_lambda: The regularization parameter
        - self.w = the weights, shape: (d, 1)
        - self.b = The bias/intercept/offset
        """
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = None

    def mse(self, X, y):
        """
        Returns the objective mean squared error of X, y, self.w, and self.b.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :return: Returns the objective evaluated with self.w and self.b.
        """
        n = X.shape[0]
        return (1/n) * np.sum((X.dot(self.w) + self.b - y)**2) + self.reg_lambda * np.sum(self.w**2)

    def compute_gradients(self, X, y):
        """
        Computes the gradient of the MSE with respect to w and
        with respect to b.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :return: tuple of the gradient evaluations with respect to w, and to b
        """
        n = X.shape[0]
        y_prediction = X.dot(self.w) + self.b
        grad_w = (2/n) * X.T.dot(y_prediction - y) + 2 * self.reg_lambda * np.sum(self.w)
        grad_b = (2/n) * np.sum(y_prediction - y)
        return grad_w, grad_b

    def train(self, X, y, step_size = 1e-4, tolerance=1e-6, max_itr=100000, batch_size=100):
        """
        Trains the model by performing stochastic gradient descent.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :param step_size: learning rate
        :param tolerance: the margin in which we considered convergence
        :param max_itr: maximum iterations
        :param batch_size: size of each batch to compute gradient on
        """
        n, d = X.shape
        self.w = np.random.normal(0, 1, size=(d, 1))
        self.b = 0 # bias term
        prev_mse = float('inf')
        cur_mse = self.mse(X, y)
        rng = np.random.default_rng()
        itr = 0
        # iterate until convergence
        while np.abs(cur_mse - prev_mse) > tolerance and itr < max_itr:
            # note that rng.choice on a=n, samples from np.arange(n)
            indices = rng.choice(a=n, size=batch_size, replace=False)
            grad_w, grad_b = self.compute_gradients(X[indices], y[indices])
            self.w = self.w - step_size * grad_w
            self.b = self.b - step_size * grad_b
            prev_mse = cur_mse
            cur_mse = self.mse(X, y)
            itr += 1

    def predict(self, X):
        """
        Predicts the output for X. Note we
        must run self.train before predicting.
        :param X: input data, shape: (n, d)
        :return: prediction output for X, or None if not trained yet
        """
        if self.w is None or self.b is None:
            return None
        return X.dot(self.w) + self.b
