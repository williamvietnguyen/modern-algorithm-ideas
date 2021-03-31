# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class RidgeRegression:
    """
    Ridge regression model, intended for data that suffers from
    multicollinearity. Contains stochastic gradient descent as its 
    primary local minimization algorithm, but can opt out
    for the normal gradient descent.
    - fit(X, y, step_size, tolerance, max_itr): trains our model with X and y
    - predict(X): predicts output for X
    """

    def __init__(self, reg_lambda=1e-6, sgd=True):
        """
        Constructor for Ridge Regression.
        :param reg_lambda: regulalization parameter
        :param sgd: whether to use sgd or normal gradient descent
        """
        self.reg_lambda = reg_lambda
        self.sgd = sgd
        self.w = None # weights
        self.b = None # bias/intercept/offset

    def objective(self, X, y):
        """
        Returns the objective (MSE + regularization) of X, y, self.w, and self.b.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :return: Returns the objective evaluated with self.w and self.b.
        """
        n = X.shape[0]
        return np.mean((X.dot(self.w) + self.b - y)**2) + self.reg_lambda * np.sum(self.w**2)

    def compute_gradients(self, X, y):
        """
        Computes the gradient of the objective with respect to w and
        with respect to b, returning the computations.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :return: tuple of the gradient evaluations with respect to w, and to b
        """
        n = X.shape[0]
        y_prediction = X.dot(self.w) + self.b
        grad_w = (2/n) * X.T.dot(y_prediction - y) + 2 * self.reg_lambda * np.sum(self.w)
        grad_b = 2 * np.mean(y_prediction - y) # same as (2/n) * np.sum(y_prediction - y)
        return grad_w, grad_b

    def fit(self, X, y, step_size = 1e-3, tolerance=1e-5, max_itr=100000, batch_size=100):
        """
        Fits the model.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :param step_size: learning rate
        :param tolerance: the margin in which we considered convergence
        :param max_itr: maximum iterations
        :param batch_size: size of each batch to compute gradient on, only for sgd
        """
        n, d = X.shape
        self.w = np.random.normal(0, 1, size=(d, 1))
        self.b = 0 # bias term
        prev_obj = float('inf')
        cur_obj = self.objective(X, y)
        rng = np.random.default_rng()
        itr = 0
        # iterate until convergence
        while np.abs(cur_obj - prev_obj) > tolerance and itr < max_itr:
            indices = None
            if self.sgd: # take random subset of data for SGD
                # note that rng.choice on a=n, samples from np.arange(n)
                indices = rng.choice(a=n, size=batch_size, replace=False)
            else: # take gradient using whole data set
                indices = np.arange(n)
            grad_w, grad_b = self.compute_gradients(X[indices], y[indices])
            # update
            self.w = self.w - step_size * grad_w
            self.b = self.b - step_size * grad_b
            prev_obj = cur_obj
            cur_obj = self.objective(X, y)
            itr += 1

    def predict(self, X):
        """
        Predicts the output for X.
        :param X: input data, shape: (n, d)
        :return: prediction output for X
        """
        return X.dot(self.w) + self.b
