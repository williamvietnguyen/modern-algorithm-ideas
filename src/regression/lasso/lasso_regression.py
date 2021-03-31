# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class LassoRegression:
    """
    Lasso regression model, intended to estimate sparse models
    by driving feature coefficients to 0. 
    Uses coordinate descent to fit the data.
    - fit(X, y, tolerance, max_itr): trains the model with given data
    - predict(X): predicts output for X
    """

    def __init__(self, reg_lambda=1e-6):
        """
        Constructor for coordinate descent for lasso.
        :param reg_lambda: regularization parameter
        """
        self.reg_lambda = reg_lambda
        self.w = None # weights
        self.b = None # bias/intercept/offset
    
    def fit(self, X, y, tolerance=1e-4, max_itr=10000):
        """
        Fits the model by performing coordinate descent for lasso.
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
        :param tolerance: the margin in which we considered convergence
        :param max_itr: maximum iterations
        """
        n, d = X.shape
        itr = 0
        w = np.random.normal(0, 1, size=(d, 1))
        w_prev = np.full(shape=(d, 1), fill_value=float('inf'))
        prev_max_diff = float('inf')
        # normalizing constants for soft thresholding
        a = 2 * np.sum(X**2, axis = 0)
        while prev_max_diff > tolerance and itr < max_itr:
            self.b = np.mean(y - X.dot(w))
            for k in range(d):
                # use this to remove column k
                remove_k = np.arange(d) != k
                # compute partial residuals
                r = y - (self.b + X[:, remove_k].dot(w[remove_k]))
                # compute the coefficient of these residuals on kth predictor
                c_k = 2 * X[:,k].T.dot(r)
                # soft thresholding
                if c_k < -1 * self.reg_lambda:
                    w[k] = (c_k + self.reg_lambda)/a[k]
                elif c_k > self.reg_lambda:
                    w[k] = (c_k - self.reg_lambda)/a[k]
                else:
                    w[k] = 0
            prev_max_diff = np.max(np.abs(w_prev - w))
            w_prev = np.copy(w)
            itr += 1
        self.w = w
    
    def predict(self, X):
        """
        Predicts the output for X. 
        :param X: input data, shape: (n, d)
        :return: prediction output for X
        """
        return X.dot(self.w) + self.b