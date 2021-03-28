# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class CoordinateLasso:

    def __init__(self, reg_lambda=1e-6):
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = None
    
    def train(self, X, y, tolerance=1e-4, max_itr=1500):
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
        if self.w is None or self.b is None:
            return None
        return X.dot(self.w) + self.b