# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class LinearLeastSquaresClosedForm:
    """
    Closed form for regularized linear least squares.
    - fit(X, y): Fits our model to the training data
    - predict(X): Gives a prediction output for X
    """

    def __init__(self, reg_lambda = 1e-6):
        """
        Constructor for linear least squares closed form.
        - self.reg_lambda: regularlization parameter
        - self.w: weights
        """
        self.reg_lambda = reg_lambda
        self.w = None

    def fit(self, X, y):
        """
        Fits the model to the training data provided.
        :param X: input data
        :param y: output data
        """
        # append the ones column for bias/intercept term
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        n, d = X_.shape
        lambda_I = np.eye(d) * self.reg_lambda
        # 0 the bias index
        lambda_I[0, 0] = 0
        self.w = np.linalg.solve(X_.T.dot(X_) + lambda_I, X_.T.dot(y))

    def predict(self, X):
        """
        Returns prediction output for X.
        :param X: input data
        :return: prediction output for X
        """
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_.dot(self.w)