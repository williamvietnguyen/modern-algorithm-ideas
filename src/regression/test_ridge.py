# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np
from ridge import RidgeRegression

def sse(y1, y2):
    return np.sum((y1 - y2)**2)

def test_sgd(X_train, y_train, X_test, y_test):
    stochastic_gradient_descent = RidgeRegression()
    stochastic_gradient_descent.fit(X_train, y_train)
    y_prediction = stochastic_gradient_descent.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for stochastic gradient descent: {}'.format(sse(y_test, y_prediction)))
    print("Sample test predictions: ", y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])

def test_gd(X_train, y_train, X_test, y_test):
    gradient_descent = RidgeRegression(sgd=False)
    gradient_descent.fit(X_train, y_train)
    y_prediction = gradient_descent.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for gradient descent: {}'.format(sse(y_test, y_prediction)))
    print("Sample test predictions: ", y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])


def test_lls_closed(X_train, y_train, X_test, y_test):
    closed_form = LinearLeastSquaresClosedForm()
    closed_form.fit(X_train, y_train)
    y_prediction = closed_form.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for linear least squares closed form: {}'.format(sse(y_test, y_prediction)))
    print('Sample test predictions: ', y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])

class LinearLeastSquaresClosedForm:
    """
    Closed form for regularized linear least squares.
    NOTE: Only used for benchmark purposes
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
        :param X: input data, shape: (n, d)
        :param y: output data, shape: (n, 1)
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
        :param X: input data, shape: (n, d)
        :return: prediction output for X
        """
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_.dot(self.w)

if __name__ == '__main__':
    n = 1200
    d = 120
    rng = np.random.default_rng()
    # true w
    w_true = rng.normal(0, 1, size=(d, 1))
    # training data
    X_train = rng.normal(0, 1, size=(n, d))
    y_train = rng.normal(0, 0.55, size=(n, 1)) + X_train.dot(w_true)
    # test data
    X_test = rng.normal(0, 1, size=(n, d))
    y_test = rng.normal(0, 0.55, size=(n, 1)) + X_test.dot(w_true)

    print('\nSample test values: ', y_test[0], y_test[1], y_test[2], y_test[-1])

    test_lls_closed(X_train, y_train, X_test, y_test)
    test_gd(X_train, y_train, X_test, y_test)
    test_sgd(X_train, y_train, X_test, y_test)
