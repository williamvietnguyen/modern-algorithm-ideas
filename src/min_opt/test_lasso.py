# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np
from lasso import CoordinateLasso

def sse(y1, y2):
    return np.sum((y1 - y2)**2)

def test_lasso(X_train, y_train, X_test, y_test):
    ratio = 1.5
    reg_lambda = 2 * np.max(X_train.T.dot(np.abs((y_train - np.mean(y_train)))))
    for i in range(20):
        lasso = CoordinateLasso(reg_lambda = reg_lambda)
        lasso.train(X_train, y_train)
        w = lasso.get_weights()
        y_prediction = lasso.predict(X_test)
        print('\n')
        print('Regularization parameter: {}'.format(reg_lambda))
        print('Sum of squared errors on test for lasso coordinate descent: {}'.format(sse(y_test, y_prediction)))
        print("Sample test predictions: ", y_prediction[0], y_prediction[1], y_prediction[-2], y_prediction[-1])
        print('Sample weight values: ', w[0], w[1], w[-2], w[-1])
        print('Non-zero weights: {}'.format(np.count_nonzero(w)))
        reg_lambda = reg_lambda/ratio


if __name__ == '__main__':
    n = 500
    d = 1000
    k = 100 # number of weights to be non-zero
    sigma = 1
    rng = np.random.default_rng()
    X_train = rng.normal(0, 1, size=(n, d))
    w = np.arange(1, d+1)/k
    w[k:] = 0 # d-k features will be 0 weight in true model
    w = np.reshape(w, (d, 1))
    y_train = X_train.dot(w) + rng.normal(0, 1, size=(n, 1))

    X_test = rng.normal(0, 1, size=(n,d))
    y_test = X_test.dot(w) + rng.normal(0, 1, size=(n, 1))

    print('\nSample test values: ', y_test[0], y_test[1], y_test[-2], y_test[-1])
    print('\nSample weight values: ', w[0], w[1], w[-2], w[-1])

    test_lasso(X_train, y_train, X_test, y_test)
