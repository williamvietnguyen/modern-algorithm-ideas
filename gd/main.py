# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np
import closed_form as cf
import gradient_descent as gd
import stochastic_gradient_descent as sgd

def sse(y1, y2):
    return np.sum((y1 - y2)**2)

def test_sgd(X_train, y_train, X_test, y_test):
    stochastic_gradient_descent = sgd.StochasticGradientDescent()
    stochastic_gradient_descent.train(X_train, y_train)
    y_prediction = stochastic_gradient_descent.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for stochastic gradient descent: {}'.format(sse(y_test, y_prediction)))
    print("Sample test predictions: ", y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])

def test_gd(X_train, y_train, X_test, y_test):
    gradient_descent = gd.GradientDescent()
    gradient_descent.train(X_train, y_train)
    y_prediction = gradient_descent.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for gradient descent: {}'.format(sse(y_test, y_prediction)))
    print("Sample test predictions: ", y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])

def test_lls_closed(X_train, y_train, X_test, y_test):
    closed_form = cf.LinearLeastSquaresClosedForm()
    closed_form.fit(X_train, y_train)
    y_prediction = closed_form.predict(X_test)
    print('\n')
    print('Sum of squared errors on test for linear least squares closed form: {}'.format(sse(y_test, y_prediction)))
    print('Sample test predictions: ', y_prediction[0], y_prediction[1], y_prediction[2], y_prediction[-1])

if __name__ == '__main__':
    n = 1200
    d = 120
    # true w
    w_true = np.random.normal(0, 1, size=(d, 1))
    # training data
    X_train = np.random.normal(0, 1, size=(n, d))
    y_train = np.random.normal(0, 0.55, size=(n, 1)) + X_train.dot(w_true)
    # test data
    X_test = np.random.normal(0, 1, size=(n, d))
    y_test = np.random.normal(0, 0.55, size=(n, 1)) + X_test.dot(w_true)

    print('\nSample test values: ', y_test[0], y_test[1], y_test[2], y_test[-1])

    test_lls_closed(X_train, y_train, X_test, y_test)
    test_gd(X_train, y_train, X_test, y_test)
    test_sgd(X_train, y_train, X_test, y_test)
