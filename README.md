# Algorithms

## About

Just some well-known algorithms and my implementations of them. These are not necessarily the most computationally efficient or optimal algorithms. Rather they are intended to present the mechanisms and routines of the ideas in a hopefully digestible and transparent way.

## Table of Contents

- **Hashing**:

  - **Count-Min Sketch Data Structure** [[Code]](src/hashing/cms/count_min_sketch.py): A probabilistic data structure that serves as
    a frequency table of events in a stream of data, using sub-linear space at the expense of overcounting some events.

- **Local Minimum Optimization**:

  - **Coordinate Descent for Lasso Regression** [[Code]](src/min_opt/lasso/coord_lasso.py): Lasso regression performs variable selection and regularization via L1 norm. Coordinate descent is state of the art for this computation, updating each feature one at a time.
  - **Gradient Descent for Ridge Regression** [[Code]](src/min_opt/ridge/gd_ridge.py): Gradient descent for ridge regression is a first-order iterative optimization algorithm for finding local minimum of the ridge regression objective.
  - **Stochastic Gradient Descent for Ridge Regression** [[Code]](src/min_opt/ridge/sgd_ridge.py): Stochastic gradient descent has the same idea as gradient descent, but instead will compute the gradient on a random subset of the data each iteration.

- **Dimensionality Reduction**:
  - **Principal Component Analysis** [[Code]](src/dimensionality_reduction/pca/pca.py): Principal component analysis (PCA) is an unsupervised linear technique for dimensionality reduction on a data set. The principal component vectors are vectors that maximize the variance of the data after it is projected onto them.
