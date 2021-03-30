# Algorithms

Just some well-known algorithm ideas and my implementations of them.

- **Count-Min Sketch Data Structure** [[Code]](src/hashing/cms/count_min_sketch.py): A probabilistic data structure that serves as
  a frequency table of events in a stream of data.
  It uses hash functions to map events to frequencies,
  but unlike a hash table - it uses only sub-linear space at
  the expense of overcounting some events due to collisions.

- **Local Minimum Optimization Algorithms**:

  - **Gradient Descent for Ridge Regression** [[Code]](src/localmin/ridge/gd_ridge.py): Gradient descent is a first-order iterative optimization algorithm for finding local minimum of a differentiable or somewhat smooth function. The idea is to take repeated steps in the opposite direction of the gradient at the current point, as this is the direction of steepest descent.
  - **Stochastic Gradient Descent for Ridge Regression** [[Code]](src/localmin/ridge/sgd_ridge.py): Stochastic gradient descent has the same idea as gradient descent, but instead SGD will compute the gradient on a random subset of the data each iteration of its training. This is often advantageous as it is less computationally burdensome than computing the gradient on the entire data set.
  - **Coordinate Descent for Lasso Regression** [[Code]](src/localmin/lasso/coord_lasso.py): Lasso (Least Absolute Shrinkage and Selection Operator) regression performs both variable selection and regularization in order to enhance prediction accuracy and interpretability of the model. Coordinate descent is state of the art for training a lasso model. We update each feature one at a time, holding all others fixed.

- **Principal Component Analysis** [[Code]](src/dim/pca/pca.py): Principal component analysis (PCA) is an unsupervised linear technique for dimensionality reduction on a data set. The principal direction are the k orthonormal directions that maximize the variance of the data after projecting onto them. Then a principal component is the result of projecting the data onto a principal direction, giving us a lower-dimensional representation of the data while preserving as much of the data's variation as possible.
