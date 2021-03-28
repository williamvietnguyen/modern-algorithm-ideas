# Modern Algorithm Ideas

Just some well-known algorithm ideas and my implementations of them.

- [[Source]](cms/src/count_min_sketch.py) **Count-Min Sketch Data Structure**: A probabilistic data structure that serves as
  a frequency table of events in a stream of data.
  It uses hash functions to map events to frequencies,
  but unlike a hash table - it uses only sub-linear space at
  the expense of overcounting some events due to collisions.

- **Local Minimum Optimization Algorithms**:
  - [[Source]](minopt/src/gd_ridge.py) **Gradient Descent for Ridge Regression**: Gradient descent is a first-order iterative optimization algorithm for finding local minimum of a differentiable or somewhat smooth function. The idea is to take repeated steps in the opposite direction of the gradient at the current point, as this is the direction of steepest descent.
  - [[Source]](minopt/src/sgd_ridge.py) **Stochastic Gradient Descent for Ridge Regression**: Stochastic gradient descent has the same idea as gradient descent, but instead SGD will compute the gradient on a random subset of the data each iteration of its training. This is often advantageous as it is less computationally burdensome than computing the gradient on the entire data set.
  - [[Source]](minopt/src/coord_lasso.py) **Coordinate Descent for Lasso Regression**: Lasso (Least Absolute Shrinkage and Selection Operator) regression performs both variable selection and regularization in order to enhance prediction accuracy and interpretability of the model. Coordinate descent is state of the art for training a lasso model. We update each feature one at a time, holding all others fixed.
