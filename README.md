# Modern Algorithm Ideas

Just some well known algorithm ideas and my implementations of them.

- [[Source]](cms/src/count_min_sketch.py) **Count-Min Sketch Data Structure**: A probabilistic data structure that serves as
  a frequency table of events in a stream of data.
  It uses hash functions to map events to frequencies,
  but unlike a hash table uses only sub-linear space, at
  the expense of overcounting some events due to collisions.

- **Gradient Descent Algorithms**: Gradient descent is a first-order iterative optimization algorithm for
  finding local minimum of a differentiable function. The idea is to take
  repeated steps in the opposite direction of the gradient at the current
  point, as this is the direction of steepest descent.
  - [[Source]](gd/src/gradient_descent.py) **Gradient Descent with Ridge Regression**: The ordinary gradient descent, but we have an L2 norm penalty to keep weights smaller.
  - [[Source]](gd/src/stochastic_gradient_descent.py) **Stochastic Gradient Descent with Ridge Regression**: Stochastic gradient descent has the same idea as gradient descent, but instead SGD will compute the gradient on a random subset of the data each iteration of its training. This is often advantageous as it is less computationally burdensome than computing the gradient on the entire data set. This implementation has an L2 norm penalty.
