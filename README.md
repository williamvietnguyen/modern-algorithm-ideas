# Modern Algorithm Ideas

Just some algorithm ideas and my implementations of them.

- [[File]](cms/count_min_sketch.py) **Count-Min Sketch Data Structure**: A probabilistic data structure that serves as
  a frequency table of events in a stream of data.
  It uses hash functions to map events to frequencies,
  but unlike a hash table uses only sub-linear space, at
  the expense of overcounting some events due to collisions.

- [[File]](gd/gradient_descent.py) **Gradient Descent**: Gradient descent is a first-order iterative optimization algorithm for
  finding local minimum of a differentiable function. The idea is to take
  repeated steps in the opposite direction of the gradient at the current
  point, as this is the direction of steepest descent. Here we use mean
  squared error for objective to minimize, and L2 norm for regularization.
