# Modern Algorithm Ideas

Just some algorithm ideas and my implementations of them.

- [cms](cms): **Count-Min Sketch Data Structure**: A probabilistic data structure that serves as
  a frequency table of events in a stream of data.
  It uses hash functions to map events to frequencies,
  but unlike a hash table uses only sub-linear space, at
  the expense of overcounting some events due to collisions.
