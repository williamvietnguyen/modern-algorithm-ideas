import numpy as np
import hashlib
import random

class CountMinSketch:
    """
    The count–min sketch (CM sketch) 
    is a probabilistic data structure that serves as 
    a frequency table of events in a stream of data. 
    It uses hash functions to map events to frequencies, 
    but unlike a hash table uses only sub-linear space, at 
    the expense of overcounting some events due to collisions.
    - increment(x): to update the count of x
    - count(x): to get an approximate frequency of x in the stream
    - clear(): clears the CMS
    """

    def __init__(self, hashes=5, buckets=256, seed = 1, is_conservative = False):
        """
        Constructor for CMS
        - self.table: the table of counts, dimensions: self.hashes by self.buckets
        - self.hashes: the number of hash functions
        - self.buckets: the number of buckets
        - self.seed: a seed for our hash function
        - self.md5: md5 hash function
        - self.is_conservative: whether to use conservative increment
        """
        self.buckets = buckets
        self.hashes = hashes
        self.table = np.zeros((self.hashes, self.buckets))
        self.seed = seed
        self.md5 = hashlib.md5
        self.is_conservative = is_conservative

    def hash(self, x, i):
        """
        Returns the hash of x using hash function i.
        Note: our hash function i, is the ith byte of 
        the md5 hash of str(x) + str(self.seed).
        :param x: element to be hashed
        :param i: the hash function to be used
        :return: the hash of x using hash function i
        """
        concat = str(x) + str(self.seed)
        concat_bytes = self.md5(concat.encode('utf-8')).digest()
        return concat_bytes[i]

    def increment(self, x):
        """
        Increments the frequency for the element x.
        :param x: element to be incremented
        """
        if self.is_conservative:
            min_freq = self.count(x)
            for i in range(self.hashes):
                if self.table[i][self.hash(x, i)] == min_freq:
                    self.table[i][self.hash(x, i)] += 1
        else:
            for i in range(self.hashes):
                self.table[i][self.hash(x, i)] += 1
    
    def count(self, x):
        """
        Returns the approximate frequency of the element x.
        :param x: element in which we want a frequency count of
        :return: count of x
        """
        min_count = float('inf')
        for i in range(self.hashes):
            min_count = min(min_count, self.table[i][self.hash(x, i)])
        return min_count
    
    def clear(self):
        """
        Resets the CMS to default state.
        """
        self.table = np.zeros((self.hashes, self.buckets))


# sample driver
if __name__ == '__main__':
    d = {} # true frequencies
    reg_cms = CountMinSketch()
    conserv_cms = CountMinSketch(is_conservative=True)
    u = 1000 # highest value in our universe for data stream
    max_freq = 100
    for i in range(u): # our data stream
        k = random.randint(0, max_freq)
        for j in range(k):
            reg_cms.increment(i)
            conserv_cms.increment(i)
            d[i] = d.get(i, 0) + 1
    # compute undercounting and average difference from true frequency
    total_stream = 0.0
    reg_total_diffs = 0.0
    reg_undercount = 0
    conserv_total_diffs = 0.0
    conserv_undercount = 0
    for i in range(u):
        reg_diff = reg_cms.count(i) - d.get(i, 0)
        conserv_diff = conserv_cms.count(i) - d.get(i, 0)
        if reg_diff < 0:
            reg_undercount += 1
        if conserv_diff < 0:
            conserv_undercount += 1
        reg_total_diffs += reg_diff
        conserv_total_diffs += conserv_diff
        total_stream += 1
    print('Regular undercounted: {} times'.format(reg_undercount))
    print('Regular average overcount: {}'.format(reg_total_diffs/total_stream))
    print('Conservative undercounted: {} times'.format(conserv_undercount))
    print('Conservative average overcount: {}'.format(conserv_total_diffs/total_stream))


    
        

