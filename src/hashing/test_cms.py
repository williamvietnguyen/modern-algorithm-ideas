# William Nguyen
# email: williamvnguyen2@gmail.com

from cms import CountMinSketch
import numpy as np

if __name__ == '__main__':
    d = {} # true frequencies
    reg_cms = CountMinSketch(is_conservative=False)
    conserv_cms = CountMinSketch(is_conservative=True)
    u = 1000 # highest value in our universe for data stream
    max_freq = 100
    rng = np.random.default_rng()
    for i in range(u): # our data stream
        k = rng.integers(0, max_freq + 1)
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
    print('Regular average difference {}'.format(reg_total_diffs/total_stream))
    print('Conservative undercounted: {} times'.format(conserv_undercount))
    print('Conservative average difference: {}'.format(conserv_total_diffs/total_stream))