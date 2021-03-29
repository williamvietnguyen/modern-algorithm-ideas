# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class PCA:
    """
    Principal component analysis (PCA) is an unsupervised linear
    technique for dimensionality reduction on a data set. Principal 
    components are a set of vectors that form an orthonormal basis, 
    ordered by their ability to maximize the variance of the 
    projected data. PCA is the computation of these components 
    and using them to perform a change of basis on the data.
    - fit(X): fits the model on the data matrix X
    - transform(X, target_d): transforms X to lower dimension target_d
    """
    
    def __init__(self):
        """
        Constructor for Principal Component Analysis.
        - self.covariance: covariance matrix of data fitted, (d, d)
        - self.eigenvectors: sorted eigenvectors (columns), descending
        - self.eigenvalues: sorted eigenvalues, descending
        - self.mean: mean of data fitted, shape: (d,)
        """
        self.covariance = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = 0

    def fit(self, X):
        """
        Fits the model to X by computing the
        eigenvectors of its covariance matrix.
        :param X: the data matrix, shape: (n, d)
        """
        self.mean = np.mean(X, axis=0)
        # demean
        X_demean = X - self.mean
        # compute covariance matrix
        self.covariance = np.cov(X_demean.T)
        # compute eigenvectors and eigenvalues of covariance
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
        # sort eigenvectors by their eigenvalues in descending order for pc's
        indices = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[indices]
        self.eigenvectors = self.eigenvectors[:,indices]

    
    def transform(self, X, target_d):
        """
        Performs PCA dimensionality reduction X to a lower dimension.
        Does this by projecting X onto the top target_d eigenvectors.
        :param X: the data matrix, shape: (n, d)
        :param target_d: the lower dimension in which we reduce to
        :return: a matrix, shape: (n, target_d)
        """
        X_demean = X - self.mean
        # projection
        X_transformed = X_demean.dot(self.eigenvectors[:,0:target_d])
        return X_transformed

        