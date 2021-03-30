# William Nguyen
# email: williamvnguyen2@gmail.com

import numpy as np

class PCA:
    """
    Principal component analysis (PCA) is an unsupervised linear
    technique for dimensionality reduction on a data set. PCA computes
    the principal directions that maximize the variance of the projected data,
    and can compute the principal components by projecting the data onto them.
    This gives us a reduced dimension of the data, while preserving as much of 
    the variance as possible.
    - fit(X): fits the model on the data matrix X
    - transform(X, target_d): transforms X to lower dimension target_d
    """
    
    def __init__(self, svd=True):
        """
        Constructor for Principal Component Analysis.
        - self.covariance: covariance matrix of data fitted, (d, d)
        - self.eigenvectors: eigenvectors of the covariance matrix
        - self.eigenvalues: eigenvalues of the covariance matrix
        - self.mean: mean of data fitted, shape: (d,)
        - self.svd: whether to use singular value decomposition to compute PCA
        - self.U: unitary matrix from svd, shape: (n, n)
        - self.S: rectangular diagonal matrix from svd, shape: (d,)
        - self.Vh: unitary matrix from svd that is transposed, shape: (d, d)
        """
        self.covariance = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = 0
        self.svd = svd
        self.Vh = None
        self.S = None
        self.U = None


    def fit(self, X):
        """
        Fits the model to X by computing the
        SVD of X or by computing the eigenvectors of its
        covariance matrix.
        :param X: the data matrix, shape: (n, d)
        """
        self.mean = np.mean(X, axis=0)
        # demean
        X_demean = X - self.mean
        if self.svd: # note that SVD is way faster at computing PCA!
            # computing our singular value decomposition
            self.U, self.S, self.Vh = np.linalg.svd(X_demean, full_matrices=True)
        else: # non-SVD approach, awfully slow
            # compute covariance matrix
            self.covariance = np.cov(X_demean.T)
            # compute eigenvectors and eigenvalues of covariance
            self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)

    
    def transform(self, X, target_d):
        """
        Performs PCA dimensionality reduction X to a lower dimension.
        :param X: the data matrix, shape: (n, d)
        :param target_d: the lower dimension in which we reduce to
        :return: a matrix, shape: (n, target_d)
        """
        X_transformed = None
        X_demean = X - self.mean
        if self.svd:
            # find the indices of the singular values in descending order
            # also truncating to get the target dimension target_d
            idx = self.S.argsort()[::-1][:target_d]
            # sort the rows of Vh, then project X onto Vh.T
            X_transformed = X_demean.dot(self.Vh[idx].T)
        else:
            # sort eigenvectors by their eigenvalues in descending order for pc's
            idx = self.eigenvalues.argsort()[::-1]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:,idx]
            # projection
            X_transformed = X_demean.dot(self.eigenvectors[:,:target_d])
        return X_transformed
    

    def fit_transform(self, X, target_d):
        """
        Performs fit and transform in one routine.
        :param X: the data matrix, shape: (n, d)
        :param target_d: the lower dimension in which we reduce to
        :return: a matrix, shape: (n, target_d)
        """
        X_transformed = None
        self.mean = np.mean(X, axis=0)
        X_demean = X - self.mean
        if self.svd: # svd is much faster
            # computing our singular value decomposition
            self.U, self.S, self.Vh = np.linalg.svd(X_demean, full_matrices=True)
            # sort the indices by the descending order of the singular values
            idx = self.S.argsort()[::-1][:target_d]
            # Notice! This is a different approach than svd in transform function.
            S_ = np.diag(self.S[idx])[:target_d,:target_d]
            U_ = self.U[:,idx]
            X_transformed = U_.dot(S_)
        else: # terribly slow approach
            # compute covariance matrix
            self.covariance = np.cov(X_demean.T)
            # compute eigenvectors and eigenvalues of covariance
            self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
            idx = self.eigenvalues.argsort()[::-1]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:,idx]
            # projection
            X_transformed = X_demean.dot(self.eigenvectors[:,:target_d])
        return X_transformed

        