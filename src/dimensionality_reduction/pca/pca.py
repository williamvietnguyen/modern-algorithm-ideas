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
    - fit_transform(X, target_d): Performs fit and transform
    """
    
    def __init__(self, svd=True):
        """
        Constructor for Principal Component Analysis.
        :param svd: whether to use SVD to compute PCA or not
        """
        self.svd = svd
        self.covariance = None # covariance matrix of data fitted, (d, d)
        self.eigenvectors = None # eigenvectors of the covariance matrix
        self.eigenvalues = None # eigenvalues of the covariance matrix
        self.mean = 0 # mean of data fitted, shape: (d,)
        self.Vh = None # unitary matrix from svd that is transposed, shape: (d, d)
        self.S = None # rectangular diagonal matrix from svd, shape: (d,)
        self.U = None # unitary matrix from svd, shape: (n, n)


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
        if self.svd: # svd is much faster
            self.fit(X)
            # sort the indices by the descending order of the singular values
            idx = self.S.argsort()[::-1][:target_d]
            # Notice that this is not necessarily redundant code! 
            # This is a different approach than svd in the transform function!
            S_ = np.diag(self.S[idx])[:target_d,:target_d]
            U_ = self.U[:,idx]
            X_transformed = U_.dot(S_)
        else: # terribly slow approach
            self.fit(X)
            X_transformed = self.transform(X, target_d)
        return X_transformed

        