import numpy as np


def batch_dot(a, b):
    """Array of dot products over the fastest axis of two arrays.

    Assuming that a and b have K + 1 matching axes,
    batch_dot(a, b)[i1, ..., iK] = np.dot(a[i1, ..., iK, :], b[i1, ..., iK, :])

    """
    return np.sum(a * b, axis=-1)


def sq_distances(X, W):
    """Find all the distances between prototypes and data-points.

    The end-result is a 3-array of shape (height, width, n_samples)
    whose (i, j, n)-th entry represents sq_dist(X[n], W[i, j]), or 

    np.dot(X[n, :] - W[i, j, :], X[n, :] - W[i, j, :]).

    """
    # diff has shape (self.height, self.width, n_samples, n_features)
    diff = X - W[..., np.newaxis, :]
    return np.sum(diff ** 2, axis=-1)
