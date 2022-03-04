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


def sq_distances1(X, W):

    height, width, _ = W.shape
    sq_dists = [[batch_dot(X - W[i, j], X - W[i, j]) for j in range(width)]
                for i in range(height)]
    return np.array(sq_dists)


def test_sq_distances():
    """ Tests whether the definition of sq_distances is correct,
    comparing it to a more transparent version.
    """
    
    gen = np.random.default_rng()

    X = gen.uniform(0, 1, size=(120 * 3,)).reshape((120, 3))
    W = gen.uniform(0, 1, size=(28 * 28 * 3,)).reshape((28, 28, 3))

    sq0 = sq_distances(X, W)
    sq1 = sq_distances1(X, W)
    
    if np.allclose(sq0, sq1, atol=1e-10):
        print("Test passed")
    else:
        print("Test not passed")
