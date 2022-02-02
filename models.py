from matplotlib import pyplot
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from .utils import *


class SelfOrganizingMap(BaseEstimator, ClusterMixin):
    """ Self Organizing Map model for data exploration and clustering.
    
    The class provides a rectangular-grid Self-Organizing Map with
    independently chosen side length (`height' and `width').
    
    The map prototypes may be initialized either at random locations or as
    a regular rectangular grid with size geometrically comparable to the
    dataset, aligned on the plane spanned by the two principal components
    of the dataset.

    The class makes use of the EM-like Batch Algorithm, which converges
    much faster than the original (1982) incremental-learning algorithm,
    without the necessity to tune and schedule learning rates.
    
    Normally, Self-Organizing Maps are used as an exploratory Data Science
    tool. For this purpose, I have added some visualization facilities. In
    particular, methods to generate numerically or visualize heat-map
    matrix representations of the learnt prototype distribution: the
    U-Matrix, which encodes the presence of clusters of prototypes, the
    P-Matrix, which encodes density estimates of the data around the
    prototypes, and the U*-Matrix, which refines the U-Matrix integrating
    P-Matrix information.

    The visualization tools may suggest the presence of data clusters to
    user's eye.  Clustering functionality has been incorporated to
    automate cluster discovery, although the use of DBSCAN as a proxy
    clustering algorithm is to be considered experimental, and may be
    changed in the future.

    Parameters
    ----------

    height : int, default=30
        Number of rows of map units (`neurons')

    width : int, default=30
        Number of columns of map units (`neurons')

    neigh_width : float, default=16.0
        The width parameter of the neighborhood function (the variance if
        the neighborhood function is Gaussian)

    neighborhood_type : (unused)
        NOT IMPLEMENTED: changes the neighborhood function in the unit-space

    init : {'random', 'pca'}, callable or ndarray of shape \
            (height, width, n_features), default='pca'
        If one of the listed str, use the respective built-in
        prototype initialization strategy; If a function is passed, it
        should have the design matrix and a random seed as arguments
        and return an initial prototype matrix. If an ndarray is
        passed, it is taken (destructively) as the initial prototype
        matrix

    Attributes
    ----------

    W_ : ndarray of shape (height, width, n_features)
        W_[i, j, :] is the prototype in data-space corresponding to the
        SOM unit (i, j)

    avg_distortion_ : float
        The mean value of the squared distances between each datapoint and
        the prototype of its best-matching unit (i.e., the closest
        prototype to the datapoint)
    
    n_iter_ : int
        The number of iterations performed by the Batch Algorithm

    n_features_ : int
        The number of features in the data (X.shape[1], where X is the
        design matrix)

    W_cluster_labels_ : ndarray of shape (height, width)
        The labels assigned to the SOM units after clustering
        (calling fit_predict)

    labels_ : ndarray of shape (n_samples,)
        The labels assigned to each datapoint after clustering 
        (calling fit_predict)
    """
   
    def __init__(
            self,
            height=30,
            width=30,
            neigh_width=16.0,
            neighborhood_type='Gaussian',
            init='pca',
            seed=None
    ):

        super(SelfOrganizingMap, self).__init__()
        
        self.height = height
        self.width = width
        self.neigh_width = neigh_width        
        self.n_weights = height * width
        self.neighborhood_type = neighborhood_type
        self.init = init
        self.seed = seed

        # Mesh-grid components reused at each update step
        ii, jj = np.ogrid[:height, :width]
        self.ii = ii[..., np.newaxis]
        self.jj = jj[..., np.newaxis]
        
        self.W_ = None
        self.avg_distortion_ = None
        self.n_iter_ = None
        self.n_features_ = None
        self.W_cluster_labels_ = None
        self.labels_ = None
        

    def W_initialize(self, X):
        """ Apply the required initialization, as specified in the constructor.
        """
        
        if self.init == 'random':
            self._W_random_normal_initialize(X)
            
        elif self.init == 'pca':
            self._W_PCA_plane_initialize(X)
            
        elif callable(self.init):
            self.W_ = self.init(X, self.seed)

        elif isinstance(self.init, np.ndarray):
            self.W_ = self.init
            
        else:
            raise ValueError('Unsupported init value or type')

        
    def fit_predict(self, X, y=None, tol=5e-3, max_iter=None, eps=0.5, min_samples=5):
        """ Train the Self-Organizing Map and clusters the dataset
        entries.

        After performing the SOM Batch Algorithm, the final prototypes are
        clustered as proxy for the datapoints, using DBSCAN.  The
        clustering for the datapoints themselves is performed under a
        nearest-prototype criterion -- the same that the best-matching
        unit for a datapoint is chosen.

        Parameters
        ----------

        X : an ndarray of shape (n_samples, n_features)
            The unlabelled input data
        
        y : default=None
            label variable, to satisfy the `fit-predict' API (ignored)
        
        tol : default=5e-3
            tolerance for the average distortion relative error
            (stopping criterion, fitting part)
        
        max_iter : int, default=None
            max number of iterations (stopping criterion, fitting part)
        
        eps: float, default=1.0
            search radius (for DBSCAN)

        min_samples: int, default=5
            minimum number of points within the search radius, to be
            considered part of a cluster (for DBSCAN)

        Returns
        -------

        self.labels_ : an ndarray of shape (n_samples,)
            The cluster labels for the datapoins (-1 for unassigned ones)
        """

        # Fitting

        self.fit(X, tol, max_iter)

        # Clustering

        n_samples, n_features = X.shape
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

        W_cluster_labs = clusterer.fit_predict(self.W_.reshape((-1, n_features)))
        W_cluster_labs = W_cluster_labs.reshape(self.W_.shape[:-1])

        self.W_cluster_labels_ = W_cluster_labs
        
        sq_dists = sq_distances(X, self.W_).reshape((-1, n_samples))
        winning_units = np.argmin(sq_dists, axis=0)
        i_win, j_win = np.unravel_index(winning_units, shape=(self.W_.shape[:-1]))

        self.labels_ = W_cluster_labs[i_win, j_win]
        
        return self.labels_
    
        
    def fit(self, X, tol=5e-3, max_iter=None):
        """ Train the Self-Organizing Map with the SOM Batch Algorithm.

        Parameters
        ----------

        X : an ndarray of shape (n_samples, n_features)
            The unlabelled input data

        tol : default=5e-3
            tolerance for the average distortion relative error
            (stopping criterion)
        
        max_iter : int, default=None
            max number of iterations (stopping criterion)

        Returns
        -------

        self : object
            The fitted SelfOrganizingMap instance
        """

        self.W_initialize(X)
        self._update_avg_distortion(X)

        counter = 0

        while True:

            # Terminate based on max iterations reached or average-distortion stationarity
            if max_iter and counter >= max_iter:
                break
            
            old_distortion = self.avg_distortion_
            self._update_W_smooth(X)
            self._update_avg_distortion(X)

            counter += 1

            if tol and np.fabs((self.avg_distortion_ - old_distortion) / old_distortion) < tol:
                break

        self.n_iter_ = counter
        self.n_features = X.shape[1]
        
        return self


    def _W_random_normal_initialize(self, X):
        """ Initialize the prototype cloud at random. 

        The prototypes are generated with independent components,
        each following a normal distribution scaled on the input data
        along each feature.
        """        

        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        # Set the means as mid-points between min and max across all features
        means = 0.5 * np.fabs(max_vals + min_vals)
        
        # Set the standard deviations as |max - min| / 2 for each feature (no covariances)        
        cov = np.diag(0.25 * (max_vals - min_vals) ** 2)
        
        gen = np.random.default_rng(self.seed)
        self.W_ = gen.multivariate_normal(means, cov, size=(self.height, self.width))
    
        
    def _W_PCA_plane_initialize(self, X):
        """ Initialize the prototypes as a rectangular mesh aligned
        to the first two principal components of the data.
        """
        
        pca = PCA() # We use sklearn for simplicity
        pca.fit(X)
        
        # Compute the half-lengths of the grid and the spanning unit-vectors
        stds = np.sqrt(pca.explained_variance_)
        new_basis = pca.components_

        # Sides of the lattice building block
        unit_y = 2 * stds[0] / (self.height - 1)
        unit_x = 2 * stds[1] / (self.width - 1)

        # Mesh construction
        self.W_ = ((self.ii * unit_y - stds[0]) * new_basis[:, 0] +
                   (self.jj * unit_x - stds[1]) * new_basis[:, 1])
        

    def _update_W_smooth(self, X):
        """ A single update step of the SOM Batch Algorithm.
        
        The SOM Batch Algorithm is analogous to the
        Expectation-Maximization (EM) Algorithm, minimizing the
        average distortion E[norm(x - w(x)) ** 2].  It cannot be,
        however, formulated exactly as EM, without substantial change
        to the BMU choice rule (see just below).

        In practice, the E-step is replaced by the following
        point-estimate:

            c(X[n, :]) = argmin( | X[n, :] - W[i, j, :] | ** 2
                                for (i, j) in units ),
      
        where c(X[n, :]) is the best-matching unit (BMU) corresponding
        to the nearest prototype to X[n, :].

        The M-step re-computes the locations of the prototypes
        W[i, j, :] as

                         sum( h(j, c(X[n, :])) * X[n, :] for n = 0, ..., n_samples )
            W[i, j, :] = -----------------------------------------------------------
                              sum( h(j, c(X[n, :])) for n = 0, ..., n_samples )

        h is the `neighborhood function' in unit space, which
        specifies the level of influence that units have among
        themselves (collaboration), irrespective of the prototype
        positions.
        
        In this case, h is a spherical Gaussian centered on the
        best-matching unit.
        
        TODO: Other choices of neighbourhood function are:
        
        Student-t pdf (scipy):  h = t(self.neigh_width).pdf(np.sqrt(output_sq_dist))
        Sharp indicator:        h = float(output_sq_dist < self.neigh_width)
        Cauchy pdf:             h = 1. / (1. + (output_sq_dist / self.neigh_width) ** 2)
        Gaussian difference:    h = a * np.exp(-0.5 * output_sq_dist / self.neigh_width_a) -
                                    b * np.exp(-0.5 * output_sq_dist / self.neigh_width_b)
        """

        n_samples, _ = X.shape

        # Approximate M-Step
        
        # Find the best-matching unit for each sample
        sq_dists = sq_distances(X, self.W_).reshape((-1, n_samples))
        winning_units = np.argmin(sq_dists, axis=0)
        i_win, j_win = np.unravel_index(winning_units, shape=(self.height, self.width))

        # Approximate E-Step
        
        # Compute all squared distances between best-matching units and all units
        # (the distances between adjacent units is conventionally 1)
        output_sq_dist = (self.ii - i_win) ** 2 + (self.jj - j_win) ** 2

        # Compute all values of the neighborhood function
        h = np.exp(-0.5 * output_sq_dist / self.neigh_width)

        # Compute the numerator of the prototype update
        weighted_sum_X = np.dot(h, X)

        # Compute the denominator
        weight_sum = np.sum(h, axis=-1)

        # Protection against division by zero
        try:
            W_new = np.divide(weighted_sum_X, weight_sum[..., np.newaxis])

        except FloatingPointError('Possible overflow or division by zero'):
            bad_indices = np.logical_or(np.isnan(W_new), np.isinf(W_new))
            W_new[bad_indices] = self.W_[bad_indices]

        self.W_ = W_new

    
    def _update_avg_distortion(self, X, rate=None):
        """ Compute a full-dataset or a stochastic expectation of the distortion,
        that is, the distance between a sample x and its reconstruction W[c(x), :].
        """

        n_samples, n_features = X.shape

        # If `rate' is provided, sample the dataset at random at such rate
        if rate:
            indices = np.random.randint(0, n_samples, size=int(rate * n_samples))
            X = X[indices, :]

        # Compute best-matching-unit indices for every input
        sq_dists = sq_distances(X, self.W_).reshape((-1, n_samples))
        winning_units = np.argmin(sq_dists, axis=0)

        # Compute prototype reconstructions for each input
        reconstructions = self.W_.reshape((-1, n_features))[winning_units, :]
        diff = X - reconstructions

        # Avg distortion as mean of squared distances of inputs and BMU prototypes
        reconstruction_errors = batch_dot(diff, diff)

        self.avg_distortion_ = np.mean(reconstruction_errors)


    def umatrix(self):
        """ Build a U-Matrix representation of the data, given the trained
        prototypes.

        Returns
        -------

        An ndarray of shape (self.height, self.width)
            The array holding the U-Matrix.
        """
        
        U = np.empty(shape=(self.height, self.width))

        # (quick and dirty code)
        for i in range(self.height):
            for j in range(self.width):

                # For units that are direct neighbors of the (i, j) one, compute the distances
                # of the correspoding prototypes from the one belonging to unit (i, j)
                # (notation: de = 'distance from Eastern unit', etc.)
                de = 0 if j == self.width - 1  else norm(self.W_[i, j, :] - self.W_[i, j + 1, :], ord=2)
                ds = 0 if i == self.height - 1 else norm(self.W_[i, j, :] - self.W_[i + 1, j, :], ord=2)
                dw = 0 if j == 0               else norm(self.W_[i, j, :] - self.W_[i, j - 1, :], ord=2)
                dn = 0 if i == 0               else norm(self.W_[i, j, :] - self.W_[i - 1, j, :], ord=2)

                # Number of neighbors of a unit: normally 4 but could be 3 or 2 if the unit is on the border
                n_neighs = (i != 0) + (j != 0) + (i != self.height) + (j != self.width)

                # Compute the map value at (i, j) as the average distance
                U[i, j] = (de + ds + dw + dn) / n_neighs

        return U


    def pmatrix(self, X):
        """ Create a P-Matrix, i.e., a visualization of the data density around
        each prototype.

        Returns
        -------

        An ndarray of shape (self.height, self.width)
            The array holding the P-Matrix.
        """

        # Compute the variances
        variances = np.var(X, axis=0)

        # The radius is 20% of the size; the size is taken as twice the variance
        sq_radius = 0.08 * np.max(variances)

        # Compute all prototype-datapoint squared distances
        sq_dists = sq_distances(X, self.W_)

        # Return the percentage of datapoints within radius (from each prototype)
        return np.mean(sq_dists <= sq_radius, axis=-1)


    def ustarmatrix(self, X):
        """ Create a U*-Matrix, i.e., a U-Matrix modulated according to data
        density (the P-Matrix).
        
        Returns
        -------

        An ndarray of shape (self.height, self.width)
            The array holding the U*-Matrix.
        """

        pmat = self.pmatrix(X)
        min_pmat = np.min(pmat)
        return self.umatrix() * (pmat - min_pmat) / (np.mean(pmat) - min_pmat)


    def plot_data_and_prototypes(self, X, draw_data=True, draw_prototypes=True, **kwargs):
        """ Plot the first three components of both the data and the SOM
        prototypes.

        The kwargs refer to the inner call to pyplot.figure. See the matplotlib
        documentation.  The user has to call pyplot.show() to draw with the
        configured matplotlib backend.
        """
        
        fig = pyplot.figure('Prototypes in the Data Space', **kwargs)
        ax = fig.add_subplot(111, projection='3d') # Idiom for 3D graphs

        # Draw the prototype mesh
        if draw_prototypes:
            ax.plot_wireframe(
                self.W_[..., 0],  self.W_[..., 1], self.W_[..., 2],
                linewidth=.3, color='k'
            )
            ax.scatter(
                self.W_[..., 0], self.W_[..., 1], self.W_[..., 2],
                'b.', s=25
            )

        # Draw the datapoints
        if draw_data:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='.', s=25)

        ax.set_axis_off()


    def __repr__(self):

        return '<Self-Organizing Map Estimator (height={}, width={}, {}) ID:0x{:x}>'.format(
            self.height,
            self.width,
            self.init,
            id(self)
        )
