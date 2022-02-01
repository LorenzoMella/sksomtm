import numpy as np
from os.path import sep, expanduser, normpath
from sys import byteorder
import urllib.request

def iris(dataset_path='data', filename='iris.dat'):
    """ Prepares the iris dataset in numpy format.

    The iris dataset is very amenable to classification tasks.
    However, it could be tricky with clustering tasks, as two of the
    three flower species aren't linearly separable.

    Parameters
    ----------
    dataset_path : str
        The directory path to the iris data-file

    filename : str, default='iris.dat'
        The name of the iris CSV file

    Returns
    -------

    X : ndarray (float64) and shape (<number of samples>, 4)
        The datapoints
    
    y : ndarray (int) of shape (<number of samples>,)
        The labels, with the following encoding:
            Iris-setosa:         0
            Iris-versicolor:     1
            Iris-virginica:      2
        
    
    name : str
        The string 'iris'
    """
    
    path = "{}{}{}".format(normpath(expanduser(dataset_path)), sep, filename)
    
    X = np.loadtxt(path, dtype=np.float64, delimiter=',',
                   skiprows=1, usecols=(0, 1, 2, 3))

    # Convert the labels (4th column) to numerical format
    conv = {4 : lambda s: (s == b'Iris-versicolor') + 2 * (s == b'Iris-virginica')}
    
    y = np.loadtxt(path, dtype=int, delimiter=',',
                   skiprows=1, usecols=4, converters=conv)

    # Standardize the values
    X = (X - X.mean()) / X.std()

    return X, y, 'iris'


def polygon_vertex_clusters(std=1.0):
    """ Generates 3D spherical Gaussian clusters of points at the
    vertices of a cube.

    The function generates eight spherical clouds of 20 points each, at
    the vertices of a cube. The dataset is meant as an easy clustering
    task for any algorithm, because the clusters are linear separable
    (if `std' isn't too large).

    Parameters
    ----------
    std : float, default=1.0
        The common standard deviation of the Gaussian components

    Returns
    -------

    X : ndarray (float64) and shape (160, 3)
        The datapoints
    
    y : ndarray (int) of shape (160,)
        The vertex labels (0 to 7)
    
    name : str
        The string 'polygon'
    """

    side = 10.2
    samples_per_cluster = 20
    num_clusters = 8    # there are 8 vertices in a cube
    max_samples = samples_per_cluster * num_clusters

    X = np.zeros(shape=(max_samples, 3))

    ii, jj, kk = np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')
    ii = ii.astype(np.float64) * side
    jj = jj.astype(np.float64) * side
    kk = kk.astype(np.float64) * side

    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                idx = 20 * (4 * i + 2 * j + k)
                X[idx:idx + 20, :] = (
                    np.array([ii[i, j, k], jj[i, j, k], kk[i, j, k]]) +
                    std * np.random.randn(samples_per_cluster, 3)
                )

    # Given the way the data is indexed, generate labels
    y = np.array([n // samples_per_cluster for n in range(max_samples)],
                 dtype=int)

    return X, y, 'polygon'


def linked_rings(std=1.0):
    """ Generates a 3D of point clouds forming two interlocking rings.

    The `interlocking-ring' dataset is supposed to be challenging for
    the most classical clustering algorithms (e.g., K-means). With
    careful tuning, local cluster search methods like DBSCAN should
    instead work well. The two rings don't touch (unless `std' is
    large enough) and are laid vertically and horizontally,
    respectively.

    Parameters
    ----------
    std : float, default=1.0
        The common standard deviation of the ring cross-section

    Returns
    -------

    X : ndarray (float64) and shape (160, 3)
        The datapoints
    
    y : ndarray (int) of shape (160,)
        The ring label (0 or 1)
    
    name : str
        The string 'rings'
    """

    samples_per_point = 10
    ring_plot_points = 100
    max_samples_per_ring = samples_per_point * ring_plot_points

    radius = 10.2
    angles = np.linspace(0, 2 * np.pi, ring_plot_points)

    # Horizontal ring
    points_h = np.vstack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    points_h = radius * points_h.T - np.array([0.5 * radius, 0, 0])
    offsets = std * np.random.randn(samples_per_point, ring_plot_points, 3)
    X_h = (points_h + offsets).reshape(max_samples_per_ring, 3)
    y_h = np.zeros(shape=(max_samples_per_ring,))

    # Horizontal ring
    points_v = np.vstack((np.cos(angles), np.zeros_like(angles), np.sin(angles)))
    points_v = radius * points_v.T + np.array([0.5 * radius, 0, 0])
    offsets = std * np.random.randn(samples_per_point, ring_plot_points, 3)
    X_v = (points_v + offsets).reshape(max_samples_per_ring, 3)
    y_v = np.ones(shape=(max_samples_per_ring,))

    X = np.vstack((X_h, X_v))
    y = np.hstack((y_h, y_v))

    return X, y, 'rings'


def mnist(dataset_path, subset='train'):
    """ Works with all MNIST (vanilla) files (training and test).

    Parameters
    ----------

    dataset_path : str
        The directory path to the four MNIST data files

    subset : str, default='train'
        The data collection (accepted values: 'train', 'test')

    Returns
    -------

    X : ndarray (float64) of shape (<number of samples>, 28, 28) the
        matrix representing the digit B&W intensity (between 0.0 and
        1.0)

    y : ndarray (int) of shape (<number of samples>,)
        the digit label (0 to 9)
    
    name: str
       description string (either 'MNIST-train' or 'MNIST-test')
    """

    assert subset in ['train', 'test']

    name = 'MNIST-{}'.format(subset)
    
    if subset == 'test':
        subset = 't10k'

    dataset_path = normpath(expanduser(dataset_path))
    
    # Fetch the input data
    filename = '{}-images.idx3-ubyte'.format(subset)
    images_path = '{}{}{}'.format(dataset_path, sep, filename)

    X = extract_idx(images_path)

    # The input data is pre-normalized between 0 and 1
    X = X.reshape((X.shape[0], -1)) / 256.

    # Fetch the labels
    filename = '{}-labels.idx1-ubyte'.format(subset)
    labels_path = '{}{}{}'.format(dataset_path, sep, filename)
    y = extract_idx(labels_path)

    return X, y, name


def extract_idx(path):

    with open(path, 'r'):

        # First 2 bytes are zero. The third signals the datatype. The 4th the
        # number of axes of the array
        first_bytes = np.fromfile(f, dtype=np.uint8, count=4)
        assert first_bytes[0] == 0 and first_bytes[1] == 0

        # This implementation works only if first_bytes[2] is 0x8, i.e., if the
        # data are 8-bit unsigned
        assert first_bytes[2] == 0x08
        ndim = first_bytes[3]

        # The next ndim 32-bit sequences are unsigned integers representing the axes sizes
        sizes = np.fromfile(f, dtype=np.uint32, count=ndim)

        # System endianness: byteorder in ['big', 'little'].
        # The vanilla MNIST files are all big-endian. Convert if needed
        if byteorder == 'little':
            sizes.byteswap(True)

        # The first feature represents samples
        max_elements = 1
        for size in sizes:
            max_elements *= size

        # The greyscale levels are encoded as unsigned byte: no byteswap needed
        data = np.fromfile(f, dtype=np.uint8, count=max_elements)

        # Data is converted to float64, put in matrix format
        data = data.reshape(sizes).astype(np.float64)
        
    f.close()

    return data
