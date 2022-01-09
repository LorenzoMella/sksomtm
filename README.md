---
title:
author:
---

# Introduction

The module aims at providing estimator classes compatible with the Scikit-Learn API, starting with the implementation of the Self-Organizing Map (`SelfOrganizingMap` class).

Self-Organizing Maps (SOM) are an early example of *topographic-mapping* unsupervised-learning algorithms, which also includes the Generative Topographic Mapping (GTM).

The class relies on the fast Batch Algorithm, entirely implemented in `numpy`, in a compact highly vectorized style that should scale well to larger problems.

Self-Organizing Maps can be thought of, in principle, as a generalization of vector quantization (K-Means) clustering, in that they aim to recreate the topological structure of the data using a finite number of prototype vectors.

The difference between K-Means and Self-Organizing Maps is that the prototypes aren't just a set of centroids for the clusters, but are interrelated through a fixed rectangular 2D graph, which dictates a regime of collaboration among its units, according to their neighborhood relationship within the graph. By collaboration, we mean that the corrective displacements applied by the algorithm to the prototypes is propagated to the ones corresponding to neighboring units.

Normally, Self-Organizing Maps are used as an exploratory Data Science tool. For this purpose, I have added some visualization facilities. In particular, methods to generate numerically or visualize heat-map matrix representations of the learnt prototype distribution: the *U*-Matrix, which encodes the presence of clusters of prototypes, the *P*-Matrix, which encodes density estimates of the data around the prototypes, and the $U^*$-Matrix, which refines the *U*-Matrix integrating *P*-Matrix information.

# Provided functionality

The class currently provides the following features:

	* Compatibility with the Scikit-Learn API (`BaseEstimator` and `ClusterMixin` classes).
	* Implementation of the EM-like Batch Algorithm, which converges much faster than the original (1982) *online algorithm*, without the necessity to tune and schedule learning rates.
	* Facilities to draw the most popular heat-map representations of a trained SOM: *U*-Matrix, but also the *P*-Matrix and the $U^*$-Matrix.
	* A clustering algorithm of the dataset combining SOM fitting and DBSCAN.

# Quick guide

## Basic fitting of the estimator

Use a `scikit-learn`-compatible style to build and fit the model:

```{python}
som = SelfOrganizingMap(height=50, width=50, initialization_type='pca')
X = some_data_provider([...])
som.fit(X)
```

There are two ways to set the stopping criterion for the Batch Algorithm. One may set a finite number of iterations (this is actually a long-standing strategy with Self-Organizing Maps), using the `max_iter` parameter:

```{python}
som.fit(X, max_iter=50)
```

Alternatively, one may stop at a relative (or pseudo-)minimum of the  *average distorion* (see below). The tolerance `tol` to compare successive average distortion values needs to be set in this case:

```{python}
som.fit(X, tol=50)
```

The two criteria aren't mutually exclusive. If both are set, the algorithm terminates at the first occurring event between the two.

## Accessing SOM properties

The main one is the array of prototypes. These are vectors in data-space, hence, their dimensionality is identical to the datapoints. Each is indexed through a pair of integer coordinates, corresponding to the row and column of the corresponding unit. 

```{python}
som.W_               # an ndarray of shape (height, width, n_features)
```

The average distortion. This is the mean squared distance between each prototype and its best-matching-unit (BMU) prototype:

\[
	\frac{1}{N} \sum_{n = 1}^{N}\Vert x_n - w_{\mathrm{BMU}(x_{n})}  \Vert^2
\]

```{python}
som.avg_distortion_             # a float
```

## Clustering

The function 

## Graphics

(`matplotlib` required) If the dataset is 3D, use the following function to display (optionally) the datapoints and (optionally) the prototype mesh.

```{python}
som.plot_data_and_prototypes()
pyplot.show()
```

More interesting are the heat-map matrices
