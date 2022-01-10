---
title:
author:
---

# Introduction

The module aims to provide estimator classes compatible with the Scikit-Learn API and implementing the class of Unsupervised Machine Learning algorithms known as *Topographic Mappings*. At the moment the module provides the implementation of the most well-known of these, the Self-Organizing Map (SOM).

The class relies on the fast Batch Algorithm, entirely implemented in `numpy`, in a compact, highly vectorized style that should scale well to larger maps and datasets.

Self-Organizing Maps can be thought of, in principle, as a generalization of *Vector Quantization* (K-Means clustering), in that they aim to recreate the topological structure of the data using a finite number of prototype vectors.

The difference between K-Means and Self-Organizing Maps is that the prototypes aren't just a set of centroids, whose corresponding Voronoi Cells are the clusters. On the contrary, they are interrelated through a fixed rectangular 2D graph, which dictates a regime of collaboration among its units, according to their neighborhood relationship within the graph: the corrective displacements applied by the algorithm to the prototypes is propagated to the ones corresponding to neighboring units.

If the graph relations are displayed in data-space connecting the prototypes with the appropriate edges, we can see that what the Self-Organizing Map does is recreating as best as possible the multidimensional distribution of the data bending and stretching a 2D blanket of discrete points.

Normally, Self-Organizing Maps are used as an exploratory Data Science tool. For this purpose, I have added some visualization facilities. In particular, methods to generate numerically or visualize heat-map matrix representations of the learnt prototype distribution: the *U*-Matrix, which encodes the presence of clusters of prototypes, the *P*-Matrix, which encodes density estimates of the data around the prototypes, and the $U^*$-Matrix, which refines the *U*-Matrix integrating *P*-Matrix information.

# Provided functionality

The class currently provides the following features:

	* Compatibility with the Scikit-Learn API (`BaseEstimator` and `ClusterMixin` classes).
	* Implementation of the EM-like Batch Algorithm, which converges much faster than the original (1982) *online algorithm*, without the necessity to tune and schedule learning rates.
	* Facilities to draw the most popular heat-map representations of a trained SOM: *U*-Matrix, but also the *P*-Matrix and the $U^*$-Matrix.
	* A clustering algorithm of the dataset which combines SOM fitting and DBSCAN clustering.

# Quick guide

## Basic fitting of the estimator

Use a `scikit-learn`-compatible style to build and fit the model:

```{python}
som = SelfOrganizingMap(height=50, width=50, initialization_type='pca')
X = some_data_provider([...])
som.fit(X)
```

There are two ways to set the stopping criterion for the Batch Algorithm. Setting a finite number of iterations (this is actually a long-standing strategy with Self-Organizing Maps), using the `max_iter` parameter:

```{python}
som.fit(X, max_iter=50)
```

Alternatively, one may use the *average distorion* (see below) as a measure of fit/energy function and try to stop at one of its local minima (as a function of the parameters). The tolerance `tol` against which to compare  average distortion variation is the learning parameter to set, in this case:

```{python}
som.fit(X, tol=50)
```

The two criteria aren't mutually exclusive. If both are set, the algorithm terminates at the first occurring event between the two.

## Accessing SOM properties

The main model property of interest (in the Scikit-Learn sense) is the fitted array of prototypes. These are vectors in data-space, hence, they have the same dimensionality as the datapoints. Each prototype is indexed with a pair of non-negative integers, corresponding to the row and column of the corresponding unit.

```{python}
som.W_               # an ndarray of shape (height, width, n_features)
```

Getting back to the *average distortion*, it is defined as the mean squared distance between each prototype and its best-matching-unit (BMU) prototype:

\[
	\frac{1}{N} \sum_{n = 1}^{N}\Vert x_n - w_{\mathrm{BMU}(x_{n})}  \Vert^2
\]

Its final value after training is accessed through

```{python}
som.avg_distortion_             # a float
```

## Clustering

Self-Organizing Maps are appealing in that, intuitively, regions dense with prototypes correspond to "soft-clusters" of the data. The number of this clusters is adaptive to the dataset and needs not be set in advance.

To make the concept precise I implemented a clustering algorithm for the datapoints, using the prototypes as proxy. Once the model is fitted to the data, the prototypes are flexibly clustered with DBSCAN (whose local-distance parameter $\epsilon$, or `eps`, is available to the user). To establish the association of a datapoint to a cluster, its BMU prototype is used instead.

Admittedly, this is a rough proof of concept. The literature provides other solutions that I will implement in the near future.

```{python}
cluster_labels = som.fit_predict(X, eps=0.5)
```

The predicted datapoint cluster labels are accessible through the parameter

```{python}
som.labels_
```

If interested in the *prototype* cluster labels, they are saved in 

```{python}
som.W_cluster_labels_
```

## Graphics

(`matplotlib` required) If the dataset is 3D, use the following function to display (optionally) the datapoints and (optionally) the prototype mesh.

```{python}
som.plot_data_and_prototypes()
pyplot.show()
```

Popular visualizations of a trained Self-Organizing Maps are accessible using

```{python}
som.plot_umatrix()
# or
som.plot_pmatrix()
# or
som.plot_ustarmatrix()
```

followed by pyplot.show(), as before.
