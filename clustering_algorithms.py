# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
import itertools
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None


    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Selecting centers at random for the first step
        r = np.random.RandomState(self.random_state)
        centers_idx = r.choice(range(len(X)), self.n_clusters)
        self.cluster_centers_ = np.vstack([X[idx] for idx in centers_idx])

        # Setting labels according to centers
        self.__update_labels(X)

        iter_cnt = 0
        eps = 0.00001

        while iter_cnt < self.max_iter:
            previous_centers = self.cluster_centers_
            self.__update_centers(X)
            self.__update_labels(X)

            distances = [
                np.linalg.norm(prev_center - self.cluster_centers_[idx])
                for idx, prev_center in enumerate(previous_centers)
            ]

            if all(distance < eps for distance in distances):
                break
            iter_cnt += 1

        return self

    # Error in the template here: typehint should be -> np.array
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_

    def __update_labels(self, X) -> None:
        self.labels_ = np.zeros(len(X))
        # Calculates labels based on self.cluster_centers_ and puts the result in self.labels_
        for idx, data_point in enumerate(X):
            distances = [
                np.linalg.norm(data_point - center)
                for center in self.cluster_centers_
            ]
            self.labels_[idx] = distances.index(min(distances))

    def __update_centers(self, X) -> None:
        # Calculates centers based on self.labels_ and puts the result in self.cluster_centers_
        for label in range(self.n_clusters):
            data_points = np.vstack([
                data_point
                for idx, data_point in enumerate(X)
                if self.labels_[idx] == label
            ])
            self.cluster_centers_[label] = np.mean(data_points)


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.distances = None
        self.processed = set()
        self.clusters = list()

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        sample_size = len(X)

        self.labels_ = np.full(sample_size, -1)
        # self.__calculate_distances(X)

        for i in range(sample_size):
            if i not in self.processed:
                self.processed.add(i)
                neighbors = self._get_neighbors(X, i)

                if len(neighbors) < self.min_samples:
                    self.labels_[i] = -1
                else:
                    self.__expand_cluster(X, i, neighbors)

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_

    def __expand_cluster(self, X, point_index, neighbors):
        cluster_index = len(self.clusters)
        self.clusters.append([point_index])
        self.labels_[point_index] = cluster_index

        i = 0

        while i < len(neighbors):
            neighbor = neighbors[i]
            if neighbor not in self.processed:
                self.processed.add(neighbor)
                new_neighbors = self._get_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, new_neighbors)

            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_index
                self.clusters[cluster_index].append(neighbor)
            i += 1

    def _get_neighbors(self, X, point_index):
        distances = np.linalg.norm(X - X[point_index], axis=1)
        return np.where(distances < self.eps)[0]


if __name__ == "__main__":
    # Foundations of Data Mining - Practical Task 1
    # Version 2.0 (2023-11-02)
    ###############################################
    # Template for a notebook that clusters pixel data of a given image.
    # This file does not have to be changed in order to complete the task.
    # That being said, you MAY change it. However, your implementation has
    # to work with the original version of this file.

    import cv2  # for image loading
    import numpy as np  # general library for numerical and scientific computing
    import matplotlib.pyplot as plt  # for plotting the images

    # For testing purposes ONLY(!), you may uncomment the following two import statements.
    # Use them to see how the program is supposed to work with your implementation and
    # what kind of content the variables should have.
    # Comment out or delete these imports before you submit your code!
    # from sklearn.cluster import KMeans
    # from sklearn.cluster import DBSCAN

    # Importing your own implementation:
    from clustering_algorithms import CustomKMeans as KMeans
    from clustering_algorithms import CustomDBSCAN as DBSCAN

    # Loading an image (replace filename if you want):
    image_path = 'giraffe.png'
    image = cv2.imread(image_path)

    # Reducing the size of the image, so that DBSCAN runs in a reasonable amount of time:
    # small_image is 0.5x the size of the original. You may change this value.
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    height, width, _ = image.shape
    pixel_data = image.reshape(-1, 3)
    # DBSCAN
    # Setting hyperparameter(s):
    eps = 5
    min_pts = 30

    # Performing the clustering:
    dbscan = CustomDBSCAN(eps=eps, min_samples=min_pts)
    dbscan_labels = dbscan.fit_predict(pixel_data)
    print(dbscan_labels)
    from sklearn.cluster import DBSCAN