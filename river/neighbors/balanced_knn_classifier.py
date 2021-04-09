
from .knn_classifier import KNNClassifier
from .base_neighbors import BaseNeighbors, KNeighborsBuffer
from river import base
import itertools
import typing
import statistics
import numpy as np
from scipy.spatial import cKDTree
from river.utils import dict2numpy
from river import base
from river.utils.skmultiflow_utils import get_dimensions


class BalancedKNNClassifier(KNNClassifier):
    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, p=2, classes=None):
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size,
                         leaf_size=leaf_size, p=p, weighted=weighted)
        self.data_window = KNeighborsBalancedBuffer(
            window_size=max_window_size, classes=classes)
        self.max_window_size = max_window_size

    def _get_neighbors(self, x, class_idx):
        X = self.data_window.features_buffer(class_idx)
        tree = cKDTree(X, leafsize=self.leaf_size)
        dist, idx = tree.query(x.reshape(1, -1), k=self.n_neighbors, p=self.p)

        # We make sure dist and idx is 2D since when k = 1 dist is one dimensional.
        if not isinstance(dist[0], np.ndarray):
            dist = [dist]
            idx = [idx]

        return dist, idx

    def predict_proba_one(self, x):
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        proba
            A dictionary which associates a probability which each label.

        """

        proba = {class_idx: 0.0 for class_idx in self.classes_}

        for idx, _ in enumerate(self.data_window.bufferClasses()):
            if self.data_window.size(0) == 0:
                return proba

        x_arr = dict2numpy(x)

        output = []

        for i, label in enumerate(self.data_window.bufferClasses()):
            if self.data_window.size(i) != 0:
                distance, _ = self._get_neighbors(x_arr, i)

                # If the closest neighbor has a distance of 0, then return it's output
                if distance[0][0] == 0:
                    proba[label] = 1.0
                    return proba

                # Select only the valid neighbors
                if self.data_window.size(i) < self.n_neighbors:
                    distance = [dist for cnt, dist in enumerate(distance[0]) if cnt < self.data_window.size(
                        i)]  # only get the valid number that are in the window
                else:
                    distance = distance[0]

                # Getting average distance away from each class
                average_distance = statistics.mean(distance)

                output.extend([(average_distance, label)])

        for index in output:
            # Inverse the value as the predict_one will find the max
            proba[index[1]] = 1.0 / index[0]

        return proba


class KNeighborsBalancedBuffer():

    def __init__(self, window_size: int = 1000, classes=None):
        self.window_size = window_size
        self.classes = classes
        self._is_initialized: bool = False
        self._n_features: int = -1
        self._next_insert = np.zeros(len(self.classes), dtype=int)
        self._oldest: int = 0
        self._size = np.zeros(len(self.classes), dtype=int)
        self._X: np.ndarray
        self._y: typing.List
        self._imask: np.ndarray

    def reset(self):
        """Reset the sliding window. """
        self._n_features = -1
        self._size = 0
        self._next_insert = 0
        self._imask = None
        self._X = None
        self._y = None
        self._is_initialized = False
        return self

    def _configure(self):  # Binary instance mask to filter data in the buffer
        self._imask = np.zeros(
            (len(self.classes), self.window_size), dtype=bool)
        self._X = np.zeros(
            (len(self.classes), self.window_size, self._n_features))
        self._is_initialized = True

    def append(self, x: np.ndarray, y: base.typing.Target):
        """Add a (single) sample to the specific class sample window.
        x
            1D-array of feature for a single sample.
        y
            The target data for a single sample. - will be associated with its target window
        """

        if not self._is_initialized:
            self._n_features = get_dimensions(x)[1]
            self._configure()

        if self._n_features != get_dimensions(x)[1]:
            raise ValueError(
                "Inconsistent number of features in X: {}, previously observed {}.".format(
                    get_dimensions(x)[1], self._n_features
                )
            )

        yIndex = self.classes.index(y)  # Gets index of class element

        self._X[yIndex, self._next_insert[yIndex], :] = x

        slot_replaced = self._imask[yIndex, self._next_insert[yIndex]]

        self._imask[yIndex, self._next_insert[yIndex]] = True

        self._next_insert[yIndex] = (self._next_insert[yIndex] + 1 if self._next_insert[yIndex]
                                     < self.window_size - 1 else 0)  # Postion oldest insert as newest

        if not slot_replaced:
            # This is not used after size == window - size of
            self._size[yIndex] += 1

        return self

    def features_buffer(self, class_idx) -> np.ndarray:
        return self._X[class_idx, self._imask[class_idx]]

    def bufferClasses(self):
        """Returns array of classes specified in the buffer """
        # print(self.classes)
        return self.classes

    def size(self, class_idx) -> int:
        """Get the window size. """
        return self._size[class_idx]


# Code that stores majority vote instead of mean value
"""
 for i, label in enumerate(self.classes_):
    distance, _ = self._get_neighbors(x_arr,i)

    # If the closest neighbor has a distance of 0, then return it's output
    if distance[0][0] == 0:
        proba[i] = 1.0
        return proba


    if self.data_window.size(i) < self.n_neighbors:  # Select only the valid neighbors
        distance = [dist for cnt, dist in enumerate(distance[0]) if cnt < self.data_window.size(i)] #only get the valid number that are in the window
    else:
        distance = distance[0]

        output.extend(list(zip(distance,[label for _ in range(self.n_neighbors)])))

# now sort the array
finalOutput = newOutput[:self.n_neighbors]

if not self.weighted:  # Uniform weights
    for index in finalOutput:
        proba[index[1]] += 1.0
else:  # Use the inverse of the distance to weight the votes
    for index in finalOutput:
        proba[index[1]] += 1.0 / index[0]
"""
