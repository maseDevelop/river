
from .knn_classifier import KNNClassifier
from .base_neighbors import BaseNeighbors
from river import base
import itertools
import typing

import numpy as np
from scipy.spatial import cKDTree
from river.utils import dict2numpy
from river.utils.math import softmax

from river import base
from river.utils.skmultiflow_utils import get_dimensions

class BalancedKNNClassifier(KNNClassifier):
    def __init__(self,n_neighbors=5,max_window_size=1000,leaf_size=30,metric='euclidean',classes=None):
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size,leaf_size=leaf_size,metric=metric)
        self.data_window = KNeighborsBalancedBuffer(window_size=max_window_size,classes=classes)
        self.max_window_size = max_window_size
        self.classes = classes
        self.metric = metric

    def _get_neighbors(self, x, classidx):
        X = self.data_window.features_buffer[class_idx]
        tree = cKDTree(X, leafsize=self.leaf_size, **self._kwargs)
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
        if self.data_window.size == 0:
            # The model is empty, default to None
            return proba

        x_arr = dict2numpy(x)

        output = []


        for c in self._classes:
            dists, neighbor_idx = self._get_neighbors(x_arr,i)
            target_buffer = self.data_window.targets_buffer[i]

            # If the closest neighbor has a distance of 0, then return it's output
            if dists[0][0] == 0:
                proba[target_buffer[neighbor_idx[0][0]]] = 1.0
                return proba

            if self.data_window.size < self.n_neighbors:  # Select only the valid neighbors
                print(neighbor_idx)
                neighbor_idx = [index for cnt, index in enumerate(neighbor_idx[0])if cnt < self.data_window.size] #only get the valid number that are in the window
                dists = [dist for cnt, dist in enumerate(dists[0]) if cnt < self.data_window.size]
            else:
                neighbor_idx = neighbor_idx[0]
                dists = dists[0]

            #creating final output array to be sorted and checked for majority
            output.extend(list(zip(dists,neighbor_idx,[c for i in range(self.n_neighbors)])))



        #now sort the array
        output = sorted(output, key=lambda x: x[0])
        #Gettting the top five
        output = output[:self.n_neighbors]









        if not self.weighted:  # Uniform weights
            for index in neighbor_idx:
                proba[target_buffer[index]] += 1.0
        else:  # Use the inverse of the distance to weight the votes
            for d, index in zip(dists, neighbor_idx):
                proba[target_buffer[index]] += 1.0 / d

        return softmax(proba)

#Need to work on indexing into array where classes are not zero and one
class KNeighborsBalancedBuffer:

    def __init__(self,window_size: int = 1000,classes=None):
        # TODO:
        self.window_size = window_size
        self.classes = classes
        self._is_initialized: bool = False
        self._n_features: int = -1
        self._next_insert = np.zeros(self.classes, dtype=int)
        self._oldest: int = 0
        self._size: int = 0

    def reset(self):
        """Reset the sliding window. """
        self._n_features = -1
        self._n_targets = -1
        self._size = 0
        self._next_insert = 0
        self._oldest = 0
        self._imask = None
        self._X = None
        self._y = None
        self._is_initialized = False

        return self

    def _configure(self):# Binary instance mask to filter data in the buffer
        self._imask = np.zeros([self.classes, self.window_size], dtype=bool)
        self._X = np.zeros((self.classes, self.window_size, self._n_features))
        self._y = np.zeros([self.classes, self.window_size], dtype=int)
        self._is_initialized = True

    def append(self, x:np.ndarray, y: base.typing.Target):
        """Add a (single) sample to the specific class sample window.
        x
            1D-array of feature for a single sample.
        y
            The target data for a single sample.
        """

        if not self._is_initialized:
            self._n_features = get_dimensions(x)[1]
            self._n_targets = get_dimensions(y)[1]
            self._configure()

        if self._n_features != get_dimensions(x)[1]:
            raise ValueError(
                "Inconsistent number of features in X: {}, previously observed {}.".format(
                    get_dimensions(x)[1], self._n_features
                )
            )

        self._X[y, self._next_insert[y], :] = x
        self._y[y, self._next_insert[y]] = y

        slot_replaced = self._imask[y, self._next_insert[y]]


        self._imask[y, self._next_insert[y]] = True
        self._next_insert[y] = (self._next_insert[y] + 1 if self._next_insert[y] < self.window_size - 1 else 0)

        if (slot_replaced):  # The oldest sample was replaced (complete cycle in the buffer)
            self._oldest = self._next_insert[y]
        else:  # Actual buffer increased
            self._size += 1 #Not used after size == window_size

        return self

    def features_buffer(self,class_idx) -> np.ndarray:
        # Corner case: The number of rows to return is
        # smaller than window_size before the buffer is full.
        # This property must return the features as an
        # np.ndarray since it will be used to search for
        # neighbors via a KDTree

        return self._X[class_idx,self._imask[class_idx]]

    def targets_buffer(self,class_idx) -> typing.List: #list should be size of sliding window when full
        """Get the targets buffer
        The shape of the buffer is (window_size, n_targets).
        """
        return list(itertools.compress(self._y[class_idx], self._imask[class_idx]))
