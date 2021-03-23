
from . import KNNClassifier
from river import base
import itertools
import typing

import numpy as np
from scipy.spatial import cKDTree

from river import base
from river.utils.skmultiflow_utils import get_dimensions

class BalancedKNNClassifier(KNNClassifier):
    def __init__(self,n_neighbors=5,max_window_size=1000,leaf_size=30,metric='euclidean',classes=None):
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size,leaf_size=leaf_size,metric=metric)
        self.data_window = KNeighborsBalancedBuffer(window_size=max_window_size,classes=classes)
        self.max_window_size = max_window_size
        self.classes = classes
        self.metric = metric



    def predict_proba_one(self, x):
        # TODO:
        raise NotImplementedError

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
        # # TODO:
        return self

    def _configure(self):# Binary instance mask to filter data in the buffer
        self._imask = np.zeros([self.classes, self.window_size], dtype=bool)
        self._X = np.zeros((self.classes, self.window_size, self._n_features))
        self._y = np.zeros([self.classes, self.window_size], dtype=bool)
        self._is_initialized = True

    def append(self, x:np.ndarray, y: base.typing.Target):
        """Add a (single) sample to the specific class sample window.
        x
            1D-array of feature for a single sample.
        y
            The target data for a single sample.
        """

        if not self._is_initialized:
            self._n_features = x.shape[1]
            self._n_targets = y.shape[1]
            self._configure()

        if self._n_features != x.shape[2]:
            raise ValueError(
                "Inconsistent number of features in X: {}, previously observed {}.".format(
                    x.shape[2], self._n_features
                )
            )

        self._X[self.classes, self._next_insert[self.classes], :] = x
        self._y[self.classes, self._next_insert[self.classes]] = y

        slot_replaced = self._imask[self.classes, self._next_insert[self.classes]]

        # Update the instance storing logic
        self._imask[self.classes, self._next_insert[self.classes]] = True  # Mark slot as filled
        self._next_insert[self.classes] = (
            self._next_insert[self.classes] + 1 if self._next_insert[self.classes] < self.window_size - 1 else 0
        )

        if (
            slot_replaced
        ):  # The oldest sample was replaced (complete cycle in the buffer)
            self._oldest = self._next_insert[self.classes]
        else:  # Actual buffer increased
            self._size += 1

        return self

    def features_buffer(self,class_idx) -> np.ndarray:
        # Corner case: The number of rows to return is
        # smaller than window_size before the buffer is full.
        # This property must return the features as an
        # np.ndarray since it will be used to search for
        # neighbors via a KDTree
        return self
