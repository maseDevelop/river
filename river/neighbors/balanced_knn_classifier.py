
from . import KNNClassifier

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

    def predict_proba_one(self, x):
        # TODO:
        raise NotImplementedError


class KNeighborsBalancedBuffer:

    def __init__(self,window_size: int = 1000,classes=None):
        # TODO:
        self.window_size = window_size
        self.classes = classes

    def reset(self):
        # # TODO:
        return self

    def append(self, x:np.ndarray, y: base.typing.Target):
        ## TODO:
        return self

    def features_buffer(self,class_idx) -> np.ndarray:

        return self
