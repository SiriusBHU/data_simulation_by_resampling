import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import utils.data_pre_process as dpp


class Pure_KNN:

    neighbors = 5
    algorithm = 'brute'   # alternative: ball_tree, kd_tree, auto
    leaf_size = 30
    distance_mold = 'minkowski'

    def __init__(self, neighbors=1, algorithm="brute", leaf_size=30, distance_mold='minkowski'):

        self.neighbors = neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.distance_mold = distance_mold

    def train(self, xs, ys, xt_arr, yt_arr, val_rate=0.5):

        from copy import deepcopy
        x_train, y_train, x_test_arr, y_test_arr = deepcopy(xs), deepcopy(ys), deepcopy(xt_arr), deepcopy(yt_arr)
        xs_train, ys_train, xs_val, ys_val, _, __ = dpp.data_preparation(x_train, y_train, val_rate)
        xs_train, xs_val = dpp.get_2d_shape(xs_train), dpp.get_2d_shape(xs_val)
        ys_train, ys_val = np.argmax(ys_train, axis=1), np.argmax(ys_val, axis=1)

        clfknn = KNeighborsClassifier(n_neighbors=self.neighbors,
                                      algorithm=self.algorithm,
                                      leaf_size=self.leaf_size,
                                      metric=self.distance_mold)
        clfknn.fit(xs_train, ys_train)
        accs = clfknn.score(xs_val, ys_val)
        acct = [clfknn.score(xt_arr[i], yt_arr[i]) for i in range(len(xt_arr))]
        print("source data accuracy: ", accs, " ", "target data accuracy: ", acct)
        return clfknn, acct, [xs_val, xt_arr]




