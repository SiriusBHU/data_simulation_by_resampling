import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import utils.data_pre_process as dpp

class Pure_SVM:

    C = 1
    kernel = "rbf"
    gamma = 0.001
    decision_function_shape = "ovo"

    def __init__(self, C=1, kernel="rbf", gamma=0.001, decision_function_shape="ovo"):
        """
        :param C:
        :param kernel:
        :param gamma:
        :param decision_function_shape:
        :param test_choice:
        :param test_array:
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape

    def PSVM_train(self, xs, ys, xt_arr, yt_arr, val_rate=0.5):

        from copy import deepcopy
        x_train, y_train, x_test_arr, y_test_arr = deepcopy(xs), deepcopy(ys), deepcopy(xt_arr), deepcopy(yt_arr)
        xs_train, ys_train, xs_val, ys_val, _, __ = dpp.data_preparation(x_train, y_train, val_rate)
        xs_train, xs_val = dpp.get_2d_shape(xs_train), dpp.get_2d_shape(xs_val)
        ys_train, ys_val = np.argmax(ys_train, axis=1), np.argmax(ys_val, axis=1)

        clfsvm = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, decision_function_shape=self.decision_function_shape)
        clfsvm.fit(xs_train, ys_train)
        accs = clfsvm.score(xs_val, ys_val)
        acct = [clfsvm.score(xt_arr[i], yt_arr[i]) for i in range(len(xt_arr))]
        print("source data accuracy: ", accs, " ", "target data accuracy: ", acct)
        return clfsvm, acct, [xs_val, xt_arr]




