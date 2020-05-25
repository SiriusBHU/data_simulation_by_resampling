"""
    debug file for each model
    author: Sirius HU
"""

from dataflow.CWRU_data_load import *
import utils.data_pre_process as dpp
import utils.data_enhance_fft as dff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    sample_size = 1024
    sample_num = 300
    # data_wc, label_wc, data_fs, label_fs = data_preparation_CWRU(sample_size, sample_num)
    data_wc, label_wc, data_fs, label_fs = data1D_reload_CWRU()
    data_wc_fft = [dpp.fast_frontier_transform(item) for item in data_wc]
    label_wc = [item.reshape(-1) for item in label_wc]
    freq_simu, time_simu, label_simu, _ = \
        dff.enhanced_set_prepare("CWRU_1D_simu_expand1", 1, data_wc, label_wc, [[1, 0.0004], [1, 0.1], [0, 0.02]])
    # freq_simu, time_simu, label_simu, time_simu_freq = dff.enhanced_set_reload("CWRU_1D_simu_expand1")

    from sklearn.decomposition import pca

    clf1 = pca.PCA(n_components=3, whiten=True)
    source_data = clf1.fit_transform(data_wc_fft[3])
    simu_data = (((clf1.transform(freq_simu[3])).reshape(10, 300, 3))[:, :300, :]).reshape(-1, 3)

    target_data_origin = np.concatenate((((data_wc_fft[1]).reshape(10, 300, -1))[:, :150, :],
                                         ((data_wc_fft[2]).reshape(10, 300, -1))[:, :150, :]), axis=1)
    target_data_origin = target_data_origin.reshape(3000, -1)
    target_data = (((clf1.transform(target_data_origin)).reshape(10, 300, 3))[:, :300, :]).reshape(-1, 3)

    a, b, c = 0, 1, 2
    label_size = 11
    # plt.figure(1)
    # ax = plt.subplot(221, projection="3d")
    # ax.scatter(source_data[:, c], source_data[:, a], source_data[:, b], c="r", marker=".", s=20, alpha=0.2)
    # ax.scatter(target_data[:, c], target_data[:, a], target_data[:, b], c="b", marker=".", s=20, alpha=0.2)
    # ax.scatter(simu_data[:, c], simu_data[:, a], simu_data[:, b], c="orange", marker=".", s=20, alpha=0.2)
    #
    # ax = plt.subplot(222, projection="3d")
    # ax.scatter(source_data[:, c], source_data[:, a], source_data[:, b], c=label_wc[0], marker=".", s=20, alpha=0.2)
    #
    # ax = plt.subplot(223, projection="3d")
    # ax.scatter(target_data[:, c], target_data[:, a], target_data[:, b], c=label_wc[0], marker=".", s=20, alpha=0.2)
    #
    # ax = plt.subplot(224, projection="3d")
    # ax.scatter(simu_data[:, c], simu_data[:, a], simu_data[:, b], c=label_wc[0], marker=".", s=20, alpha=0.2)
    #
    plt.figure(2, figsize=(6, 6), dpi=300)
    ax = plt.subplot(221)
    ax.scatter(source_data[:, a], source_data[:, b], c="r", marker=".", s=10, alpha=0.2)
    ax.scatter(target_data[:, a], target_data[:, b], c="b", marker=".", s=10, alpha=0.2)
    ax.scatter(simu_data[:, a], simu_data[:, b], c="orange", marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-2, -1, 0, 1, 2, 3, 4])
    plt.grid()

    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(222)
    ax.scatter(source_data[:, a], source_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-2, -1, 0, 1, 2, 3, 4])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(223)
    ax.scatter(target_data[:, a], target_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-2, -1, 0, 1, 2, 3, 4])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(224)
    ax.scatter(simu_data[:, a], simu_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-2, -1, 0, 1, 2, 3, 4])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    plt.savefig("0-1figure.png")

    a, b = 0, 2
    plt.figure(3, figsize=(6, 6), dpi=300)
    ax = plt.subplot(221)
    ax.scatter(source_data[:, a], source_data[:, b], c="r", marker=".", s=10, alpha=0.2)
    ax.scatter(target_data[:, a], target_data[:, b], c="b", marker=".", s=10, alpha=0.2)
    ax.scatter(simu_data[:, a], simu_data[:, b], c="orange", marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-3.5, 3.5)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(222)
    ax.scatter(source_data[:, a], source_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-3.5, 3.5)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(223)
    ax.scatter(target_data[:, a], target_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-3.5, 3.5)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]

    ax = plt.subplot(224)
    ax.scatter(simu_data[:, a], simu_data[:, b], c=label_wc[0], marker=".", s=10, alpha=0.2)
    plt.xlim(-2, 4)
    plt.ylim(-3.5, 3.5)
    plt.xticks([-2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.grid()
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]


    plt.savefig("0-2figure.png")


    plt.figure(4)
    plt.subplot(211)
    plt.scatter(1, 1, c="r", marker=".", s=100)
    plt.scatter(1, 2, c="b", marker=".", s=100)
    plt.scatter(1, 3, c="orange", marker=".",s=100)
    plt.legend([1,2,3])

    plt.subplot(212)

    plt.scatter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1,1,1,1,1,1,1,1,1], c=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], s=100)
    plt.legend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    plt.savefig("legend.png")


    plt.show()
