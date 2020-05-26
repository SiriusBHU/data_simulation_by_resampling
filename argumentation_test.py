"""
    debug file for each model
    author: Sirius HU
"""
import torch
import torch.optim as optim
import torch.nn as nn

from dataflow.CWRU_data_load import CaseWesternBearing
from utils.train_flow import NetFlow, ClassicMLModelFlow
from utils.dsr import DataSimulationByReSampling
from utils.dataset import *
from torch.utils.data import DataLoader
import logging

# import swallow model
from models.knn import KNN
from models.svm import SVM
from models.bpnn import BPNet
from models.cnn import CNN
# import deep model
from models.ticnn import TINet
from models.dncnn import DNNet
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")


def data_init(sample_num, sample_len):
    import os
    path_project = os.getcwd()

    cwdata = CaseWesternBearing(sample_num, sample_len, path_project=path_project)
    cwdata.dataset_prepare_CWRU(sample_num, sample_len)
    data_wc, labels_wc = cwdata.working_condition_transferring()

    def FFT(x):
        from scipy.fftpack import fft
        wcs, nums, chs, dims = np.shape(x)
        x = x.reshape(wcs * nums * chs, dims)
        f = np.abs(fft(x))[:, :dims//2] / np.sqrt(dims) * 2
        x = x.reshape(wcs, nums, chs, -1)
        return f.reshape(wcs, nums, chs, -1)

    return data_wc, FFT(data_wc), labels_wc


def data_augment(data, labels, dsr):

    wcs, nums, chs, dims = data.shape
    data, labels = data.reshape(wcs * nums * chs, -1), labels.reshape(wcs * nums, -1)
    simu_data, simu_freq, simu_labels = dsr(data, labels, expand_num=1)
    simu_data, simu_freq, simu_labels = simu_data.reshape(wcs, nums, chs, -1), \
                                        simu_freq.reshape(wcs, nums, chs, -1), \
                                        simu_labels.reshape(wcs, nums, -1)
    return simu_data, simu_freq, simu_labels


def data_train_val_split(x, f, y, val_split=0.5):

    wcs, nums, chs, dims = x.shape
    perm = np.arange(nums)
    np.random.shuffle(perm)
    val_num = int(nums * val_split)

    return x[:, perm[val_num:]], f[:, perm[val_num:]], y[:, perm[val_num:]], \
           x[:, perm[:val_num]], f[:, perm[:val_num]], y[:, perm[:val_num]]


def transferring_test(flow_model, data_pack, dataset, batch_size):

    train_data, train_labels, \
    val_data, val_labels, \
    train_simu_data, train_simu_labels, \
    val_simu_data, val_simu_labels = data_pack

    wcs = train_data.shape[0]
    for i in range(wcs):

        # train_set = dataset(train_data[i], train_labels[i])
        # val_set = dataset(val_data[i], val_labels[i])
        # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_set, batch_size=val_data.shape[1], shuffle=False)
        #
        # logging.info("Training with Raw Data:")
        # flow_model.train(train_loader, val_loader)
        # for j in range(wcs):
        #     if i != j:
        #         test_set = dataset(val_data[j], val_labels[j])
        #         test_loader = DataLoader(test_set, batch_size=val_data.shape[1], shuffle=False)
        #         print("from %d HP --> %d HP:" % (i, j))
        #         flow_model.evaluation(test_loader)

        train_simu_set = dataset(train_simu_data[i], train_simu_labels[i])
        val_simu_set = dataset(val_simu_data[i], val_simu_labels[i])
        train_simu_loader = DataLoader(train_simu_set, batch_size=batch_size, shuffle=True)
        val_simu_loader = DataLoader(val_simu_set, batch_size=val_simu_data.shape[1], shuffle=False)

        logging.info("Training with Augmented Data:")
        flow_model.train(train_simu_loader, val_simu_loader)
        for j in range(wcs):
            if i != j:
                test_set = dataset(val_data[j], val_labels[j])
                test_loader = DataLoader(test_set, batch_size=val_data.shape[1], shuffle=False)
                print("from %d HP --> %d HP:" % (i, j))
                flow_model.evaluation(test_loader)


if __name__ == '__main__':

    SAMPLE_NUM = 300
    SAMPLE_LEN = 1024
    BATCH_SIZE = 16
    EPOCHS = 200

    data, freq, labels = data_init(300, 1024)
    dsr = DataSimulationByReSampling(var_r=0.02, var_load=0.02, var_noise=0.02)
    simu_data, simu_freq, simu_labels = data_augment(data, labels, dsr)

    train_data, train_freq, train_labels, val_data, val_freq, val_labels = data_train_val_split(data, freq, labels)
    train_simu_data, train_simu_freq, train_simu_labels, val_simu_data, val_simu_freq, val_simu_labels = \
        data_train_val_split(simu_data, simu_freq, simu_labels)

    freq_pack = (train_freq, train_labels, val_freq, val_labels,
                 train_simu_freq, train_simu_labels, val_simu_freq, val_simu_labels)
    time_pack = (train_data, train_labels, val_data, val_labels,
                 train_simu_data, train_simu_labels, val_simu_data, val_simu_labels)

    bp_net = BPNet(SAMPLE_LEN // 2, 600, labels.shape[-1])
    bp_opt = optim.Adam(bp_net.parameters(), lr=5e-3, weight_decay=1e-5)
    bp_sche = optim.lr_scheduler.StepLR(bp_opt, step_size=30, gamma=0.3)
    bpnn = NetFlow(bp_net,
                   loss_func=nn.CrossEntropyLoss(),
                   optimizer=bp_opt,
                   lr_scheduler=bp_sche,
                   epochs=EPOCHS)

    transferring_test(bpnn, freq_pack, DealDateSetFlatten, BATCH_SIZE)

    print(1)