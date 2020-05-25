"""
    debug file for each model
    author: Sirius HU
"""

from dataflow.CWRU_data_load import *
import utils.data_pre_process as dpp
import utils.data_enhance_fft as dff
import numpy as np
import matplotlib.pyplot as plt

import models.bpnn as bpnn           # ok
import models.svm as svm             # ok
import models.knn as knn             # ok
import models.cnn as cnn             # not so good
import models.lenet5 as lenet        # time series reconstruction
import models.ticnn as ticnn         # ok
import models.dncnn as dncnn         # ok
import copy


def data_concat(x1, x2):
    x1, x2 = copy.deepcopy(x1), copy.deepcopy(x2)
    x1 = [item.reshape(10, int(np.shape(item)[0] // 10), -1) for item in x1]
    x2 = [item.reshape(10, int(np.shape(item)[0] // 10), -1) for item in x2]
    x = [np.concatenate((item1, item2), axis=1) for item1, item2 in zip(x1, x2)]
    x = [item.reshape(-1, np.shape(item)[-1]) for item in x]
    return x


def main(time_tr, time_la, time_te_list, time_la_list,
         freq_tr, freq_la, freq_te_list, freq_la_list,
         kn_param=5, sv_param=[1, 0.01], bp_param=600,
         epoch=100, val_rate=0.5, mix_up=False, a=0.2):

    # kn = knn.Pure_KNN(neighbors=kn_param, algorithm="kd_tree")
    # _, acc1, __ = kn.train(freq_tr, freq_la, freq_te_list, freq_la_list, val_rate)
    #
    # sv = svm.Pure_SVM(C=sv_param[0], kernel="rbf", gamma=sv_param[1], decision_function_shape="ovo")
    # _, acc2, __ = sv.PSVM_train(freq_tr, freq_la, freq_te_list, freq_la_list, val_rate)
    #
    # bp = bpnn.BPNN(in_num=512, hidden_num=bp_param, out_num=10, batch_size=10, lr=0.001, epoch=epoch)
    # _, __, acc3 = bp.train(freq_tr, freq_la, freq_te_list, freq_la_list, val_rate, mix_up, a)
    #
    # cn = cnn.PCNN(in_num=512, out_num=10, batch_size=10, lr=0.001, epoch=epoch)
    # _, __, acc4 = cn.train(freq_tr, freq_la, freq_te_list, freq_la_list, val_rate, mix_up, a)

    # ti_cnn = ticnn.TICNN(in_num=1024, out_num=10, batch_size=32, lr=0.001, epoch=epoch)
    # _, __, acc5 = ti_cnn.train(time_tr, time_la, time_te_list, time_la_list, val_rate, mix_up, a)

    dn_cnn = dncnn.DNCNN(in_num=1024, out_num=10, batch_size=16, lr=0.001, epoch=epoch)
    _, __, acc6 = dn_cnn.train(time_tr, time_la, time_te_list, time_la_list, val_rate, mix_up, a)

    #
    # le = lenet.LeNet(in_num=1024, out_num=10, batch_size=50, lr=0.001, epoch=epoch)
    # _, __, acc7 = le.train(time_tr, time_la, time_te_list, time_la_list, val_rate, mix_up, a)

    return np.array([acc6])#, acc5, acc6])  # acc1, acc2, acc3,


if __name__ == '__main__':

    sample_size = 1024
    sample_num = 300
    # data_wc, label_wc, data_fs, label_fs = data_preparation_CWRU(sample_size, sample_num)
    data_wc, label_wc, data_fs, label_fs = data1D_reload_CWRU()
    data_wc_fft = [dpp.fast_frontier_transform(item) for item in data_wc]

    # acc = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(data_wc[i], label_wc[i], data_wc[1:], label_wc[1:],
    #                      data_wc_fft[i], label_wc[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=True, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_mix_only.npz", acc=acc)

    choice = [
        [[1, 0], [1, 0], [0, 0.02]],
        [[1, 0], [1, 0.10], [0, 0]],
        [[1, 0], [1, 0.10], [0, 0.02]],
        [[1, 0.02], [1, 0], [0, 0]],
        [[1, 0.02], [1, 0], [0, 0.02]],
        [[1, 0.02], [1, 0.10], [0, 0]]
              ]
    parammmm = ["100", "010", "001", "101", "110", "011"]

    acc = []
    for k in range(6):

        freq_simu, time_simu, label_simu, _ = \
            dff.enhanced_set_prepare("CWRU_1D_simu_expand_ablation", 1, data_wc, label_wc, choice[k])
        paramm = parammmm[k]  #"0.1_" + str(0.02 * k + 0.02)

        for i in range(1, 4):
            for j in range(10):
                acc_i = main(time_simu[i], label_simu[i], data_wc[1:], label_wc[1:],
                             freq_simu[i], label_simu[i], data_wc_fft[1:], label_wc[1:],
                             kn_param=5, sv_param=[1, 0.01], bp_param=600,
                             epoch=100, val_rate=0.5, mix_up=True, a=0.2)
                acc.append(acc_i)

        np.savez("acc_dncnn_ablation.npz", acc=acc)


    print(-1)

    # time_mix = [np.concatenate(((item1.reshape(10, 300, -1))[:, :150, :], (item2.reshape(10, 300, -1))[:, :150, :]), axis=1)
    #             for item1, item2 in zip(data_wc, time_simu)]
    # freq_mix = [np.concatenate(((item1.reshape(10, 300, -1))[:, :150, :], (item2.reshape(10, 300, -1))[:, :150, :]), axis=1)
    #             for item1, item2 in zip(data_wc_fft, freq_simu)]
    #
    # time_mix = [item.reshape(-1, 1024) for item in time_mix]
    # freq_mix = [item.reshape(-1, 512) for item in freq_mix]

    # acc_mix = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_mix.append(main(time_mix[i], label_wc[i], data_wc[1:], label_wc[1:],
    #                             freq_mix[i], label_wc[i], data_wc_fft[1:], label_wc[1:],
    #                             kn_param=10, sv_param=[1, 0.07], bp_param=600,
    #                             epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_mix = (np.array(acc_mix)).reshape(-1, 10, 6, 3)
    # np.savez("acc_mix" + paramm+".npz", acc=acc_mix)
    #
    # acc_mixup = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_mixup.append(main(time_mix[i], label_wc[i], data_wc[1:], label_wc[1:],
    #                               freq_mix[i], label_wc[i], data_wc_fft[1:], label_wc[1:],
    #                               kn_param=10, sv_param=[1, 0.07], bp_param=600,
    #                               epoch=100, val_rate=0.5, mix_up=True, a=0.2))
    # acc_mixup = (np.array(acc_mixup)).reshape(-1, 10, 6, 3)
    # np.savez("acc_mixup" + paramm+".npz", acc=acc_mixup)


    # freq_simu, time_simu, label_simu, time_simu_freq = dff.enhanced_set_reload("CWRU_1D_simu_expand")
    # # freq_simu, time_simu, label_simu, _ = \
    # #     dff.enhanced_set_prepare("CWRU_1D_simu_expand", 1, data_wc, label_wc, [[1, 0.02], [1, 0.10], [0, 0.02]])
    #
    # acc = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(data_wc[i], label_wc[i], data_wc[1:], label_wc[1:],
    #                      data_wc_fft[i], label_wc[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(time_simu[i], label_simu[i], data_wc[1:], label_wc[1:],
    #                      freq_simu[i], label_simu[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn.npz", acc=acc)
    #
    #
    #
    # prepare few shot set....half for training half for validation
    # few_time5 = [(item.reshape(10, 300, -1))[:, :10] for item in data_wc]
    # few_freq5 = [(item.reshape(10, 300, -1))[:, :10] for item in data_wc_fft]
    # few_label5 = [(item.reshape(10, 300, -1))[:, :10] for item in label_wc]
    # few_time5 = [item.reshape(-1, 1024) for item in few_time5]
    # few_freq5 = [item.reshape(-1, 512) for item in few_freq5]
    # few_label5 = [item.reshape(-1) for item in few_label5]
    #
    # few_time10 = [(item.reshape(10, 300, -1))[:, :20] for item in data_wc]
    # few_freq10 = [(item.reshape(10, 300, -1))[:, :20] for item in data_wc_fft]
    # few_label10 = [(item.reshape(10, 300, -1))[:, :20] for item in label_wc]
    # few_time10 = [item.reshape(-1, 1024) for item in few_time10]
    # few_freq10 = [item.reshape(-1, 512) for item in few_freq10]
    # few_label10 = [item.reshape(-1) for item in few_label10]
    #
    # few_time20 = [(item.reshape(10, 300, -1))[:, :50] for item in data_wc]
    # few_freq20 = [(item.reshape(10, 300, -1))[:, :50] for item in data_wc_fft]
    # few_label20 = [(item.reshape(10, 300, -1))[:, :50] for item in label_wc]
    # few_time20 = [item.reshape(-1, 1024) for item in few_time20]
    # few_freq20 = [item.reshape(-1, 512) for item in few_freq20]
    # few_label20 = [item.reshape(-1) for item in few_label20]
    #
    # few_time50 = [(item.reshape(10, 300, -1))[:, :100] for item in data_wc]
    # few_freq50 = [(item.reshape(10, 300, -1))[:, :100] for item in data_wc_fft]
    # few_label50 = [(item.reshape(10, 300, -1))[:, :100] for item in label_wc]
    # few_time50 = [item.reshape(-1, 1024) for item in few_time50]
    # few_freq50 = [item.reshape(-1, 512) for item in few_freq50]
    # few_label50 = [item.reshape(-1) for item in few_label50]
    #
    # few_freq_simu1, few_time_simu1, few_label_simu1, _ = \
    #     dff.enhanced_set_prepare("CWRU_1D_simu_expand1", 1, few_time5, few_label5, [[1, 0.02], [1, 0.10], [0, 0.02]])
    # #
    # few_freq_simu3, few_time_simu3, few_label_simu3, _ = \
    #     dff.enhanced_set_prepare("CWRU_1D_simu_expand4", 4, few_time5, few_label5, [[1, 0.02], [1, 0.10], [0, 0.02]])
    # #
    # few_freq_simu9, few_time_simu9, few_label_simu9, _ = \
    #     dff.enhanced_set_prepare("CWRU_1D_simu_expand9", 9, few_time5, few_label5, [[1, 0.02], [1, 0.10], [0, 0.02]])
    # # freq_simu, time_simu, label_simu, time_simu_freq = dff.enhanced_set_reload("CWRU_1D_simu_expand1")
    #
    # few_time_simu1 = [item * np.sqrt(2) for item in few_time_simu1]
    # few_time_simu3 = [item * np.sqrt(2) for item in few_time_simu3]
    # few_time_simu9 = [item * np.sqrt(2) for item in few_time_simu9]
    #
    #
    # """ 最优超参数调整原则，根据源域数据进行调整！！！，source=1，那只能根据 1HP 数据的正确率调整 """
    #
    # freq_simu2 = data_concat(few_freq_simu1, few_freq5)
    # time_simu2 = data_concat(few_time_simu1, few_time5)
    # label_simu2 = data_concat(few_label_simu1, few_label5)
    # #
    # freq_simu4 = data_concat(few_freq_simu3, few_freq5)
    # time_simu4 = data_concat(few_time_simu3, few_time5)
    # label_simu4 = data_concat(few_label_simu3, few_label5)
    #
    # freq_simu10 = data_concat(few_freq_simu9, few_freq5)
    # time_simu10 = data_concat(few_time_simu9, few_time5)
    # label_simu10 = data_concat(few_label_simu9, few_label5)
    #
    #
    # acc = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(few_time5[i], few_label5[i], data_wc[1:], label_wc[1:],
    #                      few_freq5[i], few_label5[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(few_time10[i], few_label10[i], data_wc[1:], label_wc[1:],
    #                      few_freq10[i], few_label10[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(few_time20[i], few_label20[i], data_wc[1:], label_wc[1:],
    #                      few_freq20[i], few_label20[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(few_time50[i], few_label50[i], data_wc[1:], label_wc[1:],
    #                      few_freq50[i], few_label50[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(time_simu2[i], label_simu2[i], data_wc[1:], label_wc[1:],
    #                      freq_simu2[i], label_simu2[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(time_simu4[i], label_simu4[i], data_wc[1:], label_wc[1:],
    #                      freq_simu4[i], label_simu4[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_i = main(time_simu10[i], label_simu10[i], data_wc[1:], label_wc[1:],
    #                      freq_simu10[i], label_simu10[i], data_wc_fft[1:], label_wc[1:],
    #                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                      epoch=100, val_rate=0.5, mix_up=False, a=0.2)
    #         acc.append(acc_i)
    #
    # np.savez("acc_dncnn_few.npz", acc=acc)




    # acc_origin_few = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_origin_few.append(main(few_time5[i], few_label5[i], data_wc[1:], label_wc[1:],
    #                                    few_freq5[i], few_label5[i], data_wc_fft[1:], label_wc[1:],
    #                                    kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                                    epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_origin_few = (np.array(acc_origin_few)).reshape(-1, 10, 2, 3)
    # np.savez("acc_origin_few5.npz", acc=acc_origin_few)

    # acc_origin_few10 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_origin_few10.append(main(few_time10[i], few_label10[i], data_wc[1:], label_wc[1:],
    #                                      few_freq10[i], few_label10[i], data_wc_fft[1:], label_wc[1:],
    #                                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                                      epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_origin_few10 = (np.array(acc_origin_few10)).reshape(-1, 10, 6, 3)
    # np.savez("acc_origin_few10.npz", acc=acc_origin_few10)

    #
    # acc_origin_few20 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_origin_few20.append(main(few_time20[i], few_label20[i], data_wc[1:], label_wc[1:],
    #                                      few_freq20[i], few_label20[i], data_wc_fft[1:], label_wc[1:],
    #                                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                                      epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_origin_few20 = (np.array(acc_origin_few20)).reshape(-1, 10, 2, 3)
    # np.savez("acc_origin_few25.npz", acc=acc_origin_few20)
    #
    # acc_origin_few50 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_origin_few50.append(main(few_time50[i], few_label50[i], data_wc[1:], label_wc[1:],
    #                                      few_freq50[i], few_label50[i], data_wc_fft[1:], label_wc[1:],
    #                                      kn_param=5, sv_param=[1, 0.01], bp_param=600,
    #                                      epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_origin_few50 = (np.array(acc_origin_few50)).reshape(-1, 10, 2, 3)
    # np.savez("acc_origin_few50.npz", acc=acc_origin_few50)



    # acc_simu2 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_simu2.append(main(time_simu2[i], label_simu2[i], data_wc[1:], label_wc[1:],
    #                               freq_simu2[i], label_simu2[i], data_wc_fft[1:], label_wc[1:],
    #                               kn_param=5, sv_param=[10, 0.07], bp_param=600,
    #                               epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_simu2 = (np.array(acc_simu2)).reshape(-1, 10, 2, 3)
    # np.savez("acc_simu_expand2.npz", acc=acc_simu2)
    #
    # acc_simu5 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_simu5.append(main(time_simu4[i], label_simu4[i], data_wc[1:], label_wc[1:],
    #                               freq_simu4[i], label_simu4[i], data_wc_fft[1:], label_wc[1:],
    #                               kn_param=5, sv_param=[10, 0.07], bp_param=600,
    #                               epoch=100, val_rate=0.5, mix_up=False, a=0.2))
    # acc_simu5 = (np.array(acc_simu5)).reshape(-1, 10, 4, 3)
    # np.savez("acc_simu_expand5.npz", acc=acc_simu5)
    #
    # acc_simu10 = []
    # for i in range(1, 4):
    #     for j in range(10):
    #         acc_simu10.append(main(time_simu10[i], label_simu10[i], data_wc[1:], label_wc[1:],
    #                                freq_simu10[i], label_simu10[i], data_wc_fft[1:], label_wc[1:],
    #                                kn_param=5, sv_param=[10, 0.07], bp_param=600,
    #                                epoch=200, val_rate=0.5, mix_up=False, a=0.2))
    # acc_simu10 = (np.array(acc_simu10)).reshape(-1, 10, 2, 3)
    # np.savez("acc_simu_expand10_time.npz", acc=acc_simu10)








# bpnn1 = bpnn.BPNN(in_num=512, hidden_num=600, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net, test_loss, test_acc = bpnn1.train(data_wc_fft[1], label_wc[1], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# bpnn2 = bpnn.BPNN(in_num=512, hidden_num=600, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net1, test_loss1, test_acc1 = bpnn2.train(freq_simu[1], label_simu[1], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# bpnn3 = bpnn.BPNN(in_num=512, hidden_num=600, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net3, test_loss3, test_acc3 = bpnn3.train(np.concatenate((freq_simu[1], data_wc_fft[1]), axis=0),
#                                           np.concatenate((label_simu[1], label_wc[1]), axis=0),
#                                           data_wc_fft[1:], label_wc[1:], val_rate=0.75)
# ti_cnn1 = ticnn.TICNN(in_num=1024, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net1, test_loss1, test_acc1 = ti_cnn1.train(data_wc[1], label_wc[1], data_wc[1:], label_wc[1:], val_rate=0.5)
#
# ti_cnn2 = ticnn.TICNN(in_num=1024, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net2, test_loss2, test_acc2 = ti_cnn2.train(time_simu[1], label_simu[1], data_wc[1:], label_wc[1:], val_rate=0.5)
#
# ti_cnn3 = ticnn.TICNN(in_num=1024, out_num=10, batch_size=10, lr=0.001, epoch=100)
# net3, test_loss3, test_acc3 = ti_cnn3.train(np.concatenate((time_simu[1], data_wc[1]), axis=0),
#                                             np.concatenate((label_simu[1], label_wc[1]), axis=0),
#                                             data_wc[1:], label_wc[1:], val_rate=0.75)
#
# "从 lenet 的表现上看，时间上的仿真效果比较一般，或许可以考虑对相位角进行插值了 "
# lenet1 = lenet.LeNet(in_num=1024, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net1, test_loss1, test_acc1 = lenet1.train(data_wc[3], label_wc[3], data_wc[1:], label_wc[1:], val_rate=0.5)
#
# lenet2 = lenet.LeNet(in_num=1024, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net2, test_loss2, test_acc2 = lenet2.train(time_simu[3], label_simu[3], data_wc[1:], label_wc[1:], val_rate=0.5)
#
# lenet3 = lenet.LeNet(in_num=1024, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net3, test_loss3, test_acc3 = lenet3.train(np.concatenate((time_simu[3], data_wc[3]), axis=0),
#                                            np.concatenate((label_simu[3], label_wc[3]), axis=0),
#                                            data_wc[1:], label_wc[1:], val_rate=0.75)
#
# cnn1 = cnn.PCNN(in_num=512, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net1, test_loss1, test_acc1 = cnn1.train(data_wc_fft[3], label_wc[3], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# cnn2 = cnn.PCNN(in_num=512, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net2, test_loss2, test_acc2 = cnn2.train(freq_simu[3], label_simu[3], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# cnn3 = cnn.PCNN(in_num=512, out_num=10, batch_size=50, lr=0.001, epoch=100)
# net3, test_loss3, test_acc3 = cnn3.train(np.concatenate((freq_simu[3], data_wc_fft[3]), axis=0),
#                                          np.concatenate((label_simu[3], label_wc[3]), axis=0),
#                                          data_wc_fft[1:], label_wc[1:], val_rate=0.75)
#
# knn1 = knn.Pure_KNN(neighbors=5, algorithm="auto")
# model1, acc1, feature_map1 = knn1.train(data_wc_fft[1], label_wc[1], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# knn2 = knn.Pure_KNN(neighbors=15, algorithm="auto")
# model2, acc2, feature_map2 = knn2.train(freq_simu[1], label_simu[1], data_wc_fft[1:], label_wc[1:], val_rate=0.5)
#
# knn3 = knn.Pure_KNN(neighbors=5, algorithm="auto")
# model3, acc3, feature_map3 = knn3.train(np.concatenate((freq_simu[1], data_wc_fft[1]), axis=0),
#                                         np.concatenate((label_simu[1], label_wc[1]), axis=0),
#                                         data_wc_fft[1:], label_wc[1:], val_rate=0.75)
#
#
# svm1 = svm.Pure_SVM(C=1, kernel="rbf", gamma=0.01, decision_function_shape="ovo")
# model1, acc1, feature_map1 = svm1.PSVM_train(data_wc_fft[1], label_wc[1], data_wc_fft[1:], label_wc[1:])
#
# svm2 = svm.Pure_SVM(C=10, kernel="rbf", gamma=0.07, decision_function_shape="ovo")
# model2, acc2, feature_map2 = svm2.PSVM_train(freq_simu[1], label_simu[1], data_wc_fft[1:], label_wc[1:])
#
# svm3 = svm.Pure_SVM(C=10, kernel="rbf", gamma=0.07, decision_function_shape="ovo")
# model3, acc3, feature_map3 = svm3.PSVM_train(np.concatenate((freq_simu[1], data_wc_fft[1]), axis=0),
#                                              np.concatenate((label_simu[1], label_wc[1]), axis=0),
#                                              data_wc_fft[1:], label_wc[1:], val_rate=0.75)


# def show_data(data, fig_num, ylim):
#
#     _, dim = np.shape(data[0])
#     plt.figure(fig_num)
#     for i in range(4):
#         plt.subplot(4, 1, i + 1)
#         plt.plot(data[0][i + 1800])
#         plt.xlim(0, dim)
#         plt.ylim(ylim)
#
# time_ylim = (-2, 2)
# freq_ylim = (0, 2)
# show_data(data_wc, 1, time_ylim)
# show_data(time_simu, 2, time_ylim)
#
# show_data(data_wc_fft, 3, freq_ylim)
# show_data(freq_simu, 4, freq_ylim)
# show_data(time_simu_freq, 5, freq_ylim)
#
# show_data(data_wc[3:], 6, time_ylim)
# show_data(data_wc_fft[3:], 7, freq_ylim)
#
# plt.show()


print(1)