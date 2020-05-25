"""
    Author: Sirius HU
    Created Date: 2018.10.31
"""

import numpy as np
from scipy import fft, ifft


class DataSimulationByReSampling(object):

    def __init__(self, var_r=None, var_load=None, var_noise=None):

        self.mean_r = 1.
        self.mean_load = 1.
        self.mean_noise = 0.

        self.var_r = var_r
        if not self.var_r:
            self.var_r = 0.05

        self.var_load = var_load
        if not self.var_load:
            self.var_load = 0.1

        self.var_noise = var_noise

        # temporary store the re-sampling rate
        self._rates = None

    def __call__(self, signals, labels, expand_num=1):

        return self.dsr_augment(signals, labels, expand_num)

    @staticmethod
    def _check_boundary(digit):
        if digit <= 0:
            return 1.
        else:
            return digit

    def generate_params(self, dim,
                        mean_r, var_r,
                        mean_l, var_l,
                        mean_noise, var_noise):

        # prepare for r, l and noise
        r = np.random.normal(mean_r, var_r)
        r = self._check_boundary(r)
        new_dim = int(r * dim)

        l = np.random.normal(mean_l, var_l)
        l = self._check_boundary(l)

        noise = np.random.normal(mean_noise, var_noise, size=(dim,))

        return new_dim, l, noise

    def dsr_augment(self, signals, labels, expand_num=1):

        if not isinstance(signals, np.ndarray):
            signals = np.array(signals)

        if len(signals.shape) != 2:
            raise AttributeError("expected signals.shape = (num, dim), bot got shape={}"
                                 .format(signals.shape))

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        if len(labels.shape) != 2:
            raise AttributeError("expected labels.shape = (num, ?), bot got shape={}"
                                 .format(labels.shape))

        if labels.shape[0] != signals.shape[0]:
            raise AttributeError("expected number of label = number of signal, bot got %d != %d"
                                 % (labels.shape[0], signals.shape[0]))

        # get the structure info. of signals and labels
        (num, dim), s_type,  = signals.shape, signals.dtype
        classes, l_type = labels.shape[-1], labels.dtype

        # get the re-sampling info.
        mean_r, var_r = self.mean_r, self.var_r
        mean_l, var_l = self.mean_load, self.var_load
        mean_n, var_n = self.mean_noise, self.var_noise
        if not var_n:
            var_n = np.var(signals) / 20

        # allocate memory to store data
        simu_time = np.zeros((num, expand_num, dim), dtype=s_type)
        simu_freq = np.zeros((num, expand_num, dim // 2), dtype=s_type)
        simu_labels = np.zeros((num, expand_num, classes), dtype=l_type)
        self._rates = np.zeros((num, expand_num), dtype=np.float64)

        # augment
        for i, _s in enumerate(signals):
            for j in range(expand_num):

                # get random parameters
                new_dim, l, noise = self.generate_params(dim, mean_r, var_r, mean_l, var_l, mean_n, var_n)

                # two-period re-sampling process
                #   1. first-period re-sampling
                res_s = self.resample(_s, new_dim)
                #   2. calculate fft
                res_f = fft(res_s)
                #   3. second-period re-sampling
                res_res_f = self.resample(res_f, dim)
                #   4. recover the energy of the spectrum
                res_res_f *= np.sqrt(np.sum(_s ** 2) / np.sum(np.abs(res_res_f) ** 2) * dim)
                #   5. add load influence and noise
                res_res_f *= l
                res_res_f += (fft(noise) / np.sqrt(dim))

                # get freq and time representation
                simu_freq[i, j, :] = np.abs(res_res_f[:dim//2]) / np.sqrt(dim // 2)
                simu_time[i, j, :] = ifft(res_res_f).real
                simu_labels[i, j, :] = labels[i, :]
                self._rates[i, j] = dim / new_dim

                # FIXME
                # e_time = np.sum(np.abs(_s) ** 2)
                # e_res_s = np.sum(np.abs(res_s) ** 2)
                # e_res_f = np.sum(np.abs(res_f[:new_dim//2] / np.sqrt(new_dim//2)) ** 2)
                # e_res_res_f = np.sum(np.abs(res_res_f[:dim//2] / np.sqrt(dim//2)) ** 2)
                # # e_simu_f = np.sum(simu_freq[i, j] ** 2)
                # e_simu_t = np.sum(np.abs(simu_time[i, j]) ** 2)
                # print(l**2, e_res_s / e_time, e_res_f/e_time, e_res_res_f/e_time, e_simu_t/e_time)

        simu_time = simu_time.reshape(num * expand_num, dim)
        simu_freq = simu_freq.reshape(num * expand_num, dim//2)
        simu_labels = simu_labels.reshape(num * expand_num, -1)
        self._rates = self._rates.reshape(-1, 1)

        return simu_time, simu_freq, simu_labels

    @staticmethod
    def resample(signal, new_dim):

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        if len(signal.shape) != 1:
            raise AttributeError("the signal has more than 1 axis")

        # get type and dim of signal
        s_type, old_dim = signal.dtype, signal.shape[-1]

        # alloc new signal
        new_s = np.zeros((new_dim,), dtype=s_type)

        # compute the index and weight to calculate the new value
        rate = (old_dim - 1) / (new_dim - 1)
        _ind = rate * np.arange(new_dim - 1).astype(np.float64)
        _weight = _ind - _ind.astype(np.int)
        _ind = _ind.astype(np.int)

        # calculate the new value
        new_s[:-1] = signal.take(_ind) * (1 - _weight) + signal.take(_ind + 1) * _weight
        new_s[-1] = signal[-1]

        return new_s

    def show_rates(self):
        return self._rates


if __name__ == "__main__":

    import time

    # data
    phase = np.random.normal(0.3, 0.1, size=(10, 1024))
    for i in range(phase.shape[-1] - 1):
        phase[:, i + 1] += phase[:, i]

    phase2 = np.random.normal(0.1, 0.05, size=(10, 1024))
    for i in range(phase2.shape[-1] - 1):
        phase2[:, i + 1] += phase2[:, i]
    signals = np.cos(phase) * 10 * np.sin(phase2)
    labels = np.ones((10, ))

    # dsr
    dsr = DataSimulationByReSampling(var_r=0.1, var_noise=0.1, var_load=0.1)
    p1 = time.time()
    simu_time, simu_freq, simu_labels = dsr(signals, labels, expand_num=10)
    p2 = time.time()
    print(p2-p1)

    print(dsr.show_rates()[0])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(221)
    plt.plot(signals[0])
    plt.subplot(222)
    plt.plot(np.abs(fft(signals[0])[:512]) / np.sqrt(1024//2))
    plt.subplot(223)
    plt.plot(simu_time[0])
    plt.subplot(224)
    plt.plot(simu_freq[0])
    plt.show()
    print(1)







