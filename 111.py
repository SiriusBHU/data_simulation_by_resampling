
strrr = []

with open("111.txt", "r") as fp:
    for i, eachline in enumerate(fp):
        if i % 101 == 1 and i>100:
            strrr.append(eachline.strip())

strrr = [item.split(": ") for item in strrr]
strrr = [item[-1].strip() for item in strrr]
strrr = [(item.split("["))[-1] for item in strrr]
strrr = [(item.split("]"))[0] for item in strrr]
strrr = [(item.split(", ")) for item in strrr]

strrr = [[float(num) for num in item] for item in strrr]

import numpy as np
acc = np.array(strrr)
np.savez("dsr_dncnn_mix_mode1.npz", acc=acc)

acc = acc.reshape(3, 10, 3)
acc = acc.transpose(1, 0, 2)
data = acc.reshape(1, 10, 3, 3)

data = np.concatenate((data[:, :, 0, 1], data[:, :, 0, 2],
                       data[:, :, 1, 0], data[:, :, 1, 2],
                       data[:, :, 2, 0], data[:, :, 2, 1]), axis=1)

data = data.reshape(1, 6, 10)
acc = np.average(data, axis=2)
std = np.std(data, axis=2)

acc_avg = np.average(acc, axis=-1)
std_avg = np.average(std, axis=-1)


print(1)