import torch
from torch.utils.data import Dataset
import numpy as np


class DealDateSetFlatten(Dataset):

    def __init__(self, x, y):

        nums, chs, dims = np.shape(x)
        self.data = torch.from_numpy(x.reshape(nums, chs * dims)).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]


class DealDateSetConv1D(Dataset):

    def __init__(self, x, y):

        self.data = torch.from_numpy(x).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]


class DealDateSetConv2D(Dataset):

    def __init__(self, x, y, h, w):

        if not isinstance(h, int) or not isinstance(w, int):
            raise KeyError("h, w should be integer")

        nums, chs, dims = np.shape(x)
        if h * w != dims:
            raise ValueError("dimension of x not match its reshaped param (h, w)")

        self.data = torch.from_numpy(x.reshape(nums, chs, h, w)).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]

