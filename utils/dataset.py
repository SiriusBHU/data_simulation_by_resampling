import torch
from torch.utils.data import Dataset


class DealDateSetFlatten(Dataset):

    def __init__(self, x, y):

        self.data = torch.from_numpy(x).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]


class DealDateSetConv1D(Dataset):

    def __init__(self, x, y):

        self.data = torch.from_numpy(x.reshape(-1, 1, x.shape[1])).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]


class DealDateSetConv2D(Dataset):

    def __init__(self, x, y, h, w):

        if not isinstance(h, int) or not isinstance(w, int):
            raise KeyError("h, w should be integer")

        if h * w != x.shape[1]:
            raise ValueError("dimension of x not match its reshaped param (h, w)")

        self.data = torch.from_numpy(x.reshape(-1, 1, h, w)).float()
        self.label = torch.argmax(torch.from_numpy(y), dim=-1).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.size()[0]

