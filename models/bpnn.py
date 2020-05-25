import torch.nn as nn


class BPNet(nn.Module):

    def __init__(self, in_num, hidden_num, out_num):
        super(BPNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_num, hidden_num),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_num, out_num),
        )

    def forward(self, x):
        return self.sequential(x)
