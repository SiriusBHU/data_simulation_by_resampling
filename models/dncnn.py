import torch.nn as nn
import torch.nn.functional as F


class DNNet(nn.Module):

    def __init__(self, in_num, out_num):
        super(DNNet, self).__init__()

        self.in_num = in_num

        self.conv1 = nn.Conv1d(1, 16, 49, stride=1, padding=0, bias=False)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv2d(1, 16, (16, 21), stride=1, padding=0,  bias=False)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.bn2 = nn.BatchNorm1d(16)

        self.classifier = nn.Sequential(
            nn.Linear((((in_num - 48) // 4 - 20) // 4) * 16, 100),
            nn.Linear(100, out_num),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = x.view(-1, 1, 16, (self.in_num - 48) // 4)
        x = self.conv2(x)
        x = x.view(-1, 16, ((self.in_num - 48) // 4 - 20))
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = x.view(-1, (((self.in_num - 48) // 4 - 20) // 4) * 16)

        x = self.classifier(x)

        return x








