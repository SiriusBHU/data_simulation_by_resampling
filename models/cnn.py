import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, in_num, out_num):
        super(CNN, self).__init__()

        self.in_num = in_num
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=16, stride=2, padding=7, bias=False),
            nn.Conv1d(8, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_num // 2, out_num),
            nn.Softmax(dim=1))

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(features.size[0], -1)
        out = self.classifier(features)
        return out







