import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DealDateSet(Dataset):

    def __init__(self, x, y):

        classes = np.max(y) + 1
        y = torch.from_numpy(y.reshape(-1, 1)).long()
        self.data = torch.from_numpy(x.reshape(-1, 1, np.shape(x)[1])).float()
        " 注意！！！在 pytorch 自带的 criterion 中，要求 label 为长整型 long()，且为标量(not one_hot)"
        " 而自定义损失函数中, 要求 one_hot coding 而且数据类型为 float() for-mix-up "
        self.label = torch.zeros(y.size()[0], int(classes)).scatter_(1, y, 1)
        self.len = self.label.size()[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


class TINet(nn.Module):

    def __init__(self, in_num, out_num):
        super(TINet, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=8, padding=27, bias=False),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(64 * int(in_num / 64 / 8 / 2), out_num),
            nn.Softmax(dim=1))

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(-1, 64)
        out = self.classifier(features)
        return out


class TICNN(object):

    """
        structure param:
            input dim
            hidden layer dim
            output dim= 10
        hyper param:
            batch size
            learning rate
            optimizer
    """

    def __init__(self, in_num, out_num, batch_size, lr, epoch):

        self.net = TINet(in_num, out_num)
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch

    def data_split(self, x, y, val_rate):

        num, _ = np.shape(x)
        val_num = int(num * val_rate)
        perm = np.arange(num)
        np.random.shuffle(perm)

        return x[perm[val_num:]], y[perm[val_num:]], x[perm[:val_num]], y[perm[:val_num]]

    def data_prepare(self, data, label, test_data, test_label, val_rate):

        import copy
        data, label = copy.deepcopy(data), copy.deepcopy(label)
        x_tra, y_tra, x_val, y_val = self.data_split(data, label, val_rate)

        train_set = DealDateSet(x_tra, y_tra)
        val_set = DealDateSet(x_val, y_val)
        test_set_list = [DealDateSet(test_data[i], test_label[i]) for i in range(len(test_data))]

        train_loader1 = DataLoader(train_set, self.batch_size, shuffle=True)
        train_loader2 = DataLoader(train_set, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, np.shape(x_val)[0])
        test_loader_list = [DataLoader(test_set_list[i], np.shape(test_data[0])[0]) for i in range(len(test_data))]

        return train_loader1, train_loader2, val_loader, test_loader_list

    def criterion(self, outputs, labels):

        prob = torch.log(outputs + 1e-8)
        loss = torch.mean(- labels * prob)

        return loss

    def train(self, data, label, test_data, test_label, val_rate=0.3, mixup=False, a=0.2):

        train_loader1, train_loader2, val_loader, test_loader_list = self.data_prepare(data, label, test_data, test_label, val_rate)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.)  # L2 regularization
        optimizer1 = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.9)  # L2 regularization

        self.net.cuda()
        for step in range(self.epoch):

            self.net.train()
            running_loss, total_num = 0, 0
            Loss_train = []
            for (inputs1, labels1), (inputs2, labels2) in zip(train_loader1, train_loader2):

                inputs1, labels1, inputs2, labels2 = inputs1.cuda(), labels1.cuda(), inputs2.cuda(), labels2.cuda()
                mix_u = np.random.beta(a, a) if mixup else 1.
                inputs = mix_u * inputs1 + (1. - mix_u) * inputs2
                labels = mix_u * labels1 + (1. - mix_u) * labels2
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                if step < 90:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()
                running_loss += loss.item(); total_num += 1

            running_loss = running_loss / total_num * 100
            Loss_train.append(running_loss)

            # ----------- training evaluation -----------
            self.net.eval()
            with torch.no_grad():

                val_loss, val_acc = 0, 0
                for (inputs, labels) in val_loader:

                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.net(inputs)
                    val_loss = (self.criterion(outputs, labels)).item() * 100
                    predicts = torch.argmax(outputs, dim=1)
                    val_acc = np.sum(predicts.cpu().numpy() == (torch.argmax(labels, dim=1)).cpu().numpy()) / inputs.size()[0] * 100

            print("Epoch: %d,  train_loss: %.5f,  val_loss: %.5f,  val_acc: %.3f"
                  % (step, running_loss, val_loss, val_acc))

        # -------------- testing evaluation ---------------
        self.net.eval()
        with torch.no_grad():

            test_loss, test_acc = [], []
            for item in test_loader_list:
                for (inputs, labels) in item:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.net(inputs)
                    test_loss.append((self.criterion(outputs, labels)).item() * 100)
                    predicts = torch.argmax(outputs, dim=1)
                    test_acc.append(np.sum(predicts.cpu().numpy() == (torch.argmax(labels, dim=1)).cpu().numpy()) / inputs.size()[0] * 100)

        print("test_loss: ", test_loss, "  test_acc: ", test_acc)

        return self.net, test_loss, test_acc






