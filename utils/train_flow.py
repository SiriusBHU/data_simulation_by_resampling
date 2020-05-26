import torch
from torch.nn.init import kaiming_uniform_
import time


def weight_init(m):

    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        kaiming_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        kaiming_uniform_(m.weight.data)
    if class_name.find('BatchNorm') != -1:
        torch.nn.init.uniform_(m.weight)


class NetFlow(object):

    def __init__(self,
                 net,
                 loss_func,
                 optimizer,
                 lr_scheduler,
                 epochs,
                 iterations,
                 display_epoch=5):

        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # epoch to display evaluation result
        self.epochs = epochs
        self.display_epoch = display_epoch
        self.iterations = iterations

    def train(self, train_loader, val_loader, is_init=True):

        # set net, optimizer, lr_scheduler
        net = self.net
        loss_func = self.loss_func
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        # initialization
        if is_init:
            net.apply(weight_init)
        if torch.cuda.is_available():
            net.cuda()

        # training period
        LOSS_train, ACC_train, LOSS_val, ACC_val = [], [], [], []
        for epoch in range(self.epochs):

            net.train()
            acc_train, loss_train, t1 = [], [], time.time()
            for i, (samples, labels) in enumerate(train_loader):
                # feed-forward
                if torch.cuda.is_available():
                    samples, labels = samples.cuda(), labels.cuda()
                outputs = net(samples)
                # calculate loss
                _loss = loss_func(outputs, labels)
                # feed-back
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                # calculate accuracy
                if torch.cuda.is_available():
                    outputs = outputs.cpu()
                    labels = labels.cpu()
                prediction = torch.argmax(outputs).long()
                _acc = torch.mean(prediction == labels, dtype=torch.float).item()

                acc_train.append(_acc), loss_train.append(_loss.item())
                print("[iteration: %3d] -- [loss: %.5f] -- [acc: %.5f]" %
                      (i + 1, _loss.item(), _acc))
            # record result and show
            LOSS_train.append(loss_train[:self.iterations])
            ACC_train.append(acc_train[:self.iterations])

            # upgrade learning rate
            lr_scheduler.step(epoch)

            # ------- evaluate the network on validation_loader --------
            net.eval()
            acc_val, loss_val = [], []
            with torch.no_grad():
                for i, (samples, labels) in enumerate(val_loader):
                    # feed-forward
                    if torch.cuda.is_available():
                        samples, labels = samples.cuda(), labels.cuda()
                    outputs = net(samples)
                    # calculate loss
                    _loss = loss_func(outputs, labels)
                    # calculate accuracy
                    if torch.cuda.is_available():
                        outputs = outputs.cpu()
                        labels = labels.cpu()
                    prediction = torch.argmax(outputs).long()
                    _acc = torch.mean(prediction == labels, dtype=torch.float).item()

                    acc_val.append(_acc), loss_val.append(_loss.item())
            LOSS_val.append(loss_val[:self.iterations])
            ACC_val.append(acc_val[:self.iterations])
            print("Epoch: %3d -- Time Consumption: %.5fs\n"
                  "====> Training:  -- [loss: %.5f] -- [acc: %.5f]\n"
                  "====> Validation:-- [loss: %.5f] -- [acc: %.5f]\n" %
                  (epoch + 1, time.time() - t1,
                   sum(loss_train) / len(loss_train), sum(acc_train) / len(acc_train),
                   sum(loss_val) / len(loss_val), sum(acc_val) / len(acc_val)))

        return LOSS_train, ACC_train, LOSS_val, ACC_val

    def evaluation(self, test_loader):
        self.net.eval()
        acc_test, loss_test = [], []
        with torch.no_grad():
            for i, (samples, labels) in enumerate(test_loader):
                # feed-forward
                if torch.cuda.is_available():
                    samples, labels = samples.cuda(), labels.cuda()
                outputs = self.net(samples)
                # calculate loss
                _loss = self.loss_func(outputs, labels)

                # calculate accuracy
                if torch.cuda.is_available():
                    outputs = outputs.cpu()
                    labels = labels.cpu()
                prediction = torch.argmax(outputs).long()
                _acc = torch.mean(prediction == labels, dtype=torch.float).item()
                loss_test.append(_loss.item()), acc_test.append(_acc)
        print("====> Testing -- [loss: %.5f] -- [acc: %.5f]" %
              (sum(loss_test) / len(loss_test), sum(acc_test) / len(acc_test)))


class ClassicMLModelFlow(object):

    def __init__(self, classifier):
        self.clf = classifier

    def train(self, train_set, val_set):

        clf = self.clf
        cur_t = time.time()

        samples, labels = train_set
        clf.fit(samples, labels)
        acc_train = clf.score(samples, labels)

        samples, labels = val_set
        acc_val = clf.score(samples, labels)

        print("Time Consumption: %.5fs\n"
              "====> Training:   [acc: %.5f]\n"
              "====> Validation: [acc: %.5f]\n" %
              (time.time() - cur_t, acc_train, acc_val))
        return

    def evaluation(self, test_set):

        clf = self.clf
        samples, labels = test_set
        acc_test = clf.score(samples, labels)

        print("====> Testing:    [acc: %.5f]\n" % acc_test)



