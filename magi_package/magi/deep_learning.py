import torch
import torch.nn as nn
import torch.nn.functional as tfunc


class BPNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, x_train, x_test, y_train, y_test, epoch):
        super(BPNet, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        self.loss_func = nn.MSELoss()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.epoch = epoch

    def forward(self, x_input):
        y_output = tfunc.sigmoid(self.hidden(x_input))
        y_output = self.out(y_output)
        return y_output

    def bp_test(self, y_output, y_test):
        cor = 0
        result, index = torch.max(y_output.data, 1)
        cor += (index == y_test).sum()
        cor = cor.numpy()
        acc = cor / y_test.size(0)
        return acc

    def bp_train(self):
        for i in range(self.epoch):
            y_train_out = self.forward(self.x_train)
            loss = self.loss_func(y_train_out, self.y_train)
            self.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 100 == 0:
                y_test_out = self.forward(self.x_test)
                accuracy = self.bp_test(y_test_out, self.y_test)
                print("epoch: {}, loss: {:.4f}, accuracy: {:.2f}".format((i + 1), loss.item(), accuracy))

    def predict(self, x_input):
        pre_results = self.forward(x_input)
        return pre_results






