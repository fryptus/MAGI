import numpy as np
from magi_package.magi.deep_learning import *
from sklearn import datasets

dataset = datasets.load_iris()
data = dataset['data']
iris_type = dataset['target']

np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(iris_type)

inputs = torch.Tensor(data)
labels = torch.LongTensor(iris_type)

class_num = 3
batch_size = labels.shape[0]
labels_fea = labels.view(-1, 1)
temp = torch.zeros(batch_size, class_num)
lbales_fea = temp.scatter_(1, labels_fea, 1)

x_train = inputs[:50]
y_train = lbales_fea[:50]
x_test = inputs[-50:]
y_test = labels[-50:]

bp_net = BPNet(4, 20, 3, x_train, x_test, y_train, y_test, 2000)
bp_net.bp_train()
