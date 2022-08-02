from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models.nn import Linear, CrossEntropy, ReLu, normalize, SGD
from models.dataloader import Dataset, Dataloader

np.random.seed(0)


class Network:
    def __init__(self, in_dim, out_dim):
        self.fc1 = Linear(in_dim, 512)
        self.relu1 = ReLu()
        self.fc2 = Linear(512, 256)
        self.relu2 = ReLu()
        self.fc3 = Linear(256, out_dim)

    def __call__(self, x: np.ndarray):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    @property
    def state_dict(self):
        params = OrderedDict({layer: param.state_dict
                              for layer, param in self.__dict__.items() if hasattr(param, 'state_dict')})
        return params

    def backward(self, loss_func):
        g = loss_func.backward()
        layers = OrderedDict({k: v for k, v in self.__dict__.items() if hasattr(v, 'backward')})
        grad_table = OrderedDict()
        for k, v in reversed(layers.items()):
            if 'relu' in k:
                g *= v.backward()
            elif 'fc' in k:
                grad_table[k] = {'dw': v.backward(g), 'db': g.mean(axis=0)}
                g = np.dot(g, v.w)
        return grad_table


if __name__ == '__main__':
    num_epoch = 100  # 30, 100
    batch_size = 32
    dataset = 'letter'  # ['letter', 'digits', 'sat']

    train_set = Dataset(dataset, 'train', transform=normalize)
    train_loader = Dataloader(train_set, batch_size=batch_size)

    model = Network(train_set.feature_dim, train_set.num_classes)

    criterion = CrossEntropy(train_set.num_classes)
    optimizer = SGD(model.state_dict, lr=0.01)
    loss_his, acc_his = [], []
    pbar = trange(num_epoch)
    for ep in pbar:
        losses, acc = [], []
        train_loader.reset()
        for batch in train_loader:
            data, label = batch
            y = model(data)
            loss = criterion(y, label)
            acc.append(np.sum(np.argmax(y, axis=1) == label, dtype=float) / batch_size)
            grads = model.backward(loss)
            losses.append(loss.loss)
            optimizer(grads)
        losses, acc = np.array(losses).mean(), np.array(acc).mean()
        loss_his.append(losses)
        acc_his.append(acc)
        pbar.set_postfix({'loss': losses, 'acc': acc})

    axis = np.arange(num_epoch)
    l1 = plt.plot(axis, loss_his, 'r--', label='loss')
    l2 = plt.plot(axis, acc_his, 'g--', label='acc')
    plt.plot(axis, loss_his, 'ro-', axis, acc_his, 'g+-')
    plt.legend()
    plt.show()

    test_set = Dataset(dataset, 'test', transform=normalize)
    test_loader = Dataloader(test_set, batch_size=1000)
    correct, total, loss_his = 0, 0, []
    for batch in test_loader:
        data, label = batch
        y_hat = model(data)
        loss_his.append(criterion(y_hat, label, False))
        y_hat = np.argmax(y_hat, axis=1)
        correct += np.sum(y_hat == label, dtype=float)
        total += len(label)

    print('acc: ', correct / total, 'loss: ', np.array(loss_his).mean())
