import torch
from tqdm import trange
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Data(Dataset):
    data = {'letter': '../data/Letter/letter-recognition.data',
            'digits': {'train': '../data/Opt-digits/optdigits.tra', 'test': '../data/Opt-digits/optdigits.tes'},
            'sat': {'train': '../data/Statlog/sat.trn', 'test': '../data/Statlog/sat.tst'},
            'vowel': '../data/vowel/vowel-context.data',
            'iris': '../data/iris/iris.data'}

    def __init__(self, dataset, mode, transform=None):
        if mode == 'train':
            self.X, _, self.y, _ = getattr(self, dataset)()
        elif mode == 'test':
            _, self.X, _, self.y = getattr(self, dataset)()

        self.X, self.y = torch.tensor(self.X, dtype=torch.float), torch.tensor(self.y)
        self.feature_dim = self.X.shape[1]
        self.num_classes = len(np.unique(self.y))
        if transform:
            self.X = transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def letter(self):
        letter = pd.read_csv(self.data['letter'], header=None, index_col=0)
        X = letter.to_numpy()
        y = np.array([ord(n) - 65 for n in letter.index])
        return train_test_split(X, y, test_size=0.2, random_state=1107)

    def digits(self):
        digit_train = np.loadtxt(self.data['digits']['train'], delimiter=',').astype('int')
        digit_test = np.loadtxt(self.data['digits']['test'], delimiter=',').astype('int')
        return digit_train[:, :-1], digit_test[:, :-1], digit_train[:, -1], digit_test[:, -1]

    def sat(self):
        sat_train = np.loadtxt(self.data['sat']['train']).astype('int')
        sat_test = np.loadtxt(self.data['sat']['test']).astype('int')
        y_train = sat_train[:, -1]
        y_train[y_train == 7] = 6
        y_test = sat_test[:, -1]
        y_test[y_test == 7] = 6

        return sat_train[:, :-1], sat_test[:, :-1], y_train - 1, y_test - 1

    def vowel(self):
        with open(self.data['vowel']) as f:
            lines = f.readlines()
        X_train, X_test, y_train, y_test = [], [], [], []
        for line in lines:
            o = line.rstrip().split(' ')
            if o[0] == '0':
                X_train.append([float(n) for n in o[3:13]])
                y_train.append(int(o[-1]))
            elif o[0] == '1':
                X_test.append([float(n) for n in o[3:13]])
                y_test.append(int(o[-1]))
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def iris(self):
        iris = pd.read_csv(self.data['iris'], header=None)
        iris.replace('Iris-setosa', 0, inplace=True)
        iris.replace('Iris-versicolor', 1, inplace=True)
        iris.replace('Iris-virginica', 2, inplace=True)
        X = iris.iloc[:, :4].to_numpy()
        y = iris.iloc[:, 4].to_numpy()
        kf = KFold(n_splits=5, shuffle=True, random_state=1107)
        kf.get_n_splits(iris)
        folds = [(X[train_index], X[test_index], y[train_index], y[test_index])
                 for train_index, test_index in kf.split(iris)]
        return folds


class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(512, out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        x = self.fc4(x)
        return x


def seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    seeds(0)
    num_epoch = 100
    batch_size = 32
    train_set = Data('sat', 'train', F.normalize)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    loss_his, acc_his = [], []

    model = Network(train_set.feature_dim, train_set.num_classes)
    optimizer = SGD(model.parameters(), lr=0.1)
    # schedular = StepLR(optimizer, step_size=200, gamma=0.5)
    pbar = trange(num_epoch)
    for ep in pbar:
        losses, acc = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            data, label = batch
            y = model(data)
            loss = loss_func(y, label)
            acc.append((torch.argmax(y, dim=1) == label).sum().float() / batch_size)
            loss.backward()
            losses.append(loss)
            optimizer.step()

        # schedular.step()
        losses, acc = torch.stack(losses).mean().item(), torch.stack(acc).mean().item()
        loss_his.append(losses)
        acc_his.append(acc)
        pbar.set_postfix({'loss': losses, 'acc': acc})  # , 'lr': schedular.get_last_lr()[0]})

    x = np.arange(num_epoch)
    l1 = plt.plot(x, loss_his, 'r--', label='loss')
    l2 = plt.plot(x, acc_his, 'g--', label='acc')
    plt.plot(x, loss_his, 'ro-', x, acc_his, 'g+-')
    plt.legend()
    plt.show()

    test_set = Data('sat', 'test', transform=F.normalize)
    test_loader = DataLoader(test_set, batch_size=1000, pin_memory=True, shuffle=True)
    model.eval()
    correct = torch.zeros(1)
    total = torch.zeros(1)
    for batch in test_loader:
        data, label = batch
        y_hat = model(data)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += (y_hat == label).sum().float()
        total += len(label)

    print('acc: ', (correct / total).item())
