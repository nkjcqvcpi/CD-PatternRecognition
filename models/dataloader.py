import numpy as np
import numpy.random
import math
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


class Dataset:
    path = '../data/'
    data = {'letter': path + 'Letter/letter-recognition.data',
            'digits': {'train': path + 'Opt-digits/optdigits.tra',
                       'test': path + 'Opt-digits/optdigits.tes'},
            'sat': {'train': path + 'Statlog/sat.trn',
                    'test': path + 'Statlog/sat.tst'},
            'vowel': path + 'vowel/vowel-context.data',
            'iris': path + 'iris/iris.data'}

    def __init__(self, dataset, mode, transform=None):
        """

        :return: X_train, X_test, y_train, y_test
        """
        if mode == 'train':
            self.X, _, self.y, _ = getattr(self, dataset)()
        elif mode == 'test':
            _, self.X, _, self.y = getattr(self, dataset)()
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


class Dataloader:
    def __init__(self, dataset: Dataset, batch_size: int = None):
        self.dataset = dataset
        self.batch_size = batch_size or len(dataset)
        self.batch_index = [np.random.choice(np.arange(self.dataset.__len__()), batch_size, replace=False)
                            for n in range(self.__len__())]

    def __len__(self):
        return math.ceil(self.dataset.__len__() / self.batch_size)

    def __getitem__(self, item):
        return self.dataset[self.batch_index[item]]

    def reset(self):
        self.batch_index = [np.random.choice(np.arange(self.dataset.__len__()), self.batch_size, replace=False)
                            for n in range(self.__len__())]


if __name__ == '__main__':
    d = Dataset('iris', 'train', None)
    dl = Dataloader(d, 64)
    for dt in dl:
        x, y = dt
        i = 0
