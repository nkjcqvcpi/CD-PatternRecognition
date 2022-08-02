import numpy as np


class DA:
    def __init__(self, x, y):
        self.k = np.unique(y)
        self.K = len(self.k)
        self.N = len(y)
        N_k = np.bincount(y)
        self.pi = N_k / self.N

        self.miu = np.zeros(shape=(self.K, x.shape[1]))
        np.add.at(self.miu, y, x)
        self.miu /= N_k[:, None]

    def predict(self, x):
        res = np.argmax(self.delta_k(x), axis=1)
        return np.array(res)

    def score(self, x, y):
        y_pred = self.predict(x)
        score = y == y_pred
        return np.mean(score)

    def delta_k(self, x) -> np.ndarray:
        pass
