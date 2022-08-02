from .DA import *


class QDA(DA):
    def __init__(self, x, y):
        super().__init__(x, y)

        self.cov, self.cov_i = [], []
        for idx, group in enumerate(self.k):
            Xg = x[y == group, :]
            co = np.zeros(shape=(x.shape[1], x.shape[1]))
            for xi in Xg:
                s = np.atleast_2d(xi - self.miu[idx])
                co += np.dot(s.T, s)
            co /= len(Xg)
            self.cov.append(co)
            try:
                co_i = np.linalg.inv(co)
            except np.linalg.LinAlgError:
                co_i = np.linalg.pinv(co)
            self.cov_i.append(co_i)

        self.cov = np.array(self.cov)
        self.cov_i = np.array(self.cov_i)

        self.bias = np.atleast_2d(-0.5 * np.log(np.linalg.norm(
            self.cov, axis=(1, 2))))
        self.p = np.atleast_2d(np.log(self.pi))

    def delta_k(self, x):
        self.bias = np.atleast_2d(-0.5 * np.log(np.linalg.norm(
            self.cov, axis=(1, 2))))
        self.p = np.atleast_2d(np.log(self.pi))
        wx = []
        for g in x:
            std = np.atleast_2d(g).repeat(self.miu.shape[0], axis=0) \
                  - self.miu
            wx.append(np.diag(np.diagonal(0.5 * np.matmul(np.matmul(
                std, self.cov_i), std.T))))
        delta_k = self.bias.repeat(x.shape[0], axis=0) - \
                  np.array(wx) + self.p.repeat(x.shape[0], axis=0)
        return delta_k
