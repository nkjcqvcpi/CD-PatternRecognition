from .DA import *


class LDA(DA):
    def __init__(self, x, y, diag: bool):
        super().__init__(x, y)

        self.cov = np.zeros(shape=(x.shape[1], x.shape[1]))
        for idx, group in enumerate(self.k):
            Xg = x[y == group, :]
            for xi in Xg:
                s = np.atleast_2d(xi - self.miu[idx])
                self.cov += np.dot(s.T, s)
        self.cov /= (self.N - self.K)

        if diag:
            self.cov = np.diag(np.diag(self.cov))

        try:
            self.i_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self.i_cov = np.linalg.pinv(self.cov)

        self.bias = np.atleast_2d(0.5 * np.diag(np.linalg.multi_dot(
            [self.miu, self.i_cov, self.miu.T])) + np.log(self.pi))

    def delta_k(self, x: np.ndarray):
        self.bias = np.atleast_2d(0.5 * np.diag(np.linalg.multi_dot(
            [self.miu, self.i_cov, self.miu.T])) + np.log(self.pi))
        delta_k = np.linalg.multi_dot([x, self.i_cov, self.miu.T]) - \
                  self.bias.repeat(x.shape[0], axis=0)
        return delta_k
