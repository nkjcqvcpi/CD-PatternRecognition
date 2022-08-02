from collections import OrderedDict
import numpy as np


def normalize(inp):
    return (inp - np.min(inp)) / (np.max(inp) - np.min(inp))


class CrossEntropy:
    # bce_loss = -np.sum(y * np.log(y_hat) + (np.ones_like(y) - y) * np.log(np.ones_like(y_hat) - y_hat))
    def __init__(self, num_classes, reduction='mean'):
        self.num_classes = num_classes
        self.reduction = reduction
        self.grad = None

    def __call__(self, y_hat: np.ndarray, y: np.ndarray, train=True):
        self.batch_size = y_hat.shape[0]
        y_hat = self.softmax(y_hat)
        if y.ndim < y_hat.ndim:
            y = np.eye(self.num_classes)[y]
        self.y, self.y_hat = y, y_hat
        self.loss = -np.sum(self.y * np.log(self.y_hat))
        if self.reduction == 'mean':
            self.loss /= self.batch_size
        if train:
            return self
        else:
            return self.loss

    def softmax(self, inp):
        return np.exp(inp) / np.repeat(np.expand_dims(np.sum(np.exp(inp), axis=1), 1), self.num_classes, axis=1)

    def backward(self):
        self.grad = self.y_hat - self.y
        return self.grad


class ReLu:
    def __init__(self):
        self.grad = 0

    def __call__(self, inp):
        self.inp = inp
        self.oup = np.maximum(0, self.inp)
        return self.oup

    def backward(self):
        self.grad = np.ones_like(self.oup)
        self.grad[self.inp <= 0] = 0
        return self.grad


class Linear:
    def __init__(self, in_dim, out_dim):
        sqrt_k = np.sqrt(1 / in_dim)
        self.w = np.random.uniform(-sqrt_k, sqrt_k, (out_dim, in_dim))
        self.b = np.random.uniform(-sqrt_k, sqrt_k, (out_dim, ))

    def __call__(self, x):
        self.x = x
        self.y = np.dot(self.x, self.w.T) + self.b
        return self.y

    def backward(self, gr):
        return np.dot(gr.T, self.x)

    @property
    def state_dict(self):
        return {'w': self.w, 'b': self.b}


class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, grads: OrderedDict):
        for layer, grad in grads.items():
            self.params[layer]['w'] -= self.lr * grad['dw']
            self.params[layer]['b'] -= self.lr * grad['db']


if __name__ == '__main__':
    li = Linear(10, 20)
    xx = np.random.random((1, 10))
    yy = li(xx)
    i = 0
